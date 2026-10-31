"""Shared SHA-256 hashing with a per-operation verification ledger.

Design and acceptance: docs/EFFICIENCY_P1_HASHING_PREREG_20260929.md.  Every
file digest this module returns is the SHA-256 of the file's bytes, exactly as
``hashlib.file_digest(open(path, "rb"), "sha256")``; what changes is only how
often the bytes are read.

Four levers:

1. :func:`sha256_many`, :func:`first_mismatch`, :func:`verify_entries` hash
   independent files on a bounded thread pool (default ``min(4, cpu)``).
   Results come back in input order and a failure is reported exactly as a
   serial loop would have reported it: the lowest failing input index wins,
   whatever finished first, and a file that disappears while the batch runs
   fails with the caller's own check (``on_error``), as in the serial loop.
   Repeats of one path inside a batch are hashed once unless the ledger is off.
2. :func:`copy_with_digest` hashes the bytes it copies (and still reads the
   destination back for real when asked to).
3. The run ledger remembers ``path -> digest`` together with the file's
   ``(st_dev, st_ino, st_size, st_mtime_ns)``; inside :func:`run_scope` a later
   :func:`sha256_file` of an unchanged file returns the remembered digest, and
   a stack job hands its entries to the workflow worker that launched it
   (:func:`export_handover` / :func:`import_handover`).  In ``shadow`` mode
   every hit is also hashed for real and compared; a difference raises
   :class:`HashLedgerMismatch`.
4. Digest-while-write (:class:`HashingWriter`, :func:`write_with_digest`,
   :func:`save_npy_with_digest`, :func:`write_bytes_with_digest`,
   :meth:`WriteBlock.written`, :func:`register_written`) records the digest of
   bytes this process wrote, so the usual read-back becomes a ledger hit of kind
   ``write``.

Two recording rules of efficiency P1b-5 (docs/EFFICIENCY_P1B_PREREG_20261001.md,
user decisions U-H1/U-H2) close the gaps P1 left open; neither changes a digest
or a read that produces one:

* :func:`rename` is ``Path.rename`` for a folder (or file) the calling operation
  owns, and carries that operation's ledger entries to the new path - only
  entries that are unchanged, whose file has exactly the recorded stat at the
  new path, and that no :func:`writing` block holds.
* ``fresh=`` on :func:`sha256_file` / :func:`sha256_many` marks files the caller
  has just written and closed.  A real read of such a file that would be a
  "racy" entry is instead sealed as kind ``write`` (modification time moved back
  :data:`MTIME_NUDGE_NS`) in one guarded step: only if the file still has the
  stat taken before the read and no writer touched it since.

Their misses are counted in bytes (``miss_bytes``) and the largest misses of a
scope are listed in :func:`scope_report` (timing notes only).

Switches back to the former behaviour: ``AESTER_HASH_WORKERS=1`` hashes every
batch serially (lever 1).  ``AESTER_HASH_LEDGER`` = ``on`` (default) |
``shadow`` | ``off``: with ``off`` there is no ledger and no handover (lever 3),
and the product's copies and writers take their former code paths (levers 2
and 4: they ask :func:`digest_writes`, which is False with ``off`` and outside
every run scope).  Both together are the former serial behaviour; the cheap
existence and size checks of a batch still run before its hashes, and the
failure reported is the serial loop's.  The primitives of this module keep
working when they are called directly, in any mode.

Scopes belong to the context that opened them
----------------------------------------------
:func:`run_scope` binds a ledger to the current :mod:`contextvars` context
(the thread, or the asyncio task).  Other threads of the process see no
ledger - they read every file exactly as today - unless the work is handed to
them with :func:`carry_scope`.  The thread pools of this module carry the
caller's scope themselves.  Invalidation (:func:`writing`,
:func:`invalidate`) is process-wide: it drops the entries of every open scope.

What the ledger cannot see, and how each case is closed (NTFS, measured
2026-09-29, see the P1 notes):

* Writes through a memory map (``np.memmap`` ``r+``, astropy ``mode="update"``
  with ``memmap=True``, ``tifffile.memmap``) never change the modification
  time, ``os.utime`` can restore it, and ``CopyFile2`` (``shutil.copy2``) onto
  an existing file keeps the file id and copies the source's time.  Every
  in-place writer therefore wraps its write in :func:`writing` (or calls
  :func:`invalidate`).  A digest is only recorded when no :func:`writing`
  block or :func:`invalidate` touched the file (by path or by file id) at any
  time between the stat taken before the hash and the recording, and none is
  open at the recording.
* A plain rewrite (``"wb"``, ``write_bytes``) within one clock tick of the
  recorded write keeps size and modification time.  For an entry recorded by
  reading, the racy rule (as in git) catches it: an entry whose modification
  time is within :data:`RACY_SECONDS` of the moment hashing started is
  re-hashed on its next lookup.  An entry of kind ``write`` is exempt from the
  racy rule, because before recording it this module moves the file's
  modification time back by :data:`MTIME_NUDGE_NS` (100 ns, one NTFS unit):
  any later write stamps the current time, which is never that value, so the
  stat no longer matches.

The ledger is only used on local NTFS/ReFS volumes with non-zero file ids;
anywhere else every lookup misses and the file is read.

Messages raised here are developer diagnostics (shadow mode, API misuse); the
user-facing messages stay with the callers, unchanged.  The workers record
:func:`scope_report` (reads, hits, shadow checks and mismatches of their run
scope) in their timing notes as ``hash_ledger``.
"""
from __future__ import annotations

from collections import namedtuple
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
import contextvars
import functools
import hashlib
import heapq
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import threading
import time

from . import profiling

VERSION = "aester-hashing-3"
HANDOVER_VERSION = "aester-hash-handover-2"
CHUNK = 8 << 20
DEFAULT_MAX_WORKERS = 4
MAX_WORKERS = 32
# NTFS stamps a write with the coarse system time (one tick, at most 15.625 ms;
# measured: identical (size, mtime, id) for same-size rewrites up to 8 ms apart,
# never at 12 ms or more).  Half a second is 32 ticks.
RACY_SECONDS = 0.5
# Before a kind "write" entry is recorded the file's modification time is moved back by this much
# (one NTFS time unit), a value no later write can stamp.
MTIME_NUDGE_NS = 100
# Between completions of a parallel batch the caller's check() runs at least this often.
CHECK_INTERVAL_SECONDS = 0.2
# A batch whose files add up to less than this is hashed serially (threads only cost time there).
PARALLEL_MIN_BYTES = 8 << 20
# Beyond one file per worker, more small files may be queued while their total stays under this;
# it keeps the workers busy between the caller's wake-ups and bounds how much is read after a cancel.
INFLIGHT_BYTES = 256 << 20
INFLIGHT_PER_WORKER = 8
# Arrays below this many bytes in total are hashed serially by arrays_digest_many.
PARALLEL_MIN_ARRAY_BYTES = 8 << 20
# How many of a scope's largest ledger misses scope_report() lists (timing notes only).
LARGEST_MISSES = 20
MISS_STATUSES = ("no_entry", "stale", "racy", "busy")

TRUST_LEDGER = "ledger"
TRUST_REAL = "real"
_TRUSTS = (TRUST_LEDGER, TRUST_REAL)
MODES = ("on", "shadow", "off")
KINDS = ("ledger", "handover", "write")            # kinds of ledger entries
SAVED_KINDS = KINDS + ("batch",)                    # kinds of saved reads (batch: repeat inside one batch)
ENV_LEDGER = "AESTER_HASH_LEDGER"
ENV_WORKERS = "AESTER_HASH_WORKERS"

# A launcher imports its handover moments after the job ends and every launch has a fresh token, so
# a handover file older than this was left by a launcher that was stopped while its job ran.
HANDOVER_MAX_AGE_SECONDS = 24 * 3600

_HEX = frozenset("0123456789abcdef")
_DRIVE_REMOVABLE, _DRIVE_FIXED = 2, 3
_LEDGER_FILE_SYSTEMS = ("NTFS", "REFS")


class HashLedgerMismatch(RuntimeError):
    """Shadow mode: a ledger hit disagreed with the real digest of the file."""

    def __init__(self, path, source, kind, ledger_digest, real_digest):
        self.path, self.source, self.kind = str(path), source, kind
        self.ledger_digest, self.real_digest = ledger_digest, real_digest
        super().__init__(f"hash ledger mismatch ({kind}) for {self.path} [{source}]: "
                         f"ledger {ledger_digest} != real {real_digest}")


CopyResult = namedtuple("CopyResult", "src_sha256 dst_sha256 size")


class _Entry:
    __slots__ = ("digest", "sig", "recorded_at_ns", "kind", "source", "exempt")

    def __init__(self, digest, sig, recorded_at_ns, kind, source, exempt):
        self.digest, self.sig, self.recorded_at_ns = digest, sig, recorded_at_ns
        self.kind, self.source, self.exempt = kind, source, exempt


class _Scope:
    """One ledger: opened by the outermost run_scope of a context, closed when its last user leaves."""
    __slots__ = ("label", "mode", "depth", "entries", "by_id", "closed", "largest")

    def __init__(self, label, mode):
        self.label, self.mode, self.depth = label, mode, 1
        self.entries = {}           # normalized path -> _Entry
        self.by_id = {}             # (st_dev, st_ino) -> set of normalized paths
        self.closed = False
        # The LARGEST_MISSES largest ledger misses of this scope, a min-heap of
        # (bytes, status, source, last two path components); kept here, not in _counters (P1b-5c).
        self.largest = []


class _Aborted(Exception):
    """Internal: a parallel item above the lowest known failure stops reading."""


class _AfterFailed(Exception):
    """Internal: carries the exception of a batch's ``after`` callback out of a worker thread."""

    def __init__(self, error):
        super().__init__(error)
        self.error = error


# --------------------------------------------------------------------------- #
# Process state                                                                #
# --------------------------------------------------------------------------- #

_lock = threading.RLock()
_scope_var = contextvars.ContextVar("starstack_hashing_scope", default=None)
# (label, counters at entry) of every run_scope entered in this context, innermost last (scope_report).
_marks_var = contextvars.ContextVar("starstack_hashing_marks", default=())
_live = set()               # open scopes (invalidation reaches all of them)
_busy = {}                  # normalized path -> open writing() count
_busy_ids = {}              # (st_dev, st_ino) -> open writing() count
# Invalidation clock: every writing() enter/exit and invalidate() advances it.  While anyone holds a
# snapshot (a hash in progress that may be recorded, an open writing() block), the clock value of the
# latest invalidation of each path and file id is kept, so a record can tell whether its file was
# touched since its snapshot.  With no holder the map is empty (every later snapshot is newer).
_clock = 0
_holders = 0
_touched = {}               # ("k", key) | ("i", file id) -> clock value of the latest touch
_exports = {}               # normalized handover path -> normalized root (exported by this process)
_volume_by_dev = {}         # st_dev -> bool
_volume_by_root = {}        # volume root -> {"fs", "drive_type", "eligible"}
_counters = {}


def _reset_counters():
    _counters.clear()
    _counters.update({
        "real_bytes": 0, "real_files": 0,
        "saved": {kind: [0, 0] for kind in SAVED_KINDS},
        "shadow_checks": 0, "shadow_bytes": 0, "shadow_mismatches": 0, "shadow_concurrent": 0,
        "miss_no_entry": 0, "miss_stale": 0, "miss_racy": 0, "miss_busy": 0,
        "miss_bytes": {status: 0 for status in MISS_STATUSES},     # a dict of ints: _difference recurses
        "recorded": {kind: 0 for kind in KINDS},
        "refused_unstable": 0, "refused_volume": 0, "refused_busy": 0, "refused_touched": 0,
        "refused_seal": 0, "invalidated": 0, "parallel_batches": 0,
        "renamed_moved": 0, "renamed_dropped": 0, "fresh_sealed": 0, "fresh_ledger": 0, "fresh_refused": 0,
        "handover_exported": 0, "handover_revoked": 0, "handover_rejected": 0, "handover_pruned": 0,
        "memory_bytes": 0, "memory_calls": 0,
    })


_reset_counters()


def _count(key, amount=1):
    with _lock:
        _counters[key] += amount


def _counters_copy():
    with _lock:
        return json.loads(json.dumps(_counters))


# --------------------------------------------------------------------------- #
# Small helpers                                                                #
# --------------------------------------------------------------------------- #

def _sig(st):
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)


def _stat(path):
    try:
        return os.stat(path)
    except (OSError, ValueError, TypeError):
        return None


def _resolve(path, how):
    """``how(path)`` on the path as str; None when the path cannot be resolved."""
    try:
        value = os.fspath(path)
        return how(os.fsdecode(value) if isinstance(value, bytes) else value)
    except (OSError, ValueError, TypeError):
        return None


def _absolute(path):
    """Batch de-duplication key: the absolute path as spelled (no case folding, no link resolution)."""
    return _resolve(path, os.path.abspath)


def _norm(path):
    """Ledger key: normcase(realpath(path)); None when the path cannot be normalized."""
    return _resolve(path, lambda value: os.path.normcase(os.path.realpath(value)))


def _is_digest(value):
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX


def _check_trust(trust):
    if trust not in _TRUSTS:
        raise ValueError(f"hashing: trust must be one of {_TRUSTS}, not {trust!r}")


_TOKEN = re.compile(r"[A-Za-z0-9_-]{8,128}")


def _check_token(token):
    if not isinstance(token, str) or not _TOKEN.fullmatch(token):
        raise ValueError("hashing: a handover token is 8-128 characters from [A-Za-z0-9_-]")


HANDOVER_DIRECTORY = "aester-hash-handover"


def handover_path(token):
    """Default handover file for ``token``: ``<system temp>/aester-hash-handover/<token>.json``.

    Outside every cache, work, product and delivery folder, so no listing,
    size count or cleanup of those folders ever sees it.
    """
    _check_token(token)
    return Path(tempfile.gettempdir()) / HANDOVER_DIRECTORY / f"{token}.json"


# A handover file and the temporary it is written through (export_handover).
_HANDOVER_NAME = re.compile(r"[A-Za-z0-9_-]{8,128}\.json(?:\.\d+\.\d+\.tmp)?")


def prune_handovers(max_age_seconds=HANDOVER_MAX_AGE_SECONDS):
    """Delete handover files older than ``max_age_seconds`` from the default handover folder.

    A handover is normally consumed moments after its job ends; one is left
    behind when its launcher was stopped while the job ran (a resumed launch
    uses a new token, so nothing would ever read it).  Only regular files named
    like handovers (or their temporaries) are touched.  Best effort: returns
    how many were deleted, never raises.
    """
    removed = 0
    try:
        folder = Path(tempfile.gettempdir()) / HANDOVER_DIRECTORY
        limit = time.time() - float(max_age_seconds)
        with os.scandir(folder) as entries:
            for entry in entries:
                try:
                    if (_HANDOVER_NAME.fullmatch(entry.name) and entry.is_file(follow_symlinks=False)
                            and entry.stat(follow_symlinks=False).st_mtime < limit):
                        os.remove(entry.path)
                        removed += 1
                except OSError:
                    continue
    except Exception:
        pass
    if removed:
        _count("handover_pruned", removed)
    return removed


def _env_mode():
    value = os.environ.get(ENV_LEDGER)
    if value is None or not value.strip():
        return "on"
    value = value.strip().lower()
    return value if value in MODES else "off"   # an unknown value is the safe choice: today's behaviour


def resolve_workers(workers=None):
    """Worker count for a parallel batch: ``AESTER_HASH_WORKERS`` if set, else ``workers``, else min(4, cpu)."""
    env = os.environ.get(ENV_WORKERS)
    if env is not None and env.strip():
        try:
            value = int(env.strip())
        except ValueError:
            value = None
        if value is not None and value >= 1:
            return min(value, MAX_WORKERS)
    if workers is None:
        return max(1, min(DEFAULT_MAX_WORKERS, os.cpu_count() or 1))
    return max(1, min(int(workers), MAX_WORKERS))


def _file_id(st):
    return None if st is None else (st.st_dev, st.st_ino)


def _under(key, root_key):
    prefix = root_key if root_key.endswith(os.sep) else root_key + os.sep
    return key.startswith(prefix)


# --------------------------------------------------------------------------- #
# Scopes                                                                       #
# --------------------------------------------------------------------------- #

def _current():
    """The open scope of this context, or None."""
    scope = _scope_var.get()
    return scope if scope is not None and not scope.closed else None


def _live_mode(scope):
    """Mode of ``scope`` now ("off" for None or a closed scope)."""
    return "off" if scope is None or scope.closed else scope.mode


def _close(scope):
    """Lock held."""
    scope.closed = True
    scope.entries.clear()
    scope.by_id.clear()
    scope.largest.clear()
    _live.discard(scope)


@contextmanager
def run_scope(label, *, mode=None):
    """Enable a run ledger for one operation in the current context (re-entrant).

    The outermost scope of a context creates the ledger: it reads
    ``AESTER_HASH_LEDGER`` (or ``mode``) when it starts; the ledger is dropped
    when the last user leaves.  Nested scopes (in the same context, or in a
    thread given the scope by :func:`carry_scope`) share it, and their ``mode``
    and ``label`` are ignored.  Other threads see no ledger.
    """
    if mode is not None and mode not in MODES:
        raise ValueError(f"hashing: mode must be one of {MODES}, not {mode!r}")
    scope = _current()
    token = None
    with _lock:
        if scope is not None and not scope.closed:
            scope.depth += 1
        else:
            scope = _Scope(str(label), mode or _env_mode())
            _live.add(scope)
            token = _scope_var.set(scope)
        mark = _marks_var.set(_marks_var.get() + ((str(label), _counters_copy()),))
    try:
        yield
    finally:
        try:
            _marks_var.reset(mark)
        except ValueError:                # left in another context than it was entered in
            _marks_var.set(())
        if token is not None:
            try:
                _scope_var.reset(token)
            except ValueError:            # left in another context than it was entered in
                _scope_var.set(None)
        with _lock:
            scope.depth -= 1
            if scope.depth == 0:
                _close(scope)


def carry_scope(fn):
    """``fn`` bound to the scope of the calling context (or to no scope), for running in another thread.

    Use it when an operation inside :func:`run_scope` hands hashing work to a
    thread pool or thread of its own (``executor.submit(carry_scope(work),
    ...)``).  The wrapped callable may run concurrently in several threads.  If
    the scope has closed by the time it runs, it runs without a ledger.
    """
    scope = _current()

    @functools.wraps(fn)
    def carried(*args, **kwargs):
        token = _scope_var.set(scope)
        try:
            return fn(*args, **kwargs)
        finally:
            _scope_var.reset(token)
    return carried


def active_mode():
    """The ledger mode of the calling context: "on", "shadow", or "off" (also outside any scope)."""
    return _live_mode(_current())


def digest_writes():
    """Whether product copies and writers use levers 2 and 4 here (hash-while-copy, digest-while-write).

    True inside a run scope whose mode is "on" or "shadow".  With
    ``AESTER_HASH_LEDGER=off``, and outside every run scope (where no ledger
    could use the digest), they keep their former code paths.
    """
    return active_mode() != "off"


# --------------------------------------------------------------------------- #
# Volume eligibility                                                           #
# --------------------------------------------------------------------------- #

def _probe_root(path):
    """(root, file system name, drive type) of the volume holding ``path``; None when unknown."""
    if os.name != "nt":
        return None
    try:
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        path_name = kernel.GetVolumePathNameW
        path_name.restype, path_name.argtypes = wintypes.BOOL, [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
        size = 32768
        root = ctypes.create_unicode_buffer(size)
        if not path_name(path, root, size):
            return None
        root = root.value
        with _lock:
            known = _volume_by_root.get(root)
        if known is not None:
            return root, known["fs"], known["drive_type"]
        info = kernel.GetVolumeInformationW
        info.restype = wintypes.BOOL
        info.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD, wintypes.LPDWORD, wintypes.LPDWORD,
                         wintypes.LPDWORD, wintypes.LPWSTR, wintypes.DWORD]
        fs = ctypes.create_unicode_buffer(261)
        if not info(root, None, 0, None, None, None, fs, 261):
            return None
        drive = kernel.GetDriveTypeW
        drive.restype, drive.argtypes = wintypes.UINT, [wintypes.LPCWSTR]
        return root, fs.value, int(drive(root))
    except Exception:
        return None


def _volume_eligible(path, st):
    """True for a local NTFS/ReFS volume with non-zero file ids; cached per st_dev and per root."""
    if not st.st_ino:
        return False
    with _lock:
        known = _volume_by_dev.get(st.st_dev)
    if known is not None:
        return known
    probe = _probe_root(path)
    if probe is None:
        eligible, root = False, None
    else:
        root, fs, drive_type = probe
        eligible = fs.upper() in _LEDGER_FILE_SYSTEMS and drive_type in (_DRIVE_FIXED, _DRIVE_REMOVABLE)
    with _lock:
        _volume_by_dev[st.st_dev] = eligible
        if probe is not None:
            _volume_by_root.setdefault(root, {"fs": probe[1], "drive_type": probe[2], "eligible": eligible})
    return eligible


# --------------------------------------------------------------------------- #
# Invalidation clock                                                           #
# --------------------------------------------------------------------------- #

def _hold():
    """Take a snapshot of the invalidation clock and keep touches recorded until _release()."""
    global _holders
    with _lock:
        _holders += 1
        return _clock


def _release():
    global _holders
    with _lock:
        _holders -= 1
        if not _holders:
            _touched.clear()


def _bump(key, file_ids):
    """Advance the clock and mark ``key`` and ``file_ids`` touched now (lock held); returns the new value."""
    global _clock
    _clock += 1
    if _holders:
        if key is not None:
            _touched[("k", key)] = _clock
        for file_id in file_ids:
            if file_id is not None:
                _touched[("i", file_id)] = _clock
    return _clock


def _touched_since(key, file_ids, snapshot):
    """Lock held: whether ``key`` or any of ``file_ids`` was touched after ``snapshot``."""
    if key is not None and _touched.get(("k", key), 0) > snapshot:
        return True
    return any(file_id is not None and _touched.get(("i", file_id), 0) > snapshot for file_id in file_ids)


def _is_busy(key, file_ids):
    """Lock held: whether a writing() block is open for ``key`` or any of ``file_ids``."""
    return (key is not None and key in _busy) or any(file_id in _busy_ids for file_id in file_ids
                                                      if file_id is not None)


# --------------------------------------------------------------------------- #
# Ledger entries                                                               #
# --------------------------------------------------------------------------- #

def _drop(scope, key):
    """Remove one entry (lock held)."""
    entry = scope.entries.pop(key, None)
    if entry is not None:
        paths = scope.by_id.get(entry.sig[:2])
        if paths is not None:
            paths.discard(key)
            if not paths:
                del scope.by_id[entry.sig[:2]]
    return entry


def _store(scope, key, entry):
    """Insert one entry (lock held)."""
    _drop(scope, key)
    scope.entries[key] = entry
    scope.by_id.setdefault(entry.sig[:2], set()).add(key)


def _forget(key, file_ids):
    """Drop the entry of ``key`` and every entry of the given file ids, in every open scope (lock held)."""
    removed = 0
    for scope in _live:
        before = len(scope.entries)
        if key is not None:
            _drop(scope, key)
        for file_id in file_ids:
            if file_id is not None:
                for other in list(scope.by_id.get(file_id, ())):
                    _drop(scope, other)
        removed += before - len(scope.entries)
    _counters["invalidated"] += removed


def _tail(key):
    """The last two components of a ledger key ("folder/name"): enough to tell a missed file, no more."""
    head, name = os.path.split(key)
    parent = os.path.basename(head)
    return f"{parent}/{name}" if parent else name


def _miss(scope, status, st, source, key):
    """Count one ledger miss of ``status`` (lock held): its bytes, and the scope's largest misses."""
    _counters["miss_" + status] += 1
    size = st.st_size
    _counters["miss_bytes"][status] += size
    item = (size, status, str(source), _tail(key))
    if len(scope.largest) < LARGEST_MISSES:
        heapq.heappush(scope.largest, item)
    elif item > scope.largest[0]:
        heapq.heapreplace(scope.largest, item)


def _lookup(scope, key, st, source=None):
    """(entry, status) for the file now at ``key`` with stat ``st``; status is hit/no_entry/stale/racy/busy."""
    with _lock:
        if _live_mode(scope) == "off":
            return None, "no_entry"
        if _is_busy(key, ((st.st_dev, st.st_ino),)):
            _miss(scope, "busy", st, source, key)
            return None, "busy"
        entry = scope.entries.get(key)
        if entry is None:
            _miss(scope, "no_entry", st, source, key)
            return None, "no_entry"
        if entry.sig != _sig(st):
            _drop(scope, key)
            _miss(scope, "stale", st, source, key)
            return None, "stale"
        if not entry.exempt and entry.recorded_at_ns - entry.sig[3] < int(RACY_SECONDS * 1e9):
            _miss(scope, "racy", st, source, key)
            return None, "racy"
    if not _volume_eligible(key, st):
        return None, "no_entry"
    return entry, "hit"


def _record(scope, key, st_before, st_after, digest, kind, source, recorded_at_ns, snapshot):
    """Remember ``digest`` for ``key`` if the file did not change and was not touched since ``snapshot``."""
    if key is None or st_before is None or st_after is None or _live_mode(scope) == "off":
        return False
    if _sig(st_before) != _sig(st_after):
        _count("refused_unstable")
        return False
    if not _volume_eligible(key, st_after):
        _count("refused_volume")
        return False
    file_ids = ((st_after.st_dev, st_after.st_ino),)
    with _lock:
        if _live_mode(scope) == "off":
            return False
        if _is_busy(key, file_ids):
            _counters["refused_busy"] += 1
            return False
        if _touched_since(key, file_ids, snapshot):
            _counters["refused_touched"] += 1
            return False
        _store(scope, key, _Entry(digest, _sig(st_after), recorded_at_ns, kind, source, kind == "write"))
        _counters["recorded"][kind] += 1
    return True


# --------------------------------------------------------------------------- #
# Handover revocation                                                          #
# --------------------------------------------------------------------------- #

def _revoke_file(target):
    """Make a handover file unusable: delete it (or, if that fails, truncate it; else raise)."""
    revoked = True
    try:
        os.remove(target)
    except FileNotFoundError:
        revoked = False
    except OSError:
        with open(target, "wb"):
            pass
    with _lock:
        _exports.pop(_norm(target), None)
        if revoked:
            _counters["handover_revoked"] += 1


def _revoke_handovers(key):
    """Revoke every handover this process exported whose root contains ``key``."""
    if key is None or not _exports:
        return
    with _lock:
        targets = [target for target, root in _exports.items() if key == root or _under(key, root)]
    for target in targets:
        _revoke_file(target)


# --------------------------------------------------------------------------- #
# In-place writers                                                             #
# --------------------------------------------------------------------------- #

def invalidate(path):
    """Forget what every open ledger knows about ``path`` and about the file now at it (all its hard links).

    Also revokes handover files this process exported for a root containing
    ``path``.
    """
    key = _norm(path)
    file_id = _file_id(_stat(path))
    _revoke_handovers(key)
    with _lock:
        _bump(key, (file_id,))
        _forget(key, (file_id,))


def _seal(path, key, expect=None):
    """Nudge the modification time of a file just written back by MTIME_NUDGE_NS; its stat, or None.

    ``expect``: a stat the file must still have exactly (compared with the stat
    this nudge starts from, so the check and the nudge use one stat).
    """
    st = _stat(path)
    if st is None or not _volume_eligible(key, st):
        return None
    if expect is not None and _sig(st) != _sig(expect):
        return None
    target = st.st_mtime_ns - MTIME_NUDGE_NS
    try:
        os.utime(path, ns=(st.st_atime_ns, target))
    except OSError:
        return None
    sealed = _stat(path)
    if (sealed is None or sealed.st_mtime_ns != target
            or (sealed.st_dev, sealed.st_ino, sealed.st_size) != (st.st_dev, st.st_ino, st.st_size)):
        return None
    return sealed


class WriteBlock:
    """Yielded by :func:`writing`; call :meth:`written` inside the block to record the written digest."""
    __slots__ = ("path", "_digest", "_source", "registered")

    def __init__(self, path):
        self.path = path
        self._digest = self._source = None
        self.registered = False     # set when the block ends: whether a digest was recorded

    def written(self, digest, source):
        """Declare that, when the block ends, the file holds exactly the bytes with SHA-256 ``digest``.

        Recorded as kind ``write`` when the block ends without an exception,
        inside a run scope that is not "off", on a ledger volume, and only if
        no other :func:`writing` block or :func:`invalidate` touched the file
        while this block was open.
        """
        if not _is_digest(digest):
            raise ValueError(f"hashing: not a lowercase hex SHA-256 digest: {digest!r}")
        self._digest, self._source = digest, source


@contextmanager
def writing(path):
    """Wrap every in-place write (memmap, ``r+``, update mode, ``os.replace`` target, copy destination).

    Entries of the path and of the file at it are dropped from every open
    ledger on entry and on exit, handover files this process exported for a
    root containing the path are revoked on entry, and while the block runs
    lookups of the file miss and nothing about it is recorded.  Yields a
    :class:`WriteBlock`.  Works (and must be used) outside run scopes too.
    """
    with _writing(path, _current()) as block:
        yield block


@contextmanager
def _writing(path, scope, *, since=None, expect=None, revoke=True):
    """The body of :func:`writing` for an explicit ``scope`` (a fresh seal may run in a pool thread).

    With ``since`` (an invalidation-clock snapshot taken before the caller read
    the file) the block refuses to record its digest when a writing() block or
    invalidate() touched the path or the file (now, or the one with ``expect``'s
    file id) after ``since``, or one is open; this is checked under the same
    lock acquisition that marks the block busy, before its own touch.  With
    ``expect`` the seal happens only if the file still has exactly that stat.
    ``revoke=False`` skips revoking exported handovers (a nudge can only make
    an exported row stale).  ``writing(path)`` is ``_writing(path, _current())``.
    """
    key = _norm(path)
    file_id = _file_id(_stat(path))
    block = WriteBlock(path)
    if revoke:
        _revoke_handovers(key)
    stamp = _hold()
    refused = False
    try:
        with _lock:
            if since is not None:
                ids = (file_id, _file_id(expect))
                if _is_busy(key, ids) or _touched_since(key, ids, since):
                    refused = True
                    _counters["refused_touched"] += 1
            if key is not None:
                _busy[key] = _busy.get(key, 0) + 1
            if file_id is not None:
                _busy_ids[file_id] = _busy_ids.get(file_id, 0) + 1
            stamp = _bump(key, (file_id,))
            _forget(key, (file_id,))
        completed = False
        try:
            yield block
            completed = True
        finally:
            sealed = None
            if (completed and not refused and block._digest is not None and _live_mode(scope) != "off"
                    and key is not None):
                sealed = _seal(path, key, expect)
                if sealed is None:
                    _count("refused_seal")
            now_id = _file_id(_stat(path))       # a replaced or newly created file has another id
            with _lock:
                if key is not None:
                    _busy[key] -= 1
                    if not _busy[key]:
                        del _busy[key]
                if file_id is not None:
                    _busy_ids[file_id] -= 1
                    if not _busy_ids[file_id]:
                        del _busy_ids[file_id]
                contended = (_is_busy(key, (file_id, now_id)) or _touched_since(key, (file_id, now_id), stamp))
                if sealed is not None and contended:
                    _counters["refused_touched"] += 1
                end = _bump(key, (file_id, now_id))
                _forget(key, (file_id, now_id))
            if sealed is not None and not contended:
                block.registered = _record(scope, key, sealed, _stat(path), block._digest, "write",
                                           block._source, time.time_ns(), end)
    finally:
        _release()


def register_written(path, digest, source):
    """Record that ``path`` (closed now) holds exactly the bytes whose digest is ``digest``.

    Same as ``with writing(path) as block: block.written(digest, source)``
    (the modification time is nudged back :data:`MTIME_NUDGE_NS`).  Call it
    right after closing a file this thread wrote sequentially.  Returns whether
    it was recorded (never outside a run scope, in "off" mode or on a volume
    without a ledger).  Prefer calling :meth:`WriteBlock.written` inside the
    ``writing()`` block that did the write.
    """
    if not _is_digest(digest):
        raise ValueError(f"hashing: not a lowercase hex SHA-256 digest: {digest!r}")
    with writing(path) as block:
        block.written(digest, source)
    return block.registered


# --------------------------------------------------------------------------- #
# Rename                                                                       #
# --------------------------------------------------------------------------- #

def rename(src, dst):
    """``Path(src).rename(dst)`` for a folder or file the calling operation owns, carrying the ledger along.

    Exactly that call (``os.rename`` underneath; same exceptions, same
    filesystem effect, same return value), so a caller's ``stage.rename(target)``
    can become ``hashing.rename(stage, target)``.  When the rename succeeded and an open ledger
    ("on"/"shadow", in any context) holds entries at ``src`` or under it, each
    is moved to the matching path under ``dst`` only if (1) it is still the
    same entry (no writing(), invalidate() or record replaced or dropped it),
    (2) the file at the new path has exactly the recorded (st_dev, st_ino,
    st_size, st_mtime_ns), and (3) no writing() block is open for the new
    path, the old path or the file.  Every other such entry is dropped.  Kind,
    digest, recording time and racy exemption are kept (a racy entry stays
    racy).  Nothing is raised after a successful rename (an internal failure
    drops the entries).
    """
    source = Path(src)
    with _lock:
        carried = any(_live_mode(scope) != "off" for scope in _live)
    old_key = _norm(source) if carried else None
    renamed = source.rename(dst)
    if old_key is not None:
        try:
            _carry(old_key, dst)
        except Exception:
            try:
                _drop_under(old_key)
            except Exception:
                pass
    return renamed


def _carry(old_key, dst):
    """Move the entries under ``old_key`` to the same relative paths under ``dst`` (see :func:`rename`)."""
    with _lock:
        candidates = [(scope, key, entry) for scope in _live if _live_mode(scope) != "off"
                      for key, entry in scope.entries.items() if key == old_key or _under(key, old_key)]
    if not candidates:
        return
    new_root = _norm(dst)
    stats = {}
    for _, key, _ in candidates:
        if key not in stats:
            stats[key] = None if new_root is None else _stat(new_root + key[len(old_key):])
    moved = dropped = 0
    with _lock:
        for scope, key, entry in candidates:
            if scope.entries.get(key) is not entry:
                dropped += 1
                continue
            _drop(scope, key)
            new_key = None if new_root is None else new_root + key[len(old_key):]
            st = stats[key]
            file_ids = (entry.sig[:2],)
            if (new_key is not None and st is not None and _sig(st) == entry.sig and _live_mode(scope) != "off"
                    and not _is_busy(new_key, file_ids) and not _is_busy(key, file_ids)):
                _store(scope, new_key, entry)
                moved += 1
            else:
                dropped += 1
        _counters["renamed_moved"] += moved
        _counters["renamed_dropped"] += dropped


def _drop_under(old_key):
    """Drop every entry at or under ``old_key`` in every open ledger."""
    with _lock:
        for scope in _live:
            for key in [key for key in scope.entries if key == old_key or _under(key, old_key)]:
                _drop(scope, key)
                _counters["renamed_dropped"] += 1


# --------------------------------------------------------------------------- #
# Single file                                                                  #
# --------------------------------------------------------------------------- #

_buffers = threading.local()


@contextmanager
def _chunk_buffer():
    """This thread's 8 MiB chunk buffer (allocating and zeroing one per file costs ~1 ms)."""
    buffer = getattr(_buffers, "buffer", None)
    _buffers.buffer = None                 # a re-entrant use (from check()) gets its own
    if buffer is None:
        buffer = bytearray(CHUNK)
    try:
        yield buffer
    finally:
        _buffers.buffer = buffer


def _read_digest(path, source, check=None, abort=None):
    """Real read: hex digest of the whole file, counted through profiling.note_hashed."""
    with open(path, "rb", buffering=0) as handle:
        if check is None and abort is None:
            digest = hashlib.file_digest(handle, "sha256")
        else:
            digest = hashlib.sha256()
            with _chunk_buffer() as buffer:
                view = memoryview(buffer)
                try:
                    while True:
                        if abort is not None:
                            abort()
                        if check is not None:
                            check()
                        size = handle.readinto(buffer)
                        if not size:
                            break
                        digest.update(view[:size])
                finally:
                    view.release()
        nbytes = handle.tell()
    note_read(nbytes, source=source)
    return digest.hexdigest()


def note_read(nbytes, *, source):
    """Count one file read of ``nbytes`` that fed a digest (a real read, not a ledger answer).

    What every read of :func:`sha256_file` reports: ``profiling.note_hashed``
    and the ``real`` counters of :func:`snapshot`.  For a caller that reads a
    file and digests it itself (the fused pass of ``sky_validity``).
    """
    profiling.note_hashed(nbytes, source=source)
    with _lock:
        _counters["real_bytes"] += nbytes
        _counters["real_files"] += 1


def _hash_and_record(scope, path, key, source, check=None, abort=None):
    if key is None or _live_mode(scope) == "off":
        return _read_digest(path, source, check, abort)
    snapshot = _hold()
    try:
        started = time.time_ns()
        before = _stat(path)
        digest = _read_digest(path, source, check, abort)
        if before is not None:
            _record(scope, key, before, _stat(path), digest, "ledger", source, started, snapshot)
    finally:
        _release()
    return digest


def _hash_fresh(scope, path, key, source, check=None, abort=None):
    """The miss path of a ``fresh`` item: :func:`_hash_and_record`'s read, then a guarded seal if racy.

    The digest is the same real read (same bytes, same counters, same
    failures).  A miss whose entry would not be racy is recorded exactly as
    today (kind ``ledger``, modification time untouched).  One that would be
    racy is sealed as kind ``write`` only if the file kept the stat taken
    before the read and no writing()/invalidate() touched it since the
    snapshot taken before that stat (see :func:`_writing`); otherwise nothing
    is recorded.  The seal never raises an Exception.
    """
    if key is None or _live_mode(scope) == "off":
        return _read_digest(path, source, check, abort)
    snapshot = _hold()
    try:
        started = time.time_ns()
        before = _stat(path)
        digest = _read_digest(path, source, check, abort)
        if before is not None:
            after = _stat(path)
            if started - before.st_mtime_ns >= int(RACY_SECONDS * 1e9):
                if _record(scope, key, before, after, digest, "ledger", source, started, snapshot):
                    _count("fresh_ledger")
            elif _seal_fresh(scope, path, key, before, after, digest, source, snapshot):
                _count("fresh_sealed")
            else:
                _count("fresh_refused")
    finally:
        _release()
    return digest


def _seal_fresh(scope, path, key, before, after, digest, source, snapshot):
    """Record ``digest`` of a just-read fresh file as kind ``write`` (nudged); whether it was recorded."""
    if after is None or _sig(before) != _sig(after):
        _count("refused_unstable")
        return False
    try:
        if not _volume_eligible(key, after):
            _count("refused_volume")
            return False
        with _writing(path, scope, since=snapshot, expect=before, revoke=False) as block:
            block.written(digest, source)
        return block.registered
    except Exception:
        _count("refused_seal")
        return False


def _note_saved(kind, size, source):
    profiling.note_hash_saved(size, source=source, kind=kind)
    with _lock:
        _counters["saved"][kind][0] += size
        _counters["saved"][kind][1] += 1


def _shadow_verify(scope, path, key, entry, source, check=None, abort=None):
    """Shadow mode: hash the hit for real; raise HashLedgerMismatch when it differs.

    A confirmed entry is kept as it is (kind and racy state unchanged), so a
    shadow run hits exactly where an "on" run would.  A difference caused by a
    writing() block that ran during the verifying read is not a ledger error:
    the entry is dropped and the real digest returned, as a miss would.
    """
    snapshot = _hold()
    try:
        real = _read_digest(path, source, check, abort)
        with _lock:
            _counters["shadow_checks"] += 1
            _counters["shadow_bytes"] += entry.sig[2]
            if real != entry.digest:
                if scope.entries.get(key) is entry:
                    _drop(scope, key)
                file_ids = (entry.sig[:2],)
                concurrent = _is_busy(key, file_ids) or _touched_since(key, file_ids, snapshot)
                _counters["shadow_concurrent" if concurrent else "shadow_mismatches"] += 1
    finally:
        _release()
    if real != entry.digest:
        if concurrent:
            return real
        raise HashLedgerMismatch(path, source, entry.kind, entry.digest, real)
    _note_saved(entry.kind, entry.sig[2], source)
    return real


_UNSET = object()


def _sha256(path, source, check=None, trust=TRUST_LEDGER, abort=None, lookup=True, key=_UNSET, scope=_UNSET,
            fresh=False):
    if scope is _UNSET:
        scope = _current()
    mode = _live_mode(scope)
    if mode == "off":
        return _read_digest(path, source, check, abort)
    if key is _UNSET:
        key = _norm(path)
    if key is not None and trust == TRUST_LEDGER and lookup:
        st = _stat(path)
        if st is not None:
            entry, status = _lookup(scope, key, st, source)
            if status == "hit":
                if mode == "shadow":
                    return _shadow_verify(scope, path, key, entry, source, check, abort)
                if check is not None:
                    check()
                _note_saved(entry.kind, entry.sig[2], source)
                return entry.digest
    if fresh:
        return _hash_fresh(scope, path, key, source, check, abort)
    return _hash_and_record(scope, path, key, source, check, abort)


def sha256_file(path, *, source, check=None, trust=TRUST_LEDGER, fresh=False):
    """Hex SHA-256 of ``path``.

    Opens exactly ``open(path, "rb")``: a wrapper whose old helper opened
    ``Path(path).open("rb")`` must pass ``Path(path)`` so that "", ints and
    other odd values fail as they do today.  Inside a run scope with
    ``trust="ledger"`` an unchanged file may be answered from the ledger;
    ``trust="real"`` always reads (the result is still remembered).  ``check``
    is called before every 8 MiB chunk (and once on a ledger hit).  Real bytes
    are counted with ``profiling.note_hashed``; ledger hits with
    ``profiling.note_hash_saved``.

    ``fresh=True``: the calling operation has just written and closed
    ``path`` and nothing else writes it while the run lasts.  Lookups,
    digests, reads and failures are exactly those of ``fresh=False``; only
    the recording of a miss changes: an entry that would be racy (the read
    started less than :data:`RACY_SECONDS` after the file's modification time)
    is sealed as kind ``write`` (modification time moved back
    :data:`MTIME_NUDGE_NS`) when the file still has exactly the stat taken
    before the read and no writing()/invalidate() touched it since, or else
    nothing is recorded.  A non-racy miss is recorded as today (no
    modification time change).  No effect outside a scope or with "off".
    """
    _check_trust(trust)
    return _sha256(path, source, check, trust, fresh=bool(fresh))


# --------------------------------------------------------------------------- #
# Batches                                                                      #
# --------------------------------------------------------------------------- #

def _try_hit(scope, path, key, source, trust):
    """Digest when the ledger answers without a read ("on" mode only); None otherwise."""
    if key is None or trust != TRUST_LEDGER or _live_mode(scope) != "on":
        return None
    st = _stat(path)
    if st is None:
        return None
    entry, status = _lookup(scope, key, st, source)
    if status != "hit":
        return None
    _note_saved(entry.kind, entry.sig[2], source)
    return entry.digest


def _note_repeat(path, source):
    st = _stat(path)
    _note_saved("batch", 0 if st is None else st.st_size, source)


def _replacement(on_error, index, exc):
    """The exception to raise for item ``index`` whose hash raised ``exc`` (``on_error`` may replace it)."""
    if on_error is None or not isinstance(exc, Exception):
        return exc
    replacement = on_error(index, exc)          # may raise: that is then the failure of the item
    return exc if replacement is None else replacement


def _fresh_flags(fresh, count):
    """``fresh`` of :func:`sha256_many` as one bool per path (ValueError when a sequence has another length)."""
    if fresh is None or isinstance(fresh, bool):
        return [bool(fresh)] * count
    flags = [bool(flag) for flag in fresh]
    if len(flags) != count:
        raise ValueError("hashing: fresh flags and paths differ in length")
    return flags


def _many(paths, *, source, workers, check, trust, expected=None, on_error=None, after=None, fresh=False):
    """(digests, first mismatching index) with serial-loop failure semantics.

    Serial reference: ``for i: check(); d[i] = sha256_file(p[i]); after(i); if
    expected and d[i] != expected[i]: return i``.  An exception of item i is
    raised only if no item below i failed or mismatched; the caller's
    ``check()`` failing counts as a failure of the next item not yet started.

    ``on_error(i, exc)``: called (in the caller's thread) when an exception
    ``exc`` raised while hashing item i is the failure to report; it returns
    the exception to raise instead, or None to keep ``exc``, and it may raise.
    A caller whose serial loop checked the file just before hashing it (it
    exists, it is a regular file, its size) re-runs that check here, so a file
    removed while earlier items were hashed fails as it did in the serial loop.
    ``after(i)``: called right after item i's digest is known (a read, a ledger
    hit, or the first occurrence of a repeated path), in the thread that got it;
    its exception is the failure of item i (``on_error`` is not asked).
    ``fresh``: see :func:`sha256_many` (a repeated path uses the flag of its
    first occurrence).
    """
    _check_trust(trust)
    paths = list(paths)
    count = len(paths)
    if expected is not None:
        expected = list(expected)
        if len(expected) != count:
            raise ValueError("hashing: expected digests and paths differ in length")
    fresh = _fresh_flags(fresh, count)
    results = [None] * count
    if not count:
        return results, None
    scope = _current()
    mode = _live_mode(scope)
    # Repeats are found by the absolute path as spelled (no case folding, no link resolution):
    # cheap, and never merges two files that differ only in case inside a case-sensitive
    # directory.  A repeat that is spelled differently is simply hashed again (or hits the ledger).
    # With the ledger off (in scope, or AESTER_HASH_LEDGER=off outside one) every item is read.
    dedupe = mode != "off" if scope is not None else _env_mode() != "off"
    keys = [_norm(path) for path in paths] if mode != "off" else [None] * count
    primary = list(range(count))
    if dedupe:
        seen = {}
        for index, name in enumerate(_absolute(path) for path in paths):
            if name is not None:
                primary[index] = seen.setdefault(name, index)
    unique = sum(1 for index in range(count) if primary[index] == index)
    workers = resolve_workers(workers)
    sizes = [0] * count
    if workers > 1 and unique > 1:
        for index in range(count):
            if primary[index] == index:
                st = _stat(paths[index])
                sizes[index] = 0 if st is None else st.st_size

    if workers <= 1 or unique <= 1 or sum(sizes) < PARALLEL_MIN_BYTES:
        for index in range(count):
            if check is not None:
                check()
            if primary[index] != index:
                results[index] = results[primary[index]]
                _note_repeat(paths[index], source)
            else:
                failure = None
                try:
                    results[index] = _sha256(paths[index], source, None, trust, key=keys[index], scope=scope,
                                             fresh=fresh[index])
                except Exception as exc:
                    if on_error is None:
                        raise
                    failure = exc
                if failure is not None:
                    raise _replacement(on_error, index, failure)
            if after is not None:
                after(index)
            if expected is not None and results[index] != expected[index]:
                return results, index
        return results, None

    _count("parallel_batches")
    failures = {}
    hash_failures = set()            # items whose failure came from hashing (on_error may replace it)
    state = {"stop": None}           # lowest index known to fail or mismatch
    lookup_in_job = mode == "shadow"

    def fail(index, exc=None):
        if exc is not None:
            failures[index] = exc
        if state["stop"] is None or index < state["stop"]:
            state["stop"] = index

    def job(index):
        def abort():
            stop = state["stop"]
            if stop is not None and index > stop:
                raise _Aborted()
        digest = _sha256(paths[index], source, None, trust, abort, lookup=lookup_in_job, key=keys[index],
                         scope=scope, fresh=fresh[index])
        if after is not None:
            try:
                after(index)
            except BaseException as exc:
                raise _AfterFailed(exc) from None
        return digest

    inflight = {}
    inflight_bytes = 0
    next_index = 0
    executor = ThreadPoolExecutor(max_workers=min(workers, unique), thread_name_prefix="aester-hash")

    def room(index):
        if len(inflight) < workers:
            return True
        return (len(inflight) < workers * INFLIGHT_PER_WORKER
                and inflight_bytes + sizes[index] <= INFLIGHT_BYTES)

    try:
        while True:
            while state["stop"] is None and next_index < count and room(next_index):
                index = next_index
                try:
                    if check is not None:
                        check()
                    if primary[index] != index:      # a repeated path: resolved from its first occurrence
                        next_index += 1
                        continue
                    hit = _try_hit(scope, paths[index], keys[index], source, trust)
                    if hit is not None and after is not None:
                        after(index)
                except BaseException as exc:
                    fail(index, exc)
                    break
                next_index += 1
                if hit is not None:
                    results[index] = hit
                    if expected is not None and hit != expected[index]:
                        fail(index)
                    continue
                inflight[executor.submit(job, index)] = index
                inflight_bytes += sizes[index]
            if not inflight:
                break
            done, _ = wait(inflight, timeout=CHECK_INTERVAL_SECONDS, return_when=FIRST_COMPLETED)
            for future in done:
                index = inflight.pop(future)
                inflight_bytes -= sizes[index]
                try:
                    results[index] = future.result()
                except _Aborted:
                    continue
                except _AfterFailed as wrapped:
                    fail(index, wrapped.error)
                    continue
                except BaseException as exc:
                    fail(index, exc)
                    hash_failures.add(index)
                    continue
                if expected is not None and results[index] != expected[index]:
                    fail(index)
            if state["stop"] is None and check is not None and next_index < count:
                try:
                    check()
                except BaseException as exc:
                    fail(next_index, exc)
    except BaseException:
        state["stop"] = -1               # e.g. KeyboardInterrupt while waiting: every item stops reading
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    for index in range(count):
        if index in failures:
            if index in hash_failures:
                raise _replacement(on_error, index, failures[index])
            raise failures[index]
        if primary[index] != index:
            results[index] = results[primary[index]]
            _note_repeat(paths[index], source)
            if after is not None:
                after(index)
        if results[index] is None:        # only items above the lowest failure are left unfinished
            raise RuntimeError(f"hashing: internal error, item {index} has no result and no failure")
        if expected is not None and results[index] != expected[index]:
            return results, index
    if count in failures:
        raise failures[count]
    return results, None


def sha256_many(paths, *, source, workers=None, check=None, trust=TRUST_LEDGER, on_error=None, after=None,
                fresh=False):
    """Digests of ``paths`` in input order, misses hashed on a bounded thread pool.

    Repeats of one absolute path (as spelled) are hashed once (not when the
    ledger is off); batches under :data:`PARALLEL_MIN_BYTES` in total run
    serially.  ``check()`` runs before each file is started and between
    completions.  On failure every lower-index item is finished first and the
    exception of the lowest failing index is raised - the one a serial loop
    would have raised.  ``on_error`` and ``after``: see :func:`_many` (re-run
    the serial loop's own check of a file that failed to hash; do per-file
    work, e.g. a stat, right after its digest).

    ``fresh``: True, or a sequence of bools (one per path; ValueError when the
    length differs), marking files the calling operation has just written and
    closed and that nothing else writes while the run lasts.  Lookups,
    digests, reads, order and failures are exactly those of ``fresh=False``;
    only the recording of a miss changes, as for :func:`sha256_file`.  The
    seal runs in whatever thread hashed the item, with the batch's scope.
    """
    return _many(paths, source=source, workers=workers, check=check, trust=trust, on_error=on_error,
                 after=after, fresh=fresh)[0]


def first_mismatch(paths, expected, *, source, workers=None, check=None, trust=TRUST_LEDGER, on_error=None):
    """Index of the first path whose digest differs from ``expected[i]`` (None if all match).

    Same order semantics as a serial ``for`` loop that stops at the first
    mismatch: an exception of a later item is not raised when an earlier item
    mismatches, and vice versa.  ``on_error``: as in :func:`sha256_many`.
    """
    return _many(paths, source=source, workers=workers, check=check, trust=trust, expected=expected,
                 on_error=on_error)[1]


def _entry_path(entry):
    return Path(entry["path"])


def _entry_digest(entry):
    return entry["sha256"]


def _entry_precheck(entry, path):
    return path.is_file() and path.stat().st_size == entry["bytes"]


def verify_entries(entries, *, source, error, check=None, trust=TRUST_LEDGER, workers=None,
                   path=_entry_path, expected=_entry_digest, precheck=_entry_precheck, validate=None):
    """Verify ``{path, sha256, bytes}`` entries; parallel, but failing like the serial loop.

    Serial reference (``workflow._verify_files`` with an optional check; with
    ``validate`` also ``finite_stack._verify_sources``)::

        for entry in entries:
            validate(entry)            # may raise; runs before check()
            check()
            p = path(entry)
            if not precheck(entry, p) or sha256_file(p) != expected(entry):
                raise error(entry)

    The cheap ``validate``/``path``/``precheck`` run for every entry in order
    first; digests of the entries before the first failed one are then
    computed in parallel; the first failure in input order is raised, with the
    caller's exception from ``error(entry)`` for a mismatch.  When the hash of
    an entry fails with an OSError (the file went away, or was replaced by a
    folder, while earlier entries were hashed), its ``path`` and ``precheck``
    are run again at that point, as the serial loop ran them just before
    hashing: their exception, or ``error(entry)`` when the precheck now fails,
    is raised instead.
    """
    entries = list(entries)

    def recheck(index, exc):
        """The serial loop's failure for ``entries[index]`` whose hash raised ``exc`` (None: keep ``exc``)."""
        if not isinstance(exc, OSError):
            return None
        entry = entries[index]
        if not precheck(entry, path(entry)):    # either may raise, as in the serial loop
            return error(entry)
        return None

    paths, digests = [], []
    limit, early, hash_first, before_check = len(entries), None, None, False
    for index, entry in enumerate(entries):
        if validate is not None:
            try:
                validate(entry)
            except BaseException as exc:  # raised before this entry's check(), as in the serial loop
                limit, early, before_check = index, exc, True
                break
        try:
            candidate = path(entry)
            ok = precheck(entry, candidate)
        except BaseException as exc:      # the failure of this entry, raised only if nothing before it fails
            limit, early = index, exc
            break
        if not ok:
            limit = index
            break
        try:
            wanted = expected(entry)
        except BaseException as exc:      # the serial loop hashes this file before it looks at the digest
            limit, early, hash_first = index, exc, candidate
            break
        paths.append(candidate)
        digests.append(wanted)
    index = first_mismatch(paths, digests, source=source, workers=workers, check=check, trust=trust,
                           on_error=recheck)
    if index is not None:
        raise error(entries[index])
    if limit < len(entries):
        if check is not None and not before_check:
            check()
        if hash_first is not None:
            failure = None
            try:
                sha256_file(hash_first, source=source, trust=trust)
            except Exception as exc:
                failure = exc
            if failure is not None:
                raise _replacement(recheck, limit, failure)
        if early is not None:
            raise early
        raise error(entries[limit])


# --------------------------------------------------------------------------- #
# Copy                                                                         #
# --------------------------------------------------------------------------- #

def _same_file(src, dst):
    probe = getattr(shutil, "_samefile", None)
    if probe is not None:
        return probe(src, dst)
    try:
        return os.path.samefile(src, dst)
    except OSError:
        return False


def copy_with_digest(src, dst, *, source, expected=None, check=None, exclusive=True, fsync=True, readback=True):
    """Copy ``src`` to ``dst`` hashing the bytes read; returns ``CopyResult(src_sha256, dst_sha256, size)``.

    ``exclusive``: ``dst`` is opened ``"xb"`` (FileExistsError if it exists).
    Otherwise ``"wb"``, with ``shutil.copyfile``'s guards: ``SameFileError``
    when ``dst`` is ``src`` or a hard link to it (checked before anything is
    opened), and FileNotFoundError for a missing destination directory.
    ``check()`` runs before every 8 MiB chunk and once after the copy is
    closed (and fsynced), before any comparison.  When ``expected`` is given
    and the copied bytes differ, nothing is read back and ``dst_sha256`` is
    None: the caller raises its own error.  With ``readback`` the destination
    is re-read for real (never from the ledger, no ``check()`` during the
    read-back) and, when it equals the copied bytes, recorded as written by
    this process; without it the copy digest is recorded for ``dst``
    directly.  The source digest is recorded as kind ``ledger`` when the
    source did not change during the copy.
    """
    if not exclusive and _same_file(src, dst):
        raise shutil.SameFileError("{!r} and {!r} are the same file".format(src, dst))
    scope = _current()
    active = _live_mode(scope) != "off"
    src_key = _norm(src) if active else None
    snapshot = _hold()
    try:
        started = time.time_ns()
        before = _stat(src) if active else None
        digest = hashlib.sha256()
        size = 0
        with writing(dst) as block:
            with open(src, "rb", buffering=0) as incoming:
                try:
                    outgoing = open(dst, "xb" if exclusive else "wb")
                except IsADirectoryError as exc:
                    if not exclusive and not os.path.exists(dst):
                        raise FileNotFoundError(f'Directory does not exist: {dst}') from exc
                    raise
                with outgoing:
                    with _chunk_buffer() as buffer:
                        view = memoryview(buffer)
                        try:
                            while True:
                                if check is not None:
                                    check()
                                count = incoming.readinto(buffer)
                                if not count:
                                    break
                                chunk = view[:count]
                                digest.update(chunk)
                                outgoing.write(chunk)
                                chunk.release()
                                size += count
                        finally:
                            view.release()
                    outgoing.flush()
                    if fsync:
                        os.fsync(outgoing.fileno())
            after = _stat(src) if active else None
            note_read(size, source=source)
            src_sha256 = digest.hexdigest()
            if check is not None:
                check()
            if before is not None and after is not None and before.st_size == size:
                _record(scope, src_key, before, after, src_sha256, "ledger", source, started, snapshot)
            if expected is not None and src_sha256 != expected:
                return CopyResult(src_sha256, None, size)
            if not readback:
                block.written(src_sha256, source)
                return CopyResult(src_sha256, src_sha256, size)
            dst_before = _stat(dst)
            dst_sha256 = _read_digest(dst, source)
            dst_after = _stat(dst)
            if (dst_sha256 == src_sha256 and dst_before is not None and dst_after is not None
                    and _sig(dst_before) == _sig(dst_after)):
                block.written(dst_sha256, source)
        return CopyResult(src_sha256, dst_sha256, size)
    finally:
        _release()


# --------------------------------------------------------------------------- #
# Digest while writing                                                         #
# --------------------------------------------------------------------------- #

class HashingWriter:
    """Sequential-only binary writer that hashes what it writes.

    Wraps a file freshly opened ``"wb"``/``"xb"``.  Writes at the end of the
    data and forward seeks (the gap reads back as zeros and is hashed as
    such) keep the digest; any backward seek, out-of-order write, truncate or
    read gives it up (:meth:`digest` then returns None and the caller reads
    the file back).  ``fileno()`` is not exposed, so libraries cannot write
    around the wrapper.  ``hashed_bytes`` counts the bytes fed to SHA-256.
    """

    def __init__(self, raw):
        self._raw = raw
        self._pos = 0
        self._end = 0
        self._hash = hashlib.sha256()
        self.hashed_bytes = 0
        self.gave_up = None
        self.attempted = False      # whether a write or truncate was attempted (successful or not)
        try:
            start = raw.tell()
            size = os.fstat(raw.fileno()).st_size
        except Exception as exc:
            self._give_up(f"cannot establish the start: {type(exc).__name__}")
        else:
            if start or size:
                self._give_up("the file was not empty")
        self.name = getattr(raw, "name", None)
        self.mode = getattr(raw, "mode", "wb")

    def _give_up(self, why):
        if self._hash is not None:
            self._hash = None
            self.gave_up = why

    # -- io surface ------------------------------------------------------------
    @property
    def closed(self):
        return self._raw.closed

    def writable(self):
        return True

    def seekable(self):
        return True

    def readable(self):
        return False

    def fileno(self):
        import io
        raise io.UnsupportedOperation("HashingWriter does not expose a file descriptor")

    def tell(self):
        return self._pos

    def seek(self, offset, whence=0):
        target = self._raw.seek(offset, whence)
        if target < self._end:
            self._give_up("backward seek")
        self._pos = target
        return target

    def write(self, data):
        self.attempted = True
        view = memoryview(data)
        if view.ndim != 1 or view.format not in ("B", "b", "c") or not view.c_contiguous:
            view = view.cast("B") if view.c_contiguous else memoryview(view.tobytes())
        written = self._raw.write(view)
        if written is None:
            written = 0
        if self._hash is not None:
            if self._pos < self._end:
                self._give_up("out-of-order write")
            else:
                gap = self._pos - self._end
                zeros = bytes(min(gap, CHUNK)) if gap else b""
                while gap:
                    step = min(gap, CHUNK)
                    self._hash.update(zeros[:step])
                    self.hashed_bytes += step
                    gap -= step
                self._hash.update(view[:written])
                self.hashed_bytes += written
        self._pos += written
        self._end = max(self._end, self._pos)
        return written

    def writelines(self, lines):
        for line in lines:
            self.write(line)

    def truncate(self, size=None):
        self.attempted = True
        size = self._pos if size is None else size
        if size != self._end:
            self._give_up("truncate")
        result = self._raw.truncate(size)
        self._end = size
        return result

    def read(self, *args):
        self._give_up("read")
        import io
        raise io.UnsupportedOperation("HashingWriter is write-only")

    readinto = readline = read

    def flush(self):
        if not self._raw.closed:
            self._raw.flush()

    def close(self):
        self._raw.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- result ----------------------------------------------------------------
    @property
    def size(self):
        return self._end

    def digest(self):
        """Hex digest of the bytes written, or None when the writing was not sequential."""
        return None if self._hash is None else self._hash.copy().hexdigest()


def _note_memory(nbytes, source):
    profiling.note_hashed_memory(nbytes, source=source)
    with _lock:
        _counters["memory_bytes"] += nbytes
        _counters["memory_calls"] += 1


def note_memory(nbytes, *, source):
    """Count ``nbytes`` hashed from memory by the caller (``profiling.note_hashed_memory`` and :func:`snapshot`)."""
    _note_memory(nbytes, source)


DISCARD_CREATED = "created"
DISCARD_UNWRITTEN = "unwritten"
_DISCARDS = (None, DISCARD_CREATED, DISCARD_UNWRITTEN)


def _discard(handle, path):
    """Close ``handle`` and delete the file this call created, as ``PIL.Image.save`` does on a failed save."""
    handle.close()
    try:
        os.remove(path)
    except PermissionError:
        pass


def write_with_digest(path, writer, *, source, mode="wb", fsync=False, discard=None):
    """Open ``path`` (``"wb"``/``"xb"``), call ``writer(file)`` with a :class:`HashingWriter`, close.

    Returns the digest of the bytes written, recorded as kind ``write`` (see
    :meth:`WriteBlock.written`), or None when the writer was not sequential
    (the caller then reads the file back).  The file's bytes are exactly what
    ``writer`` would write to a plain file object that has no ``fileno()``.
    Runs (and returns the digest) outside a run scope too.

    ``discard`` says what a failure leaves behind, to match the writer the
    caller replaces: None leaves the file (a plain ``open`` then write, e.g.
    ``np.save`` into an open file); ``"unwritten"`` deletes a file this call
    created when ``writer`` fails before attempting any write (astropy's
    ``writeto`` validates before it creates the file); ``"created"`` deletes a
    file this call created on any failure (``Image.save``).
    """
    if mode not in ("wb", "xb"):
        raise ValueError("hashing: write_with_digest needs mode 'wb' or 'xb'")
    if discard not in _DISCARDS:
        raise ValueError(f"hashing: discard must be one of {_DISCARDS}, not {discard!r}")
    with writing(path) as block:
        created = discard is not None and not os.path.exists(path)
        with open(path, mode) as raw:
            wrapper = HashingWriter(raw)
            try:
                writer(wrapper)
                if not raw.closed:
                    raw.flush()
                    if fsync:
                        os.fsync(raw.fileno())
            except BaseException as exc:
                if created and (discard == DISCARD_UNWRITTEN and not wrapper.attempted
                                or discard == DISCARD_CREATED and isinstance(exc, Exception)):
                    _discard(raw, path)
                raise
            finally:
                _note_memory(wrapper.hashed_bytes, source)
        digest = wrapper.digest()
        if digest is not None:
            block.written(digest, source)
    return digest


def save_npy_with_digest(path, array, *, source, mode="wb", fsync=False):
    """``np.save`` to exactly ``path`` (no ``.npy`` is appended); digest as :func:`write_with_digest`."""
    import numpy as np
    return write_with_digest(path, lambda handle: np.save(handle, array), source=source, mode=mode, fsync=fsync)


def write_bytes_with_digest(path, data, *, source, mode="wb", fsync=False, discard=None):
    """``Path(path).write_bytes(data)`` for bytes already encoded in memory; returns their digest.

    ``discard="created"``: a file this call created is deleted when the write
    fails (an ``Exception``), as ``Image.save`` deletes the file it created;
    None leaves whatever was written, as ``write_bytes`` does.
    """
    if mode not in ("wb", "xb"):
        raise ValueError("hashing: write_bytes_with_digest needs mode 'wb' or 'xb'")
    if discard not in (None, DISCARD_CREATED):
        raise ValueError(f"hashing: discard must be None or {DISCARD_CREATED!r}, not {discard!r}")
    view = memoryview(data)
    digest = hashlib.sha256(view).hexdigest()
    _note_memory(view.nbytes, source)
    with writing(path) as block:
        created = discard is not None and not os.path.exists(path)
        with open(path, mode) as handle:
            try:
                handle.write(view)
                handle.flush()
                if fsync:
                    os.fsync(handle.fileno())
            except Exception:
                if created:
                    _discard(handle, path)
                raise
        block.written(digest, source)
    return digest


# --------------------------------------------------------------------------- #
# In-memory digests                                                            #
# --------------------------------------------------------------------------- #

def sha256_bytes(data, *, source):
    """``hashlib.sha256(data).hexdigest()`` (same errors), counted with ``profiling.note_hashed_memory``."""
    digest = hashlib.sha256(data).hexdigest()
    _note_memory(memoryview(data).nbytes, source)
    return digest


def array_digest(array, *, source):
    """Exactly ``photographic.array_identity(array)`` (same bytes, same errors), counted as in-memory hashing.

    Format: SHA-256 of ``str((shape, dtype.str))`` in ASCII followed by the
    C-ordered bytes of the array in blocks of 128 rows.
    """
    import numpy as np
    h = hashlib.sha256()
    h.update(str((array.shape, array.dtype.str)).encode('ascii'))
    for y in range(0, array.shape[0], 128):
        h.update(np.ascontiguousarray(array[y:y+128]).view(np.uint8))
    digest = h.hexdigest()
    _note_memory(array.nbytes, source)
    return digest


def arrays_digest_many(arrays, *, source, workers=None, check=None):
    """``[array_digest(a) for a in arrays]`` in input order, on a bounded thread pool.

    Serial reference: ``for a in arrays: check(); out.append(array_digest(a))``.
    ``check()`` runs before each array in input order (before its hash starts);
    the exception raised is the one of the lowest failing index (a ``check()``
    failure counts for the array it precedes).  Small totals run serially.
    """
    arrays = list(arrays)
    workers = resolve_workers(workers)
    total = 0
    for array in arrays:
        total += int(getattr(array, "nbytes", 0) or 0)
    if workers <= 1 or len(arrays) <= 1 or total < PARALLEL_MIN_ARRAY_BYTES:
        out = []
        for array in arrays:
            if check is not None:
                check()
            out.append(array_digest(array, source=source))
        return out
    futures = []
    stop = None
    executor = ThreadPoolExecutor(max_workers=min(workers, len(arrays)), thread_name_prefix="aester-array-hash")
    try:
        for array in arrays:
            if check is not None:
                try:
                    check()
                except BaseException as exc:
                    stop = exc
                    break
            futures.append(executor.submit(array_digest, array, source=source))
        out = []
        for future in futures:
            out.append(future.result())       # raises the lowest-index failure first
        if stop is not None:
            raise stop
        return out
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


# --------------------------------------------------------------------------- #
# Cross-process handover                                                       #
# --------------------------------------------------------------------------- #

def export_handover(path, root, *, token):
    """Write the ledger entries of files under ``root`` to ``path`` (JSON, atomic); returns the count.

    ``path`` None means :func:`handover_path` ``(token)`` (its folder is
    created).  ``token`` identifies the run attempt; :func:`import_handover`
    accepts the file only with the same token.  Only entries whose file still has the
    recorded stat are written.  When nothing is exported (outside a run scope,
    "off" mode, an unusable root, no entries) any existing file at ``path`` is
    removed and None (or 0 for no entries) is returned.  Until the file is
    consumed, a later :func:`writing` / :func:`invalidate` in this process of
    any path under ``root`` revokes (deletes) it, so export after the last
    in-place write is the efficient order, but a later one stays safe.
    """
    _check_token(token)
    target = os.fspath(handover_path(token) if path is None else path)
    scope = _current()
    root_key = _norm(root) if _live_mode(scope) != "off" else None
    if root_key is None:
        _revoke_file(target)
        return None
    with _lock:
        candidates = [(key, entry) for key, entry in scope.entries.items() if _under(key, root_key)]
    rows = []
    for key, entry in sorted(candidates, key=lambda item: item[0]):
        st = _stat(key)
        if st is None or _sig(st) != entry.sig:
            continue
        dev, ino, size, mtime_ns = entry.sig
        rows.append({"path": key, "sha256": entry.digest, "dev": dev, "ino": ino, "size": size,
                     "mtime_ns": mtime_ns, "recorded_at_ns": entry.recorded_at_ns, "kind": entry.kind,
                     "racy_exempt": bool(entry.exempt)})
    if not rows:
        _revoke_file(target)
        return 0
    document = {"version": HANDOVER_VERSION, "token": token, "root": root_key, "pid": os.getpid(),
                "created_at_ns": time.time_ns(), "racy_seconds": RACY_SECONDS, "entries": rows}
    if path is None:
        os.makedirs(os.path.dirname(target), exist_ok=True)
    temporary = f"{target}.{os.getpid()}.{threading.get_ident()}.tmp"
    with writing(target):
        try:
            with open(temporary, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(document, handle, ensure_ascii=False, indent=1, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        except BaseException:
            try:
                os.remove(temporary)
            except OSError:
                pass
            raise
    target_key = _norm(target)
    with _lock:
        if target_key is not None:
            _exports[target_key] = root_key
        _counters["handover_exported"] += len(rows)
    return len(rows)


def import_handover(path, root, *, token):
    """Load another process's entries for files under ``root`` as kind ``handover``; returns the count.

    ``path`` None means :func:`handover_path` ``(token)``.  The file is
    always consumed: read and then deleted, whatever it holds; outside a run
    scope or in "off" mode it is only deleted and 0 is returned.  It is
    accepted only with version :data:`HANDOVER_VERSION` and the same
    ``token``, and only its well-formed rows under ``root`` are loaded.  Imported entries still need
    an exact stat match at lookup; the racy rule uses the exporter's
    ``recorded_at_ns`` unless the row was a nudged write entry.  Entries this
    ledger already has, and files being written now, are skipped.
    """
    _check_token(token)
    source = handover_path(token) if path is None else path
    scope = _current()
    if _live_mode(scope) == "off":
        try:
            os.remove(source)
        except OSError:
            pass
        return 0
    root_key = _norm(root)
    try:
        with open(source, "rb") as handle:
            raw = handle.read()
    except OSError:
        return 0
    try:
        os.remove(source)
    except OSError:
        pass
    if root_key is None:
        return 0
    try:
        document = json.loads(raw.decode("utf-8"))
    except ValueError:
        _count("handover_rejected")
        return 0
    if (not isinstance(document, dict) or document.get("version") != HANDOVER_VERSION
            or document.get("token") != token or not isinstance(document.get("entries"), list)):
        _count("handover_rejected")
        return 0
    imported = 0
    for row in document["entries"]:
        if not isinstance(row, dict):
            continue
        key = row.get("path")
        numbers = [row.get(name) for name in ("dev", "ino", "size", "mtime_ns", "recorded_at_ns")]
        if (not isinstance(key, str) or not _is_digest(row.get("sha256")) or row.get("kind") not in KINDS
                or not all(type(value) is int for value in numbers) or not numbers[1]
                or key != os.path.normcase(key) or not _under(key, root_key)):
            continue
        dev, ino, size, mtime_ns, recorded_at_ns = numbers
        exempt = row.get("kind") == "write" and row.get("racy_exempt") is True
        with _lock:
            if _live_mode(scope) == "off":
                return imported
            if key in scope.entries or _is_busy(key, ((dev, ino),)):
                continue
            _store(scope, key, _Entry(row["sha256"], (dev, ino, size, mtime_ns), recorded_at_ns, "handover",
                                      "handover", exempt))
            _counters["recorded"]["handover"] += 1
        imported += 1
    return imported


# --------------------------------------------------------------------------- #
# Diagnostics                                                                  #
# --------------------------------------------------------------------------- #

def _counter_view(counters):
    """The grouped form of a counters dict (as in snapshot())."""
    return {
        "real": {"bytes": counters["real_bytes"], "files": counters["real_files"]},
        "saved": {kind: {"bytes": value[0], "files": value[1]} for kind, value in counters["saved"].items()},
        "memory": {"bytes": counters["memory_bytes"], "calls": counters["memory_calls"]},
        "shadow": {"checks": counters["shadow_checks"], "bytes": counters["shadow_bytes"],
                   "mismatches": counters["shadow_mismatches"], "concurrent": counters["shadow_concurrent"]},
        "misses": {"no_entry": counters["miss_no_entry"], "stale": counters["miss_stale"],
                   "racy": counters["miss_racy"], "busy": counters["miss_busy"]},
        "miss_bytes": counters["miss_bytes"],
        "renamed": {"moved": counters["renamed_moved"], "dropped": counters["renamed_dropped"]},
        "fresh": {"sealed": counters["fresh_sealed"], "ledger": counters["fresh_ledger"],
                  "refused": counters["fresh_refused"]},
        "recorded": counters["recorded"],
        "refused": {"unstable": counters["refused_unstable"], "volume": counters["refused_volume"],
                    "busy": counters["refused_busy"], "touched": counters["refused_touched"],
                    "seal": counters["refused_seal"]},
        "invalidated": counters["invalidated"], "parallel_batches": counters["parallel_batches"],
        "handover": {"exported": counters["handover_exported"], "revoked": counters["handover_revoked"],
                     "rejected": counters["handover_rejected"], "pruned": counters["handover_pruned"]},
    }


def snapshot():
    """The calling context's scope plus process-wide counters since start (callers diff two snapshots)."""
    scope = _current()
    with _lock:
        counters = json.loads(json.dumps(_counters))
        return {
            "version": VERSION, "active": scope is not None, "label": None if scope is None else scope.label,
            "mode": _live_mode(scope), "entries": 0 if scope is None else len(scope.entries),
            "open_scopes": len(_live), "workers": resolve_workers(None), "racy_seconds": RACY_SECONDS,
            **_counter_view(counters),
            "volumes": {root: dict(value) for root, value in sorted(_volume_by_root.items())},
        }


def _difference(start, now):
    if isinstance(now, dict):
        return {key: _difference((start or {}).get(key), value) for key, value in now.items()}
    if isinstance(now, list):
        start = start if isinstance(start, list) and len(start) == len(now) else [0] * len(now)
        return [value - before for value, before in zip(now, start)]
    return now - (start or 0)


def scope_report():
    """What hashing did since the innermost :func:`run_scope` of this context was entered; None outside one.

    For the timing notes of a worker (``hash_ledger``): the ledger's label
    and mode, real reads, saved reads by kind, in-memory hashing, shadow
    checks and mismatches (``concurrent``: differences blamed on a writer
    that ran during the verifying read), misses (count and bytes per
    status), refusals, renames carried, fresh seals and handovers.  The
    counters are process-wide, so work of other threads in that time is
    included.  ``largest_misses``: the ledger's (not only this nested
    scope's) :data:`LARGEST_MISSES` largest misses, largest first, as
    ``[bytes, status, source, "folder/name"]``.  JSON-safe.
    """
    marks = _marks_var.get()
    scope = _current()
    if not marks or scope is None:
        return None
    label, start = marks[-1]
    now = _counters_copy()
    with _lock:
        largest = [list(item) for item in sorted(scope.largest, reverse=True)]
    return {"version": VERSION, "label": label, "ledger": scope.label, "mode": _live_mode(scope),
            "workers": resolve_workers(None), **_counter_view(_difference(start, now)),
            "largest_misses": largest}


NOTE_KEY = "hash_ledger"


def note_scope_report():
    """Attach :func:`scope_report` to the running timers' notes as ``hash_ledger``; never raises.

    Call it inside the run scope, before the timer of the operation finishes.
    """
    try:
        report = scope_report()
        if report is not None:
            profiling.note(NOTE_KEY, report)
    except Exception:
        pass
