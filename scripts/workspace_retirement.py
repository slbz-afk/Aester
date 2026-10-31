"""Explicit-file workspace retirement, with a reviewable plan and JSONL receipt.

This is a developer maintenance tool, not a recursive cleaner. Planning hashes
every listed file; execution verifies *all* files before deleting any of them.
Only an explicit --apply enables deletion. The caller must also hold the relevant
product/job writer locks and record the loss of resume/recompose capability.
This tool's workspace lock coordinates only other instances of this tool.

Windows file handles deny writes/replacement during each check and deletion;
workspace directory handles prevent renames. Handles are bounded to one candidate
at a time. After full preflight, each file is reopened and fully checked again
before deletion. Multiple hard links account for this plan's link-count changes.
The POSIX fallback requires quiescent/cooperating writers; it is not a hostile
filesystem race security boundary. Byte totals are logical, NOT reclaimed space.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import time

VERSION = "workspace-retirement-1"
LOCK = ".workspace-retirement.lock"
PROTECTED = ("raw", "raw_1", "raw_2_sony", "research", ".git", ".venv", "tools", LOCK)
ENTRY_KEYS = {"path", "bytes", "mtime_ns", "device", "inode", "nlink", "sha256"}


def _now():
    return datetime.now(timezone.utc).isoformat()


def _relative(value):
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ValueError("Expected an explicit forward-slash relative file path")
    parts = value.split("/")
    if PurePosixPath(value).is_absolute() or any(p in ("", ".", "..") or p.endswith((".", " ")) for p in parts):
        raise ValueError("Unsafe relative path: " + value)
    if any(re.fullmatch(r"(?i)(con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\..*)?", p) for p in parts):
        raise ValueError("Reserved Windows path: " + value)
    return value


def _no_links(path):
    """Check every existing ancestor, including those above the workspace."""
    for part in reversed((path, *path.parents)):
        info = part.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("Symlink/reparse path refused: " + str(part))


def _root(value):
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("Workspace must be an absolute path without '..'")
    _no_links(path)
    if not path.is_dir():
        raise ValueError("Workspace must be an existing non-volume directory")
    if path.parent == path:
        raise ValueError("Volume roots cannot be cleaned")
    return path


def _identity(info):
    return {"bytes": info.st_size, "mtime_ns": info.st_mtime_ns,
            "device": str(info.st_dev), "inode": str(info.st_ino), "nlink": info.st_nlink}


def _root_identity(root):
    info = root.stat()
    return {"device": str(info.st_dev), "inode": str(info.st_ino)}


def _protected(values):
    return sorted(set(PROTECTED) | {_relative(v) for v in values})


def _candidate(root, relative, protected):
    relative = _relative(relative)
    value = relative.casefold()
    if any(value == p.casefold() or value.startswith(p.casefold() + "/") or p.casefold().startswith(value + "/") for p in protected):
        raise ValueError("Protected path: " + relative)
    path = root.joinpath(*relative.split("/"))
    path.relative_to(root)  # lexical bound, before any filesystem operation
    _no_links(path)
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError("Only individual ordinary files may be retired: " + relative)
    return path


@contextmanager
def maintenance_lock(root):
    path = root / LOCK
    if path.exists() or path.is_symlink():
        _no_links(path)
        if not path.is_file() or path.stat().st_nlink != 1:
            raise ValueError("Unsafe maintenance lock")
    with path.open("a+b") as stream:
        if os.fstat(stream.fileno()).st_nlink != 1:
            raise ValueError("Maintenance lock cannot be hard-linked")
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        try:
            if os.name == "nt":
                import msvcrt
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise ValueError("Another workspace retirement operation is active") from error
        try:
            yield
        finally:
            if os.name == "nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


if os.name == "nt":
    import ctypes
    from ctypes import wintypes
    _win = ctypes.WinDLL("kernel32", use_last_error=True)
    _win.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    _win.CreateFileW.restype = wintypes.HANDLE
    _win.CloseHandle.argtypes = [wintypes.HANDLE]
    _win.ReadFile.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
    _win.SetFilePointerEx.argtypes = [wintypes.HANDLE, ctypes.c_longlong, ctypes.c_void_p, wintypes.DWORD]
    _win.SetFileInformationByHandle.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]

    class _Info(ctypes.Structure):
        _fields_ = [("attributes", wintypes.DWORD), ("created", wintypes.FILETIME), ("accessed", wintypes.FILETIME),
                    ("written", wintypes.FILETIME), ("volume", wintypes.DWORD), ("size_high", wintypes.DWORD),
                    ("size_low", wintypes.DWORD), ("nlink", wintypes.DWORD), ("index_high", wintypes.DWORD), ("index_low", wintypes.DWORD)]
    _win.GetFileInformationByHandle.argtypes = [wintypes.HANDLE, ctypes.POINTER(_Info)]

    def _ok(result):
        if not result:
            raise ctypes.WinError(ctypes.get_last_error())
        return result

    class _Handle:
        def __init__(self, path, *, deleting=False, directory=False, alias=False):
            self.path = path
            access = 0x80 if alias else 0x80000000
            if deleting:
                access |= 0x10000
            share = 3 if directory else (5 if alias else 1)
            name = str(path)
            if not name.startswith("\\\\?\\"):
                name = "\\\\?\\UNC\\" + name[2:] if name.startswith("\\\\") else "\\\\?\\" + name
            self.handle = _win.CreateFileW(name, access, share, None, 3, 0x00200000 | (0x02000000 if directory else 0), None)
            if self.handle == wintypes.HANDLE(-1).value:
                raise ctypes.WinError(ctypes.get_last_error())
            try:
                self.info()
            except BaseException:
                self.close()
                raise

        def info(self):
            info = _Info()
            _ok(_win.GetFileInformationByHandle(self.handle, ctypes.byref(info)))
            if info.attributes & 0x400:
                raise ValueError("Reparse handle refused: " + str(self.path))
            return {"bytes": (info.size_high << 32) | info.size_low,
                    "mtime_ns": (((info.written.dwHighDateTime << 32) | info.written.dwLowDateTime) - 116444736000000000) * 100,
                    "device": str(info.volume), "inode": str((info.index_high << 32) | info.index_low), "nlink": info.nlink}

        def digest(self):
            _ok(_win.SetFilePointerEx(self.handle, 0, None, 0))
            data = ctypes.create_string_buffer(4 * 1024 * 1024)
            count = wintypes.DWORD()
            digest = hashlib.sha256()
            while True:
                _ok(_win.ReadFile(self.handle, data, len(data), ctypes.byref(count), None))
                if not count.value:
                    return digest.hexdigest()
                digest.update(data.raw[:count.value])

        def delete(self):
            disposition = wintypes.BOOL(1)
            _ok(_win.SetFileInformationByHandle(self.handle, 4, ctypes.byref(disposition), ctypes.sizeof(disposition)))
            self.close()

        def close(self):
            if self.handle is not None:
                _win.CloseHandle(self.handle)
                self.handle = None
else:
    class _Handle:
        def __init__(self, path, **kwargs):
            self.path = path
            self.fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))

        def info(self):
            return _identity(os.fstat(self.fd))

        def digest(self):
            os.lseek(self.fd, 0, os.SEEK_SET)
            digest = hashlib.sha256()
            while data := os.read(self.fd, 4 * 1024 * 1024):
                digest.update(data)
            return digest.hexdigest()

        def delete(self):
            self.path.unlink()
            self.close()

        def close(self):
            if self.fd is not None:
                os.close(self.fd)
                self.fd = None


def _hold(stack, path, **kwargs):
    handle = _Handle(path, **kwargs)
    stack.callback(handle.close)
    return handle


def _directories(stack, root, paths):
    if os.name == "nt":
        directories = {parent for path in paths for parent in path.parents if parent == root or root in parent.parents}
        for directory in sorted(directories, key=lambda p: (len(p.parts), str(p))):
            _no_links(directory)
            _hold(stack, directory, directory=True)


def _snapshot(handle, relative):
    before = handle.info()
    digest = handle.digest()
    if handle.info() != before:
        raise ValueError("File changed while hashing: " + relative)
    return {"path": relative, **before, "sha256": digest}


def _plan_id(plan):
    data = {key: value for key, value in plan.items() if key != "plan_id"}
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _progress(callback, phase, index, total, byte_count, relative=None):
    if callback:
        callback({"phase": phase, "files_done": index, "files_total": total, "logical_bytes_done": byte_count, "path": relative})


def make_plan(root, files, *, protected=(), progress=None):
    root = _root(root)
    pins = _protected(protected)
    names = list(files)
    if len({name.casefold() for name in names if isinstance(name, str)}) != len(names):
        raise ValueError("Duplicate or invalid file paths")
    with maintenance_lock(root), ExitStack() as stack:
        paths = [_candidate(root, name, pins) for name in names]
        _directories(stack, root, paths)
        entries = []
        byte_count = 0
        for relative, path in zip(names, paths):
            with ExitStack() as one:
                entries.append(_snapshot(_hold(one, path), relative))
            byte_count += entries[-1]["bytes"]
            _progress(progress, "plan", len(entries), len(names), byte_count, relative)
    plan = {"version": VERSION, "root": str(root), "root_identity": _root_identity(root), "created": _now(),
            "protected": pins, "entries": entries, "logical_bytes": sum(e["bytes"] for e in entries),
            "notice": "Caller must stop/lock affected jobs and record retirement before apply. Logical bytes are not reclaimed bytes."}
    plan["plan_id"] = _plan_id(plan)
    return plan


def _validate_plan(root, plan, protected):
    if not isinstance(plan, dict) or plan.get("version") != VERSION or plan.get("plan_id") != _plan_id(plan):
        raise ValueError("Invalid or edited retirement plan")
    if plan.get("root") != str(root) or plan.get("root_identity") != _root_identity(root):
        raise ValueError("Plan workspace does not match")
    if not isinstance(plan.get("protected"), list) or not isinstance(plan.get("entries"), list):
        raise ValueError("Invalid retirement plan lists")
    pins = _protected([*plan["protected"], *protected])
    seen = set()
    for entry in plan["entries"]:
        if not isinstance(entry, dict) or set(entry) != ENTRY_KEYS:
            raise ValueError("Invalid file entry")
        relative = _relative(entry["path"])
        if relative.casefold() in seen:
            raise ValueError("Duplicate entry")
        seen.add(relative.casefold())
        if any(type(entry[k]) is not int for k in ("bytes", "mtime_ns", "nlink")) or entry["bytes"] < 0 or entry["nlink"] < 1:
            raise ValueError("Invalid file metadata")
        if any(not isinstance(entry[k], str) or not entry[k].isdigit() for k in ("device", "inode")):
            raise ValueError("Invalid file identity")
        if not isinstance(entry["sha256"], str) or not re.fullmatch("[a-f0-9]{64}", entry["sha256"]):
            raise ValueError("Invalid digest")
        _candidate(root, relative, pins)
    if plan.get("logical_bytes") != sum(e["bytes"] for e in plan["entries"]):
        raise ValueError("Invalid byte total")
    return pins


def _preflight(root, plan, pins, stack, *, progress=None):
    paths = [_candidate(root, entry["path"], pins) for entry in plan["entries"]]
    _directories(stack, root, paths)
    byte_count = 0
    for index, (entry, path) in enumerate(zip(plan["entries"], paths), 1):
        with ExitStack() as one:
            handle = _hold(one, path)
            actual = _snapshot(handle, entry["path"])
            if actual != entry:
                raise ValueError("File changed since planning: " + entry["path"])
        byte_count += entry["bytes"]
        _progress(progress, "preflight", index, len(paths), byte_count, entry["path"])


def check_plan(root, plan, *, protected=(), progress=None):
    root = _root(root)
    with maintenance_lock(root), ExitStack() as stack:
        pins = _validate_plan(root, plan, protected)
        _preflight(root, plan, pins, stack, progress=progress)
    return {"state": "checked", "dry_run": True, "files": len(plan["entries"]), "logical_bytes": plan["logical_bytes"], "plan_id": plan["plan_id"]}


def execute_plan(root, plan, receipt_path, *, protected=(), apply=False, progress=None):
    """Check by default. Apply creates a NEW receipt, fsyncing every event.

    A failed preflight deletes nothing. A later failure stops further deletion;
    the receipt lists completed entries and the failed entry. To retry, make a
    new plan for remaining files instead of weakening the original snapshot.
    """
    if not apply:
        return check_plan(root, plan, protected=protected, progress=progress)
    if not isinstance(plan, dict):
        raise ValueError("Plan must be an object")
    root = _root(root)
    receipt = Path(receipt_path)
    if not receipt.is_absolute():
        raise ValueError("Receipt must be an absolute workspace path")
    relative = receipt.relative_to(root).as_posix()
    _relative(relative)
    _no_links(receipt.parent)
    pins = _protected([*protected, relative])
    # Receipts cannot overwrite sources, user pins, or old receipts.
    for pin in _protected([*protected, *plan.get("protected", [])]):
        if relative.casefold() == pin.casefold() or relative.casefold().startswith(pin.casefold() + "/"):
            raise ValueError("Receipt path is protected")
    with maintenance_lock(root), ExitStack() as stack:
        _directories(stack, root, [receipt])
        with receipt.open("x", encoding="utf-8", newline="\n") as log:
            def write(event):
                log.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
                log.flush()
                os.fsync(log.fileno())

            result = {"state": "blocked", "dry_run": False, "plan_id": plan.get("plan_id"), "removed_files": 0, "removed_logical_bytes": 0}
            write({"event": "begin", "at": _now(), "root": str(root), "plan_id": plan.get("plan_id")})
            try:
                pins = _validate_plan(root, plan, pins)
                _preflight(root, plan, pins, stack, progress=progress)
            except (ValueError, OSError) as error:
                result["error"] = str(error)
                write({"event": "finish", "at": _now(), **result})
                return result
            write({"event": "preflight_passed", "at": _now(), "files": len(plan["entries"])})
            removed_links = {}
            for entry in plan["entries"]:
                key = (entry["device"], entry["inode"])
                with ExitStack() as one:
                    try:
                        path = _candidate(root, entry["path"], pins)
                        handle = _hold(one, path, deleting=True)
                        expected = {**entry, "nlink": entry["nlink"] - removed_links.get(key, 0)}
                        actual = _snapshot(handle, entry["path"])
                        if actual != expected:
                            raise ValueError("File changed after preflight: " + entry["path"])
                        with ExitStack() as alias_stack:
                            alias = _hold(alias_stack, path, alias=True)
                            if alias.info() != handle.info():
                                raise ValueError("File replaced after preflight: " + entry["path"])
                        write({"event": "delete_intent", "at": _now(), **entry})
                        handle.delete()
                        removed_links[key] = removed_links.get(key, 0) + 1
                        result["removed_files"] += 1
                        result["removed_logical_bytes"] += entry["bytes"]
                        write({"event": "deleted", "at": _now(), **entry})
                        _progress(progress, "delete", result["removed_files"], len(plan["entries"]), result["removed_logical_bytes"], entry["path"])
                    except (ValueError, OSError) as error:
                        result.update(state="partial_failure", error=str(error), failed_path=entry["path"])
                        write({"event": "failed", "at": _now(), "path": entry["path"], "error": str(error)})
                        write({"event": "finish", "at": _now(), **result})
                        return result
            result["state"] = "completed"
            write({"event": "finish", "at": _now(), **result})
            return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("plan", "check", "apply"))
    parser.add_argument("--root", required=True)
    parser.add_argument("--files", help="JSON array of explicit relative file paths")
    parser.add_argument("--plan", required=True, help="Plan output (plan) or existing plan input")
    parser.add_argument("--protect", action="append", default=[], help="Additional relative path to preserve (repeatable)")
    parser.add_argument("--receipt", help="NEW absolute JSONL receipt path inside workspace")
    parser.add_argument("--apply", action="store_true", help="Required with apply to enable deletion")
    args = parser.parse_args(argv)
    last_progress = 0.0

    def progress(event):
        nonlocal last_progress
        current = time.monotonic()
        if current - last_progress >= 10 or event["files_done"] == event["files_total"]:
            print(json.dumps(event, ensure_ascii=False), file=sys.stderr, flush=True)
            last_progress = current
    try:
        if args.operation == "plan":
            if not args.files or args.apply:
                parser.error("plan requires --files and never accepts --apply")
            files = json.loads(Path(args.files).read_text(encoding="utf-8-sig"))
            if not isinstance(files, list):
                raise ValueError("--files must contain a JSON array")
            result = make_plan(args.root, files, protected=args.protect, progress=progress)
            target = Path(args.plan)
            root = _root(args.root)
            if not target.is_absolute():
                raise ValueError("Plan output must be an absolute workspace path")
            relative = _relative(target.relative_to(root).as_posix())
            _no_links(target.parent)
            if any(relative.casefold() == pin.casefold() or relative.casefold().startswith(pin.casefold() + "/") for pin in result["protected"]):
                raise ValueError("Plan output is protected")
            # New output only; never overwrite a plan or candidate file.
            with target.open("x", encoding="utf-8") as stream:
                json.dump(result, stream, ensure_ascii=False, indent=2)
            print(json.dumps({"state": "planned", "plan_id": result["plan_id"], "files": len(result["entries"]), "logical_bytes": result["logical_bytes"]}))
        else:
            plan = json.loads(Path(args.plan).read_text(encoding="utf-8-sig"))
            if args.operation == "apply" and args.apply and not args.receipt:
                parser.error("Deletion requires --receipt")
            result = execute_plan(args.root, plan, args.receipt, protected=args.protect, apply=args.operation == "apply" and args.apply, progress=progress)
            print(json.dumps(result, ensure_ascii=False))
            return 0 if result["state"] in ("checked", "completed") else 2
    except (ValueError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
