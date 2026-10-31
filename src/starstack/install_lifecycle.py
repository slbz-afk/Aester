"""Installation-wide leases, also imported directly by the standalone uninstaller.

Every installed Python process holds a shared lease from package import until
exit. Maintenance takes the exclusive lease, then leaves a pending marker so
new starts cannot enter while Inno finishes deleting its registered files.
No process names, PIDs or global Python installations are used as ownership.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import atexit
import json
import os
import stat
import sys
import threading
import uuid

PROTOCOL = "lifecycle.json"
LOCK = "runtime-use.lock"
PENDING = "uninstall.pending"
_held = None
_process_lock = threading.RLock()
_exclusive_roots = set()


class InstallationBusy(RuntimeError):
    """An active application or maintenance operation owns this installation."""


def _safe(path):
    path = Path(os.path.abspath(path))
    if path == Path(path.anchor):
        raise ValueError("需要具体的 Aester 安装目录")
    for part in (path, *path.parents):
        try:
            info = part.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("安装使用记录不能经过链接：" + str(part))
        if stat.S_ISREG(info.st_mode) and info.st_nlink != 1:
            raise ValueError("安装使用记录不能使用硬链接：" + str(part))
    return path


def _validate(root):
    root = _safe(root)
    metadata = _safe(root / PROTOCOL)
    if metadata.stat().st_size > 8192:
        raise ValueError("安装使用记录无效，请修复安装")
    data = json.loads(metadata.read_text(encoding="utf-8"))
    if (not isinstance(data, dict) or data.get("schema_version") != 1
            or not isinstance(data.get("install_root"), str)
            or _safe(data["install_root"]) != root):
        raise ValueError("安装使用记录不属于当前安装")
    gate = _safe(root / LOCK)
    if not gate.is_file():
        raise ValueError("安装使用锁丢失，请修复安装")
    return root


def initialize_install_root(root):
    """Install-time initialization; never clears a pending uninstall implicitly."""
    root = _safe(root)
    root.mkdir(parents=True, exist_ok=True)
    _safe(root)
    gate = _safe(root / LOCK)
    try:
        with gate.open("xb") as stream:
            stream.write(b"0")
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        if not gate.is_file():
            raise ValueError("安装使用锁不是普通文件")
    metadata = _safe(root / PROTOCOL)
    if not metadata.exists():
        # A concurrent initializer may win. Link-free creation never overwrites
        # an established installation's identity or active lock file.
        data = {"schema_version": 1, "install_root": str(root)}
        try:
            with metadata.open("x", encoding="utf-8") as stream:
                json.dump(data, stream)
                stream.flush()
                os.fsync(stream.fileno())
        except FileExistsError:
            pass
    return _validate(root)


@contextmanager
def _open_lease(root, exclusive):
    root = _validate(root)
    path = _safe(root / LOCK)
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes as wt
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateFileW.argtypes = [wt.LPCWSTR, wt.DWORD, wt.DWORD, wt.LPVOID,
                                      wt.DWORD, wt.DWORD, wt.HANDLE]
        kernel.CreateFileW.restype = wt.HANDLE
        kernel.CloseHandle.argtypes = [wt.HANDLE]
        kernel.CloseHandle.restype = wt.BOOL
        # OPEN_EXISTING + OPEN_REPARSE_POINT, no inheritable handles. Shared
        # readers deny write/delete; maintenance requests read/write, no sharing.
        absolute = str(path)
        if not absolute.startswith("\\\\?\\"):
            absolute = "\\\\?\\UNC\\" + absolute[2:] if absolute.startswith("\\\\") else "\\\\?\\" + absolute
        handle = kernel.CreateFileW(absolute, 0xC0000000 if exclusive else 0x80000000,
                                    0 if exclusive else 1, None, 3, 0x00200000, None)
        if handle == ctypes.c_void_p(-1).value:
            error = ctypes.get_last_error()
            if error in (32, 33):
                raise InstallationBusy("Aester 或后台任务仍在使用此安装，请先退出或等待任务结束")
            raise ctypes.WinError(error)
        try:
            # Check the opened object, not just the name inspected before open.
            class Info(ctypes.Structure):
                _fields_ = [("attributes", wt.DWORD), ("created", wt.FILETIME),
                            ("accessed", wt.FILETIME), ("written", wt.FILETIME),
                            ("volume", wt.DWORD), ("size_high", wt.DWORD),
                            ("size_low", wt.DWORD), ("links", wt.DWORD),
                            ("index_high", wt.DWORD), ("index_low", wt.DWORD)]
            kernel.GetFileInformationByHandle.argtypes = [wt.HANDLE, ctypes.POINTER(Info)]
            kernel.GetFileInformationByHandle.restype = wt.BOOL
            info = Info()
            if not kernel.GetFileInformationByHandle(handle, ctypes.byref(info)):
                raise ctypes.WinError(ctypes.get_last_error())
            if info.attributes & (0x400 | 0x10) or info.links != 1:
                raise ValueError("安装使用锁已被替换为链接或目录")
            _safe(root)
            yield root
        finally:
            kernel.CloseHandle(handle)
    else:
        import fcntl
        descriptor = os.open(path, os.O_RDWR if exclusive else os.O_RDONLY)
        try:
            try:
                fcntl.flock(descriptor, (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise InstallationBusy("Aester 或后台任务仍在使用此安装") from error
            yield root
        finally:
            os.close(descriptor)


@contextmanager
def runtime_lease(root):
    with _open_lease(root, False) as owned:
        if _safe(owned / PENDING).exists():
            raise InstallationBusy("此 Aester 安装正在维护或上次维护尚未完成，请完成安装、卸载或修复后再启动")
        yield owned


@contextmanager
def exclusive_maintenance(root, *, mark_pending=True):
    """Keep the pending marker on success OR error; the caller decides rollback.

    Only a caller that knows no destructive work happened may clear_pending.
    On success Inno removes it last, after all launchable app files are gone.
    """
    with _open_lease(root, True) as owned:
        if mark_pending:
            marker = _safe(owned / PENDING)
            temporary = _safe(owned / (PENDING + "." + uuid.uuid4().hex + ".tmp"))
            try:
                with temporary.open("x", encoding="utf-8") as stream:
                    json.dump({"schema_version": 1, "install_root": str(owned)}, stream)
                    stream.flush()
                    os.fsync(stream.fileno())
                temporary.replace(marker)
            finally:
                temporary.unlink(missing_ok=True)
        with _process_lock:
            _exclusive_roots.add(owned)
        try:
            yield owned
        finally:
            with _process_lock:
                _exclusive_roots.discard(owned)


def clear_pending(root):
    root = _validate(root)
    with _process_lock:
        if root not in _exclusive_roots:
            raise RuntimeError("只能持有安装维护独占锁时清除卸载标记")
    _safe(root / PENDING).unlink(missing_ok=True)


def installed_root(module_file):
    """Recognize installed layout; never trust an environment-provided delete path."""
    source = Path(os.path.abspath(module_file))
    if len(source.parents) < 5:
        return None
    package, src, version, versions, app = source.parents[:5]
    if package.name != "starstack" or src.name != "src" or versions.name != "versions":
        return None
    # The new package must fail closed when its maintenance metadata disappears.
    # Historical packages do not call this function; development checkouts do
    # not use the app/versions/<version>/src/starstack installed layout.
    return _safe(app)


def ensure_process_lease(module_file=None):
    """Called at the very start of an installed starstack import, including MCP."""
    global _held
    root = installed_root(module_file or __file__)
    if root is None:
        return False
    with _process_lock:
        if _held is not None:
            return True
        lease = runtime_lease(root)
        lease.__enter__()
        _held = lease
        sys.dont_write_bytecode = True
        def release():
            global _held
            with _process_lock:
                held, _held = _held, None
            if held is not None:
                held.__exit__(None, None, None)
        atexit.register(release)
    return True
