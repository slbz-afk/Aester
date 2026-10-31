"""Runtime Siril-environment guard for the whole test session (efficiency P1b W1b; installed by conftest.py).

Every process start in this pytest process whose executable is a Siril binary (``siril`` or ``siril-cli``, with or
without ``.exe``) must carry, in its effective environment (the ``env`` it passes, else this process's):
  R1  an XDG_DATA_HOME that is set and is not the real per-user data folder (FOLDERID_LocalAppData, which GLib uses
      when XDG_DATA_HOME is unset), so Siril's Python venv work can never reach the user's %LOCALAPPDATA%\\siril
      (S-4b);
  R2  DBUS_SESSION_BUS_ADDRESS=disabled:, so Siril neither starts nor joins the user's D-Bus session bus (S-4d).
A launch whose argv is exactly ``[exe, "-v"]`` is exempt: the version probe exits during option parsing, before Siril
looks at a data folder or a bus (main-cli.c:86-93).

Hooks, all looked up at call time, so no import alias of the caller can bypass them:
  process_guard._JobObjectProcess.start  every Windows guarded_process start (skipped when a test passes a fake
                                         ``api=``: tests/test_process_guard.py creates no process that way);
  process_guard._PopenProcess.start      POSIX and explicitly unguarded starts;
  subprocess.Popen.__init__              subprocess.run/call/check_*/Popen and asyncio's Windows Popen subclass.
A violation raises SirilEnvironmentViolation, a BaseException, so a product ``except Exception`` cannot hide it, and is
recorded; pytest_sessionfinish (conftest.py) fails the session when any was recorded, which also catches launches
from worker threads. The static counterpart (every launch site in the code base) is test_siril_launch_sites.py.
"""
import ctypes
import inspect
import ntpath
import os
import re
import subprocess
import uuid

_SIRIL = re.compile(r"^siril(-cli)?(\.exe)?$", re.IGNORECASE)
NO_SESSION_BUS = "disabled:"
VIOLATIONS: list[str] = []


class SirilEnvironmentViolation(BaseException):
    """A Siril process was about to start with the user's data folder or session bus."""


def _real_local_appdata() -> str:
    if os.name == "nt":
        class GUID(ctypes.Structure):
            _fields_ = [("Data1", ctypes.c_uint32), ("Data2", ctypes.c_uint16), ("Data3", ctypes.c_uint16),
                        ("Data4", ctypes.c_ubyte * 8)]

        known = uuid.UUID("F1B32785-6FBA-4FCF-9D55-7B8E7F157091")          # FOLDERID_LocalAppData
        guid = GUID(known.fields[0], known.fields[1], known.fields[2], (ctypes.c_ubyte * 8)(*known.bytes[8:]))
        path = ctypes.c_wchar_p()
        shell32, ole32 = ctypes.windll.shell32, ctypes.windll.ole32
        if shell32.SHGetKnownFolderPath(ctypes.byref(guid), 0, None, ctypes.byref(path)) == 0:
            try:
                return path.value
            finally:
                ole32.CoTaskMemFree(path)
    return os.path.join(os.path.expanduser("~"), ".local", "share")


REAL_DATA = os.path.normcase(os.path.abspath(_real_local_appdata()))


def _argv(args) -> list[str]:
    if isinstance(args, (str, bytes)):
        text = os.fsdecode(args).strip()
        if text.startswith('"'):
            head, _, rest = text[1:].partition('"')
            return [head] + rest.split()
        return text.split()
    return [os.fsdecode(os.fspath(a)) for a in args]


def _lookup(env, key: str):
    """Case-insensitive, as Windows environment blocks are."""
    for name, value in env.items():
        if os.fsdecode(name).upper() == key:
            return os.fsdecode(value)
    return None


def is_siril(executable) -> bool:
    return bool(_SIRIL.match(ntpath.basename(os.fsdecode(os.fspath(executable)).strip().strip('"'))))


def problems(args, env, executable=None) -> list[str]:
    """The rules a start of ``args`` with ``env`` breaks; [] for anything that is not Siril."""
    argv = _argv(args)
    exe = executable if executable else (argv[0] if argv else "")
    if not exe or not is_siril(exe) or argv[1:] == ["-v"]:
        return []
    effective = os.environ if env is None else env
    found = []
    xdg = _lookup(effective, "XDG_DATA_HOME")
    if not xdg or os.path.normcase(os.path.abspath(xdg)) == REAL_DATA:
        found.append(f"R1 XDG_DATA_HOME={xdg!r} is the user's data folder")
    bus = _lookup(effective, "DBUS_SESSION_BUS_ADDRESS")
    if bus != NO_SESSION_BUS:
        found.append(f"R2 DBUS_SESSION_BUS_ADDRESS={bus!r}, not {NO_SESSION_BUS!r}")
    return found


def check(args, env, executable=None) -> None:
    found = problems(args, env, executable)
    if found:
        message = (f"Siril started with the user's environment ({'inherited' if env is None else 'explicit env'}; "
                   f"{'; '.join(found)}): {_argv(args)!r}. Launch Siril only through siril.run_script, or pass "
                   "env=siril.siril_environment(<private folder>).")
        VIOLATIONS.append(message)
        raise SirilEnvironmentViolation(message)


def install(monkeypatch) -> None:
    """Wrap the three start points (undone with ``monkeypatch.undo()``)."""
    from starstack import process_guard as pg

    def wrap_start(owner):
        original = owner.__dict__["start"].__func__

        def start(cls, args, *a, **kw):
            if kw.get("api") is None:          # a fake kernel32 creates no process
                check(args, kw.get("env"))
            return original(cls, args, *a, **kw)

        monkeypatch.setattr(owner, "start", classmethod(start))

    wrap_start(pg._JobObjectProcess)
    wrap_start(pg._PopenProcess)
    original_init = subprocess.Popen.__init__
    signature = inspect.signature(original_init)

    def init(self, *a, **kw):
        bound = signature.bind(self, *a, **kw)
        check(bound.arguments.get("args"), bound.arguments.get("env"), bound.arguments.get("executable"))
        return original_init(self, *a, **kw)

    monkeypatch.setattr(subprocess.Popen, "__init__", init)
