"""Read-only discovery and opt-in version checks for an external GIMP install.

No settings, installation ownership or preferences are changed. Discovery does
not start executables. The only optional process is an owned ``--version`` call.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time

from .finishing_assets import ordinary_path
from .process_guard import guarded_process, guard_kind, GUARD_JOB_OBJECT

TESTED_GIMP_VERSION = "3.2.6"
INSTALL_URL = "https://www.gimp.org/downloads/"
COMMAND_REFERENCE_URL = "https://www.gimp.org/man/gimp.html"
_CONSOLE_NAMES = ("gimp-console-3.2.exe", "gimp-console-3.0.exe", "gimp-console-2.10.exe")
_UNINSTALL_KEYS = tuple("Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\" + name
                        for name in ("GIMP-3_is1", "GIMP-2_is1"))
_OUTPUT_LIMIT = 64 * 1024


def _registry_locations():
    """Read at most eight named keys; never enumerate the uninstall registry."""
    locations, diagnostics = [], []
    if os.name != "nt":
        return locations, diagnostics
    import winreg
    for hive_name in ("HKEY_CURRENT_USER", "HKEY_LOCAL_MACHINE"):
        hive = getattr(winreg, hive_name)
        for view_name in ("KEY_WOW64_64KEY", "KEY_WOW64_32KEY"):
            for subkey in _UNINSTALL_KEYS:
                label = f"{hive_name}/{view_name}/{subkey}"
                try:
                    with winreg.OpenKey(hive, subkey, 0, winreg.KEY_READ | getattr(winreg, view_name)) as key:
                        name = winreg.QueryValueEx(key, "DisplayName")[0]
                        location = winreg.QueryValueEx(key, "InstallLocation")[0]
                        try:
                            version = winreg.QueryValueEx(key, "DisplayVersion")[0]
                        except OSError:
                            version = None
                    if not isinstance(name, str) or not re.match(r"^GIMP(?:\s|$)", name, re.I):
                        diagnostics.append({"code": "registry_name_unrecognized", "location": label})
                        continue
                    if not isinstance(location, str) or not location.strip():
                        diagnostics.append({"code": "registry_location_missing", "location": label})
                        continue
                    locations.append((os.path.expandvars(location), label,
                                      version if isinstance(version, str) else None))
                except FileNotFoundError:
                    continue
                except OSError as exc:
                    diagnostics.append({"code": "registry_check_failed", "location": label, "message": str(exc)})
    return locations, diagnostics


def _standard_locations():
    """A bounded list of standard install roots; no PATH search or disk walk."""
    if os.name != "nt":
        return []
    roots = [os.environ.get("ProgramFiles"), os.environ.get("ProgramFiles(x86)")]
    local = os.environ.get("LOCALAPPDATA")
    if local:
        roots.append(str(Path(local) / "Programs"))
    return [str(Path(root) / name) for root in roots if root for name in ("GIMP 3", "GIMP 2")]


def _candidate(value, source, *, registry_version=None):
    record = {"requested_path": None,
              "path": None, "source": source, "exists": False, "error": None,
              "registry_version_hint": registry_version}
    try:
        if not isinstance(value, (str, os.PathLike)):
            raise ValueError("GIMP path must be an absolute executable path")
        raw = os.fspath(value)
        if not isinstance(raw, str) or not raw.strip() or "\x00" in raw:
            raise ValueError("GIMP path must be nonempty text without NUL")
        record["requested_path"] = raw
        path = Path(raw).expanduser()
        if not path.is_absolute() or ".." in path.parts:
            raise ValueError("GIMP path must be absolute and contain no parent traversal")
        record["path"] = str(path)
        if path.suffix.lower() != ".exe":
            raise ValueError("Choose the Windows GIMP console .exe, not a command or script")
        checked = ordinary_path(path)
        record["path"] = str(checked)
        record["exists"] = True
    except FileNotFoundError:
        record["error"] = "Selected executable does not exist"
    except (OSError, ValueError, TypeError) as exc:
        record["error"] = str(exc)
    return record


def _automatic_candidates():
    locations, diagnostics = _registry_locations()
    candidates, seen = [], set()
    entries = [(path, "registry", label, version) for path, label, version in locations]
    entries += [(path, "standard", None, None) for path in _standard_locations()]
    for location, source, label, version in entries:
        directory = Path(location)
        if not directory.is_absolute() or ".." in directory.parts:
            diagnostics.append({"code": "install_location_invalid", "location": str(location)})
            continue
        names = list(_CONSOLE_NAMES)
        # A registry hint can suggest one fixed console basename, but cannot
        # certify the version, choose command arguments or trigger execution.
        match = re.fullmatch(r"(\d{1,3})\.(\d{1,3})\.\d{1,3}", version or "")
        if match:
            hinted = f"gimp-console-{match[1]}.{match[2]}.exe"
            names = [hinted, *[name for name in names if name != hinted]]
        for name in names:
            path = directory / "bin" / name
            identity = os.path.normcase(str(path))
            if identity in seen:
                continue
            seen.add(identity)
            record = _candidate(path, source, registry_version=version)
            if label is not None:
                record["registry_key"] = label
            candidates.append(record)
    return candidates, diagnostics


_VERSION_TOKEN = r"(?<![\w.])(\d+\.\d+\.\d+(?:[-+][A-Za-z0-9.-]+)?)(?![\w.])"


def _parse_version(text):
    pattern = r"^\s*(?:GNU Image Manipulation Program version|GIMP(?: version)?)\s+(\d+\.\d+\.\d+(?:[-+][A-Za-z0-9.-]+)?)\s*$"
    versions = set(re.findall(pattern, text, re.MULTILINE | re.IGNORECASE))
    if not versions:
        # GIMP translates this line with the Windows display language when LANG is
        # unset (e.g. "GNU 图像处理程序 版本 3.2.6"). Accept a localized answer only
        # when it is a single line that keeps GNU/GIMP and exactly one version number.
        lines = [line for line in text.splitlines() if line.strip()]
        if len(lines) == 1 and re.search(r"\b(?:GNU|GIMP)\b", lines[0], re.IGNORECASE):
            versions = set(re.findall(_VERSION_TOKEN, lines[0]))
    if len(versions) != 1:
        raise ValueError("Version output did not identify exactly one GIMP version")
    return versions.pop()


def _probe_version(path, timeout):
    if guard_kind() != GUARD_JOB_OBJECT:
        raise RuntimeError("Owned GIMP version checks are currently supported on Windows only")
    executable = ordinary_path(path)
    command = [str(executable), "--version"]
    started = time.monotonic()
    # --version is documented to print and exit without opening a GUI. Do not
    # load files, run batches or alter application/profile configuration here.
    with tempfile.TemporaryFile("w+b") as stdout, tempfile.TemporaryFile("w+b") as stderr:
        with guarded_process(command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                             cwd=str(executable.parent),
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)) as process:
            while process.poll() is None:
                if time.monotonic() - started >= timeout:
                    process.kill()
                    raise TimeoutError("GIMP version check timed out")
                if os.fstat(stdout.fileno()).st_size + os.fstat(stderr.fileno()).st_size > _OUTPUT_LIMIT:
                    process.kill()
                    raise ValueError("GIMP version output exceeded its size budget")
                time.sleep(0.02)
            code = process.returncode
        stdout.seek(0)
        stderr.seek(0)
        data = stdout.read(_OUTPUT_LIMIT + 1) + stderr.read(_OUTPUT_LIMIT + 1)
    if len(data) > _OUTPUT_LIMIT:
        raise ValueError("GIMP version output exceeded its size budget")
    text = data.decode("utf-8", errors="replace")
    if code != 0:
        raise RuntimeError(f"GIMP version check exited with status {code}: {text[:2000]}")
    return {"version": _parse_version(text), "seconds": time.monotonic() - started,
            "output": text, "returncode": code}


def discover_gimp(selected_path=None, *, probe=False, timeout=5.0) -> dict:
    """Discover an external install; optionally check only its reported version.

    Explicit configuration and AESTER_GIMP pin the selection, including missing
    selections. Automatic alternatives never replace a pinned choice. No state
    is saved. Callers must separately request and verify actual render products.
    """
    result = {"schema": "aester-finishing-component-1", "status": "not_found",
              "selected_path": None, "selection_source": None, "selection_locked": False,
              "path_exists": False, "version": None, "version_checked": False,
              "version_compatible": None, "operator_verified": False,
              "managed_by_aester": False, "ownership": "external_unmanaged",
              "install_url": INSTALL_URL, "command_reference_url": COMMAND_REFERENCE_URL,
              "tested_version": TESTED_GIMP_VERSION, "candidates": [], "diagnostics": []}
    if (type(probe) is not bool or type(timeout) not in {int, float}
            or not math.isfinite(timeout) or not 0 < timeout <= 15):
        result["status"] = "check_failed"
        result["diagnostics"].append({"code": "invalid_check_options", "message": "probe must be bool; timeout must be finite, >0 and <=15 seconds"})
        return result
    if selected_path is not None:
        candidates = [_candidate(selected_path, "explicit")]
        result["selection_locked"] = True
    elif os.environ.get("AESTER_GIMP"):
        candidates = [_candidate(os.environ["AESTER_GIMP"], "environment")]
        result["selection_locked"] = True
    else:
        candidates, result["diagnostics"] = _automatic_candidates()
    result["candidates"] = candidates
    selected = candidates[0] if result["selection_locked"] else next((item for item in candidates if item["exists"]), None)
    if selected is None:
        incomplete = any(item["error"] not in {None, "Selected executable does not exist"} for item in candidates)
        incomplete |= any(item["code"] == "registry_check_failed" for item in result["diagnostics"])
        result["status"] = "check_failed" if incomplete else "not_found"
        result["diagnostics"].append({"code": result["status"], "message": "No usable executable found in the bounded registry/standard locations; select an existing GIMP console path or use the official installation page"})
        return result
    result.update(selected_path=selected["path"] or selected["requested_path"],
                  selection_source=selected["source"], path_exists=selected["exists"])
    if not selected["exists"]:
        result["status"] = "selected_missing" if selected["error"] == "Selected executable does not exist" else "check_failed"
        result["diagnostics"].append({"code": result["status"], "message": selected["error"]})
        return result
    result["status"] = "available"
    if not probe:
        result["diagnostics"].append({"code": "version_not_checked", "message": "Executable found; neither its version nor its operators have been verified"})
        return result
    try:
        receipt = _probe_version(selected["path"], float(timeout))
        result.update(version=receipt["version"], version_checked=True, version_probe=receipt)
        result["version_compatible"] = receipt["version"] == TESTED_GIMP_VERSION
        if not result["version_compatible"]:
            result["status"] = "incompatible_version"
        result["diagnostics"].append({"code": "operator_validation_required",
            "message": "Version inspection does not validate TIFF, masks, operations or XCF replay"})
    except (OSError, ValueError, RuntimeError) as exc:
        result["status"] = "check_failed"
        result["diagnostics"].append({"code": "version_check_failed", "message": str(exc)})
    return result
