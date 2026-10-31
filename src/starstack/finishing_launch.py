"""Fixed, guarded child entrypoint for the optional GIMP backend.

This file is launched by absolute path with Python -I. It does not import the
host package or accept scripts/commands from recipes. The Windows job inherited
from the parent owns both this launcher and GIMP's plugin descendants.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from xml.sax.saxutils import escape


def isolated_environment(work: Path, request: Path) -> dict:
    env = os.environ.copy()
    for key in tuple(env):
        if (key.upper().startswith(("PYTHON", "GIMP", "XDG_", "FONTCONFIG_", "GEGL_",
                                    "BABL_", "GTK_", "GIO_", "GSETTINGS_"))
                or key.upper() == "GI_TYPELIB_PATH"):
            env.pop(key)
    names = {"GIMP3_DIRECTORY": "profile", "XDG_CONFIG_HOME": "xdg-config",
             "XDG_CACHE_HOME": "xdg-cache", "GIMP3_CACHEDIR": "cache",
             "GIMP3_TEMPDIR": "temp", "TEMP": "temp", "TMP": "temp"}
    for key, name in names.items():
        directory = work / name
        directory.mkdir(exist_ok=True)
        env[key] = str(directory)
    (work / "profile" / "gimprc").write_text(
        "(check-updates no)\n(use-opencl no)\n", encoding="utf-8")
    fontcache = work / "font-cache"
    fontcache.mkdir(exist_ok=True)
    fontconfig = work / "fonts.conf"
    fontconfig.write_text(
        '<?xml version="1.0"?><!DOCTYPE fontconfig SYSTEM "urn:fontconfig:fonts.dtd">'
        '<fontconfig><cachedir>' + escape(fontcache.as_posix()) + '</cachedir></fontconfig>',
        encoding="utf-8")
    env.update({"FONTCONFIG_FILE": str(fontconfig), "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONIOENCODING": "utf-8", "AESTER_FINISHING_REQUEST": str(request),
                "AESTER_FINISHING_DRIVER": str(Path(__file__).with_name("finishing_gimp_script.py"))})
    return env


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        raise ValueError("Expected host-configured GIMP executable and private request")
    executable, request = (Path(item).resolve(strict=True) for item in args)
    work = request.parent
    data = json.loads(request.read_text(encoding="utf-8"))
    if data.get("schema") not in {"aester-finishing-render-1", "aester-finishing-render-2"}:
        raise ValueError("Invalid finishing render request")
    env = isolated_environment(work, request)
    code = "import os,runpy;runpy.run_path(os.environ['AESTER_FINISHING_DRIVER'],run_name='__main__')"
    command = [str(executable), "--new-instance", "--no-interface", "--no-splash",
               "--no-fonts", "--console-messages", "--batch-interpreter=python-fu-eval",
               "--batch=" + code, "--quit"]
    with subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL,
                          creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)) as process:
        return process.wait()


if __name__ == "__main__":
    raise SystemExit(main())
