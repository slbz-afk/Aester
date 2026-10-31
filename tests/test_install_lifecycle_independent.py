"""Independent installed entrypoint checks, including Python without -B."""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys

from starstack import install_lifecycle as life


def installed_package(tmp_path):
    root = tmp_path / "private-app"
    life.initialize_install_root(root)
    package = root / "versions/new/src/starstack"
    package.mkdir(parents=True)
    for name in ("__init__.py", "install_lifecycle.py"):
        shutil.copyfile(Path(life.__file__).parent / name, package / name)
    return root, package


def run_plain(package, extra="", *, no_bytecode=False):
    env = dict(os.environ, PYTHONPATH=str(package.parent))
    env.pop("PYTHONDONTWRITEBYTECODE", None)
    return subprocess.run([sys.executable, "-X", "utf8", *(["-B"] if no_bytecode else []), "-c", "import starstack; " + extra],
        capture_output=True, text=True, encoding="utf-8", env=env, timeout=15,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def test_plain_python_first_import_only_may_cache_init_before_its_code_executes(tmp_path):
    root, package = installed_package(tmp_path)
    result = run_plain(package)
    assert result.returncode == 0, result.stderr
    # CPython compiles/writes __init__ before it can execute dont_write_bytecode.
    # This one owned cache is handled by uninstall; no later module may cache.
    assert all(path.name.startswith("__init__.") for path in package.rglob("*.pyc"))


def test_official_no_bytecode_launch_does_not_leave_any_application_pyc(tmp_path):
    root, package = installed_package(tmp_path)
    result = run_plain(package, no_bytecode=True)
    assert result.returncode == 0, result.stderr
    assert not list(package.rglob("*.pyc"))


def test_damaged_new_install_does_not_downgrade_to_unleased_legacy(tmp_path):
    root, package = installed_package(tmp_path)
    (root / life.PROTOCOL).unlink()
    assert not (root / life.PENDING).exists()
    result = run_plain(package, no_bytecode=True)
    assert result.returncode != 0


def test_cancellation_before_mutation_only_reopens_under_exclusive(tmp_path):
    root, package = installed_package(tmp_path)
    source = (package / "__init__.py").read_bytes()
    with life.exclusive_maintenance(root):
        blocked = run_plain(package)
        assert blocked.returncode != 0
        life.clear_pending(root)
        still_blocked = run_plain(package)
        assert still_blocked.returncode != 0
    reopened = run_plain(package)
    assert reopened.returncode == 0, reopened.stderr
    assert (package / "__init__.py").read_bytes() == source


def test_finished_helper_pending_blocks_new_import_until_inno_finishes(tmp_path):
    root, package = installed_package(tmp_path)
    with life.exclusive_maintenance(root):
        pass
    blocked = run_plain(package)
    assert blocked.returncode != 0
    assert "InstallationBusy" in blocked.stderr
