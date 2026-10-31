"""Real OS sharing tests. Child interpreters model separate installed processes."""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys

import pytest
from starstack import install_lifecycle as life


def child_code(root, statement):
    return ("import importlib.util; from pathlib import Path; "
            f"s=importlib.util.spec_from_file_location('lease', {str(Path(life.__file__).resolve())!r}); "
            "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
            f"root=Path({str(root)!r}); " + statement)


def run(root, statement):
    return subprocess.run([sys.executable, '-X', 'utf8', '-B', '-c', child_code(root, statement)],
                          capture_output=True, text=True, encoding='utf-8', timeout=15,
                          creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))


@pytest.fixture
def installation(tmp_path):
    root = tmp_path / '含空格 Aester'
    life.initialize_install_root(root)
    return root


def test_shared_readers_can_coexist_but_exclusive_waits(installation):
    with life.runtime_lease(installation):
        child = run(installation, "lease=m.runtime_lease(root); lease.__enter__(); print('ok')")
        assert child.returncode == 0 and 'ok' in child.stdout
        with pytest.raises(life.InstallationBusy):
            with life.exclusive_maintenance(installation):
                pytest.fail('active application permitted destructive maintenance')
    assert not (installation / life.PENDING).exists()
    with life.exclusive_maintenance(installation):
        life.clear_pending(installation)


def test_exclusive_blocks_new_process_and_pending_blocks_after_release(installation):
    with life.exclusive_maintenance(installation):
        child = run(installation, "lease=m.runtime_lease(root); lease.__enter__()")
        assert child.returncode != 0
        assert 'InstallationBusy' in child.stderr
    with pytest.raises(life.InstallationBusy):
        with life.runtime_lease(installation):
            pytest.fail('new process bypassed pending uninstall')
    with life.exclusive_maintenance(installation):
        life.clear_pending(installation)
    with life.runtime_lease(installation):
        pass


def test_failed_maintenance_keeps_pending_until_explicit_safe_recovery(installation):
    with pytest.raises(ValueError, match='delete failed'):
        with life.exclusive_maintenance(installation):
            raise ValueError('delete failed')
    assert (installation / life.PENDING).exists()
    with pytest.raises(RuntimeError):
        life.clear_pending(installation)
    life.initialize_install_root(installation)
    assert (installation / life.PENDING).exists()


def test_non_destructive_maintenance_can_leave_no_pending(installation):
    with life.exclusive_maintenance(installation, mark_pending=False):
        assert not (installation / life.PENDING).exists()
    with life.runtime_lease(installation):
        pass


def test_import_installed_package_holds_until_process_dies(installation):
    package = installation / 'versions/test1/src/starstack'
    package.mkdir(parents=True)
    for name in ['__init__.py', 'install_lifecycle.py']:
        shutil.copyfile(Path(life.__file__).parent / name, package / name)
    env = dict(os.environ, PYTHONPATH=str(package.parent))
    proc = subprocess.Popen([sys.executable, '-X', 'utf8', '-B', '-u', '-c',
        "import starstack, time; print('ready', flush=True); time.sleep(60)"],
        env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding='utf-8', creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    try:
        # Poll pipe readability via a bounded thread to avoid a hung test.
        from concurrent.futures import ThreadPoolExecutor
        pool = ThreadPoolExecutor(1)
        try:
            assert pool.submit(proc.stdout.readline).result(timeout=15).strip() == 'ready'
        finally:
            pool.shutdown(wait=False)
        with pytest.raises(life.InstallationBusy):
            with life.exclusive_maintenance(installation):
                pytest.fail('installed direct Python import was not leased')
    finally:
        proc.kill()
        proc.wait(timeout=10)
        proc.stdout.close()
        proc.stderr.close()
    with life.exclusive_maintenance(installation):
        life.clear_pending(installation)
    assert not list(package.rglob('*.pyc'))


def test_protocol_root_mismatch_fails_without_touching_files(installation, tmp_path):
    protocol = installation / life.PROTOCOL
    protocol.write_text(json.dumps({'schema_version': 1, 'install_root': str(tmp_path / 'other')}))
    before = protocol.read_bytes()
    with pytest.raises(ValueError):
        with life.exclusive_maintenance(installation):
            pass
    assert protocol.read_bytes() == before
    assert not (installation / life.PENDING).exists()


def test_dev_layout_does_not_get_runtime_leases(tmp_path):
    assert life.installed_root(Path(life.__file__)) is None
    assert life.ensure_process_lease(tmp_path / 'src/starstack/__init__.py') is False


def test_installed_layout_with_deleted_metadata_fails_closed(installation):
    module = installation / 'versions/v1/src/starstack/__init__.py'
    (installation / life.PROTOCOL).unlink()
    assert life.installed_root(module) == installation
    with pytest.raises(FileNotFoundError):
        life.ensure_process_lease(module)


def test_identity_and_gate_are_not_replaced_on_reinitialize(installation):
    records = [(installation / name).stat() for name in [life.PROTOCOL, life.LOCK]]
    with life.runtime_lease(installation):
        life.initialize_install_root(installation)
    after = [(installation / name).stat() for name in [life.PROTOCOL, life.LOCK]]
    assert [(s.st_ino, s.st_mtime_ns) for s in records] == [(s.st_ino, s.st_mtime_ns) for s in after]


def test_missing_gate_refuses_to_silently_open_an_unprotected_install(installation):
    (installation / life.LOCK).unlink()
    with pytest.raises(ValueError):
        with life.runtime_lease(installation):
            pass


def test_hardlink_gate_rejected(installation, tmp_path):
    os.link(installation / life.LOCK, tmp_path / 'gate-alias')
    with pytest.raises(ValueError, match='硬链接'):
        with life.exclusive_maintenance(installation):
            pass
    assert not (installation / life.PENDING).exists()


@pytest.mark.skipif(os.name != 'nt', reason='Windows native sharing rule')
def test_active_lease_prevents_gate_replacement(installation):
    with life.runtime_lease(installation):
        with pytest.raises(PermissionError):
            (installation / life.LOCK).unlink()


@pytest.mark.skipif(os.name != 'nt', reason='Windows junction')
def test_junction_install_root_rejected(installation, tmp_path):
    junction = tmp_path / 'redirected'
    # Directory junction creation is a single native PowerShell operation.
    # These test-only paths are passed as PowerShell single-quoted literals.
    quoted = lambda p: "'" + str(p).replace("'", "''") + "'"
    cmd = f"New-Item -ItemType Junction -Path {quoted(junction)} -Target {quoted(installation)} | Out-Null"
    result = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', cmd],
                            capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr
    try:
        with pytest.raises(ValueError, match='链接'):
            with life.runtime_lease(junction):
                pass
    finally:
        # rmdir here removes the verified junction itself; never its target tree.
        assert junction.is_junction()
        junction.rmdir()
