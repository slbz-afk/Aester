"""Self-test of the session's runtime Siril guard (tests/siril_guard.py, efficiency P1b W1b).

Every launch form the static scan could miss is stopped at run time. The "Siril" here is a copy of hostname.exe
named siril-cli.exe, so even a guard failure could not touch a Siril data folder or bus; the expected violations of
these tests are removed from the session record again.
"""
import asyncio
import os
import shutil
import subprocess
import threading
from pathlib import Path

import pytest

import siril_guard as guard
from starstack import process_guard, siril

windows = pytest.mark.skipif(os.name != "nt", reason="the stand-in is hostname.exe")
PRIVATE = {"DBUS_SESSION_BUS_ADDRESS": "disabled:"}


@pytest.fixture
def fake(tmp_path, monkeypatch):
    exe = tmp_path / "bin" / "siril-cli.exe"
    exe.parent.mkdir()
    shutil.copyfile(Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "hostname.exe", exe)
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.delenv("MSYSTEM", raising=False)
    monkeypatch.delenv("DBUS_SESSION_BUS_ADDRESS", raising=False)
    monkeypatch.delenv(siril.SIRIL_PIN_ENV, raising=False)
    before = len(guard.VIOLATIONS)
    yield str(exe)
    del guard.VIOLATIONS[before:]      # this file's expected violations are not session failures


def blocked(fn, rule="R1"):
    count = len(guard.VIOLATIONS)
    with pytest.raises(guard.SirilEnvironmentViolation, match=rule):
        fn()
    assert len(guard.VIOLATIONS) == count + 1


def test_the_rules_without_starting_anything(tmp_path):
    private = {"XDG_DATA_HOME": str(tmp_path), **PRIVATE}
    assert guard.problems(["C:/x/siril-cli.exe", "-s", "a.ssf"], private) == []
    assert guard.problems(["C:/x/SIRIL.EXE", "-s", "a"], {"xdg_data_home": str(tmp_path), **PRIVATE}) == []
    assert [p[:2] for p in guard.problems(["siril-cli", "-s", "a"], {})] == ["R1", "R2"]
    assert [p[:2] for p in guard.problems(["siril", "-d", "x"], {"XDG_DATA_HOME": guard.REAL_DATA + os.sep,
                                                                  **PRIVATE})] == ["R1"]
    assert [p[:2] for p in guard.problems(["siril-cli.exe", "-s", "a"], {"XDG_DATA_HOME": str(tmp_path),
                                                                         "DBUS_SESSION_BUS_ADDRESS": "autolaunch:"})] == ["R2"]
    assert guard.problems(["siril-cli.exe", "-v"], {}) == []                     # the version probe
    assert guard.problems(["siril-cli.exe", "-v", "-s", "a"], {}) != []
    assert guard.problems(["sirilx.exe", "-s", "a"], {}) == [] and guard.problems(["python.exe", "siril-cli"], {}) == []
    assert guard.problems('"C:/Program Files/Siril/bin/siril-cli.exe" -s a.ssf', {}) != []
    assert guard.problems(["anything", "-s", "x"], {}, executable="D:/siril-cli.exe") != []
    assert os.path.isabs(guard.REAL_DATA)


@windows
def test_plain_subprocess_run(fake):
    blocked(lambda: subprocess.run([fake, "-s", "x.ssf"], capture_output=True))


@windows
def test_aliased_imports_and_a_string_command_line(fake):
    import subprocess as sp
    from subprocess import Popen as P, run as start
    blocked(lambda: sp.check_call([fake, "-s", "a.ssf"]))
    blocked(lambda: start([fake, "-s", "b.ssf"]))
    blocked(lambda: P([fake, "-s", "c.ssf"]))
    blocked(lambda: sp.check_output(f'"{fake}" -s d.ssf'))


@windows
def test_executable_keyword(fake):
    blocked(lambda: subprocess.run(["anything", "-s", "x"], executable=fake))


@windows
def test_guarded_process_under_any_name(fake):
    def via(launch):
        with launch([fake, "-s", "x.ssf"], stdout=subprocess.DEVNULL):
            pass

    blocked(lambda: via(process_guard.guarded_process))
    blocked(lambda: via(siril.guarded_process))                     # the name bound in siril.py at import


@windows
def test_asyncio(fake):
    async def go():
        process = await asyncio.create_subprocess_exec(fake, "-s", "x.ssf")
        await process.wait()

    blocked(lambda: asyncio.run(go()))


@windows
def test_an_explicit_environment_with_the_users_folder_or_bus(fake, tmp_path):
    blocked(lambda: subprocess.run([fake, "-s", "x"], env={**os.environ, "XDG_DATA_HOME": guard.REAL_DATA + "\\",
                                                           **PRIVATE}))
    blocked(lambda: subprocess.run([fake, "-s", "x"], env={**os.environ, "XDG_DATA_HOME": str(tmp_path)}), "R2")


@windows
def test_allowed_forms_start_and_record_nothing(fake, tmp_path):
    count = len(guard.VIOLATIONS)
    subprocess.run([fake, "-v"], capture_output=True)                                       # version probe
    subprocess.run([fake, "-s", "x"], capture_output=True,
                   env={**os.environ, "XDG_DATA_HOME": str(tmp_path / "desktop"), **PRIVATE})
    with process_guard.guarded_process([fake, "-s", "x"], stdout=subprocess.DEVNULL,
                                       env=siril.siril_environment(tmp_path)) as process:
        process.wait(timeout=30)
    job = tmp_path / "job"
    job.mkdir()
    with pytest.raises(RuntimeError, match="失败"):                    # hostname.exe rejects '-i': a non-zero exit
        siril.run_script(fake, job, "x", ["close"], lambda: None)
    assert len(guard.VIOLATIONS) == count


@windows
def test_other_programs_are_untouched(fake, tmp_path):
    other = tmp_path / "bin" / "sirilx.exe"
    shutil.copyfile(fake, other)
    count = len(guard.VIOLATIONS)
    subprocess.run([str(other), "-s", "x"], capture_output=True)
    assert len(guard.VIOLATIONS) == count


@windows
@pytest.mark.parametrize("lost", ["the whole block", "the bus line"])
def test_a_product_path_regression_is_caught(fake, tmp_path, monkeypatch, lost):
    if lost == "the whole block":                                     # S-4b lost in a refactor
        monkeypatch.setattr(siril, "siril_environment", lambda *a, **k: None)
    else:                                                             # S-4d lost in a refactor
        monkeypatch.setattr(siril, "_NO_SESSION_BUS", {})
    job = tmp_path / "job"
    job.mkdir()
    blocked(lambda: siril.run_script(fake, job, "x", ["close"], lambda: None),
            "R1" if lost == "the whole block" else "R2")


@windows
def test_a_violation_in_a_worker_thread_is_recorded_for_the_session(fake):
    caught = []

    def worker():
        try:
            subprocess.run([fake, "-s", "x"], capture_output=True)
        except Exception:                                             # what product code might do
            caught.append("Exception")
        except BaseException as exc:                                  # noqa: B036 - the guard's own class
            caught.append(type(exc).__name__)

    count = len(guard.VIOLATIONS)
    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    assert caught == ["SirilEnvironmentViolation"] and len(guard.VIOLATIONS) == count + 1


def test_the_session_hooks_are_installed():
    assert "start" in vars(process_guard._JobObjectProcess) and "start" in vars(process_guard._PopenProcess)
    assert subprocess.Popen.__init__.__module__ == guard.__name__
    assert vars(process_guard._JobObjectProcess)["start"].__func__.__module__ == guard.__name__
    assert vars(process_guard._PopenProcess)["start"].__func__.__module__ == guard.__name__
