"""Real protected-scanner lifecycle tests with finite harmless worker payloads.

Windows owner/child identity is pinned by creation time and open process
handles before termination. No name-based or later-PID-based cleanup is used.
The scan algorithm itself is covered separately by synthetic FITS tests.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest
import numpy as np

from starstack import transient_scan as scan
from starstack import process_guard
from test_process_guard import IDENTITY, OwnedProcess


PAYLOAD = IDENTITY + '''
from pathlib import Path
import sys, time
request = Path(sys.argv[1])
staged = request.with_suffix('.identity-tmp')      # written whole, then renamed: a reader never sees a partial file
staged.write_text(json.dumps(identity(os.getpid())))
os.replace(staged, request.with_suffix('.identity'))
deadline = time.monotonic() + 20
with request.with_suffix('.heartbeat').open('ab', buffering=0) as stream:
    while time.monotonic() < deadline and not (request.parent.parent.parent / 'STOP').exists():
        stream.write(b'x')
        time.sleep(.05)
'''


def wait_for(predicate, timeout=8):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(.025)
    raise AssertionError("bounded test wait expired")


def load_identities(root: Path, expected=2):
    paths = list(root.glob("scan-runs/*/*.identity"))
    if len(paths) != expected:
        return None
    try:
        return [json.loads(path.read_text()) for path in paths]
    except (ValueError, OSError):
        return None


def load_identity(path: Path):
    """The identity a process reported, once its file is complete (None until then: a reader may otherwise catch the
    file between its creation and its write -- the race that made test_parent_force_kill_stops_all_scanners flaky)."""
    try:
        return json.loads(path.read_text())
    except (ValueError, OSError):
        return None


@pytest.fixture
def payload(tmp_path):
    if process_guard.guard_kind() == process_guard.GUARD_NONE:
        pytest.skip("no supported process guard")
    file = tmp_path / "finite_payload.py"
    file.write_text(PAYLOAD)
    return file


def test_cancel_checks_while_no_frame_has_finished(tmp_path, payload, monkeypatch):
    monkeypatch.setattr(scan, "_worker_command", lambda request: [sys.executable, str(payload), str(request)])
    owned = []
    started = time.monotonic()

    def cancel():
        identities = load_identities(tmp_path)
        if identities:
            owned.extend(OwnedProcess(identity) for identity in identities)
            raise RuntimeError("cancel from caller")

    try:
        with pytest.raises(RuntimeError, match="cancel from caller"):
            scan.scan_frames({"a": {}, "b": {}}, workers=2, output_dir=tmp_path, check=cancel)
        assert time.monotonic() - started < 8
        assert len(owned) == 2
        wait_for(lambda: not any(proc.alive() for proc in owned))
    finally:
        (tmp_path / "STOP").touch()
        for proc in owned:
            proc.kill()
            proc.close()


def test_startup_failure_stops_already_started_worker(tmp_path, payload, monkeypatch):
    owned = []
    calls = 0

    def command(request):
        nonlocal calls
        calls += 1
        if calls == 2:
            identities = wait_for(lambda: load_identities(tmp_path, 1))
            owned.append(OwnedProcess(identities[0]))
            raise OSError("injected second startup failure")
        return [sys.executable, str(payload), str(request)]

    monkeypatch.setattr(scan, "_worker_command", command)
    try:
        with pytest.raises(OSError, match="second startup"):
            scan.scan_frames({"a": {}, "b": {}}, workers=2, output_dir=tmp_path)
        wait_for(lambda: not owned[0].alive())
    finally:
        (tmp_path / "STOP").touch()
        for proc in owned:
            proc.kill()
            proc.close()


def test_parent_force_kill_stops_all_scanners(tmp_path, payload):
    if os.name != "nt":
        pytest.skip("Windows product guarantee; Linux direct-child fallback is tested separately")
    owner_file = tmp_path / "owner.py"
    owner_file.write_text(IDENTITY + '''
from pathlib import Path
import sys
from starstack import transient_scan as scan
root, payload = Path(sys.argv[1]), sys.argv[2]
(root / 'owner.identity-tmp').write_text(json.dumps(identity(os.getpid())))
os.replace(root / 'owner.identity-tmp', root / 'owner.identity')     # complete before it is visible
scan._worker_command = lambda request: [sys.executable, payload, str(request)]
scan.scan_frames({'a': {}, 'b': {}}, workers=2, output_dir=root, total_timeout=15)
''')
    owner = None
    children = []
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(scan.__file__).resolve().parents[1]) + os.pathsep + env.get("PYTHONPATH", "")
    with (tmp_path / "owner.log").open("wb") as logfile:
        launcher = subprocess.Popen([sys.executable, str(owner_file), str(tmp_path), str(payload)],
                                    stdin=subprocess.DEVNULL, stdout=logfile, stderr=subprocess.STDOUT,
                                    env=env,
                                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            owner = OwnedProcess(wait_for(lambda: load_identity(tmp_path / "owner.identity")))
            identities = wait_for(lambda: load_identities(tmp_path))
            children = [OwnedProcess(identity) for identity in identities]
            assert all(child.alive() for child in children)
            owner.kill()
            wait_for(lambda: not owner.alive())
            wait_for(lambda: not any(child.alive() for child in children))
            sizes = [p.stat().st_size for p in tmp_path.glob("scan-runs/*/*.heartbeat")]
            time.sleep(.15)
            assert sizes == [p.stat().st_size for p in tmp_path.glob("scan-runs/*/*.heartbeat")]
        finally:
            (tmp_path / "STOP").touch()
            for child in children:
                child.kill()
                child.close()
            if owner is not None:
                owner.kill()
                owner.close()
            if launcher.poll() is None:
                launcher.kill()
            launcher.wait(timeout=8)


def test_worker_failure_includes_retained_diagnostics(tmp_path):
    with pytest.raises(scan.ScanWorkerError, match="失败"):
        scan.scan_frames({"bad": {"frame_path": str(tmp_path / "missing.fit"),
                                  "master_path": str(tmp_path / "missing.fit"),
                                  "detection": {}, "grid_marks": [], "stats_step": 8}},
                         workers=1, output_dir=tmp_path, total_timeout=10)
    logs = list(tmp_path.glob("scan-runs/*/*.log"))
    assert len(logs) == 1
    assert "FileNotFoundError" in logs[0].read_text()


def test_total_scan_timeout_is_bounded(tmp_path, payload, monkeypatch):
    monkeypatch.setattr(scan, "_worker_command", lambda request: [sys.executable, str(payload), str(request)])
    started = time.monotonic()
    with pytest.raises(TimeoutError, match="总时限"):
        scan.scan_frames({"a": {}}, workers=1, output_dir=tmp_path, total_timeout=.2)
    assert time.monotonic() - started < 5


def test_real_scan_is_numerically_unchanged_with_one_or_two_workers(tmp_path):
    from starstack import synthetic_starfield as sf
    from starstack import transient_stage as stage
    from starstack import transient_trails as trails

    rng = np.random.default_rng(125)
    yy, xx = np.mgrid[:180, :320]
    master = (.02 + rng.normal(0, .0005, (3, 180, 320))).astype(np.float32)
    master_path = tmp_path / "master.fit"
    sf.write_fits_rgb32(master_path, master)
    jobs = {}
    # Deliberately nonalphabetic order to catch completion/order assumptions.
    for name, amplitude in [("z", .04), ("a", 0.0), ("q", .02)]:
        trail = amplitude * np.exp(-((yy - 85) ** 2) / 4) * ((xx >= 60) & (xx <= 250))
        frame = (master + trail + rng.normal(0, .0005, master.shape)).astype(np.float32)
        path = tmp_path / f"{name}.fit"
        sf.write_fits_rgb32(path, frame)
        jobs[name] = {"frame_path": str(path), "master_path": str(master_path),
                      "detection": dict(trails.DEFAULT_DETECTION), "grid_marks": [], "stats_step": 8}
    expected = {name: stage._scan_frame(**args) for name, args in jobs.items()}
    assert expected["z"]["result"]["segments"], "fixture must actually exercise a detected trail"
    for workers in (1, 2):
        results = scan.scan_frames(jobs, workers=workers, output_dir=tmp_path / f"w{workers}", log=lambda _: None)
        assert list(results) == list(jobs)
        for name in jobs:
            assert results[name]["stats"] == expected[name]["stats"]
            assert results[name]["result"] == expected[name]["result"]


@pytest.mark.parametrize("workers", [0, -1, True, 1.5])
def test_invalid_worker_count_rejected_before_start(tmp_path, workers):
    with pytest.raises(ValueError):
        scan.scan_frames({}, workers=workers, output_dir=tmp_path)
    assert not list(tmp_path.iterdir())
