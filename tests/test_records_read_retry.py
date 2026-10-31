"""records.read_json: bounded retry of Windows access denials while another process replaces a record.

P0 benchmark R2b (2026-09-28) failed a 48-frame workflow after 565 s when the
workflow worker's poll of job.json landed inside the job worker's os.replace.
"""
import errno
import os
import threading
import time
from pathlib import Path

import pytest

from starstack import core, records

windows = pytest.mark.skipif(os.name != "nt", reason="Windows replace/open sharing semantics")


def denial(winerror=None):
    exc = PermissionError(errno.EACCES, "Permission denied")
    if winerror is not None:
        exc.winerror = winerror
    return exc


def test_transient_denials_are_retried(tmp_path, monkeypatch):
    path = tmp_path / "job.json"
    records.atomic_json(path, {"state": "running", "n": 1})
    real, calls = Path.read_text, []

    def flaky(self, *args, **kwargs):
        calls.append(self)
        if len(calls) <= 2:
            raise denial(None if len(calls) == 1 else 5)
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", flaky)
    assert records.read_json(path) == {"state": "running", "n": 1}
    assert len(calls) == 3


def test_other_errors_are_not_retried(tmp_path, monkeypatch):
    path = tmp_path / "job.json"
    calls = []

    def refused(self, *args, **kwargs):
        calls.append(self)
        raise denial(1)                          # not an access/sharing denial

    monkeypatch.setattr(Path, "read_text", refused)
    with pytest.raises(PermissionError):
        records.read_json(path)
    assert len(calls) == 1
    with pytest.raises(FileNotFoundError):
        monkeypatch.undo()
        records.read_json(tmp_path / "missing.json")


def test_persistent_denial_raises_after_deadline(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "read_text", lambda self, *a, **k: (_ for _ in ()).throw(denial(32)))
    started = time.monotonic()
    with pytest.raises(PermissionError):
        records.read_json(tmp_path / "job.json", retry_seconds=.1)
    assert .09 <= time.monotonic() - started < 2


@windows
def test_polling_readers_survive_concurrent_replacement(tmp_path):
    """Real race: one thread replaces job.json / workflow.json as fast as it can while load_job polls."""
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    records.atomic_json(job_dir / "job.json", {"state": "succeeded", "n": 0})
    stop, errors = threading.Event(), []

    def writer():
        n = 0
        while not stop.is_set():
            n += 1
            try:
                records.atomic_json(job_dir / "job.json", {"state": "succeeded", "n": n, "pad": "x" * 4096})
            except Exception as exc:              # the writer's own 2 s retry must hold as well
                errors.append(exc)

    thread = threading.Thread(target=writer)
    thread.start()
    reads = 0
    try:
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            assert core.load_job(job_dir)["state"] == "succeeded"
            reads += 1
    finally:
        stop.set()
        thread.join()
    assert not errors and reads > 100
