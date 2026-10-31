"""Readers must never see partially replaced job metadata."""
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from starstack import core


@pytest.mark.skipif(os.name != "nt", reason="Windows delete-sharing semantics")
def test_save_waits_for_existing_reader_and_preserves_complete_document(tmp_path, monkeypatch):
    target = tmp_path / "job.json"
    before = {"state": "running", "message": "处理中的完整旧文档"}
    after = {"state": "succeeded", "artifacts": ["母版.tiff"]}
    core.atomic_json(target, before)
    real_replace = core.os.replace
    sharing_failure = threading.Event()

    def observed_replace(source, destination):
        try:
            return real_replace(source, destination)
        except PermissionError:
            sharing_failure.set()
            raise

    monkeypatch.setattr(core.os, "replace", observed_replace)
    reader = target.open("r", encoding="utf-8")
    with ThreadPoolExecutor(max_workers=1) as pool:
        saving = pool.submit(core.atomic_json, target, after)
        try:
            assert sharing_failure.wait(1), "Expected actual Windows sharing conflict"
            assert json.load(reader) == before
            assert not saving.done()
        finally:
            reader.close()
        saving.result(timeout=3)
    assert json.loads(target.read_text(encoding="utf-8")) == after
    assert list(tmp_path.glob("*.tmp")) == []


def test_persistent_denial_is_bounded_and_keeps_previous_document(tmp_path, monkeypatch):
    target = tmp_path / "job.json"
    core.atomic_json(target, {"state": "running"})
    before = target.read_bytes()
    denied = PermissionError("simulated persistent Windows denial")
    denied.winerror = 5
    clock = iter([0, .5, 1, 2])
    monkeypatch.setattr(core.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(core.time, "sleep", lambda _: None)

    def forbidden_replace(*_):
        raise denied

    monkeypatch.setattr(core.os, "replace", forbidden_replace)
    with pytest.raises(PermissionError):
        core.atomic_json(target, {"state": "succeeded"})
    assert target.read_bytes() == before
    assert list(tmp_path.glob("*.tmp")) == []
