"""Independent writer-race guard: revision is checked after acquiring the lock."""
from contextlib import contextmanager
from copy import deepcopy

import pytest

from starstack import core


def test_expected_revision_rechecked_under_writer_lock(tmp_path, monkeypatch):
    old = {"id": "one", "record": "one.json", "sha256": "1" * 64}
    new = {"id": "two", "record": "two.json", "sha256": "2" * 64}
    state = {"state": "succeeded", "transient_revision": old}
    monkeypatch.setattr(core, "load_job", lambda path: deepcopy(state))
    monkeypatch.setattr(core, "gc_refusal", lambda *a: None)
    monkeypatch.setattr(core, "is_locked", lambda p: False)

    @contextmanager
    def writer_lock(*a, **kw):
        state["transient_revision"] = new
        yield

    monkeypatch.setattr(core, "file_lock", writer_lock)
    monkeypatch.setattr(core, "_recompose_locked", lambda *a: pytest.fail("stale review reached pixels"))
    with pytest.raises(ValueError, match="新版本"):
        core.recompose_transients(str(tmp_path), expected_revision=old)


def test_matching_revision_reaches_recompose_once(tmp_path, monkeypatch):
    revision = {"id": "one", "record": "one.json", "sha256": "1" * 64}
    state = {"state": "succeeded", "transient_revision": revision}
    monkeypatch.setattr(core, "load_job", lambda path: deepcopy(state))
    monkeypatch.setattr(core, "gc_refusal", lambda *a: None)
    monkeypatch.setattr(core, "is_locked", lambda p: False)
    calls = []

    @contextmanager
    def writer_lock(*a, **kw):
        yield

    monkeypatch.setattr(core, "file_lock", writer_lock)
    monkeypatch.setattr(core, "_recompose_locked", lambda *a: calls.append(a) or {"ok": True})
    assert core.recompose_transients(str(tmp_path), labels={"a": "unknown"},
                                    overrides={"a": "keep"}, expected_revision=revision) == {"ok": True}
    assert len(calls) == 1
    assert calls[0][2:4] == ({"a": "unknown"}, {"a": "keep"})
