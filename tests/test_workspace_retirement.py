"""Synthetic-file checks only: never invokes cleanup on project assets."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess

import pytest


_SPEC = importlib.util.spec_from_file_location("workspace_retirement", Path(__file__).parents[1] / "scripts" / "workspace_retirement.py")
retirement = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(retirement)


def put(root, name, data=b"image data"):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def events(receipt):
    return [json.loads(line) for line in receipt.read_text(encoding="utf-8").splitlines()]


@pytest.mark.parametrize("name", ["../outside.bin", "/outside.bin", "C:/outside.bin", "outputs/../raw/file", "outputs//file", "outputs\\file", "outputs/file:stream", "outputs/NUL", "outputs/file."])
def test_rejects_unsafe_relative(tmp_path, name):
    with pytest.raises((ValueError, FileNotFoundError)):
        retirement.make_plan(tmp_path, [name])


@pytest.mark.parametrize("prefix", ["raw", "raw_1", "raw_2_sony", "research", ".git", ".venv", "tools"])
def test_immutable_roots_always_protected(tmp_path, prefix):
    path = put(tmp_path, prefix + "/source.bin")
    with pytest.raises(ValueError, match="Protected"):
        retirement.make_plan(tmp_path, [prefix + "/source.bin"])
    assert path.exists()


def test_user_pin_overlapping_directory_and_file(tmp_path):
    put(tmp_path, "outputs/retained/frame.bin")
    for pin in ("outputs/retained", "outputs/retained/frame.bin", "OUTPUTS/RETAINED"):
        with pytest.raises(ValueError, match="Protected"):
            retirement.make_plan(tmp_path, ["outputs/retained/frame.bin"], protected=[pin])


def test_extra_execution_pin_blocks_entire_plan(tmp_path):
    a = put(tmp_path, "outputs/a.bin")
    b = put(tmp_path, "outputs/b.bin")
    plan = retirement.make_plan(tmp_path, ["outputs/a.bin", "outputs/b.bin"])
    result = retirement.execute_plan(tmp_path, plan, tmp_path / "receipt.jsonl", apply=True, protected=["outputs/b.bin"])
    assert result["state"] == "blocked"
    assert a.exists() and b.exists()


def test_directory_is_not_a_candidate(tmp_path):
    (tmp_path / "outputs").mkdir()
    with pytest.raises(ValueError, match="individual"):
        retirement.make_plan(tmp_path, ["outputs"])


def test_missing_is_not_silently_skipped(tmp_path):
    with pytest.raises(FileNotFoundError):
        retirement.make_plan(tmp_path, ["unknown.bin"])


def test_duplicate_entries_rejected(tmp_path):
    put(tmp_path, "a.bin")
    with pytest.raises(ValueError, match="Duplicate"):
        retirement.make_plan(tmp_path, ["a.bin", "A.BIN"])


def test_dry_run_hashes_and_never_creates_receipt(tmp_path):
    path = put(tmp_path, "outputs/a.bin")
    plan = retirement.make_plan(tmp_path, ["outputs/a.bin"])
    receipt = tmp_path / "receipt.jsonl"
    result = retirement.execute_plan(tmp_path, plan, receipt)
    assert result["state"] == "checked" and result["dry_run"]
    assert path.read_bytes() == b"image data"
    assert not receipt.exists()
    assert plan["entries"][0]["sha256"] == retirement.hashlib.sha256(b"image data").hexdigest()


def test_normal_explicit_deletion_leaves_unknown_sibling(tmp_path):
    a = put(tmp_path, "outputs/a.bin")
    b = put(tmp_path, "outputs/b.bin", b"second")
    keep = put(tmp_path, "outputs/unknown.bin", b"keep")
    plan = retirement.make_plan(tmp_path, ["outputs/a.bin", "outputs/b.bin"])
    receipt = tmp_path / "receipt.jsonl"
    result = retirement.execute_plan(tmp_path, plan, receipt, apply=True)
    assert result["state"] == "completed", result
    assert result["removed_files"] == 2 and result["removed_logical_bytes"] == 16
    assert not a.exists() and not b.exists() and keep.read_bytes() == b"keep"
    assert (tmp_path / "outputs").is_dir()
    assert [e["event"] for e in events(receipt)] == ["begin", "preflight_passed", "delete_intent", "deleted", "delete_intent", "deleted", "finish"]


def test_any_changed_file_blocks_all_deletion_even_same_size_mtime(tmp_path):
    a = put(tmp_path, "a.bin", b"aaaa")
    b = put(tmp_path, "b.bin", b"bbbb")
    plan = retirement.make_plan(tmp_path, ["a.bin", "b.bin"])
    info = b.stat()
    b.write_bytes(b"evil")
    os.utime(b, ns=(info.st_atime_ns, info.st_mtime_ns))
    result = retirement.execute_plan(tmp_path, plan, tmp_path / "receipt.jsonl", apply=True)
    assert result["state"] == "blocked" and result["removed_files"] == 0
    assert a.exists() and b.read_bytes() == b"evil"


def test_replacement_same_bytes_and_time_rejected(tmp_path):
    original = put(tmp_path, "a.bin")
    plan = retirement.make_plan(tmp_path, ["a.bin"])
    before = original.stat()
    replacement = put(tmp_path, "other.bin")
    os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns))
    replacement.replace(original)
    with pytest.raises(ValueError, match="changed"):
        retirement.check_plan(tmp_path, plan)


def test_plan_change_not_accepted(tmp_path):
    put(tmp_path, "a.bin")
    plan = retirement.make_plan(tmp_path, ["a.bin"])
    plan["entries"][0]["bytes"] += 1
    with pytest.raises(ValueError, match="edited"):
        retirement.check_plan(tmp_path, plan)


def test_wrong_workspace_and_root_replacement(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    put(root, "a.bin")
    plan = retirement.make_plan(root, ["a.bin"])
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(ValueError, match="workspace"):
        retirement.check_plan(other, plan)
    root.rename(tmp_path / "saved")
    root.mkdir()
    put(root, "a.bin")
    with pytest.raises(ValueError, match="workspace"):
        retirement.check_plan(root, plan)


def _link_directory(link, target):
    try:
        link.symlink_to(target, target_is_directory=True)
        return
    except OSError:
        if os.name != "nt":
            pytest.skip("Symlink creation not available")
    # Junctions are used only to create a synthetic test fixture, never for cleanup.
    completed = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], capture_output=True)
    if completed.returncode:
        pytest.skip("Neither symlinks nor junctions available")


def test_parent_link_and_linked_workspace_refused(tmp_path):
    external = tmp_path / "external"
    external.mkdir()
    put(external, "keep.bin")
    root = tmp_path / "workspace"
    root.mkdir()
    _link_directory(root / "linked", external)
    try:
        with pytest.raises(ValueError, match="reparse"):
            retirement.make_plan(root, ["linked/keep.bin"])
        with pytest.raises(ValueError, match="reparse"):
            retirement.make_plan(root / "linked", ["keep.bin"])
        assert (external / "keep.bin").exists()
    finally:
        if (root / "linked").is_symlink():
            (root / "linked").unlink()
        else:
            (root / "linked").rmdir()


def test_linked_parent_substituted_after_plan_refused(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    put(root, "outputs/a.bin")
    plan = retirement.make_plan(root, ["outputs/a.bin"])
    (root / "outputs").rename(tmp_path / "saved")
    _link_directory(root / "outputs", tmp_path / "saved")
    try:
        with pytest.raises(ValueError, match="reparse"):
            retirement.check_plan(root, plan)
        assert (tmp_path / "saved/a.bin").exists()
    finally:
        if (root / "outputs").is_symlink():
            (root / "outputs").unlink()
        else:
            (root / "outputs").rmdir()


def test_partial_failure_stops_and_records_completed_entries(tmp_path, monkeypatch):
    for name in ("a.bin", "b.bin", "c.bin"):
        put(tmp_path, name)
    plan = retirement.make_plan(tmp_path, ["a.bin", "b.bin", "c.bin"])
    original_delete = retirement._Handle.delete

    def fail_second(handle):
        if handle.path.name == "b.bin":
            raise PermissionError("synthetic locked file")
        original_delete(handle)

    monkeypatch.setattr(retirement._Handle, "delete", fail_second)
    receipt = tmp_path / "receipt.jsonl"
    result = retirement.execute_plan(tmp_path, plan, receipt, apply=True)
    assert result["state"] == "partial_failure" and result["removed_files"] == 1
    assert not (tmp_path / "a.bin").exists()
    assert (tmp_path / "b.bin").exists() and (tmp_path / "c.bin").exists()
    assert [e["event"] for e in events(receipt)] == ["begin", "preflight_passed", "delete_intent", "deleted", "delete_intent", "failed", "finish"]
    assert events(receipt)[-1]["failed_path"] == "b.bin"


def test_hardlink_names_and_unselected_alias_preserved(tmp_path):
    a = put(tmp_path, "a.bin")
    os.link(a, tmp_path / "b.bin")
    os.link(a, tmp_path / "keep.bin")
    plan = retirement.make_plan(tmp_path, ["a.bin", "b.bin"])
    assert all(e["nlink"] == 3 for e in plan["entries"])
    receipt = tmp_path / "receipt.jsonl"
    result = retirement.execute_plan(tmp_path, plan, receipt, apply=True)
    assert result["state"] == "completed", result
    assert not a.exists() and not (tmp_path / "b.bin").exists()
    assert (tmp_path / "keep.bin").read_bytes() == b"image data"
    assert (tmp_path / "keep.bin").stat().st_nlink == 1


def test_link_count_changes_invalidate_plan(tmp_path):
    a = put(tmp_path, "a.bin")
    plan = retirement.make_plan(tmp_path, ["a.bin"])
    os.link(a, tmp_path / "new-link.bin")
    with pytest.raises(ValueError, match="changed"):
        retirement.check_plan(tmp_path, plan)


def test_receipt_outside_or_existing_refused(tmp_path):
    path = put(tmp_path, "a.bin")
    plan = retirement.make_plan(tmp_path, ["a.bin"])
    with pytest.raises(ValueError):
        retirement.execute_plan(tmp_path, plan, tmp_path.parent / "outside.jsonl", apply=True)
    receipt = put(tmp_path, "old.jsonl", b"do not overwrite")
    with pytest.raises(FileExistsError):
        retirement.execute_plan(tmp_path, plan, receipt, apply=True)
    assert receipt.read_bytes() == b"do not overwrite" and path.exists()


def test_nested_maintenance_lock_refused(tmp_path):
    put(tmp_path, "a.bin")
    with retirement.maintenance_lock(tmp_path):
        with pytest.raises(ValueError, match="active"):
            retirement.make_plan(tmp_path, ["a.bin"])


@pytest.mark.skipif(os.name != "nt", reason="Windows sharing semantics")
def test_windows_candidate_handle_prevents_write_and_replacement(tmp_path):
    path = put(tmp_path, "a.bin")
    replacement = put(tmp_path, "replacement.bin")
    handle = retirement._Handle(path, deleting=True)
    try:
        with pytest.raises(OSError):
            path.write_bytes(b"bad")
        with pytest.raises(OSError):
            replacement.replace(path)
        with pytest.raises(OSError):
            path.rename(tmp_path / "moved.bin")
    finally:
        handle.close()
    assert path.read_bytes() == b"image data"


@pytest.mark.skipif(os.name != "nt", reason="Windows sharing semantics")
def test_windows_directory_guard_prevents_parent_swap(tmp_path):
    directory = tmp_path / "outputs"
    directory.mkdir()
    handle = retirement._Handle(directory, directory=True)
    try:
        with pytest.raises(OSError):
            directory.rename(tmp_path / "renamed")
    finally:
        handle.close()


def test_cli_apply_without_switch_only_checks(tmp_path, capsys):
    path = put(tmp_path, "a.bin")
    plan = retirement.make_plan(tmp_path, ["a.bin"])
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    assert retirement.main(["apply", "--root", str(tmp_path), "--plan", str(plan_path)]) == 0
    assert json.loads(capsys.readouterr().out)["dry_run"] is True
    assert path.exists()


def test_progress_reports_both_preflight_and_delete(tmp_path):
    put(tmp_path, "a.bin")
    reported = []
    plan = retirement.make_plan(tmp_path, ["a.bin"], progress=reported.append)
    retirement.execute_plan(tmp_path, plan, tmp_path / "receipt.jsonl", apply=True, progress=reported.append)
    assert [entry["phase"] for entry in reported] == ["plan", "preflight", "delete"]
    assert all(entry["files_done"] == 1 for entry in reported)


def test_post_preflight_change_is_rehashed_and_stops(tmp_path):
    a = put(tmp_path, "a.bin", b"aaaa")
    b = put(tmp_path, "b.bin", b"bbbb")
    plan = retirement.make_plan(tmp_path, ["a.bin", "b.bin"])
    original_time = b.stat()

    def mutate_after_preflight(event):
        if event["phase"] == "preflight" and event["files_done"] == 2:
            b.write_bytes(b"evil")
            os.utime(b, ns=(original_time.st_atime_ns, original_time.st_mtime_ns))

    result = retirement.execute_plan(tmp_path, plan, tmp_path / "receipt.jsonl", apply=True, progress=mutate_after_preflight)
    assert result["state"] == "partial_failure" and result["removed_files"] == 1
    assert not a.exists() and b.read_bytes() == b"evil"


def test_receipt_write_failure_before_delete_preserves_file(tmp_path, monkeypatch):
    path = put(tmp_path, "a.bin")
    plan = retirement.make_plan(tmp_path, ["a.bin"])
    original_fsync = retirement.os.fsync
    calls = 0

    def fail_intent(fd):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("synthetic disk full")
        original_fsync(fd)

    monkeypatch.setattr(retirement.os, "fsync", fail_intent)
    result = retirement.execute_plan(tmp_path, plan, tmp_path / "receipt.jsonl", apply=True)
    assert result["state"] == "partial_failure" and result["removed_files"] == 0
    assert path.exists()


def test_preflight_file_handles_are_bounded(tmp_path, monkeypatch):
    names = [f"{i}.bin" for i in range(30)]
    for name in names:
        put(tmp_path, name)
    plan = retirement.make_plan(tmp_path, names)
    original_init = retirement._Handle.__init__
    original_close = retirement._Handle.close
    alive = set()
    peak = 0

    def track_init(handle, path, **kwargs):
        nonlocal peak
        original_init(handle, path, **kwargs)
        if not kwargs.get("directory"):
            alive.add(id(handle))
            peak = max(peak, len(alive))

    def track_close(handle):
        alive.discard(id(handle))
        original_close(handle)

    monkeypatch.setattr(retirement._Handle, "__init__", track_init)
    monkeypatch.setattr(retirement._Handle, "close", track_close)
    result = retirement.execute_plan(tmp_path, plan, tmp_path / "receipt.jsonl", apply=True)
    assert result["state"] == "completed" and result["removed_files"] == 30
    assert peak <= 2 and not alive


def test_hardlinks_interspersed_with_regular_file(tmp_path):
    a = put(tmp_path, "a.bin")
    os.link(a, tmp_path / "b.bin")
    put(tmp_path, "unrelated.bin")
    plan = retirement.make_plan(tmp_path, ["a.bin", "unrelated.bin", "b.bin"])
    result = retirement.execute_plan(tmp_path, plan, tmp_path / "receipt.jsonl", apply=True)
    assert result["state"] == "completed" and result["removed_files"] == 3


@pytest.mark.skipif(os.name != "nt", reason="Windows attributes")
def test_readonly_file_not_force_unlocked_or_deleted(tmp_path):
    path = put(tmp_path, "a.bin")
    path.chmod(0o444)
    try:
        plan = retirement.make_plan(tmp_path, ["a.bin"])
        result = retirement.execute_plan(tmp_path, plan, tmp_path / "receipt.jsonl", apply=True)
        assert result["state"] == "partial_failure" and result["removed_files"] == 0
        assert path.exists()
    finally:
        path.chmod(0o666)
