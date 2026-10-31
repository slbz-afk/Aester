"""Aester process tree (governor resident memory, efficiency P1b-2): root walk, PID-reuse guard, real tree."""
import os
import subprocess
import sys
import time

import pytest

from starstack import resources

windows = pytest.mark.skipif(os.name != "nt", reason="Toolhelp32 process tree")


def created_from(table):
    return lambda pid: table.get(pid)


def test_root_walk_stops_at_a_non_aester_image():
    snapshot = {1: (0, "explorer.exe"), 10: (1, "Aester.exe"), 20: (10, "pythonw.exe"), 30: (20, "python.exe"),
                40: (30, "siril-cli.exe"), 41: (40, "gdbus.exe"), 50: (1, "notepad.exe")}
    created = created_from({1: 0, 10: 5, 20: 6, 30: 7, 40: 8, 41: 9, 50: 3})
    root, members = resources.tree_members(snapshot, 30, created)
    assert root == 10
    assert sorted(members) == [10, 20, 30, 40, 41]          # the whole tree below the GUI, Siril included
    root, members = resources.tree_members(snapshot, 50, created)
    assert root == 50 and members == [50]                  # explorer.exe is not an Aester image: no walk up
    root, _ = resources.tree_members(snapshot, 10, created)
    assert root == 10


def test_pid_reuse_is_excluded_both_ways():
    # 20's recorded parent 10 started after it: a reused PID, so the walk stops at 20
    snapshot = {10: (1, "python.exe"), 20: (10, "python.exe"), 30: (20, "python.exe"), 31: (20, "python.exe")}
    created = created_from({10: 50, 20: 6, 30: 7, 31: 2})
    root, members = resources.tree_members(snapshot, 20, created)
    assert root == 20
    assert sorted(members) == [20, 30]                     # 31 started before 20: a stale child entry, excluded


def test_unknown_creation_times_stop_the_walk_and_cycles_terminate():
    snapshot = {10: (20, "python.exe"), 20: (10, "python.exe")}
    root, members = resources.tree_members(snapshot, 10, created_from({10: 1, 20: 1}))
    assert root in (10, 20) and sorted(members) == [10, 20]
    root, members = resources.tree_members({10: (20, "python.exe"), 20: (0, "python.exe")}, 10, created_from({}))
    assert root == 10 and members == [10]


@pytest.mark.parametrize("name, ok", [("python.exe", True), ("PYTHONW.EXE", True), ("Aester.exe", True),
                                      ("aester-gui.exe", True), ("siril-cli.exe", False), ("bash.exe", False),
                                      (None, False)])
def test_aester_images(name, ok):
    assert resources.is_aester_image(name) is ok


@windows
def test_a_real_child_tree_is_found_and_its_private_bytes_summed():
    code = "import sys, time; data = bytearray(64 << 20); sys.stdout.write('ready\\n'); sys.stdout.flush(); time.sleep(30)"
    child = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == "ready"
        before = time.perf_counter()
        resident = resources.aester_resident(detail=True)
        elapsed = time.perf_counter() - before
        assert resident["members"] >= 2
        pids = {row["pid"] for row in resident["detail"]}
        assert os.getpid() in pids
        # the venv launcher may sit between: the child (or its own child) is in the tree
        assert any(row["exe"].lower().startswith("python") for row in resident["detail"])
        assert resident["commit"] >= 64 << 20 and resident["ws"] > 0
        assert all(row["shared_commit"] is None or row["shared_commit"] >= 0 for row in resident["detail"])
        assert elapsed < 1.0
    finally:
        child.kill()
        child.wait(timeout=10)
