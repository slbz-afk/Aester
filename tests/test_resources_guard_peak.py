"""process_guard: the direct child's peak working set and commit in its accounting (efficiency P1b-2 §3.11)."""
import os
import subprocess
import sys

import pytest

from starstack import process_guard as pg, profiling

MB = 10 ** 6


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object accounting")
def test_a_real_child_reports_its_peak_working_set_and_commit():
    code = "import time\ndata = bytearray(300 * 1000 * 1000)\nfor i in range(0, len(data), 4096): data[i] = 1\n"
    timer = profiling.StageTimer()
    with timer.operation("script:probe"):
        with pg.guarded_process([sys.executable, "-c", code], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL) as process:
            assert process.wait(timeout=60) == 0
    accounting = process.accounting
    # the venv's python.exe may be a launcher: then the direct child is small and the Job peak is the useful value
    assert accounting["peak_job_memory_bytes"] >= 250 * MB
    assert accounting["child_peak_working_set_bytes"] is not None and accounting["child_peak_commit_bytes"] is not None
    if accounting["descendant_processes"] == 1:
        assert accounting["child_peak_working_set_bytes"] >= 250 * MB
        assert accounting["child_peak_commit_bytes"] >= 250 * MB
    children = timer.finish()["resources"]["operations"]["script:probe"]["children"]
    assert children["child_peak_working_set_bytes"] == accounting["child_peak_working_set_bytes"]


@pytest.mark.skipif(os.name != "nt", reason="Windows process handles")
def test_the_direct_child_of_a_launcher_tree_is_measured_too():
    """A child that itself allocates (no launcher in between) is measured directly."""
    import shutil
    exe = shutil.which("powershell") or shutil.which("pwsh")
    if not exe:
        pytest.skip("no PowerShell")
    code = "$a = New-Object byte[] (300MB); for ($i = 0; $i -lt $a.Length; $i += 4096) { $a[$i] = 1 }"
    with pg.guarded_process([exe, "-NoProfile", "-NonInteractive", "-Command", code], stdin=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) as process:
        assert process.wait(timeout=120) == 0
    assert process.accounting["child_peak_working_set_bytes"] >= 250 * MB
    assert process.accounting["child_peak_commit_bytes"] >= 250 * MB


def test_child_peak_memory_is_best_effort_for_doubles():
    class NoQuery:
        pass

    class Refusing:
        def K32GetProcessMemoryInfo(self, *args):
            return 0

    class Raising:
        def K32GetProcessMemoryInfo(self, *args):
            raise OSError("denied")

    assert pg.child_peak_memory(NoQuery(), 5) is None
    assert pg.child_peak_memory(Refusing(), 5) is None
    assert pg.child_peak_memory(Raising(), 5) is None
    assert pg.child_peak_memory(Refusing(), None) is None
    import ctypes
    expected = 72 if ctypes.sizeof(ctypes.c_void_p) == 8 else 40
    assert ctypes.sizeof(pg.PROCESS_MEMORY_COUNTERS) == expected
