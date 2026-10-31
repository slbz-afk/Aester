"""The desktop app's thread, pool and process sites (the closed twin of test_p1c_static_guards' open-engine list)."""
from pathlib import Path

from test_p1c_static_guards import sites_in

DESKTOP = Path(__file__).resolve().parents[2] / "src" / "aester_desktop"

# (file, qualname, construct): count -- moved here from the engine's POOL_SITES at the open/closed split
DESKTOP_SITES = {
    ("experience.py", "Experience._restart", "process"): 1,
    # GUI threads: the Qt pools of the main and finishing windows, their tasks, the trail review dialog's thread
    ("gui.py", "MainWindow.__init__", "qt_pool"): 1,
    ("gui.py", "_Worker", "qt_runnable"): 1,
    ("finishing_gui.py", "FinishingWindow.__init__", "qt_pool"): 1,
    ("finishing_gui.py", "_Work", "qt_runnable"): 1,
    ("trail_review_dialog.py", "_ContextWorker", "qt_thread"): 1,
}


def test_desktop_pool_and_process_sites_are_the_listed_ones():
    found = {}
    for path in sorted(p for p in DESKTOP.rglob("*.py") if "__pycache__" not in p.parts):
        found.update(sites_in(path.read_text(encoding="utf-8"), path.relative_to(DESKTOP).as_posix()))
    assert {key: len(lines) for key, lines in found.items()} == DESKTOP_SITES
