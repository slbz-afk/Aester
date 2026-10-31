"""Process launches in the desktop app (the closed twin of test_siril_launch_sites, which pins the open tree)."""
import launch_scan

DESKTOP_SITES = {
    ('src/aester_desktop/experience.py', 'Experience._restart', 'subprocess.Popen', 'call'): 1,  # GUI restart (python)
}


def test_desktop_process_launches_are_pinned():
    found = launch_scan.launch_sites(launch_scan.REPO, roots=("src/aester_desktop",))
    assert launch_scan.counter(found) == launch_scan.Counter(DESKTOP_SITES)
