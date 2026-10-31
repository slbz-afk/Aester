"""Every process launch in src/starstack, packaging and scripts is pinned (efficiency P1b W1, static guard).

Only siril._started starts a Siril script run, always with its own environment block (siril.siril_environment,
S-4b/S-4d); the version probes run exactly the pinned argv, and the two in src/ pass a private environment too (W4).
A new launch fails test_every_process_launch_is_pinned_with_its_count until it is classified: if it can start Siril,
route it through siril.run_script; otherwise regenerate the table (tests/launch_scan.py --update, see its docstring)
and say what it starts. The runtime counterpart, which checks every launch the tests actually make, is
tests/siril_guard.py. Research scripts (research/, gitignored) are covered by neither.
"""
from collections import Counter
from pathlib import Path
import ast
import re

import launch_scan

REPO = launch_scan.REPO

# (file, enclosing scope, canonical launcher, call|ref) -> count, and what each one starts.
SITES = Counter({
    # BEGIN SITES (tests/launch_scan.py --update)
    ('packaging/aester_p2/installer_worker.py', '_run', 'subprocess.run', 'call'): 1,  # runtime python import probe
    ('scripts/bench_workflow.py', '_git_commit', 'subprocess.run', 'call'): 2,  # git
    ('scripts/bench_workflow.py', 'check_environment', 'subprocess.run', 'call'): 1,  # python affinity probe
    ('scripts/bench_workflow.py', 'fingerprint_run', 'subprocess.run', 'call'): 1,  # python verifier
    ('scripts/bench_workflow.py', 'main', 'subprocess.Popen', 'call'): 1,  # the bench re-executes itself (python)
    ('scripts/bench_workflow.py', 'sha_probe', 'subprocess.run', 'call'): 1,  # python SHA probe
    ('scripts/build_aester_installer.py', '_smoke_desktop', 'subprocess.run', 'call'): 1,  # bundled python: import smoke of the desktop source
    ('scripts/build_aester_installer.py', '_source_commit', 'subprocess.run', 'call'): 1,  # git rev-parse HEAD
    ('scripts/build_aester_installer.py', 'build', 'subprocess.run', 'call'): 2,  # csc; Inno Setup
    ('scripts/build_aester_installer.py', 'release_gates', 'subprocess.run', 'call'): 1,  # git status (release gate G5)
    ('scripts/convert_z8_test_dng.py', 'main', 'process_guard.guarded_process', 'call'): 1,  # Adobe DNG Converter
    ('scripts/split/split_desktop.py', '_git', 'subprocess.run', 'call'): 1,  # git mv (internal open/closed split tool)
    ('scripts/validate_guard_crash.py', 'main', 'process_guard.guarded_process', 'call'): 1,  # python owner
    ('scripts/validate_mcp.py', 'main', 'process_guard.guarded_process', 'call'): 1,  # python MCP server
    ('scripts/validate_mcp_next.py', 'main', 'process_guard.guarded_process', 'call'): 1,  # python MCP server
    ('scripts/validate_mcp_raw.py', 'main', 'process_guard.guarded_process', 'call'): 1,  # python MCP server
    ('scripts/validate_sony_device.py', 'adobe_decode', 'subprocess.run', 'call'): 1,  # Adobe DNG Converter
    ('scripts/validate_sony_device.py', 'e0_engine', 'subprocess.run', 'call'): 1,  # powershell version query
    ('src/starstack/components.py', '_validate', 'subprocess.run', 'call'): 2,  # runtime python -c; siril -v (W4)
    ('src/starstack/core.py', '_spawn_worker', 'subprocess.Popen', 'call'): 1,  # job worker (python)
    ('src/starstack/finishing_backend.py', 'render_recipe', 'process_guard.guarded_process', 'call'): 1,  # finishing_launch (python -> GIMP)
    ('src/starstack/finishing_components.py', '_probe_version', 'process_guard.guarded_process', 'call'): 1,  # GIMP --version
    ('src/starstack/finishing_launch.py', 'main', 'subprocess.Popen', 'call'): 1,  # GIMP batch render
    ('src/starstack/process_guard.py', '_JobObjectProcess.start', 'api.CreateProcessW', 'call'): 1,  # the guard itself
    ('src/starstack/process_guard.py', '_PopenProcess.start', 'subprocess.Popen', 'call'): 1,  # the guard itself
    ('src/starstack/raw_import.py', 'prepare_import', 'process_guard.guarded_process', 'call'): 1,  # Adobe DNG Converter
    ('src/starstack/retouch_distribution.py', '_run_upstream_installer', 'process_guard.guarded_process', 'call'): 1,  # GIMP installer
    ('src/starstack/siril.py', '_started', 'process_guard.guarded_process', 'call'): 1,  # THE Siril script run
    ('src/starstack/siril.py', 'doctor', 'subprocess.run', 'call'): 1,  # siril -v (W4)
    ('src/starstack/transient_scan.py', 'scan_frames', 'process_guard.guarded_process', 'call'): 1,  # transient scan worker (python)
    ('src/starstack/workflow.py', '_spawn', 'subprocess.Popen', 'call'): 1,  # workflow worker (python)
    # END SITES
})
# The complete argv of every launch in the version-probe functions (ast.unparse of the argv list), and the private
# environment the two in src/ must pass (W4). (The legacy trial ZIP's packaging/bootstrap.py was retired at the
# open/closed split.)
PROBES = {
    ("src/starstack/siril.py", "doctor"): {"[str(p.resolve()), '-v']": "siril_environment("},
    ("src/starstack/components.py", "_validate"): {"[str(runtime / 'python.exe'), '-B', '-X', 'utf8', '-c', code]": None,
                                                   "[str(siril), '-v']": "_siril_probe_environment("},
}
# FFTW wisdom lives in g_get_user_cache_dir(), which siril_environment does not isolate (settings.c:315-322; fft.c:453;
# deconvolution.c:487, 590). Adding one of these commands needs XDG_CACHE_HOME (and the config dir) there first.
FFTW = re.compile(r"^\s*(fftd|ffti|rl|sb|wiener|seqrl|seqsb|seqwiener|makepsf)(\s|$)")
_FOUND = launch_scan.launch_sites(REPO)


def _keywords(call):
    return {keyword.arg: ast.unparse(keyword.value) for keyword in call.keywords}


def test_every_process_launch_is_pinned_with_its_count():
    seen = launch_scan.counter(_FOUND)
    extra, missing = seen - SITES, SITES - seen
    assert not extra, ("new process launch (or another one in a pinned function): if it can start Siril, route it "
                       "through siril.run_script; otherwise run tests/launch_scan.py --update and say what it "
                       "starts: %s" % dict(extra))
    assert not missing, "a pinned launch moved or was removed; run tests/launch_scan.py --update: %s" % dict(missing)


def test_only_started_runs_siril_scripts_with_its_own_environment_block():
    started = [node for key, node in _FOUND if key[:2] == ("src/starstack/siril.py", "_started")]
    assert len(started) == 1
    assert _keywords(started[0]).get("env", "").startswith("siril_environment("), ast.unparse(started[0])
    run_script = [node for key, node in _FOUND if key[:2] == ("src/starstack/siril.py", "run_script")]
    assert run_script == []                       # run_script launches only through _started


def test_version_probes_run_exactly_the_pinned_argv_with_a_private_environment():
    for (file, scope), argvs in PROBES.items():
        calls = [node for key, node in _FOUND if key[:2] == (file, scope)]
        assert sorted(ast.unparse(call.args[0]) for call in calls) == sorted(argvs), (file, scope)
        for call in calls:
            wanted = argvs[ast.unparse(call.args[0])]
            if wanted:
                assert _keywords(call).get("env", "").startswith(wanted), f"{file}:{scope}: {ast.unparse(call)}"


def test_product_siril_scripts_use_no_fftw_command():
    hits = []
    for path in sorted((REPO / "src/starstack").rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):
            text = None
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                text = node.value
            elif isinstance(node, ast.JoinedStr) and node.values and isinstance(node.values[0], ast.Constant):
                text = node.values[0].value
            if text is not None and any(FFTW.match(line) for line in text.splitlines() or [text]):
                hits.append(f"{path.name}:{node.lineno}")
    assert hits == []


def test_installer_sources_start_only_python():
    sources = sorted((REPO / "packaging/aester_p2").glob("*.iss")) + [REPO / "packaging/aester_p2/Launcher.cs"]
    assert len(sources) >= 2
    for path in sources:
        for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
            if re.search(r"\b(Exec|ExecAndLogOutput|ShellExec|RunWorker)\s*\(|Process\.Start|ProcessStartInfo\(", line):
                assert "siril" not in line.lower(), f"{path.name}:{number}: {line.strip()}"


SYNTHETIC = '''
import asyncio, ctypes, os, subprocess
import subprocess as sp
from functools import partial
from subprocess import run as start
from .process_guard import guarded_process as G
from PySide6.QtCore import QProcess
from . import siril as backend, core
from .siril import guarded_process as g2
import starstack.siril as ss
import starstack.process_guard as pg
import starstack.process_guard


def launcher(siril):
    sp.run([siril, "-s", "a.ssf"])
    start([siril, "-s", "b.ssf"])
    with G([siril, "-s", "c.ssf"]):
        pass
    runner = subprocess.run
    later = partial(subprocess.Popen, [siril])
    ctypes.windll.kernel32.CreateProcessW(None, siril, None, None, False, 0, None, None, None, None)
    os.startfile(siril)
    asyncio.create_subprocess_exec(siril, "-s", "d.ssf")
    QProcess.startDetached(siril)
    return runner, later


def through_other_modules(siril):
    with backend.guarded_process([siril, "-s", "e.ssf"]):
        pass
    with g2([siril, "-s", "f.ssf"]):
        pass
    backend.subprocess.run([siril, "-s", "g.ssf"])
    core.subprocess.Popen([siril, "-s", "h.ssf"])
    ss.subprocess.check_call([siril])
    with pg.guarded_process([siril]):
        pass
    with starstack.process_guard.guarded_process([siril]):
        pass
    backend.os.startfile(siril)
    core.asyncio.create_subprocess_exec(siril)
    return backend.subprocess.Popen
'''


def test_the_scanner_finds_every_launch_form():
    found = launch_scan.counter(launch_scan.scan_source(SYNTHETIC, "x.py"))
    assert found == Counter({
        ("x.py", "launcher", "subprocess.run", "call"): 2,                 # sp.run, start (an aliased run)
        ("x.py", "launcher", "process_guard.guarded_process", "call"): 1,  # G
        ("x.py", "launcher", "subprocess.run", "ref"): 1,                  # runner = subprocess.run
        ("x.py", "launcher", "subprocess.Popen", "ref"): 1,                # partial(subprocess.Popen, ...)
        ("x.py", "launcher", "ctypes.windll.kernel32.CreateProcessW", "call"): 1,
        ("x.py", "launcher", "os.startfile", "call"): 1,
        ("x.py", "launcher", "asyncio.create_subprocess_exec", "call"): 1,
        ("x.py", "launcher", "PySide6.QtCore.QProcess.startDetached", "call"): 1,
        # P1b review SIRIL-4: primitives reached through another module's namespace
        ("x.py", "through_other_modules", "siril.guarded_process", "call"): 2,        # backend.*, g2
        ("x.py", "through_other_modules", "siril.subprocess.run", "call"): 1,
        ("x.py", "through_other_modules", "core.subprocess.Popen", "call"): 1,
        ("x.py", "through_other_modules", "siril.subprocess.check_call", "call"): 1,  # import starstack.siril as ss
        ("x.py", "through_other_modules", "process_guard.guarded_process", "call"): 2,  # pg.*, starstack.process_guard.*
        ("x.py", "through_other_modules", "siril.os.startfile", "call"): 1,
        ("x.py", "through_other_modules", "core.asyncio.create_subprocess_exec", "call"): 1,
        ("x.py", "through_other_modules", "siril.subprocess.Popen", "ref"): 1,
    })
    assert sum(found.values()) == 9 + 10


def test_an_extra_launch_in_a_pinned_function_or_an_aliased_one_is_caught():
    path = REPO / "src/starstack/components.py"
    source = path.read_text(encoding="utf-8-sig")
    anchor = "    if b\"1.4.4\" not in result.stdout + result.stderr:\n"
    assert source.count(anchor) == 1
    injected = source.replace(anchor, "    subprocess.run([str(siril), \"-s\", \"selftest.ssf\"], check=True)\n" + anchor)
    injected += "\n\ndef sneaky(siril):\n    import subprocess as sp\n    sp.check_call([siril, '-s', 'x.ssf'])\n"
    seen = launch_scan.counter(launch_scan.scan_source(injected, "src/starstack/components.py"))
    pinned = Counter({k: v for k, v in SITES.items() if k[0] == "src/starstack/components.py"})
    assert seen - pinned == Counter({("src/starstack/components.py", "_validate", "subprocess.run", "call"): 1,
                                     ("src/starstack/components.py", "sneaky", "subprocess.check_call", "call"): 1})


def test_every_pinned_launch_says_what_it_starts():
    lines, start, end = launch_scan._block_lines(Path(launch_scan.TABLE).read_bytes().decode("utf-8"))
    entries = [line for line in lines[start + 1:end] if line.strip()]
    assert len(entries) == len(SITES)
    unclassified = [line.strip() for line in entries if not re.search(r"\):\s*\d+,\s+# (?!NEW:)\S", line)]
    assert unclassified == [], "replace each '# NEW: ...' with what the launch starts"


def test_the_table_regenerates_to_itself(tmp_path):
    copy = tmp_path / "table.py"
    copy.write_bytes(Path(launch_scan.TABLE).read_bytes())
    result = launch_scan.update(copy, REPO)
    assert result["added"] == {} and result["removed"] == {}
    assert copy.read_bytes() == Path(launch_scan.TABLE).read_bytes()
