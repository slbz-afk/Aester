"""Efficiency P1b-3: L mode runs Siril's export and its stack as two processes (controlled engine, never Siril).

The three L paths (3a own registration, 3b fallback with quality-gate exclusions, 3c plain fallback) launch
``02_export`` (registration commands) and then ``02_stack`` (``core.LEGACY_STACK_SCRIPT``); the diagnostics-only
switch ``AESTER_L_SIRIL_LAYOUT=combined`` restores the pre-P1b single ``02_stack`` script.  Products, records and
the job identity do not depend on the layout.

These tests run jobs admitted before the EFF-D4 stack-settings contract (efficiency P1c §4.2; ``world.run``'s
``contract=False``, the default here), which keep exactly these scripts; a contract job refuses the combined layout and
adds its settings lines to ``02_stack`` (tests/test_stack_settings.py, which reuses this engine).
"""
from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

from starstack import core, imaging, siril, stack_settings
from starstack import registration_calibration as rc
import d4_fakes
import test_core_two_pass_reference as two_pass   # not "from ... import": pytest would collect its TestCase again
from test_core_two_pass_reference import _seq_text

FRAMES = 4
EXCLUDED = "light_00003"
REGISTER = "register light -2pass -transf=homography"
APPLY = "seqapplyreg light -filter-included"


def _calibration(path):
    """What registration_calibration.calibrate returns for each L path (and its record file)."""
    def fake(work, **kwargs):
        work = Path(work)
        if path == "3a":
            return two_pass.JobDiagnosticsTest._fake_calibration(work, {f"light_{i:05d}" for i in range(1, FRAMES + 1)},
                                                        count=FRAMES)
        record = {"status": "no_improvement" if path == "3b" else "unavailable", "reason": "controlled fallback"}
        if path == "3b":
            record["quality_gate"] = {"applied": True, "excluded_frames": [EXCLUDED],
                                      "excluded": [{"stem": EXCLUDED, "reasons": ["controlled blurred frame"]}]}
        core.atomic_json(work / rc.DISTORTION_JSON, record)
        return record
    return fake


class Engine:
    """Siril's file contract for the L scripts, dispatched on the script name and checked on its commands."""

    def __init__(self):
        self.scripts = []
        self.options = []        # (name, keyword options of the launch): config / stack_settings for a contract stack
        self.seen = []           # (name, r_light frames present, r_light_.seq present, selection) at launch
        self.after = None        # hook(name, directory) after a script
        self.stack_log = {}      # keyword arguments for d4_fakes.stack_log (what a contract stack prints)

    def __call__(self, executable, directory, name, commands, cancel_check, **options):
        commands = list(commands)
        work = directory / "work"
        # As run_script does: the launch (its .ssf written, Siril started) comes before the first cancel poll, so a
        # launch the product starts after a cancel request is recorded here (review lsplit-3).
        self.scripts.append((name, commands))
        self.options.append((name, options))
        self.seen.append((name, sorted(p.name for p in work.glob("r_light_*.fit")), (work / "r_light_.seq").exists()))
        (directory / f"{name}.ssf").write_text(siril.script_text(commands), encoding="utf-8")
        cancel_check()
        if name == "01_convert":
            for i, source in enumerate(sorted((directory / "stage").glob("*")), 1):
                shutil.copyfile(source, work / f"light_{i:05d}.fit")
        elif name == "02_probe":
            assert commands == core.PROBE_SCRIPT
        elif name == "02_register_fallback":
            assert commands == core.PROBE_SCRIPT
            (work / core.SEQUENCE_FILE).write_text(_seq_text(FRAMES, 1), encoding="utf-8")
        elif name == "02_export":
            assert commands[0] == "cd work" and commands[-1] == APPLY and core.STACK_COMMAND not in commands
            self._export(work, commands)
        elif name == "02_stack":
            plain, settings = stack_settings.without_settings(commands)
            assert (settings is not None) == bool(options.get("stack_settings")), (commands, options)
            if plain == list(core.LEGACY_STACK_SCRIPT):        # split layout: the stack process only
                assert (work / "r_light_.seq").exists(), "the stack process needs the export's sequence"
            else:                                             # combined layout: the pre-P1b single script
                assert commands[-1] == core.STACK_COMMAND and "seqclean r_light -stat" not in commands
                self._export(work, commands[:-1])
            self._stack(work)
            d4_fakes.stack_log(directory, name, commands, **self.stack_log)   # a contract stack's two log lines
        else:
            raise AssertionError(name)
        if self.after:
            self.after(name, directory)

    @staticmethod
    def _export(work, commands):
        if REGISTER in commands:   # 3c: Siril's own two-pass registration writes the sequence
            (work / core.SEQUENCE_FILE).write_text(_seq_text(FRAMES, 1), encoding="utf-8")
        selected = [image["stem"] for image in core.read_seq(work / core.SEQUENCE_FILE)["images"] if image["selected"]]
        for number, stem in enumerate(selected, 1):
            shutil.copyfile(work / f"{stem}.fit", work / f"r_light_{number:05d}.fit")
        (work / "r_light_.seq").write_text(_seq_text(len(selected), 0), encoding="utf-8")

    @staticmethod
    def _stack(work):
        frames = sorted(work.glob("r_light_*.fit"))
        assert frames, "the stack reads the exported frames from disk"
        data = np.mean([imaging.read_rgb(p) for p in frames], axis=0).astype(np.float32)
        imaging.write_linear(work / "sky_master.fit", data, "Controlled engine, not Siril", extra={"STACKCNT": len(frames)})

    def names(self):
        return [name for name, _ in self.scripts]

    def commands(self, name):
        found = [commands for script, commands in self.scripts if script == name]
        assert len(found) == 1, (name, self.names())
        return found[0]


@pytest.fixture
def world(tmp_path, monkeypatch):
    source = tmp_path / "素材"
    source.mkdir()
    for number in range(FRAMES):
        data = np.full((18, 24, 3), .03 + number * .001, dtype=np.float32)
        data[3:6, 5:8] += .2
        imaging.write_linear(source / f"原片 {number}.fit", data, "Test fixture")
    engine_info = {"available": True, "executable": str(tmp_path / "fake engine.exe"), "version": "1.4.4",
                   "message": "Test engine"}
    monkeypatch.setattr(core, "doctor", lambda *_: dict(engine_info))
    monkeypatch.setattr(core, "_spawn_worker", lambda directory: None)
    monkeypatch.setattr(core.shutil, "disk_usage", lambda _: SimpleNamespace(free=10**12))
    monkeypatch.delenv(core.L_LAYOUT_ENV, raising=False)
    engine = Engine()
    monkeypatch.setattr(core, "run_script", engine)

    def run(path, output="out", layout=None, engine_=None, contract=False):
        monkeypatch.setattr(rc, "calibrate", _calibration(path))
        if engine_ is not None:
            monkeypatch.setattr(core, "run_script", engine_)
        if layout is None:
            monkeypatch.delenv(core.L_LAYOUT_ENV, raising=False)
        else:
            monkeypatch.setenv(core.L_LAYOUT_ENV, layout)
        with d4_fakes.admitted_before_contract(not contract):
            job = core.submit_job(str(source), str(tmp_path / output), idempotency_key="lsplit", transient_mode="off")
        core.run_job(job["job_dir"])
        return core.load_job(job["job_dir"])

    return SimpleNamespace(source=source, tmp=tmp_path, engine=engine, run=run, monkeypatch=monkeypatch)


EXPECTED_SPLIT = {
    "3a": (["01_convert", "02_probe", "02_export", "02_stack"], core.export_script(core.DISTORTION_WCS, True)),
    "3b": (["01_convert", "02_probe", "02_register_fallback", "02_export", "02_stack"], ["cd work", APPLY]),
    "3c": (["01_convert", "02_probe", "02_export", "02_stack"], ["cd work", REGISTER, APPLY]),
}
EXPECTED_COMBINED = {
    "3a": (["01_convert", "02_probe", "02_stack"], core.stack_script(core.DISTORTION_WCS, own_registration=True)),
    "3b": (["01_convert", "02_probe", "02_register_fallback", "02_stack"], ["cd work", APPLY, core.STACK_COMMAND]),
    "3c": (["01_convert", "02_probe", "02_stack"], core.stack_script(None)),
}


def _succeeded(job):
    assert job["state"] == "succeeded", job.get("error")
    return job


def test_own_registration_exports_then_stacks(world):
    job = _succeeded(world.run("3a"))
    engine = world.engine
    assert engine.names() == ["01_convert", "02_probe", "02_export", "02_stack"]
    assert engine.commands("02_export") == core.export_script(core.DISTORTION_WCS, own_registration=True) \
        == ["cd work", APPLY]
    assert engine.commands("02_stack") == list(core.LEGACY_STACK_SCRIPT) \
        == ["cd work", "seqclean r_light -stat", core.STACK_COMMAND]
    folder = Path(job["job_dir"])
    assert (folder / "02_export.ssf").read_text(encoding="utf-8") == siril.script_text(core.export_script(core.DISTORTION_WCS, True))
    assert (folder / "02_stack.ssf").read_text(encoding="utf-8") == siril.script_text(core.LEGACY_STACK_SCRIPT)
    assert job["diagnostics"]["registration_source"] == core.REGISTRATION_SOURCE_OWN
    assert job["diagnostics"]["registration"] == ["seqapplyreg light -filter-included"]   # unchanged record
    assert job["diagnostics"]["stacked_count"] == FRAMES
    assert job["checkpoints"] == ["convert", "distortion", "stack"]


def test_fallback_with_exclusions_registers_exports_then_stacks(world):
    seen = {}

    def after(name, directory):
        if name == "02_register_fallback":
            seen["application_before_export"] = (directory / "work" / "quality_gate_application.json").exists()
    world.engine.after = after
    original = world.engine._export

    def export(work, commands):
        seen["selected"] = [i["stem"] for i in core.read_seq(work / core.SEQUENCE_FILE)["images"] if i["selected"]]
        original(work, commands)
    world.engine._export = export
    job = _succeeded(world.run("3b"))
    assert world.engine.names() == EXPECTED_SPLIT["3b"][0]
    assert world.engine.commands("02_export") == ["cd work", APPLY]
    assert world.engine.commands("02_stack") == list(core.LEGACY_STACK_SCRIPT)
    assert seen["application_before_export"] is False      # written by the product right after the fallback register
    assert seen["selected"] == [f"light_{i:05d}" for i in range(1, FRAMES + 1) if f"light_{i:05d}" != EXCLUDED]
    assert job["diagnostics"]["stacked_count"] == FRAMES - 1
    assert job["diagnostics"]["quality_gate"]["in_effect"]


def test_plain_fallback_moves_register_into_the_export(world):
    job = _succeeded(world.run("3c"))
    assert world.engine.names() == ["01_convert", "02_probe", "02_export", "02_stack"]
    assert world.engine.commands("02_export") == core.export_script(None) == ["cd work", REGISTER, APPLY]
    assert world.engine.commands("02_stack") == list(core.LEGACY_STACK_SCRIPT)
    assert job["diagnostics"]["registration_source"] == core.REGISTRATION_SOURCE_SIRIL
    assert job["diagnostics"]["registration"] == list(core.REGISTRATION_COMMANDS)
    assert job["diagnostics"]["stacked_count"] == FRAMES


def _literal_lists(function, script_name, namespace):
    """The command lists of every launch named ``script_name`` in ``function`` (second-to-last argument)."""
    tree = ast.parse(inspect.getsource(function))
    found = []
    for call in (n for n in ast.walk(tree) if isinstance(n, ast.Call)):
        constants = [a.value for a in call.args if isinstance(a, ast.Constant)]
        if script_name in constants and isinstance(call.args[-2], ast.List):
            found.append(eval(compile(ast.Expression(call.args[-2]), "<launch>", "eval"), dict(namespace)))
    return found


def test_split_scripts_compose_the_combined_script():
    for distortion in (None, core.DISTORTION_WCS):
        for own in (False, True):
            assert core.stack_script(distortion, own) == core.export_script(distortion, own) + [core.STACK_COMMAND]
    assert core.STACK_SCRIPT == core.export_script(None) + [core.STACK_COMMAND]
    assert core.OWN_REGISTRATION_STACK_SCRIPT == core.export_script(core.DISTORTION_WCS, True) + [core.STACK_COMMAND]
    assert core.LEGACY_STACK_SCRIPT == ("cd work", "seqclean r_light -stat", core.STACK_COMMAND)
    # strict-legacy's own stack script (left untouched by P1b-3) is the same stack process
    from starstack import registered_support as rs
    strict = _literal_lists(core._supported_stack, "02_stack", {**vars(core), "rs": rs})
    assert strict == [list(core.LEGACY_STACK_SCRIPT)]
    assert core.legacy_stack_layout("") == core.legacy_stack_layout("split") == "split"
    assert core.legacy_stack_layout("combined") == "combined"
    for bogus in ("Combined", "split ", "both", "1"):
        with pytest.raises(ValueError, match=core.L_LAYOUT_ENV):
            core.legacy_stack_layout(bogus)


@pytest.mark.parametrize("path", ["3a", "3b", "3c"])
def test_combined_layout_switch_restores_the_single_script(world, path):
    _succeeded(world.run(path, layout="combined"))
    names, script = EXPECTED_COMBINED[path]
    assert world.engine.names() == names
    assert world.engine.commands("02_stack") == script


@pytest.mark.parametrize("path", ["3a", "3c"])
@pytest.mark.parametrize("value", ["bogus", "Combined", "split "])
def test_malformed_layout_fails_before_any_siril_run(world, path, value):
    job = world.run(path, layout=value)
    assert job["state"] == "failed"
    assert core.L_LAYOUT_ENV in job["error"]
    assert world.engine.scripts == []


def _notes(job):
    return ((job.get("diagnostics") or {}).get("timings") or {}).get("notes") or {}


def _cancel_after_the_stack_checkpoint(world):
    """Request a cancel at the first step after checkpoint('stack') (the exposure diagnostics), once."""
    original, armed = core.exposure_diagnostics, [True]

    def hook(work, *args, **kwargs):
        if armed[0]:
            armed[0] = False
            core.cancel_job(str(Path(work).parent))
        return original(work, *args, **kwargs)
    world.monkeypatch.setattr(core, "exposure_diagnostics", hook)


@pytest.mark.parametrize("value", [None, "bogus", "Combined", "split ", "combined"])
def test_resume_after_the_stack_checkpoint_ignores_the_layout(world, value):
    """Review lsplit-2: an attempt resumed past the stack checkpoint never runs L's export and stack, so it ignores
    AESTER_L_SIRIL_LAYOUT (malformed or not), as before P1b."""
    _cancel_after_the_stack_checkpoint(world)
    job = world.run("3a", output="resume")
    assert job["state"] == "cancelled" and job["checkpoints"] == ["convert", "distortion", "stack"], job.get("error")
    world.engine.scripts.clear()
    if value is None:
        world.monkeypatch.delenv(core.L_LAYOUT_ENV, raising=False)
    else:
        world.monkeypatch.setenv(core.L_LAYOUT_ENV, value)
    core.resume_job(job["job_dir"])
    core.run_job(job["job_dir"])
    resumed = _succeeded(core.load_job(job["job_dir"]))
    assert world.engine.scripts == []                                      # nothing re-stacked
    assert "l_siril_layout" not in _notes(resumed)
    assert len(resumed["diagnostics"]["timing_attempts"]) == 1


def _first_launch_stops(world):
    """An engine that records each launch and fails the first one: how far an attempt gets before Siril."""
    launched = []

    def engine(executable, directory, name, commands, cancel_check):
        launched.append(name)
        raise RuntimeError("controlled stop at the first Siril launch")
    world.monkeypatch.setattr(core, "run_script", engine)
    return launched


@pytest.mark.parametrize("mode", ["R", "F", "strict"])
def test_range_finite_and_strict_jobs_ignore_the_layout(world, mode):
    """Review lsplit-2: F, R and strict jobs never use L's layout, so a malformed (or any) AESTER_L_SIRIL_LAYOUT
    leaves their attempt exactly as without it: the same first Siril launch, the same error, no layout note."""
    from starstack import finite_stack as fs, registered_support as rs, stack_range as sr
    mp = world.monkeypatch
    mp.setattr(sr, "require_engine", lambda *_: None)
    mp.setattr(rs, "require_engine", lambda *_: None)
    mp.setattr(fs, "describe_engine", lambda path: {"executable": str(path), "sha256": "0" * 64})
    mp.setattr(fs, "verify_engine", lambda *_: None)
    mask = world.tmp / "sky.png"
    mask.write_bytes(b"the attempt only copies the mask and checks its digest before the first launch")
    engine = world.tmp / "finite-siril-cli.exe"
    engine.write_bytes(b"stand-in")
    options = {"R": {"stack_range_mode": sr.MODE},
               "F": {"support_mode": rs.MODE, "sky_mask_path": str(mask), "stack_range_mode": fs.MODE,
                     "stack_engine_path": str(engine)},
               "strict": {"support_mode": rs.MODE, "sky_mask_path": str(mask)}}[mode]
    outcomes = {}
    for number, value in enumerate((None, "bogus", "Combined", "split ", "combined")):
        root = world.tmp / f"{mode}-{number}"
        launched = _first_launch_stops(world)
        if value is None:
            mp.delenv(core.L_LAYOUT_ENV, raising=False)
        else:
            mp.setenv(core.L_LAYOUT_ENV, value)
        job = core.submit_job(str(world.source), str(root), idempotency_key=f"lsplit-{mode}", transient_mode="off",
                              **options)
        core.run_job(job["job_dir"])
        job = core.load_job(job["job_dir"])
        assert job["state"] == "failed" and "controlled stop" in job["error"], (value, job.get("error"))
        assert core.L_LAYOUT_ENV not in job["error"] and "l_siril_layout" not in _notes(job)
        outcomes[value] = (launched, job["error"].replace(str(root), "<ROOT>"))
    assert outcomes[None][0] == ["01_convert"]
    assert all(outcome == outcomes[None] for outcome in outcomes.values()), outcomes


def test_only_l_attempts_before_the_stack_checkpoint_read_the_layout():
    from starstack import finite_stack as fs, registered_support as rs, stack_range as sr
    l_job = {"support_mode": "legacy", "stack_range_mode": "legacy", "checkpoints": ["convert", "distortion"]}
    assert core._may_run_legacy_stack(l_job)
    assert core._may_run_legacy_stack({"checkpoints": []})                # a job written before both fields existed
    assert not core._may_run_legacy_stack({**l_job, "checkpoints": ["convert", "distortion", "stack"]})
    assert not core._may_run_legacy_stack({**l_job, "support_mode": rs.MODE})                              # strict
    assert not core._may_run_legacy_stack({**l_job, "stack_range_mode": sr.MODE})                          # R
    assert not core._may_run_legacy_stack({**l_job, "support_mode": rs.MODE, "stack_range_mode": fs.MODE})  # F


def _without_timers(value):
    """Drop what the workflow verifier excludes as timers: timing receipts, durations and wall-clock times."""
    if isinstance(value, dict):
        return {key: _without_timers(item) for key, item in value.items()
                if key not in ("timings", "timing_attempts", "seconds") and not key.endswith(("_seconds", "_at"))}
    if isinstance(value, list):
        return [_without_timers(item) for item in value]
    return value


def _normalised(job, root):
    """job.json without timers; run roots and the random revision folder named."""
    assert "timings" in job["diagnostics"] and "timing_attempts" in job["diagnostics"]
    data = _without_timers(copy.deepcopy(job))
    text = json.dumps(data, ensure_ascii=False).replace(json.dumps(str(root))[1:-1], "<ROOT>")
    revisions = {Path(a["path"]).parent.name for a in job["artifacts"] if Path(a["path"]).parent.parent.name == "products"}
    for name in sorted(revisions):
        text = text.replace(name, "<REV>")
    return json.loads(text)


def _differences(a, b, path=""):
    """Leaf paths where two JSON values differ (readable assertion messages)."""
    if isinstance(a, dict) and isinstance(b, dict):
        return [d for key in sorted(set(a) | set(b), key=str)
                for d in (_differences(a.get(key, "<absent>"), b.get(key, "<absent>"), f"{path}/{key}")
                          if key in a and key in b else [f"{path}/{key}: only on one side"])]
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in _differences(x, y, f"{path}[{i}]")]
    return [] if a == b else [f"{path}: {str(a)[:120]!r} != {str(b)[:120]!r}"]


def _digests(job):
    """The master and every published artifact by name: decoded data for images (a JPEG's embedded ICC profile
    carries its creation time), raw bytes otherwise.  The products/<revision> folder name is random."""
    from PIL import Image
    from astropy.io import fits
    folder = Path(job["job_dir"])
    files = {"work/sky_master.fit": folder / "work" / "sky_master.fit",
             **{f"artifact/{a['label']}/{Path(a['path']).name}": Path(a["path"]) for a in job["artifacts"]}}
    assert len(files) == 1 + len(job["artifacts"])
    digests = {}
    for key, path in files.items():
        if path.suffix.lower() in (".fit", ".fits"):
            with fits.open(path, memmap=False) as hdus:
                payload = b"".join(hdu.data.tobytes() for hdu in hdus if hdu.data is not None)
        elif path.suffix.lower() in (".tif", ".tiff"):
            import tifffile
            payload = tifffile.imread(path).tobytes()
        elif path.suffix.lower() in (".jpg", ".jpeg", ".png"):
            with Image.open(path) as image:
                payload = image.tobytes()
        else:
            payload = path.read_bytes()
        digests[key] = hashlib.sha256(payload).hexdigest()
    return digests


@pytest.mark.parametrize("path", ["3a", "3b", "3c"])
def test_layout_switch_is_diagnostic_only(world, path):
    split = _succeeded(world.run(path, output="split"))
    combined = _succeeded(world.run(path, output="combined", layout="combined"))
    assert split["signature"] == combined["signature"]
    assert split["checkpoint_files"] == combined["checkpoint_files"]
    assert _digests(split) == _digests(combined)
    assert _differences(_normalised(split, world.tmp / "split"), _normalised(combined, world.tmp / "combined")) == []
    assert "l_siril_layout" not in split["diagnostics"]["timings"].get("notes", {})
    assert combined["diagnostics"]["timings"]["notes"]["l_siril_layout"] == "combined"
    outside = json.dumps({k: v for k, v in combined.items() if k != "diagnostics"})
    assert "l_siril_layout" not in outside and core.L_LAYOUT_ENV not in outside
    split_ops = split["diagnostics"]["timings"]["operation_counts"]
    combined_ops = combined["diagnostics"]["timings"]["operation_counts"]
    assert split_ops["script:02_export"] == 1 and split_ops["script:02_stack"] == 1
    assert "script:02_export" not in combined_ops and combined_ops["script:02_stack"] == 1


def test_range_finite_and_strict_scripts_are_unchanged():
    from starstack import finite_stack as fs, registered_support as rs, stack_range as sr
    namespace = {**vars(core), "rs": rs}
    # R mode's export call sites in the job worker (own registration, then fallback) and its stack (stack_range)
    assert _literal_lists(core._run_locked, "02_export", namespace) == [
        ["cd work", "seqapplyreg light -filter-included"], ["cd work", "seqapplyreg light -filter-included"]]
    assert ['setext fit', 'set32bits', 'setcompress 0', 'seqclean r_light -stat', sr.STACK] == [
        "setext fit", "set32bits", "setcompress 0", "seqclean r_light -stat",
        "stack r_light rej w 3 3 -norm=addscale -32b -filter-included -out=encoded_master.fit"]
    source = inspect.getsource(sr)
    assert "commands = ['setext fit', 'set32bits', 'setcompress 0', 'seqclean r_light -stat', STACK]" in source
    # strict-legacy (and F/strict-R through _supported_stack): export, then the same stack process as L
    assert _literal_lists(core._supported_stack, "02_export", namespace) == [
        ["cd work", "seqapplyreg light -filter-included -interp=lanczos4 -layer=1"]]
    assert _literal_lists(core._supported_stack, "02_stack", namespace) == [
        ["cd work", "seqclean r_light -stat", core.STACK_COMMAND]]
    # F: the finite build's stack script
    assert fs.STACK == "stack q rej w 3 3 -norm=addscale -32b -signed_output -nan_missing_v1 -out=finite_master.fit"
    assert ("commands = ['offline', 'setext fit', 'set32bits', 'setcompress 0',\n"
            "                    'set gui.icc_autoassignment=0', 'set gui.icc_autoconversion=0', f'setref q {ref}', STACK]"
            ) in inspect.getsource(fs)
    # no R/F/strict launch goes through the L helper
    run_locked = ast.parse(inspect.getsource(core._run_locked))
    helper_calls = [n for n in ast.walk(run_locked) if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Name) and n.func.id == "export_then_stack"]
    assert len(helper_calls) == 3


def test_cancel_between_export_and_stack(world):
    reference = _succeeded(world.run("3a", output="reference"))
    engine = Engine()

    def cancel_after_export(name, directory):
        if name == "02_export":
            core.cancel_job(str(directory))
    engine.after = cancel_after_export
    job = world.run("3a", output="cancelled", engine_=engine)
    assert job["state"] == "cancelled", job.get("error")
    # no stack process after the cancel: core's explicit check() between the two launches stops the attempt before
    # 02_stack is launched (the engine records a launch before its first cancel poll, as run_script does)
    assert engine.names() == ["01_convert", "02_probe", "02_export"]
    assert not (Path(job["job_dir"]) / "02_stack.ssf").exists()
    assert "stack" not in job["checkpoints"]
    engine.after = None
    engine.scripts.clear(); engine.seen.clear()
    core.resume_job(job["job_dir"])
    core.run_job(job["job_dir"])
    resumed = _succeeded(core.load_job(job["job_dir"]))
    assert engine.names() == ["02_export", "02_stack"]                    # convert/distortion checkpoints reused
    assert engine.seen[0] == ("02_export", [], False)                     # the attempt re-exports from a clean slate
    assert resumed["checkpoint_files"] == reference["checkpoint_files"]
    assert _digests(resumed) == _digests(reference)
    assert len(resumed["diagnostics"]["timing_attempts"]) == 1


def test_pre_split_job_folder_resumes_into_the_split(world):
    """A folder left by the pre-P1b single script (combined 02_stack.ssf, stale r_light frames, no stack checkpoint)."""
    fresh = _succeeded(world.run("3a", output="fresh"))
    old = Engine()

    def crash_after_export(name, directory):
        if name == "02_stack":     # the old single process exported, then died before the master was accepted
            (directory / "work" / "sky_master.fit").unlink()
            raise RuntimeError("Siril 02_stack 失败（1）")
    old.after = crash_after_export
    job = world.run("3a", output="old", layout="combined", engine_=old)
    folder = Path(job["job_dir"])
    assert job["state"] == "failed" and old.names() == ["01_convert", "02_probe", "02_stack"]
    assert (folder / "02_stack.ssf").read_text(encoding="utf-8") == siril.script_text(core.OWN_REGISTRATION_STACK_SCRIPT)
    assert sorted(p.name for p in (folder / "work").glob("r_light_*.fit")) and (folder / "work" / "r_light_.seq").exists()
    new = Engine()
    world.monkeypatch.setattr(core, "run_script", new)
    world.monkeypatch.delenv(core.L_LAYOUT_ENV)
    core.resume_job(job["job_dir"])
    core.run_job(job["job_dir"])
    resumed = _succeeded(core.load_job(job["job_dir"]))
    assert new.names() == ["02_export", "02_stack"]
    assert new.seen[0] == ("02_export", [], False)                        # stale exports removed first
    assert (folder / "02_stack.ssf").read_text(encoding="utf-8") == siril.script_text(core.LEGACY_STACK_SCRIPT)
    assert (folder / "02_export.ssf").read_text(encoding="utf-8") == siril.script_text(core.export_script(core.DISTORTION_WCS, True))
    assert resumed["signature"] == fresh["signature"]
    assert resumed["checkpoint_files"] == fresh["checkpoint_files"]
    assert _digests(resumed) == _digests(fresh)


def test_real_siril_fingerprint_allows_one_missing_statistics_line_per_layer():
    """test_legacy_stack_split_siril compares r_light_.seq's statistics (M) lines on their intersection because Siril's
    unlocked statistics cache can drop an entry; at most one line per layer may be missing, and each is a warning."""
    from test_legacy_stack_split_siril import _differing
    full = {f"M{layer}-{image}": f"M{layer}-{image} 1 2 3" for layer in range(3) for image in range(10)}
    side = {"r_light_.seq": "S 'r_light_'", "r_light_.seq M": full}

    def other(drop=(), change=None):
        lines = {tag: line for tag, line in full.items() if tag not in drop}
        if change:
            lines[change] += " 4"
        return {**side, "r_light_.seq M": lines}
    assert _differing(side, other()) == []
    for drop in (["M0-4"], ["M0-4", "M1-4", "M2-7"]):                       # one per layer, either side
        with pytest.warns(UserWarning, match="M0-4"):
            assert _differing(side, other(drop)) == [] and _differing(other(drop), side) == []
    for drop in (["M0-4", "M0-5"], list(full)):                             # two in one layer; every line missing
        with pytest.warns(UserWarning):
            assert _differing(side, other(drop)) == ["r_light_.seq M"]
    with pytest.warns(UserWarning):                                         # one missing on each side, same layer
        assert _differing(other(["M1-2"]), other(["M1-3"])) == ["r_light_.seq M"]
    assert _differing(side, other(change="M2-9")) == ["r_light_.seq M"]   # a line present on both sides differs
