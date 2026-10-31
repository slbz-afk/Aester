"""Efficiency P1c F-3: core's side (docs/EFFICIENCY_P1C_PREREG_20261005.md §3.3, U-C2).

- AESTER_FINITE_WRITE (auto | reference): read once, at the start of an attempt that may write the finite stack's q_
  inputs; a malformed value fails that attempt before any Siril run; every other attempt ignores it.
- core._FinitePools: each pool runs in its timer operation and is noted under timings only (frame_pool.<site>, R19),
  nothing new in the record; a pool its own plan made serial carries the pre-registered reason.
- Job level (§6.2 (c)): an F job run through core with a controlled engine (strict-support export, MeanEngine as the
  finite engine) that fails at frame k in either finite pool records the serial run's job.json and receipt, for the
  pre-P1c writer serially, core's default and 16 workers; resumed, it ends as an uninterrupted job.
"""
from __future__ import annotations

import json
import shutil
import threading
from pathlib import Path

import numpy as np
import pytest

from finite_fixtures import MeanEngine, fake_runtime, make_work, q_digests, sequence_text
from starstack import core, finite_stack as fs, frame_pool, profiling, registered_support as rs, stack_range as sr
from test_legacy_stack_split import (_differences, _digests, _first_launch_stops, _without_timers,  # noqa: F401
                                     world)


# ================================================================== AESTER_FINITE_WRITE

@pytest.mark.parametrize("value, setting", [(None, "auto"), ("", "auto"), ("auto", "auto"), (" Reference ", "reference"),
                                            ("AUTO", "auto")])
def test_the_knob_accepts_auto_and_reference(monkeypatch, value, setting):
    if value is None:
        monkeypatch.delenv(core.FINITE_WRITE_ENV, raising=False)
    else:
        monkeypatch.setenv(core.FINITE_WRITE_ENV, value)
    assert core.finite_write_setting() == setting


@pytest.mark.parametrize("value", ["ref", "streamed", "off", "1", "ｒｅｆｅｒｅｎｃｅ"])
def test_a_malformed_knob_is_refused_by_name(value):
    with pytest.raises(ValueError, match=core.FINITE_WRITE_ENV):
        core.finite_write_setting(value)


def test_only_f_attempts_before_the_stack_checkpoint_read_the_knob():
    f_job = {"stack_range_mode": fs.MODE, "support_mode": rs.MODE, "checkpoints": ["convert", "distortion"]}
    assert core._may_run_finite_write(f_job)
    assert not core._may_run_finite_write({**f_job, "checkpoints": ["convert", "distortion", "stack"]})
    assert not core._may_run_finite_write({**f_job, "stack_range_mode": sr.MODE})                   # R
    assert not core._may_run_finite_write({"support_mode": rs.MODE, "checkpoints": []})             # strict L
    assert not core._may_run_finite_write({"checkpoints": []})                                      # L


def _options(world, mode, monkeypatch):
    monkeypatch.setattr(sr, "require_engine", lambda *_: None)
    monkeypatch.setattr(rs, "require_engine", lambda *_: None)
    monkeypatch.setattr(fs, "describe_engine", lambda path: {"executable": str(path), "sha256": "0" * 64})
    monkeypatch.setattr(fs, "verify_engine", lambda *_: None)
    mask = world.tmp / "sky.png"
    mask.write_bytes(b"the attempt only copies the mask and checks its digest before the first launch")
    engine = world.tmp / "finite-siril-cli.exe"
    engine.write_bytes(b"stand-in")
    return {"L": {}, "R": {"stack_range_mode": sr.MODE},
            "F": {"support_mode": rs.MODE, "sky_mask_path": str(mask), "stack_range_mode": fs.MODE,
                  "stack_engine_path": str(engine)},
            "strict": {"support_mode": rs.MODE, "sky_mask_path": str(mask)}}[mode]


@pytest.mark.parametrize("mode", ["F", "L", "R", "strict"])
def test_a_malformed_knob_fails_only_an_f_attempt_and_before_any_siril_run(world, monkeypatch, mode):
    options = _options(world, mode, monkeypatch)
    outcomes = {}
    for number, value in enumerate((None, "bogus", "reference", "auto")):
        launched = _first_launch_stops(world)
        if value is None:
            monkeypatch.delenv(core.FINITE_WRITE_ENV, raising=False)
        else:
            monkeypatch.setenv(core.FINITE_WRITE_ENV, value)
        root = world.tmp / f"{mode}-{number}"
        job = core.submit_job(str(world.source), str(root), idempotency_key=f"fwrite-{mode}", transient_mode="off",
                              **options)
        core.run_job(job["job_dir"])
        job = core.load_job(job["job_dir"])
        assert job["state"] == "failed", job
        outcomes[value] = (launched, job["error"].replace(str(root), "<ROOT>"))
    assert outcomes[None][0] == ["01_convert"] and "controlled stop" in outcomes[None][1]
    if mode == "F":
        assert outcomes["bogus"][0] == [] and core.FINITE_WRITE_ENV in outcomes["bogus"][1]
        del outcomes["bogus"]
    assert all(outcome == outcomes[None] for outcome in outcomes.values()), outcomes


# ================================================================== notes, operations, records

def _f_run(tmp_path, monkeypatch, setting, workers=None):
    if workers is None:
        monkeypatch.delenv(frame_pool.ENV_WORKERS, raising=False)
    else:
        monkeypatch.setenv(frame_pool.ENV_WORKERS, str(workers))
    job = make_work(tmp_path / f"job-{setting}-{workers}", count=7)
    runtime = fake_runtime(tmp_path / "runtime", monkeypatch)
    timer = profiling.StageTimer()
    record = fs.run(runtime, job.directory, job.work, lambda: None, MeanEngine(7, 2),
                    pools=core._FinitePools(timer, setting))
    return record, timer.finish(), job


@pytest.mark.parametrize("setting, workers", [("auto", None), ("reference", None), ("auto", 1), ("reference", "max")])
def test_each_pool_is_noted_under_timings_in_its_operation_and_the_record_is_unchanged(tmp_path, monkeypatch, setting,
                                                                                         workers):
    record, timings, job = _f_run(tmp_path, monkeypatch, setting, workers)
    plain = fs.run(fake_runtime(tmp_path / "runtime", monkeypatch), job.directory, job.work, lambda: None,
                   MeanEngine(7, 2))
    assert list(record) == list(plain)                                   # no new key outside timings (R19)
    assert harness_json(record["sources"]) == harness_json(plain["sources"])
    notes = timings["notes"]
    inventory, write = notes["frame_pool.finite_inventory"], notes["frame_pool.finite_write"]
    assert len(inventory) == len(write) == 1
    write = write[0]
    assert set(timings["operations_seconds"]) >= {"finite_inventory", "finite_write"}
    assert write["body"] == ("reference" if setting == "reference" else "streamed") and write["fallbacks"] == {}
    assert write["unit_bytes"] > 0 and "bound" in write and "fp_env_applied" in write
    if setting == "reference":
        assert write["serial"] and write["reason"] == "route" and write["workers"] == 1
    elif workers == 1:
        assert write["serial"] and write["reason"] == "workers"         # --parallel serial keeps run_ordered's reason
        assert inventory[0]["serial"] and inventory[0]["reason"] == "workers"
    else:
        assert not write["serial"] and write["workers"] == min(core.FINITE_POOL_CAP, 7)
        assert not inventory[0]["serial"]


def harness_json(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)


# ================================================================== the wiring: _run_locked -> _supported_stack -> run

@pytest.mark.parametrize("given, environment, expected", [
    ("reference", None, "reference"), ("auto", "reference", "auto"),    # the attempt's own reading wins
    (None, "reference", "reference"), (None, None, "auto")])            # a caller without one: read now
def test_the_supported_stack_hands_run_core_s_pools(tmp_path, monkeypatch, given, environment, expected):
    seen = {}

    def run(engine, directory, work, check, runner, *, pools=None):
        seen.update(engine=engine, pools=pools)
    monkeypatch.setattr(fs, "run", run)
    monkeypatch.setattr(rs, "prepare_exports", lambda *args, **kwargs: seen.update(retain=kwargs))
    monkeypatch.setattr(core, "_validate_frame", lambda *args, **kwargs: (3, 6, 8))
    monkeypatch.setattr(core, "_validity_source", lambda *args, **kwargs: None)
    monkeypatch.setattr(core, "_grid_model", lambda *args, **kwargs: None)
    monkeypatch.setattr(core, "_update", lambda *args, **kwargs: None)
    if environment is None:
        monkeypatch.delenv(core.FINITE_WRITE_ENV, raising=False)
    else:
        monkeypatch.setenv(core.FINITE_WRITE_ENV, environment)
    work = tmp_path / "work"
    work.mkdir()
    (work / core.SEQUENCE_FILE).write_text("a sequence", encoding="utf-8")
    (work / "light_00001.fit").write_bytes(b"")
    launches = []

    def runner(executable, directory, name, commands, check, **options):
        launches.append((name, frame_pool.exclusive_sections()))
    timer = profiling.StageTimer()
    job = {"stack_range_mode": fs.MODE, "stack_engine": {"executable": "finite-siril-cli.exe"}}
    core._supported_stack("siril", tmp_path, work, job, {}, True, {"dilation_px": 0}, lambda: None, runner=runner,
                          timer=timer, finite_write=given)
    pools = seen["pools"]
    assert isinstance(pools, core._FinitePools) and pools.timer is timer and pools.write_setting == expected
    assert seen["engine"] == job["stack_engine"] and seen["retain"]["retain_frame_support"] is True
    assert launches == [("02_export", ())]                              # the export only; not a stock stack


# ================================================================== job level: a failure in a finite pool (§6.2 (c))

class _FiniteEngine:
    """The world's controlled engine for an F job (never Siril): strict support's export (the selected frames as
    registered by Siril, here copied, and an identity r_light_.seq) and the finite engine's stack (MeanEngine); every
    other launch goes to the world's engine."""

    def __init__(self, base):
        self.base, self.stacks = base, []          # stacks: the attempt folder of each finite stack launch

    def __call__(self, executable, directory, name, commands, cancel_check, **options):
        if name == "02_export":
            assert list(commands) == ["cd work", rs.EXPORT_COMMAND]
            work = directory / "work"
            selected = [i["stem"] for i in core.read_seq(work / core.SEQUENCE_FILE)["images"] if i["selected"]]
            for number, stem in enumerate(selected, 1):
                shutil.copyfile(work / f"{stem}.fit", work / f"r_light_{number:05d}.fit")
            (work / "r_light_.seq").write_text(sequence_text(len(selected), 0), encoding="utf-8")
            cancel_check()
            return
        if name == "stack":
            self.stacks.append(directory.name)
            return MeanEngine(len(list(directory.glob("q_*.fit"))), 0)(executable, directory, name, commands,
                                                                         cancel_check)
        return self.base(executable, directory, name, commands, cancel_check, **options)


# (AESTER_FINITE_WRITE, AESTER_FRAME_WORKERS): the pre-P1c writer serially, then core's default and 16 workers
_SETTINGS = (("reference", "1"), ("auto", None), ("auto", "16"))


F_HEIGHT, F_WIDTH = 48, 64     # the world's 18 x 24 frames leave no strictly supported sky under the fallback's shifts


def _f_world(world, monkeypatch):
    """F-job stand-ins on the world: engines and runtime admitted, Siril's two-pass fallback (strict support with
    own=False: shifts of 1.5 px per frame), a source folder of 48 x 64 frames and its sky mask (upper two thirds)."""
    from PIL import Image
    from starstack import imaging, registration_calibration as rc
    from test_legacy_stack_split import FRAMES, _calibration
    for target, name in ((sr, "require_engine"), (rs, "require_engine"), (fs, "verify_engine")):
        monkeypatch.setattr(target, name, lambda *_: None)
    monkeypatch.setattr(fs, "describe_engine", lambda path: {"executable": str(path), "sha256": "0" * 64})
    monkeypatch.setattr(rc, "calibrate", _calibration("3c"))
    source = world.tmp / "F 素材"
    source.mkdir()
    rng = np.random.default_rng(7)
    for number in range(FRAMES):
        data = (.03 + number * .001 + rng.normal(0, .002, (F_HEIGHT, F_WIDTH, 3))).astype(np.float32)
        data[10:14, 20:24] += .2
        imaging.write_linear(source / f"原片 {number}.fit", data, "Test fixture")
    mask = np.zeros((F_HEIGHT, F_WIDTH), dtype=np.uint8)
    mask[:32] = 255
    Image.fromarray(mask).save(world.tmp / "sky.png")
    engine = world.tmp / "finite-siril-cli.exe"
    engine.write_bytes(b"stand-in")
    finite = _FiniteEngine(world.engine)
    monkeypatch.setattr(core, "run_script", finite)
    return source, {"support_mode": rs.MODE, "sky_mask_path": str(world.tmp / "sky.png"), "stack_range_mode": fs.MODE,
                    "stack_engine_path": str(engine)}, finite


def _set(monkeypatch, write, workers):
    monkeypatch.setenv(core.FINITE_WRITE_ENV, write)
    if workers is None:
        monkeypatch.delenv(frame_pool.ENV_WORKERS, raising=False)
    else:
        monkeypatch.setenv(frame_pool.ENV_WORKERS, workers)


def _f_job(world, f_world, name):
    source, options, _ = f_world
    job = core.submit_job(str(source), str(world.tmp / name), idempotency_key="finite-job", transient_mode="off",
                          **options)
    core.run_job(job["job_dir"])
    return core.load_job(job["job_dir"])


def _checkpoints(job):
    """The job's checkpoint records: each file's digest, except a JSON record's, which carries wall-clock fields
    (sky_validity.json's applied_at, the support record's and the finite receipt's seconds) and, for the finite
    receipt, the random attempt folder: such a record by its content without those, the job folder named."""
    folder = Path(job["job_dir"])

    def record(path):
        value = _without_timers(json.loads((folder / path).read_text(encoding="utf-8")))
        if Path(path).name == fs.RECORD:
            del value["attempt"]
        return json.dumps(value, ensure_ascii=False).replace(json.dumps(str(folder))[1:-1], "<JOB>")
    return {stage: [(f["path"], record(f["path"]) if f["path"].endswith(".json") else f["sha256"]) for f in files]
            for stage, files in job["checkpoint_files"].items()}


def _finite_outcome(job, root):
    """job.json without timers (its checkpoint records as _checkpoints gives them), its root named; the finite
    attempts' receipts (state, error) in creation order."""
    value = {**_without_timers(job), "checkpoint_files": _checkpoints(job)}
    text = json.dumps(value, ensure_ascii=False).replace(json.dumps(str(root))[1:-1], "<ROOT>")
    attempts = sorted(Path(job["job_dir"]).glob("finite-*"), key=lambda p: p.stat().st_ctime_ns)
    receipts = [json.loads((a / "receipt.json").read_text(encoding="utf-8")) for a in attempts]
    return json.loads(text), [(r["state"], r.get("error")) for r in receipts]


def _finite_products(job):
    """What a resumed F job must share with an uninterrupted one: its checkpoint records (_checkpoints), the q_ copies
    of the published attempt and every product's data."""
    folder = Path(job["job_dir"])
    attempt = json.loads((folder / "work" / fs.RECORD).read_text(encoding="utf-8"))["attempt"]
    return _checkpoints(job), q_digests(folder / attempt), _digests(job)


@pytest.fixture
def clean_f_job(world, monkeypatch):
    f_world = _f_world(world, monkeypatch)
    _set(monkeypatch, "auto", None)
    job = _f_job(world, f_world, "clean")
    assert job["state"] == "succeeded", job.get("error")
    return f_world, _finite_products(job)


@pytest.mark.parametrize("k", [0, 1, 2, 3])                      # world: 4 frames, so {0, 1, middle, last} is all
@pytest.mark.parametrize("site", ["finite_inventory", "finite_write"])
def test_a_job_failing_in_a_finite_pool_records_the_serial_error_and_resumes_to_the_clean_result(
        world, monkeypatch, clean_f_job, site, k):
    """Prereg §6.2 (c) at job level: an F job through core.run_job with a controlled engine fails at frame k in a
    finite pool (the inventory: frame k's export digest; the q_ writes: frame k's support map, loaded a second time)
    under the pre-P1c writer serially, core's default and 16 workers: job.json (timers aside) and the finite attempt's
    receipt (state, error) are the serial run's; resumed, each job ends as the uninterrupted one."""
    f_world, clean = clean_f_job
    finite = f_world[2]
    monkeypatch.setattr(frame_pool, "_AUDIT_CONTEXT", True)      # §6.2 (e)
    failing = f"r_light_{k + 1:05d}.fit" if site == "finite_inventory" else f"frame_support_light_{k + 1:05d}.npy"
    outcomes = {}
    for index, (write, workers) in enumerate(_SETTINGS):
        _set(monkeypatch, write, workers)
        name = f"{site[7]}{k}{index}"            # short: the finite engine's longest path must fit (S-8)
        threads = set()                          # the threads the injected site's calls ran on
        with monkeypatch.context() as patch:
            if site == "finite_inventory":
                real_sha = fs.sha

                def sha(path, check=None):
                    if Path(path).name.startswith("r_light_"):
                        threads.add(threading.current_thread().name.startswith("aester-"))
                    if Path(path).name == failing:
                        raise RuntimeError(f"injected failure hashing {failing}")
                    return real_sha(path, check=check)
                patch.setattr(fs, "sha", sha)
            else:
                real_load, loads = rs.load_frame_support, {}

                def load(work, descriptor):
                    loads[descriptor["file"]] = loads.get(descriptor["file"], 0) + 1
                    if loads[descriptor["file"]] == 2:   # a q_ write unit
                        threads.add(threading.current_thread().name.startswith("aester-"))
                    if descriptor["file"] == failing and loads[failing] == 2:   # the first load is the inventory's
                        raise RuntimeError(f"injected failure loading {failing}")
                    return real_load(work, descriptor)
                patch.setattr(rs, "load_frame_support", load)
            launched = len(finite.stacks)
            job = _f_job(world, f_world, name)
        assert job["state"] == "failed" and "injected failure" in job["error"], job.get("error")
        assert len(finite.stacks) == launched                         # it failed before the finite engine started
        assert threads == ({False} if workers == "1" else {True}) or workers is None, (write, workers, threads)
        outcomes[(write, workers)] = _finite_outcome(job, world.tmp / name)
        core.resume_job(job["job_dir"])
        core.run_job(job["job_dir"])
        resumed = core.load_job(job["job_dir"])
        assert resumed["state"] == "succeeded", resumed.get("error")
        found = _finite_products(resumed)
        assert found == clean, (write, workers, _differences(list(found), list(clean)))
    serial = outcomes[_SETTINGS[0]]
    verb = "hashing" if site == "finite_inventory" else "loading"
    assert serial[1] == [("failed", f"injected failure {verb} {failing}")]
    for setting, found in outcomes.items():
        assert found == serial, (setting, _differences(found[0], serial[0]), found[1])
