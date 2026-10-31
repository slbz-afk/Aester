"""Hardening package (docs/HARDENING_PACKAGE_20260922.md): H2 ``force_restart``, H3 ``starstack gc`` and the disk gate by
frames — synthetic jobs on the controlled engine, never live Siril."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import shutil
import sys

import numpy as np
import pytest

from starstack import cli, core, disk_budget, gc, imaging
from starstack.locking import file_lock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import color_fixtures as cf  # noqa: E402

# Controlled engines here model Siril before the EFF-D4 stack-settings contract (efficiency P1c §4.2): jobs are
# admitted without it (tests/conftest.py pre_contract_jobs); contract jobs: tests/test_stack_settings.py.
pytestmark = pytest.mark.usefixtures("pre_contract_jobs")


@pytest.fixture
def environment(tmp_path, monkeypatch):
    source = tmp_path / "素材 with spaces"
    source.mkdir()
    for number in range(3):
        data = np.full((18, 24, 3), .03 + number * .001, dtype=np.float32)
        data[3:6, 5:8] += .2
        imaging.write_linear(source / f"原片 {number}.fit", data, "Test fixture")
    engine = {"available": True, "executable": str(tmp_path / "fake engine.exe"), "version": "1.4.4", "message": "Test engine"}
    monkeypatch.setattr(core, "doctor", lambda *_: dict(engine))
    monkeypatch.setattr(core.shutil, "disk_usage", lambda _: SimpleNamespace(free=10 ** 12))
    launched = []
    monkeypatch.setattr(core, "_spawn_worker", lambda directory: launched.append(directory))
    return SimpleNamespace(source=source, output=tmp_path / "results with spaces", engine=engine, launched=launched)


class ControlledEngine:
    def __init__(self, monkeypatch):
        self.calls = []
        self.drop_frame = False
        monkeypatch.setattr(core, "run_script", self.run)

    def run(self, executable, directory, name, commands, cancel_check):
        phase = core.load_job(directory)["stage"]
        self.calls.append(phase)
        cancel_check()
        work = directory / "work"
        if phase == "convert":
            sources = sorted((directory / "stage").glob("*"))
            for i, path in enumerate(sources[:-1] if self.drop_frame else sources, 1):
                shutil.copyfile(path, work / f"light_{i:05d}.fit")
            (work / "light_.seq").write_text("controlled sequence\n", encoding="utf-8")
        elif phase == "calibrate":
            pass
        elif phase == "stack" and name == "02_export":   # P1b-3: L's export process
            for i, p in enumerate(sorted(work.glob("light_*.fit")), 1):
                shutil.copyfile(p, work / f"r_light_{i:05d}.fit")     # the registered copies Siril's seqapplyreg would write
            (work / "light_.seq").write_text("controlled sequence after registration\n", encoding="utf-8")
            (work / "r_light_.seq").write_text("controlled registered sequence\n", encoding="utf-8")
        elif phase == "stack" and name == "02_stack":    # P1b-3: L's stack process
            inputs = [imaging.read_rgb(p) for p in sorted(work.glob("light_*.fit"))]
            imaging.write_linear(work / "sky_master.fit", np.mean(inputs, axis=0).astype(np.float32), "Controlled engine")
        else:
            raise AssertionError(f"Unexpected engine phase: {phase}")


def submit(env, key=None, **kwargs):
    return core.submit_job(str(env.source), str(env.output), idempotency_key=key, **kwargs)


def tree_digest(root: Path, *, exclude=()) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name not in exclude):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def tree_bytes(root: Path, *, exclude=()) -> int:
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file() and p.name not in exclude)


def finished_job(environment, monkeypatch):
    fake = ControlledEngine(monkeypatch)
    job = submit(environment, "hardening")
    core.run_job(job["job_dir"])
    result = core.load_job(job["job_dir"])
    assert result["state"] == "succeeded", result.get("error")
    return fake, result


# ---------------------------------------------------------------- H2: force_restart

def test_force_restart_retires_a_failed_job_and_runs_a_fresh_one_under_the_same_key(environment, monkeypatch):
    fake = ControlledEngine(monkeypatch)
    fake.drop_frame = True
    job = submit(environment, "retry-me")
    core.run_job(job["job_dir"])
    failed = core.load_job(job["job_dir"])
    assert failed["state"] == "failed"
    old_dir = Path(job["job_dir"])
    old_digest = tree_digest(old_dir)
    # plain resubmission keeps returning the failed job (unchanged behaviour)
    assert submit(environment, "retry-me")["state"] == "failed" and len(environment.launched) == 1
    fake.drop_frame = False
    fresh = submit(environment, "retry-me", force_restart=True)
    assert fresh["job_dir"] == str(old_dir) and fresh["state"] == "queued" and len(environment.launched) == 2
    assert fresh["restarted_from"]["previous_state"] == "failed" and fresh["restarted_from"]["previous_error"] == failed["error"]
    retired = Path(fresh["restarted_from"]["directory"])
    assert retired.parent == old_dir.parent and retired.name.startswith(old_dir.name + ".restarted-")
    assert tree_digest(retired) == old_digest, "the retired directory is renamed only, byte for byte"
    assert json.loads((retired / "job.json").read_text(encoding="utf-8"))["state"] == "failed"
    core.run_job(fresh["job_dir"])
    assert core.load_job(fresh["job_dir"])["state"] == "succeeded"
    assert fake.calls == ["convert", "convert", "calibrate", "stack", "stack"]   # P1b-3: export, then stack


def test_force_restart_accepts_a_crashed_worker_but_refuses_live_or_changed_jobs(environment, monkeypatch):
    ControlledEngine(monkeypatch)
    job = submit(environment, "crashed")
    directory = Path(job["job_dir"])
    # queued and possibly starting: refused, nothing renamed
    with pytest.raises(ValueError, match="仍在运行或排队"):
        submit(environment, "crashed", force_restart=True)
    assert directory.is_dir() and not list(directory.parent.glob("*.restarted-*"))
    # a worker that holds the lock: refused
    file = directory / "job.json"
    stored = json.loads(file.read_text(encoding="utf-8"))
    stored.update(state="running", stage="stack", worker_pid=999999999)
    file.write_text(json.dumps(stored), encoding="utf-8")
    with file_lock(directory / "worker.lock"):
        with pytest.raises(ValueError, match="仍在运行或排队"):
            submit(environment, "crashed", force_restart=True)
    assert not list(directory.parent.glob("*.restarted-*"))
    # the same persisted state with no lock and a dead pid is a crashed worker (recovery_blocked): resume refuses, restart works
    assert core.load_job(directory)["recovery_blocked"]
    with pytest.raises(ValueError):
        core.resume_job(str(directory))
    fresh = submit(environment, "crashed", force_restart=True)
    assert fresh["restarted_from"]["previous_state"] == "failed" and len(environment.launched) == 2
    assert len(list(directory.parent.glob("job-*.restarted-*"))) == 1
    # changed material under the same key is still refused, with or without force_restart, and nothing is renamed
    p = next(environment.source.glob("*.fit"))
    with p.open("ab") as handle:
        handle.write(b"changed")
    for force in (False, True):
        with pytest.raises(ValueError, match="不同素材"):
            submit(environment, "crashed", force_restart=force)
    assert len(list(directory.parent.glob("job-*.restarted-*"))) == 1 and len(environment.launched) == 2


def test_force_restart_is_a_strict_bool_in_core_and_mcp(environment, monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="bool"):
        submit(environment, "k", force_restart="yes")
    with pytest.raises(ValueError, match="bool"):
        submit(environment, "k", force_restart=1)
    from starstack.mcp_server import WorkspaceTools
    inputs, outputs = tmp_path / "in", tmp_path / "out"
    inputs.mkdir()
    outputs.mkdir()
    service = WorkspaceTools(input_root=inputs, output_root=outputs)
    monkeypatch.setattr(core, "submit_job", lambda *a, **k: pytest.fail("validation must happen before core"))
    with pytest.raises(ValueError, match="bool"):
        service.submit_job("scene", force_restart="true")
    with pytest.raises(ValueError, match="idempotency_key"):
        service.submit_job("scene", force_restart=True)


def test_cli_run_forwards_force_restart(monkeypatch, capsys):
    seen = {}

    def fake_submit(*args, **kwargs):
        seen.update(kwargs)
        return {"job_dir": "x", "state": "queued"}
    monkeypatch.setattr(cli, "submit_job", fake_submit)
    monkeypatch.setattr(sys, "argv", ["starstack", "run", "--input", "a", "--output", "b", "--key", "k", "--force-restart"])
    assert cli.main() == 0
    assert seen["force_restart"] is True
    monkeypatch.setattr(sys, "argv", ["starstack", "run", "--input", "a", "--output", "b", "--key", "k"])
    assert cli.main() == 0 and seen["force_restart"] is False


# ---------------------------------------------------------------- H3: gc

def test_gc_dry_run_changes_nothing_and_apply_removes_exactly_the_listed_bytes(environment, monkeypatch):
    fake, job = finished_job(environment, monkeypatch)
    directory = Path(job["job_dir"])
    products = directory / "products"
    referenced = gc.referenced_products(directory, job)
    assert len(referenced) == 1
    kept_dir = products / next(iter(referenced))
    # superseded exports of the job's own shape, an empty attempt, and foreign directories that must be kept
    superseded = products / "deadbeef0123"
    superseded.mkdir()
    shutil.copyfile(kept_dir / "master_linear.fit", superseded / "master_linear.fit")
    (superseded / "natural.jpg").write_bytes(b"old jpeg")
    (products / "0123456789ab").mkdir()
    foreign = products / "master-abc123"
    foreign.mkdir()
    (foreign / "master_scene_linear.tif").write_bytes(b"user's standalone export, not the job's")
    (products / "notes.txt").write_bytes(b"user note")
    plain_digest = tree_digest(directory)
    plan = gc.plan(directory, work_cache=True)
    assert plan["dry_run"] and tree_digest(directory) == plain_digest, "a dry run must not modify the tree"
    assert plan["kept_products"] == [kept_dir.name] and set(plan["unrecognised_kept"]) == {"master-abc123", "notes.txt"}
    categories = {e["category"] for e in plan["entries"]}
    assert categories == {"superseded_products", "work_cache", "stage_copies"}
    # The plan records native relative paths; compare their path identity on
    # Windows and POSIX without changing the deletion or byte-count contract.
    listed = {Path(e["path"]).as_posix() for e in plan["entries"]}
    assert "products/deadbeef0123" in listed and "products/0123456789ab" in listed
    assert all(f"work/light_{i:05d}.fit" in listed and f"work/r_light_{i:05d}.fit" in listed for i in (1, 2, 3))
    assert {"work/light_.seq", "work/light_.initial.seq", "work/r_light_.seq"} <= listed and len([p for p in listed if p.startswith("stage/")]) == 3
    assert "work/sky_master.fit" not in listed
    expected = (superseded / "master_linear.fit").stat().st_size + len(b"old jpeg") \
        + sum(p.stat().st_size for p in (directory / "work").glob("*light*")) \
        + sum(p.stat().st_size for p in (directory / "stage").iterdir())
    assert plan["bytes"] == expected
    assert plan["files"] == sum(e["files"] for e in plan["entries"]) == 2 + len([e for e in plan["entries"] if e["category"] != "superseded_products"])
    # products-only dry run leaves resume / recompose possible and lists no work files
    only_products = gc.plan(directory)
    assert {e["category"] for e in only_products["entries"]} == {"superseded_products"} and "不受影响" in only_products["consequence"]
    before_bytes = tree_bytes(directory, exclude=("job.json", "metadata.lock", "worker.lock"))
    kept_digest = tree_digest(kept_dir)
    master_digest = hashlib.sha256((directory / "work" / "sky_master.fit").read_bytes()).hexdigest()
    result = gc.apply(directory, work_cache=True)
    after_bytes = tree_bytes(directory, exclude=("job.json", "metadata.lock", "worker.lock"))
    record = core.load_job(directory)["gc"]
    assert result["removed_bytes"] == plan["bytes"] == record["bytes"] == before_bytes - after_bytes
    assert record["state"] == "done" and record["work_cache_removed"] and record["removed_categories"] == sorted(categories)
    assert [Path(p).as_posix() for p in record["products_removed"]] == ["products/0123456789ab", "products/deadbeef0123"] and record["history"] == []
    assert tree_digest(kept_dir) == kept_digest and hashlib.sha256((directory / "work" / "sky_master.fit").read_bytes()).hexdigest() == master_digest
    assert not superseded.exists() and not (products / "0123456789ab").exists()
    assert foreign.is_dir() and (products / "notes.txt").is_file()
    assert not list((directory / "work").glob("*light*")) and not list((directory / "stage").iterdir())
    assert (directory / "worker.log").is_file() and (directory / "job.json").is_file()
    for artifact in core.load_job(directory)["artifacts"]:
        assert Path(artifact["path"]).is_file(), artifact
    # afterwards resume and the bright-trail recompose are refused with the gc reason
    stored = json.loads((directory / "job.json").read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="gc --work-cache"):
        core.recompose_transients(str(directory), overrides={"x": "keep"})
    stored["state"] = "failed"
    (directory / "job.json").write_text(json.dumps(stored), encoding="utf-8")
    with pytest.raises(ValueError, match="gc --work-cache"):
        core.resume_job(str(directory))
    assert len(environment.launched) == 1
    # a second apply is idempotent and keeps the first record in its history
    again = gc.apply(directory, work_cache=False)
    assert again["removed_bytes"] == 0 and again["gc"]["state"] == "done"


def test_gc_work_cache_refuses_unfinished_jobs_and_running_workers(environment, monkeypatch):
    fake = ControlledEngine(monkeypatch)
    fake.drop_frame = True
    job = submit(environment, "gc-failed")
    core.run_job(job["job_dir"])
    directory = Path(job["job_dir"])
    assert core.load_job(directory)["state"] == "failed"
    with pytest.raises(ValueError, match="succeeded"):
        gc.apply(directory, work_cache=True)
    assert list((directory / "work").glob("light_*.fit")), "nothing removed on refusal"
    products_only = gc.apply(directory)             # superseded exports of a failed job may go; there are none here
    assert products_only["removed_bytes"] == 0 and core.load_job(directory).get("gc") is None
    plan = gc.plan(directory, work_cache=True)       # listing is always allowed
    assert plan["work_cache_allowed"] is False and plan["entries"]
    fake.drop_frame = False
    fresh = submit(environment, "gc-running")
    running = Path(fresh["job_dir"])
    with file_lock(running / "worker.lock"):
        stored = json.loads((running / "job.json").read_text(encoding="utf-8"))
        stored.update(state="running", stage="stack")
        (running / "job.json").write_text(json.dumps(stored), encoding="utf-8")
        with pytest.raises(ValueError, match="不能清理"):
            gc.apply(running)


def test_gc_refuses_a_retired_or_copied_job_directory(environment, monkeypatch):
    fake, job = finished_job(environment, monkeypatch)
    directory = Path(job["job_dir"])
    copied = directory.parent / "job-copy"
    shutil.copytree(directory, copied)
    with pytest.raises(ValueError, match="实际位置不一致"):
        gc.plan(copied)
    with pytest.raises(ValueError, match="实际位置不一致"):
        gc.apply(copied, work_cache=True)
    assert list((copied / "work").glob("light_*.fit"))


def test_gc_cli_defaults_to_dry_run(environment, monkeypatch, capsys):
    fake, job = finished_job(environment, monkeypatch)
    directory = Path(job["job_dir"])
    digest = tree_digest(directory)
    capsys.readouterr()                      # drop the worker's own console lines
    monkeypatch.setattr(sys, "argv", ["starstack", "gc", str(directory), "--work-cache"])
    assert cli.main() == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["dry_run"] is True and printed["bytes"] > 0 and tree_digest(directory) == digest
    monkeypatch.setattr(sys, "argv", ["starstack", "gc", str(directory), "--work-cache", "--apply"])
    assert cli.main() == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["dry_run"] is False and printed["removed_bytes"] == printed["bytes"]
    assert not list((directory / "work").glob("light_*.fit"))


# ---------------------------------------------------------------- H3.3: disk gate by frames

def test_disk_gate_estimates_from_frame_geometry_and_refuses_with_numbers(environment, monkeypatch):
    ControlledEngine(monkeypatch)
    job = submit(environment, "gate")
    estimate = job["disk_estimate"]
    assert estimate["basis"] == "frames_x_frame_volume" and estimate["frame_shape_hw"] == [18, 24] and estimate["frames"] == 3
    pixels = 18 * 24
    assert estimate["converted_frame_bytes"] == pixels * 12 and estimate["registered_frame_bytes"] == pixels * 12   # FITS input: float in, float out
    assert estimate["required_bytes"] == int((3 * pixels * 24 + 8 * pixels * 12 + estimate["source_bytes"]) * 1.1 + 512 * 1024 ** 2)
    assert estimate["free_bytes_at_admission"] == 10 ** 12
    monkeypatch.setattr(core.shutil, "disk_usage", lambda _: SimpleNamespace(free=10 ** 6))
    with pytest.raises(ValueError) as info:
        submit(environment, "gate-2")
    message = str(info.value)
    assert "估算需要" in message and "3 帧" in message and "24×18" in message and "starstack gc" in message
    assert not (environment.output / f"job-{hashlib.sha256(b'gate-2').hexdigest()[:24]}").exists()


def test_disk_estimate_reads_dng_geometry_and_falls_back_for_unreadable_raw(tmp_path):
    dng = cf.write_reference_dng(tmp_path / "frame00001.dng")
    import tifffile
    with tifffile.TiffFile(dng) as tf:
        page = max((p for p in tf.pages if int(p.photometric) == 32803), key=lambda p: p.imagewidth * p.imagelength)
        height, width = int(page.imagelength), int(page.imagewidth)
    sources = {"kind": "raw", "count": 12, "bytes": 12 * dng.stat().st_size, "files": [str(dng)]}
    estimate = disk_budget.estimate(sources, None)
    assert estimate["basis"] == "frames_x_frame_volume" and estimate["frame_shape_hw"] == [height, width]
    assert estimate["converted_frame_bytes"] == height * width * 6 and estimate["registered_frame_bytes"] == height * width * 12
    ranged = disk_budget.estimate(sources, None, stack_range_mode="positive_power2_v1", range_attempts=3)
    assert ranged["range_trial_bytes"] == 3 * 12 * height * width * 12 and ranged["required_bytes"] > estimate["required_bytes"]
    stub = tmp_path / "frame.nef"
    stub.write_bytes(b"not a raw file")
    legacy = disk_budget.estimate({"kind": "raw", "count": 4, "bytes": 4 * 14, "files": [str(stub)]}, None)
    assert legacy["basis"] == "legacy_source_bytes" and legacy["required_bytes"] == 4 * 14 * 20 + 512 * 1024 ** 2
    assert "旧的原片字节估算" in disk_budget.shortfall_message(legacy, 10)
    assert disk_budget.shortfall_message(estimate, 10).count("GB") >= 6
