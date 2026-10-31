"""P1c finishing wiring (docs/EFFICIENCY_P1C_PREREG_20261005.md §2.2, §3.5): environment checks at the workflow's
admission, in finish_milkyway and in the trail apply; nox separation as an exclusive section; the workflow's cancel
checks reaching the Milky Way render and the reduction; cancelled steps leaving no staging folder (bounded removal,
then a marker the next run sweeps); failed steps keeping their records only; pools running inside their named timer
operations.
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest

import p1c_finishing_data as fd
import p1c_site_harness as h
from starstack import (frame_pool, performance, photographic_job as job, profiling, star_reduction as sr,
                       starless_runtime as rt, workflow as wf, workflow_edit as edit)
from test_workflow import admit, setup, stub_model  # noqa: F401  (the workflow fixture)


# ================================================================== performance.operation

def test_operation_times_the_block_on_every_timer_active_in_this_thread():
    outer, inner, idle = profiling.StageTimer(), profiling.StageTimer(), profiling.StageTimer()
    with profiling.activate(outer), profiling.activate(inner):
        with performance.operation("demo_pool"):
            pass
        with pytest.raises(KeyError), performance.operation("demo_pool"):
            raise KeyError("errors pass through")
    for timer in (outer, inner):
        assert timer.finish()["operation_counts"]["demo_pool"] == 2
    assert "demo_pool" not in idle.finish().get("operation_counts", {})
    with performance.operation("no_timer"):                    # no active timer: a no-op
        pass


def test_without_an_active_timer_the_operation_goes_to_the_governor_scope_timer(monkeypatch):
    """The job worker hands its timer around explicitly (never profiling.activate): a pool site there (the
    composition's inward_weight) opens its operation on the process's governor scope timer; an activated timer wins."""
    from starstack import resources
    monkeypatch.setenv(resources.MODE_ENV, "shadow")
    job_timer = profiling.StageTimer()
    resources.shadow_start(resources.JOB_SCOPE, timer=job_timer,
                           job={"inputs": [{"path": "C:\\x.dng"}] * 4, "disk_estimate": {"frame_pixels": 45_700_000}})
    assert resources.scope_timer() is job_timer
    with performance.operation("demo_pool"):
        pass
    active = profiling.StageTimer()
    with profiling.activate(active), performance.operation("demo_pool"):
        pass
    assert job_timer.finish()["operation_counts"]["demo_pool"] == 1
    assert active.finish()["operation_counts"]["demo_pool"] == 1
    assert resources.scope_timer() is None                      # a finished timer is no longer live
    with performance.operation("after"):                        # no timer at all: a no-op
        pass


def test_a_finishing_pool_runs_inside_its_operation(tmp_path):
    stack = fd.layer_stack(11, (300, 24), paired=False)
    result = fd.make_separation(tmp_path / "sep", stack)
    timer = profiling.StageTimer()
    with profiling.activate(timer), fd.forced(h.Case(3, None, "reuse", None)):
        sr.reduce_stars(result, tmp_path / "reduction", background_strength=.5)
    counts = timer.finish()["operation_counts"]
    assert {"starless_scope", "star_reduction", "jpeg_rows"} <= set(counts)


# ================================================================== environment checks

@pytest.mark.parametrize("name, value", [(frame_pool.ENV_WORKERS, "lots"), (frame_pool.ENV_OVERLAP, "sometimes")])
def test_a_malformed_knob_fails_the_workflow_at_admission(setup, monkeypatch, name, value):
    record = admit(setup)
    monkeypatch.setenv(name, value)
    done = wf.run_workflow(record["workflow_dir"])
    assert done["state"] == "failed" and name in done["error"]
    assert setup["calls"]["submits"] == 0                        # before any stack work


def test_finish_milkyway_refuses_a_malformed_knob_before_any_work(tmp_path, monkeypatch):
    files = fd.milkyway_files(tmp_path / "in", shape=(64, 32))
    monkeypatch.setenv(frame_pool.ENV_WORKERS, "0")
    with pytest.raises(ValueError, match=frame_pool.ENV_WORKERS):
        job.finish_milkyway(files["source"], tmp_path / "out", files["sky"], files["protected"], files["milkyway"],
                            milkyway_evidence=files["evidence"])
    assert not (tmp_path / "out").exists()


def test_reduce_stars_refuses_a_malformed_knob_before_any_work(tmp_path, monkeypatch):
    result = fd.make_separation(tmp_path / "sep", fd.layer_stack(12, (64, 24)))
    monkeypatch.setenv(frame_pool.ENV_OVERLAP, "never")
    with pytest.raises(ValueError, match=frame_pool.ENV_OVERLAP):
        sr.reduce_stars(result, tmp_path / "reduction", background_strength=.5)
    assert not (tmp_path / "reduction").exists()


@pytest.mark.usefixtures("pre_contract_jobs")
def test_the_trail_apply_refuses_a_malformed_knob_before_its_staging_folder(setup, monkeypatch):
    from test_trail_default_and_extras import build
    record, state = build(setup, monkeypatch)
    assert wf.run_workflow(record["workflow_dir"])["state"] == "succeeded"
    monkeypatch.setenv(frame_pool.ENV_OVERLAP, "maybe")
    before = sorted(p.name for p in Path(record["workflow_dir"]).iterdir())
    with pytest.raises(ValueError, match=frame_pool.ENV_OVERLAP):
        edit.apply_review(record["workflow_dir"], state["session"])
    assert sorted(p.name for p in Path(record["workflow_dir"]).iterdir()) == before


# ================================================================== nox: an exclusive section

def test_the_starless_provider_runs_the_model_in_the_nox_section(monkeypatch):
    seen = {}

    class Model:
        identity = {"model_sha256": "0" * 64}

        def __call__(self, x):
            pool_stats = {}
            frame_pool.run_ordered(lambda item, unit: item, [1, 2, 3], workers=3, name="probe", stats=pool_stats)
            seen.setdefault("sections", frame_pool.exclusive_sections())
            seen.setdefault("pool", pool_stats)
            return np.ascontiguousarray(x, np.float32)

    estimate, _ = rt.provider(Model(), tile=64, stride=32)
    source = np.full((80, 90, 3), 1000, np.uint16)
    out = np.zeros_like(source)
    monkeypatch.delenv(frame_pool.ENV_WORKERS, raising=False)
    estimate(source, out, None)
    assert seen["sections"] == ("nox",)
    assert seen["pool"]["serial"] and seen["pool"]["reason"] == "exclusive" and seen["pool"]["exclusive"] == "nox"
    assert frame_pool.exclusive_sections() == ()


def test_the_workflow_separation_runs_the_model_in_the_nox_section(setup, monkeypatch):
    seen = []

    class Model:
        def __init__(self, path, **kw):
            self.identity = {"model_sha256": wf.core._sha(Path(path)), "backend": "test-half", **kw}

        def __call__(self, x):
            seen.append(frame_pool.exclusive_sections())
            return np.ascontiguousarray(x * .8, np.float32)

    monkeypatch.setattr(rt, "OnnxModel", Model)
    record = admit(setup, sky_mask_path=setup["mask"], star_model_path=setup["model"], stars_enabled=True,
                   backend_options={"tile": 64, "stride": 32})
    done = wf.run_workflow(record["workflow_dir"])
    assert done["state"] == "succeeded", done.get("error")
    assert seen and set(seen) == {("nox",)}
    assert frame_pool.exclusive_sections() == ()


# ================================================================== the workflow's cancel checks

def test_the_workflow_hands_its_cancel_check_to_the_milkyway_render(tmp_path, monkeypatch):
    directory = tmp_path / "wf"
    directory.mkdir()
    seen = {}

    def finish(source, root, sky, protect, **kwargs):
        seen.update(kwargs)
        return {"state": "succeeded", "artifacts": {}}

    monkeypatch.setattr(job, "finish_milkyway", finish)
    monkeypatch.setattr(wf, "_cached", lambda d, name: None)
    monkeypatch.setattr(wf, "_display_for_finishing", lambda d, stack: "source.tif")
    monkeypatch.setattr(wf, "_milkyway_inputs", lambda d, r, s, src: {"milkyway_mask": "m.tif"})
    monkeypatch.setattr(wf, "_masks", lambda d, stack: {"sky_mask": "s.tif", "protected_mask": "p.png"})
    monkeypatch.setattr(wf, "_update", lambda d, **kw: None)
    monkeypatch.setattr(wf, "_checkpoint", lambda *a, **kw: None)
    wf._milkyway_finish(directory, {"milkyway_enabled": True}, {})
    check = seen["check"]
    check()                                                       # no request: nothing
    (directory / "cancel.request").write_text("stop")
    with pytest.raises(wf.WorkflowCancelled):
        check()


def test_a_stop_during_the_reduction_resumes_to_the_uncancelled_result(setup, monkeypatch):
    """C3 in miniature: the stop request reaches the reduction's row blocks; the workflow is cancelled with no
    staging folder left, and its resumption publishes the reduction an uninterrupted workflow publishes."""
    stub_model(monkeypatch)
    options = dict(sky_mask_path=setup["mask"], star_model_path=setup["model"], stars_enabled=True,
                   background_strength=.6, backend_options={"tile": 64, "stride": 32})
    plain = wf.run_workflow(admit(setup, idempotency_key="plain", **options)["workflow_dir"])
    assert plain["state"] == "succeeded", plain.get("error")
    record = admit(setup, idempotency_key="stopped", **options)
    directory = Path(record["workflow_dir"])
    original, calls = sr._reduce_rows, []

    def stop_first(*args, **kwargs):
        calls.append(kwargs.get("check"))
        if len(calls) == 1:
            wf.cancel_workflow(record["workflow_dir"])
        return original(*args, **kwargs)

    monkeypatch.setattr(sr, "_reduce_rows", stop_first)
    stopped = wf.run_workflow(record["workflow_dir"])
    assert stopped["state"] == "cancelled" and calls[0] is not None
    assert not list((directory / "finishing" / "reduction").glob(".partial-*"))
    assert not list((directory / "finishing" / "reduction").iterdir())
    wf.resume_workflow(record["workflow_dir"])
    done = wf.run_workflow(record["workflow_dir"])
    assert done["state"] == "succeeded", done.get("error")
    assert not list(directory.rglob(".partial-*"))

    # two workflows' files differ in their descriptions and embedded ICC (path-bearing record digests, the profile's
    # creation time) even uninterrupted: the decoded pixels compare
    import tifffile
    from PIL import Image

    def reduced(result):
        artifacts = result["finishing"]["reduction"]["artifacts"]
        with Image.open(artifacts["star_reduced.jpg"]["path"]) as jpeg:
            codes = np.asarray(jpeg)
        return (codes, tifffile.imread(artifacts["star_reduced_srgb16.tif"]["path"]),
                tifffile.imread(artifacts["stars_reduced_signed_codes.tif"]["path"]))
    h.assert_same(reduced(plain), reduced(done))


# ================================================================== a cancelled step's staging folder

def test_a_folder_that_cannot_be_removed_at_once_is_marked_and_swept_next_time(tmp_path, monkeypatch):
    stage = tmp_path / ".partial-star-reduction-abc"
    stage.mkdir()
    (stage / "a.tif").write_bytes(b"x" * 10)
    attempts = []
    real = job.shutil.rmtree

    def locked(path, *args, **kwargs):
        attempts.append(path)
        if kwargs.get("ignore_errors"):
            return real(path, *args, **kwargs)
        raise PermissionError("in use")

    monkeypatch.setattr(job.shutil, "rmtree", locked)
    sleeps = []
    monkeypatch.setattr(job.time, "sleep", sleeps.append)
    assert job._discard_partial(stage, attempts=3) is False
    assert len(attempts) == 3 and json.loads((stage / job.DISCARD_MARKER).read_text())["state"] == "cancelled"
    assert len(sleeps) == 2                                       # no wait after the last attempt
    other = tmp_path / ".partial-star-reduction-failed"          # a failed attempt's folder stays for diagnosis
    other.mkdir()
    (other / "result.json").write_text("{}")
    job._sweep_discarded(tmp_path)
    assert not stage.exists() and other.exists()


def test_removing_a_held_folder_gives_up_well_inside_the_cancel_budget(tmp_path, monkeypatch):
    """The bounded retry ends within DISCARD_SECONDS (plus the attempts themselves), well inside the workflow's 3 s
    from a cancel request to 'cancelled' (prereg §6.4), and never waits after its last attempt."""
    import threading
    stage = tmp_path / ".partial-milkyway-held"
    stage.mkdir()
    attempts, sleeps, real_sleep, main = [], [], job.time.sleep, threading.get_ident()

    def held(path, *args, **kwargs):
        attempts.append(path)
        raise PermissionError("in use")

    def sleep(seconds):
        if threading.get_ident() == main:
            sleeps.append(seconds)
        real_sleep(seconds)

    monkeypatch.setattr(job.shutil, "rmtree", held)
    monkeypatch.setattr(job.time, "sleep", sleep)
    started = job.time.perf_counter()
    assert job._discard_partial(stage) is False
    elapsed = job.time.perf_counter() - started
    assert job.DISCARD_SECONDS <= 1.0 and sum(sleeps) <= job.DISCARD_SECONDS + .001
    assert elapsed < job.DISCARD_SECONDS + .5, elapsed
    assert len(sleeps) == len(attempts) - 1 and len(attempts) >= 2
    assert (stage / job.DISCARD_MARKER).is_file()


def test_removing_a_folder_that_is_already_gone_is_fine(tmp_path):
    assert job._discard_partial(tmp_path / "missing") is True


def test_a_failed_attempt_keeps_its_records_and_drops_its_pixels(tmp_path):
    """§3.5: rows a stopped pool wrote past the failing block live only in the staged pixel files; without them the
    failed folder is the same at any worker count."""
    stage = tmp_path / ".partial-reviewed-abc"
    (stage / "fixed-colour").mkdir(parents=True)
    for name in ("final_srgb16.tif", "final.jpg", "failure.json", "fixed-colour/natural_display_16bit.tif",
                 "fixed-colour/record.JSON", "fixed-colour/preview.png"):
        (stage / name).write_bytes(b"x")
    job._discard_failed(stage)
    kept = sorted(p.relative_to(stage).as_posix() for p in stage.rglob("*"))
    assert kept == ["failure.json", "fixed-colour", "fixed-colour/record.JSON"]
    job._discard_failed(tmp_path / "missing")                     # never raises
