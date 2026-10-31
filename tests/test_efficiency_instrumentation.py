"""P0 timing instrumentation (docs/EFFICIENCY_P0_PREREG_20260928.md, deliverable 1).

Timings are diagnostics: they must not change products, signatures, checkpoint
data digests, idempotency or delivered receipts, and records without them must
still load. Workflow runs reuse the synthetic fixtures of test_workflow and
test_trail_default_and_extras (RAW stacking, Siril and the model stubbed).
"""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading

import numpy as np
import pytest
import tifffile
from PIL import Image

from starstack import core, profiling, process_guard as pg
from starstack import workflow as wf, workflow_edit as edit
from test_workflow import setup, stub_model  # noqa: F401  (fixture)
from test_trail_default_and_extras import build
from test_core_selected_inputs import admitted  # noqa: F401  (fixture)

# Controlled engines here model Siril before the EFF-D4 stack-settings contract (efficiency P1c §4.2): jobs are
# admitted without it (tests/conftest.py pre_contract_jobs); contract jobs: tests/test_stack_settings.py.
pytestmark = pytest.mark.usefixtures("pre_contract_jobs")

WINDOWS = os.name == "nt"
V1_KEYS = {"version", "scope", "unit", "total_seconds", "phases_seconds", "operations_seconds",
           "operation_counts", "semantics"}


class FakeSampler:
    """Deterministic process counters: each reading advances by a fixed step."""

    def __init__(self):
        self.n = 0

    def __call__(self):
        self.n += 1
        n = self.n
        return {"cpu_user_seconds": .5 * n, "cpu_kernel_seconds": .25 * n,
                "io_read_operations": n, "io_write_operations": 2 * n, "io_other_operations": 0,
                "io_read_bytes": 100 * n, "io_write_bytes": 10 * n, "io_other_bytes": 0,
                "hashed_bytes": 1000 * n, "hashed_files": n,
                "working_set_bytes": 5000 + n, "private_bytes": 6000 + n,
                "peak_working_set_bytes": 7000 + 10 * n, "peak_private_bytes": 8000}


def _json_roundtrip(value):
    return json.loads(json.dumps(value, allow_nan=False))


def _walk(value, path=()):
    if isinstance(value, dict):
        for key, item in value.items():
            yield path + (key,), item
            yield from _walk(item, path + (key,))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk(item, path + (index,))


# --------------------------------------------------------------------------- #
# StageTimer v2                                                                #
# --------------------------------------------------------------------------- #

def test_v2_keeps_v1_keys_and_adds_per_phase_resources():
    now = [0.0]
    timer = profiling.StageTimer(clock=lambda: now[0], sampler=FakeSampler())   # reading 1
    now[0] = 2
    timer.phase("convert")                                                     # reading 2
    with pytest.raises(RuntimeError):
        with timer.operation("script"):                                        # readings 3, 4
            now[0] = 5
            raise RuntimeError("test")
    now[0] = 7
    report = timer.finish()                                                    # reading 5
    assert V1_KEYS <= set(report)
    assert report["version"] == profiling.VERSION and report["compatible_with"] == profiling.LEGACY_VERSION
    assert report["phases_seconds"] == {"setup": 2, "convert": 5}
    assert report["operations_seconds"] == {"script": 3} and report["operation_counts"] == {"script": 1}
    assert report["phase_entries"] == {"setup": 1, "convert": 1}
    phases = report["resources"]["phases"]
    assert phases["setup"]["cpu_seconds"] == .75 and phases["setup"]["hashed_bytes"] == 1000
    assert phases["convert"]["cpu_user_seconds"] == 1.5 and phases["convert"]["io_read_bytes"] == 300
    assert phases["convert"]["hashed_files"] == 3 and phases["convert"]["working_set_bytes"] == 5005
    assert phases["convert"]["peak_working_set_bytes"] == 7050
    assert phases["convert"]["peak_working_set_rise_bytes"] == 30 and phases["convert"]["peak_private_rise_bytes"] == 0
    assert report["resources"]["operations"]["script"]["io_write_bytes"] == 10
    total = report["resources"]["total"]
    assert total["cpu_seconds"] == 3.0 and total["hashed_bytes"] == 4000 and total["peak_working_set_bytes"] == 7050
    assert "children" not in phases["convert"]
    # The receipt is fixed at the first finish and returned as an independent copy.
    now[0] = 10
    report["phases_seconds"]["setup"] = -1
    assert timer.finish()["phases_seconds"] == {"setup": 2, "convert": 5}
    timer.phase("late")
    assert timer.finish()["phases_seconds"] == {"setup": 2, "convert": 5}
    assert _json_roundtrip(timer.finish()) == timer.finish()


def test_unavailable_or_non_finite_readings_never_break_the_receipt():
    def broken():
        raise OSError("no counters")
    report = profiling.StageTimer(sampler=broken).finish()
    assert report["resources"]["total"]["cpu_seconds"] is None
    odd = profiling.StageTimer(sampler=lambda: {"cpu_user_seconds": float("nan"), "cpu_kernel_seconds": 0.0})
    _json_roundtrip(odd.finish())
    real = profiling.StageTimer()
    real.phase("work")
    sum(range(20000))
    result = real.finish()
    if WINDOWS:
        work = result["resources"]["phases"]["work"]
        assert work["peak_working_set_bytes"] > 0 and work["private_bytes"] > 0
        assert work["io_read_bytes"] is not None and work["cpu_seconds"] is not None


def test_nested_stages_are_exclusive_and_names_are_relative_to_the_innermost_stage():
    now = [0.0]
    outer = profiling.StageTimer(clock=lambda: now[0], sampler=FakeSampler(), initial="unstaged")

    @profiling.staged("masks")
    def masks():
        now[0] += 1
        return "m"

    profiling.switch(".ignored")            # no active timer: no-op
    with profiling.stage("ignored"):
        pass
    with profiling.activate(outer):
        with profiling.stage("milkyway_region"):
            now[0] += 2
            assert masks() == "m"
            now[0] += 3
        with profiling.stage("publish"):
            profiling.switch(".stage")
            now[0] += 4
            profiling.switch(".verify")
            now[0] += 5
        with profiling.stage("trail_addback"):
            profiling.switch(".entry_verify")
            inner = profiling.StageTimer(clock=lambda: now[0], sampler=FakeSampler(), initial="entry_verify")
            with profiling.activate(inner):
                now[0] += 6
                with profiling.stage(".recompose"):
                    profiling.switch(".compose")
                    now[0] += 7
                profiling.switch(".commit")
                now[0] += 8
            inner_report = inner.finish()
        now[0] += 9
    report = outer.finish()
    assert report["phases_seconds"] == {"unstaged": 9, "milkyway_region": 5, "masks": 1, "publish": 0,
                                        "publish.stage": 4, "publish.verify": 5, "trail_addback": 0,
                                        "trail_addback.entry_verify": 6, "trail_addback.recompose": 0,
                                        "trail_addback.recompose.compose": 7, "trail_addback.commit": 8}
    assert sum(report["phases_seconds"].values()) == report["total_seconds"] == 45
    assert report["phase_entries"]["milkyway_region"] == 1
    assert inner_report["phases_seconds"] == {"entry_verify": 6, "recompose": 0, "recompose.compose": 7, "commit": 8}
    assert set(report["resources"]["phases"]) == set(report["phases_seconds"])
    assert profiling.active() == ()


def test_stage_exits_on_exceptions_and_decorator_keeps_the_signature():
    now = [0.0]
    timer = profiling.StageTimer(clock=lambda: now[0], sampler=FakeSampler(), initial="unstaged")

    @profiling.staged("separation")
    def separate(directory, request, stack, source=None):
        """doc"""
        now[0] += 1
        raise ValueError("boom")

    assert separate.__name__ == "separate" and separate.__doc__ == "doc"
    with profiling.activate(timer):
        with pytest.raises(ValueError):
            separate(1, 2, 3)
        now[0] += 2
    assert timer.finish()["phases_seconds"] == {"unstaged": 2, "separation": 1}


def test_notes_attach_to_running_timers_only():
    running = profiling.StageTimer(sampler=FakeSampler())
    done = profiling.StageTimer(sampler=FakeSampler())
    before = done.finish()
    value = {"threads": 4, "memory_ratio": .5, "bad": float("inf")}
    profiling.note("siril_pin", value)
    value["threads"] = 99
    report = running.finish()
    assert report["notes"] == {"siril_pin": {"threads": 4, "memory_ratio": .5, "bad": None}}
    assert done.finish() == before and "notes" not in before
    assert "notes" not in profiling.StageTimer(sampler=FakeSampler()).finish()


def test_hash_counter_counts_bytes_without_changing_digests(tmp_path):
    files = []
    for i in range(6):
        path = tmp_path / f"f{i}.bin"
        path.write_bytes(os.urandom(1000 + 37 * i))
        files.append(path)
    before = profiling.hashed_totals()
    digests = {}

    def worker(path):
        digests[path] = core._sha(path)

    threads = [threading.Thread(target=worker, args=(p,)) for p in files]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    after = profiling.hashed_totals()
    assert all(digests[p] == hashlib.sha256(p.read_bytes()).hexdigest() for p in files)
    assert after["hashed_bytes"] - before["hashed_bytes"] >= sum(p.stat().st_size for p in files)
    assert after["hashed_files"] - before["hashed_files"] >= len(files)
    timer = profiling.StageTimer(initial="verify")
    entry = wf._file(files[0])
    wf._verify_files([entry])
    assert timer.finish()["resources"]["phases"]["verify"]["hashed_bytes"] >= 2 * files[0].stat().st_size


def test_hashed_sources_name_what_the_lower_bound_includes(tmp_path):
    path = tmp_path / "blob.bin"
    path.write_bytes(os.urandom(300_000))
    timer = profiling.StageTimer(initial="verify")
    assert profiling.sha256_file(path, source="test.helper") == hashlib.sha256(path.read_bytes()).hexdigest()
    assert core._sha(path) == hashlib.sha256(path.read_bytes()).hexdigest()
    profiling.note_hashed(10)
    profiling.note_hashed("not a number", source="test.bad")
    report = timer.finish()
    resources = report["resources"]
    sources = resources["hashed_sources"]
    assert sources["test.helper"] == {"hashed_bytes": 300_000, "hashed_files": 1}
    assert sources["core._sha"]["hashed_bytes"] >= 300_000
    assert sources[profiling.UNLABELLED_SOURCE]["hashed_bytes"] >= 10 and "test.bad" not in sources
    assert resources["phases"]["verify"]["hashed_bytes"] >= 600_010
    assert resources["total"]["hashed_bytes"] == sum(item["hashed_bytes"] for item in sources.values())
    assert "lower bound" in resources["semantics"]
    _json_roundtrip(report)
    # Fake or unavailable counters report no per-helper breakdown rather than a wrong one.
    assert "hashed_sources" not in profiling.StageTimer(sampler=FakeSampler()).finish()["resources"]


def test_host_profile_reports_processors_and_a_sha_rate_once():
    first, second = profiling.host_profile(), profiling.host_profile()
    assert first == second and first["version"] == profiling.HOST_VERSION
    assert first["logical_processors"] == os.cpu_count()
    assert first["sha256_rate_gbps"] > 0 and first["sha256_probe_bytes"] == 64 << 20
    assert first["sha_ext_guess"] == (first["sha256_rate_gbps"] >= profiling.SHA_EXT_THRESHOLD_GBPS)
    if WINDOWS:
        assert sum(first["processor_groups"]) == os.cpu_count() and first["process_affinity_processors"] >= 1
    assert profiling.StageTimer(host=True).finish()["host"] == first
    assert "host" not in profiling.StageTimer().finish()


# --------------------------------------------------------------------------- #
# Child (Job Object) accounting                                                #
# --------------------------------------------------------------------------- #

def test_child_accounting_is_charged_to_the_current_phase_and_open_operations():
    now = [0.0]
    timer = profiling.StageTimer(clock=lambda: now[0], sampler=FakeSampler())
    timer.phase("stack")
    with timer.operation("script:02_stack"):
        profiling.note_child_accounting({"cpu_user_seconds": 2.0, "cpu_kernel_seconds": .5, "io_read_bytes": 10,
                                         "io_write_bytes": 20, "peak_job_memory_bytes": 300,
                                         "descendant_processes": 1, "wall_seconds": 3.0})
    profiling.note_child_accounting({"cpu_user_seconds": 1.0, "cpu_kernel_seconds": 0.0, "peak_job_memory_bytes": 900})
    profiling.note_child_accounting("not a dict")
    report = timer.finish()
    child = report["resources"]["phases"]["stack"]["children"]
    assert child["processes"] == 2 and child["cpu_seconds"] == 3.5 and child["peak_job_memory_bytes"] == 900
    op = report["resources"]["operations"]["script:02_stack"]["children"]
    assert op["processes"] == 1 and op["io_write_bytes"] == 20 and op["peak_job_memory_bytes"] == 300
    assert report["resources"]["total"]["children"]["processes"] == 2
    profiling.note_child_accounting({"cpu_user_seconds": 9.0})   # finished timers ignore late children
    assert timer.finish() == report


def test_job_accounting_is_best_effort_for_doubles_and_failing_queries():
    class NoQuery:
        pass

    class Raising:
        def QueryInformationJobObject(self, *args):
            raise OSError("denied")

    class Refusing:
        def QueryInformationJobObject(self, *args):
            return 0

    assert pg.job_accounting(NoQuery(), 5) is None
    assert pg.job_accounting(Raising(), 5) is None
    assert pg.job_accounting(Refusing(), 5) is None
    assert pg.job_accounting(Refusing(), None) is None
    assert pg.EXPECTED_STRUCT_SIZES[8]["JOBOBJECT_BASIC_AND_IO_ACCOUNTING_INFORMATION"] == 96


@pytest.mark.skipif(not WINDOWS, reason="Windows Job Object accounting")
def test_job_accounting_of_a_real_short_child(tmp_path):
    target = tmp_path / "written.bin"
    code = ("import hashlib, sys\n"
            "data = bytes(8 << 20)\n"
            "for _ in range(40): hashlib.sha256(data).digest()\n"
            "open(sys.argv[1], 'wb').write(data)\n")
    seen = []
    pg.add_accounting_listener(seen.append)
    timer = profiling.StageTimer()
    timer.phase("child")
    with timer.operation("script:probe"):
        with pg.guarded_process([sys.executable, "-c", code, str(target)], stdin=subprocess.DEVNULL,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) as process:
            assert process.wait(timeout=60) == 0
    accounting = process.accounting
    assert accounting["source"] == "windows_job_object" and accounting["returncode"] == 0
    assert accounting["cpu_user_seconds"] + accounting["cpu_kernel_seconds"] > 0.05
    assert accounting["io_write_bytes"] >= 8 << 20
    assert accounting["peak_job_memory_bytes"] >= 8 << 20 and accounting["peak_process_memory_bytes"] >= 8 << 20
    assert accounting["descendant_processes"] >= 1 and accounting["active_processes"] == 0
    assert 0 < accounting["wall_seconds"] < 60 and accounting["executable"] == Path(sys.executable).name
    assert any(item == accounting for item in seen)
    _json_roundtrip(accounting)
    report = timer.finish()
    op = report["resources"]["operations"]["script:probe"]["children"]
    assert op["processes"] == 1 and op["io_write_bytes"] >= 8 << 20
    assert report["resources"]["phases"]["child"]["children"]["cpu_seconds"] > 0.05


def test_core_job_receipt_is_v2_with_host_and_keeps_v1_semantics(admitted):
    source, paths, out = admitted
    job = core.submit_job(None, str(out), input_files=paths[:3], transient_mode="off")
    core.run_job(job["job_dir"])
    result = core.load_job(job["job_dir"])
    assert result["state"] == "succeeded", result.get("error")
    timing = result["diagnostics"]["timings"]
    assert V1_KEYS <= set(timing) and timing["version"] == profiling.VERSION
    assert sum(timing["phases_seconds"].values()) == pytest.approx(timing["total_seconds"])
    assert set(timing["resources"]["phases"]) == set(timing["phases_seconds"])
    assert timing["resources"]["phases"]["input_stage"]["hashed_bytes"] >= sum(Path(p).stat().st_size for p in paths[:3])
    assert "script:01_convert" in timing["resources"]["operations"]
    assert timing["host"]["sha256_rate_gbps"] > 0
    colour = result["diagnostics"].get("color_export") or {}
    if colour.get("timings"):
        assert "host" not in colour["timings"]
    assert "siril_pin" not in timing.get("notes", {})


@pytest.mark.parametrize("value", ["cpu=4,mem=8", "cpu=zero"])
def test_a_siril_pin_is_noted_in_the_stack_job_timings_only(admitted, monkeypatch, value):
    from starstack import siril
    source, paths, out = admitted
    plain = core.submit_job(None, str(out / "plain"), input_files=paths[:3], transient_mode="off")
    monkeypatch.setenv(siril.SIRIL_PIN_ENV, value)
    job = core.submit_job(None, str(out / "pinned"), input_files=paths[:3], transient_mode="off")
    assert job["signature"] == plain["signature"]            # a pin never enters the job identity
    core.run_job(job["job_dir"])                             # run_script is stubbed by the fixture
    result = core.load_job(job["job_dir"])
    assert result["state"] == "succeeded", result.get("error")
    note = result["diagnostics"]["timings"]["notes"]["siril_pin"]
    if value == "cpu=4,mem=8":
        assert note == siril.pin_diagnostics(value) and note["cpu"] == 4 and note["mem_gb"] == 8
        assert "setcpu 4" in note["script_commands"]
    else:  # malformed: noted with its error (the real Siril run refuses it)
        assert note["source"] == siril.SIRIL_PIN_ENV and note["value"] == value and "ValueError" in note["error"]
    assert result["signature"] == plain["signature"]
    outside = json.dumps({k: v for k, v in result.items() if k != "diagnostics"})
    assert '"siril_pin"' not in outside and siril.SIRIL_PIN_ENV not in outside


# --------------------------------------------------------------------------- #
# Workflow worker                                                              #
# --------------------------------------------------------------------------- #

WORKFLOW_PHASES = {"unstaged", "admission", "stack_checkpoint", "stack_wait", "masks", "separation", "advice",
                   "reduction", "layers", "publish.stage", "publish.verify", "publish.commit",
                   "trail_addback.review_open", "trail_addback.entry_verify", "trail_addback.recompose",
                   "trail_addback.reuse_check", "trail_addback.render_fixed", "trail_addback.write_pair",
                   "trail_addback.stage_delivery", "trail_addback.verify", "trail_addback.commit"}
APPLY_PHASES = {"entry_verify", "recompose", "reuse_check", "render_fixed", "write_pair", "stage_delivery",
                "verify", "commit"}


def _full_run(setup, monkeypatch, **kwargs):
    record, state = build(setup, monkeypatch, stars=True, save_starless=True, **kwargs)
    done = wf.run_workflow(record["workflow_dir"])
    assert done["state"] == "succeeded", done.get("error")
    return record, done, state


def _assert_no_timing_in_frozen_data(done):
    """Workflow timings stay out of the request, checkpoint data and receipts."""
    assert done["signature"] == wf._digest(done["request"])
    assert not any(key in ("timings", "timing_attempts") for key, _ in
                   ((p[-1], v) for p, v in _walk(done["request"])))
    for name, checkpoint in done["checkpoints"].items():
        assert checkpoint["data_sha256"] == wf._digest(checkpoint["data"])
        data = checkpoint["data"]
        if name == "stack":
            # Only the child job's own receipt (diagnostics since v1) lives in its frozen snapshot.
            data = {k: v for k, v in data.items() if k != "job"}
            data["colour"] = {k: v for k, v in data["colour"].items() if k != "timings"}
        assert not any(path[-1] in ("timings", "timing_attempts") for path, _ in _walk(data)), name


def test_workflow_records_well_formed_stage_timings(setup, monkeypatch):
    record, done, state = _full_run(setup, monkeypatch)
    timings = done["timings"]
    assert V1_KEYS <= set(timings) and timings["state"] == "succeeded" and done["timing_attempts"] == []
    assert timings["recorded_at"] and timings["host"]["sha256_rate_gbps"] > 0
    phases = timings["phases_seconds"]
    assert WORKFLOW_PHASES <= set(phases), sorted(WORKFLOW_PHASES - set(phases))
    assert sum(phases.values()) == pytest.approx(timings["total_seconds"])
    assert all(value >= 0 for value in phases.values())
    resources = timings["resources"]["phases"]
    assert set(resources) == set(phases)
    # The admission check always reads the request files for real (P1 hashing).
    assert resources["admission"]["hashed_bytes"] >= sum(item["bytes"] for item in done["request"]["files"])
    # Later checks run too, but may be answered by the run ledger (hash_saved_*) instead of a read.
    assert all(resources[phase]["hashed_bytes"] + resources[phase]["hash_saved_bytes"] > 0
               for phase in ("publish.verify", "trail_addback.verify"))
    assert timings["resources"]["hashed_sources"]["workflow._verify_files"]["hashed_bytes"] > 0
    assert "siril_pin" not in timings.get("notes", {})
    if WINDOWS:
        assert resources["publish.stage"]["io_write_bytes"] > 0
        assert timings["resources"]["total"]["peak_working_set_bytes"] > 0
    # The published trails version carries its own apply timings in the workflow index.
    entry = done["review_deliveries"][-1]
    assert entry["started_at"] <= entry["finished_at"]
    assert entry["timings"]["scope"] == "trail_apply" and APPLY_PHASES <= set(entry["timings"]["phases_seconds"])
    # ...but never in a delivered or cached receipt.
    receipt_text = Path(entry["receipt"]).read_text(encoding="utf-8")
    assert "timings" not in receipt_text and "started_at" not in receipt_text
    for folder in (done["base_delivery_dir"], done["delivery_dir"]):
        for path in Path(folder).rglob("*.json"):
            assert "phases_seconds" not in path.read_text(encoding="utf-8"), path
    _assert_no_timing_in_frozen_data(done)
    wf._verify(done)
    assert wf.run_workflow(record["workflow_dir"]) == wf.workflow_status(record["workflow_dir"])


def test_manual_apply_adds_timestamps_and_timings_to_its_delivery_entry(setup, monkeypatch):
    record, done, state = _full_run(setup, monkeypatch)
    session = Path(done["stack_job_dir"]) / "review" / "draft-manual.json"
    session.write_text("{}")
    result = edit.apply_review(done["workflow_dir"], session, expected_draft_revision="manual")
    assert result["published"] is True
    entries = result["workflow"]["review_deliveries"]
    assert len(entries) == 2 and entries[0] == done["review_deliveries"][0]
    manual = entries[-1]
    assert manual["timings"]["scope"] == "trail_apply"
    assert APPLY_PHASES <= set(manual["timings"]["phases_seconds"])
    assert not any(key.startswith("trail_addback") for key in manual["timings"]["phases_seconds"])
    # The worker's own attempt receipt is not replaced by a later manual apply.
    assert result["workflow"]["timings"] == done["timings"]
    _json_roundtrip(result["workflow"])
    wf._verify(result["workflow"])


def test_instrumentation_does_not_change_signature_idempotency_or_products(setup, monkeypatch):
    def run(disabled):
        if setup["output"].exists():
            shutil.rmtree(setup["output"])
        with monkeypatch.context() as patch:
            if disabled:
                patch.setattr(profiling, "active", lambda: ())
                patch.setattr(wf, "_record_timings", lambda directory, timer, prior, result: result)
                patch.delenv("AESTER_SIRIL_PIN", raising=False)
            else:  # the pinned, instrumented run must still match the plain one
                patch.setenv("AESTER_SIRIL_PIN", "cpu=4,mem=8")
            record, done, _ = _full_run(setup, monkeypatch)
        again = wf.submit_workflow(setup["inputs"], setup["output"], sky_mask_path=setup["mask"],
                                   color_adjustments={"warmth": 12, "tint": 5, "exposure_ev": .3},
                                   stars_enabled=True, star_model_path=setup["model"], background_strength=.7,
                                   backend_options={"tile": 64, "stride": 32}, save_starless=True,
                                   trails_enabled=True, trail_default="meteor", idempotency_key="fixed")
        first = wf.submit_workflow(setup["inputs"], setup["output"], sky_mask_path=setup["mask"],
                                   color_adjustments={"warmth": 12, "tint": 5, "exposure_ev": .3},
                                   stars_enabled=True, star_model_path=setup["model"], background_strength=.7,
                                   backend_options={"tile": 64, "stride": 32}, save_starless=True,
                                   trails_enabled=True, trail_default="meteor", idempotency_key="fixed")
        assert again["workflow_dir"] == first["workflow_dir"] and again["signature"] == done["signature"]
        # File bytes embed this run's random workflow/stage ids in paths and
        # source hashes (with or without timings), so compare decoded pixels
        # and sizes here; byte-level run equivalence is the verifier's job.
        pixels, sizes = {}, {}
        for folder in (done["base_delivery_dir"], done["delivery_dir"]):
            for item in sorted(Path(folder).iterdir()):
                key = (folder == done["delivery_dir"], item.name)
                if item.name == "review_delivery.json":
                    # A receipt of random stage names and paths; timings never enter it.
                    assert "timings" not in item.read_text(encoding="utf-8")
                elif item.is_file():
                    sizes[key] = item.stat().st_size
                if item.suffix == ".tif":
                    pixels[key] = tifffile.imread(item)
                elif item.suffix == ".jpg":
                    pixels[key] = np.asarray(Image.open(item))
        return done, pixels, sizes

    instrumented, pixels_a, sizes_a = run(False)
    plain, pixels_b, sizes_b = run(True)
    assert "timings" in instrumented and "timings" not in plain
    assert "timings" in instrumented["review_deliveries"][-1]
    assert instrumented["timings"]["notes"]["siril_pin"]["cpu"] == 4
    assert instrumented["timings"]["notes"]["siril_pin"]["mem_gb"] == 8
    assert instrumented["signature"] == plain["signature"]
    assert set(instrumented["checkpoints"]) == set(plain["checkpoints"])
    assert len(pixels_a) >= 6 and pixels_a.keys() == pixels_b.keys()
    assert all(np.array_equal(pixels_a[k], pixels_b[k]) for k in pixels_a)
    assert sizes_a == sizes_b
    for done in (instrumented, plain):
        _assert_no_timing_in_frozen_data(done)


def test_attempts_are_kept_and_records_without_timings_still_load(setup, monkeypatch):
    record, state = build(setup, monkeypatch)
    publish = wf._publish
    failing = {"left": 1}

    def flaky(*args):
        if failing["left"]:
            failing["left"] -= 1
            raise RuntimeError("synthetic publish failure")
        return publish(*args)

    monkeypatch.setattr(wf, "_publish", flaky)
    failed = wf.run_workflow(record["workflow_dir"])
    assert failed["state"] == "failed" and failed["timings"]["state"] == "failed"
    assert "publish" not in failed["timings"]["phases_seconds"] and failed["timing_attempts"] == []
    wf.resume_workflow(record["workflow_dir"])
    done = wf.run_workflow(record["workflow_dir"])
    assert done["state"] == "succeeded", done.get("error")
    assert done["timing_attempts"] == [failed["timings"]] and done["timings"] != failed["timings"]
    assert done["timings"]["phases_seconds"]["stack_checkpoint"] >= 0   # reused checkpoint re-verified
    assert "stack_wait" not in done["timings"]["phases_seconds"]

    # Strip every diagnostic this package adds: an older record must behave as before.
    directory = Path(record["workflow_dir"])
    legacy = wf._read(directory)
    for key in ("timings", "timing_attempts"):
        legacy.pop(key)
    for entry in legacy.get("review_deliveries", []):
        for key in ("timings", "started_at", "finished_at"):
            entry.pop(key, None)
    wf.atomic_json(directory / "workflow.json", legacy)
    status = wf.workflow_status(directory)
    assert status["state"] == "succeeded" and "timings" not in status
    wf._verify(status)
    assert wf.run_workflow(directory) == status        # succeeded: nothing re-run, nothing written

    # A failed old record without timings resumes into a first receipt with no prior attempts.
    old = wf._read(directory)
    old.update(state="failed", error="old failure")
    wf.atomic_json(directory / "workflow.json", old)
    assert wf.resume_workflow(directory)["state"] == "queued"
    rerun = wf.run_workflow(directory)
    assert rerun["state"] == "succeeded", rerun.get("error")
    assert rerun["timing_attempts"] == [] and rerun["timings"]["state"] == "succeeded"


def test_timing_write_failure_never_fails_the_worker(setup, monkeypatch):
    record, _ = build(setup, monkeypatch, trails=False)

    class Broken(profiling.StageTimer):
        def finish(self):
            raise RuntimeError("diagnostics unavailable")

    monkeypatch.setattr(wf.profiling, "StageTimer", Broken)
    done = wf.run_workflow(record["workflow_dir"])
    assert done["state"] == "succeeded" and "timings" not in done
    assert "diagnostics unavailable" in (Path(record["workflow_dir"]) / "worker.log").read_text(encoding="utf-8")
