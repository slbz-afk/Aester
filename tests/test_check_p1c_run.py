"""scripts/check_p1c_run.py (the P1c run check, prereg 6 "运行检查") on synthetic bench result folders."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
JOB = "{WF}/stack/job-0123456789abcdef01234567"
PIN = {"cpu": 16, "mem": 16}
P1C_PINS = {"parallel": None, "overlap": None, "trails": "on", "memory_view_gib": None, "commit_view_gib": None,
            "python_openblas": None}


def _module():
    spec = importlib.util.spec_from_file_location("check_p1c_run", ROOT / "scripts" / "check_p1c_run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


check_p1c = _module()
bench = check_p1c.bench_module()


def _environment(parallel=None):
    env = {"AESTER_CACHE_ROOT": "C:\\b\\runs\\L20\\c", "AESTER_COMPONENT_ROOT": "C:\\b\\runs\\L20\\k",
           "AESTER_SIRIL_PIN": None, "OPENSSL_ia32cap": None,
           **{name: None for name in bench.PARALLEL_VARIABLES},
           bench.MEMORY_VIEW_ENV: None, bench.COMMIT_VIEW_ENV: None, "OPENBLAS_CORETYPE": None,
           "OPENBLAS_NUM_THREADS": None, bench.PYTHON_ONLY_ENV: None, "AESTER_L_SIRIL_LAYOUT": None,
           "AESTER_GOVERNOR": "shadow-full", "AESTER_GOVERNOR_FACTS": None, "MSYSTEM": "MINGW64",
           "XDG_DATA_HOME": "C:\\b\\runs\\L20\\x", "DBUS_SESSION_BUS_ADDRESS": "disabled:",
           "PYTHONPATH": "C:\\repo\\src;C:\\b\\runs\\L20\\h", bench.HOOK_LOG_ENV: "C:\\b\\runs\\L20\\h\\imports",
           bench.HOOK_SHIM_ENV: None, bench.HOOK_EVENTS_ENV: "C:\\b\\runs\\L20\\h\\events",
           bench.THREAD_LOG_ENV: "C:\\b\\runs\\L20\\h\\threads", bench.BLAS_LOG_ENV: "1"}
    if parallel:
        env.update(bench.PARALLEL_TABLE[parallel])
    return env


def _plan(mode="L", parallel=None, **changes):
    plan = {"run_id": "R", "mode": mode, "frames": 20, "slot": f"{mode}20", "profile": None, "simulation": None,
            "pins": {"siril": None, "ort_intra_op_threads": None, "affinity_logical": None, "mask_sha": False,
                     "render_workers": None, "frame_workers": None, "governor": "shadow-full", "governor_facts": None,
                     **P1C_PINS, "parallel": parallel},
            "parallel": {"mode": parallel, "effective": parallel or "default",
                         "environment": bench.PARALLEL_TABLE[parallel] if parallel else
                         {name: None for name in bench.PARALLEL_VARIABLES}, "knobs": [], "tier": []},
            "submit": {"siril_path": "C:\\repo\\tools\\siril-cli.exe", "star_model_path": "C:\\m\\nox.onnx",
                       "stack_engine_path": None,
                       "stack_range_mode": {"L": "legacy", "F": "finite_validity_v1", "R": "positive_power2_v1"}[mode],
                       "backend_options": {"tile": 512, "intra_op_threads": None}, "trails_enabled": True,
                       "trail_default": "meteor", "stars_enabled": True, "regions_path": "C:\\b\\assets\\r.json",
                       "delivery_root": "C:\\b\\runs\\L20\\d"},
            "environment": _environment(parallel), "siril_launch": {"MSYSTEM": "MINGW64"}, "hook": True}
    plan.update(changes)
    return plan


def _entry(workers=4, units=87, **changes):
    entry = {"requested": workers, "workers": workers, "units": units, "serial": workers == 1,
             "max_running": workers, "max_held": workers, "inflight": 2 * workers, "wall_seconds": 1.2,
             "unit_bytes": 60_000_000, "cpu_user_s": 4.0, "cpu_kernel_s": .4, "max_unit_s": .1}
    if workers == 1:
        entry["reason"] = "workers"
    else:
        entry["fp_env_applied"] = False
    entry.update(changes)
    return entry


def _o1_note(eligible=True, reason=None, **changes):
    """An overlap.o1 note as core writes it (core._run_locked's o1_begin; tests/test_p1c_site_overlap.py reads real
    ones): ineligible notes carry the schema, eligible and the reason; eligible ones the plan and the outcome."""
    note = {"schema": check_p1c.OVERLAP_SCHEMA, "eligible": eligible, "reason": reason}
    if eligible:
        note.update(mode="lr", pixels=45_705_600, frames=20, n_o1=8, n_serial=8, override=None, cov_plan=8,
                    crop_cap=8, native_workers=0, scan_unit_bytes=1_800_000_000, a_reserve_bytes=2_745_000_000,
                    b_inproc_bytes=301_989_888, headroom_bytes={"physical": 1 << 36, "commit": 1 << 37},
                    predicted_s={"scan_cpu": 72.0, "a": 34.0, "scan_serial": 9.0, "scan_o1": 9.0, "extra": 0.0,
                                 "margin": 5.0},
                    pieces={"a": 548_000_000, "coverage": 2_197_000_000, "native_mask": 0, "scan": 14_400_000_000},
                    b_done_at_join=True, join_wait_s=0.0, outcome="joined",
                    window_s={"b_start_s": 120.1, "b_end_s": 140.2, "join_start_s": 150.3, "join_end_s": 150.3})
    note.update(changes)
    return note


def _job_notes(mode="L", workers=4, **extra):
    notes = {"hash_ledger": {"version": "aester-hashing-3", "miss_bytes": {"no_entry": 0}, "largest_misses": []},
             "governor": {"mode": "shadow-full", "compact": True,
                          "pools": {"py.median": json.dumps({"actual": workers, "recommended": 4, "bound_by": "cap"})}},
             "frame_pool.median": [_entry(workers)], "frame_pool.sky_zero": [_entry(workers, units=20)],
             "overlap.o1": _o1_note() if workers > 1 else _o1_note(False, "overlap_off")}
    if mode in ("L", "R"):
        notes["frame_pool.coverage"] = {"requested": 8, "workers": 8, "units": 20, "serial": False, "max_running": 8}
        notes["siril_stack_settings"] = {"setcpu": 20, "mem_amount": 153, "per_image_mb": 523, "frames": 20}
    else:
        notes["frame_pool.strict_support"] = [_entry(min(workers, 8) if workers > 1 else 1, units=20)]
    notes.update(extra)
    return notes


def _workflow_notes(workers=4):
    return {"hash_ledger": {"version": "aester-hashing-3", "miss_bytes": {}, "largest_misses": []},
            "governor": {"mode": "shadow-full"}, "nox_threads": 8,
            "frame_pool.milkyway_render": [_entry(workers, units=44)],
            "frame_pool.star_reduction": [_entry(workers, units=345), _entry(workers, units=345)]}


D4_STACK = "requires 1.4.4 1.5.0\ncd work\nsetcpu 20\nset core.mem_mode=1\nset core.mem_amount=153\nstack r_light rej 3 3\n"


def _manifest(created, *, mode="L", scan_workers=4, ssf=None, docs=None, tiff_layout="aa", extra_files=None):
    files = {}
    if mode == "L":
        files[f"{JOB}/02_stack.ssf"] = {"class": "text", "content": {"text": ssf if ssf is not None else D4_STACK}}
    elif mode == "R":
        files[f"{JOB}/02_export.ssf"] = {"class": "text", "content": {"text": "requires 1.4.4 1.5.0\ncd work\n"}}
        files[f"{JOB}/range-0123456789ab/trial1/stack.ssf"] = {
            "class": "text", "content": {"text": ssf if ssf is not None else
                                         "requires 1.4.4 1.5.0\nsetext fit\nsetcpu 20\nset core.mem_mode=1\n"
                                         "set core.mem_amount=153\nseqclean r_light -stat\nstack r_light\n"}}
    else:
        files[f"{JOB}/finite-0123456789abcdef/stack.ssf"] = {
            "class": "text", "content": {"text": ssf if ssf is not None else "requires 1.4.4 1.5.0\nstack q\n"}}
    files[f"{JOB}/products/0123456789abcdef0123456789abcdef/transients/candidates.json"] = {
        "class": "json", "sha256": "c" * 64,
        "content": {"doc": {"version": "x", "frame_order": [f"f{i}" for i in range(20)], "workers": scan_workers,
                            "candidates": []}}}
    files["{WF}/workflow.json"] = {"class": "json", "sha256": "d" * 64, "content": {"doc": docs if docs is not None else
                                   {"state": "succeeded", "request": {"a": 1, "b": 2},
                                    "timings": {"notes": {"frame_pool.x": [1]}}}}}
    files["{DELIV}/delivery-0123456789ab/final_srgb16.tif"] = {
        "class": "tiff", "sha256": "e" * 64,
        "content": {"pages": [{"tags": [], "dropped": {"StripOffsets": {"$value_sha256": tiff_layout},
                                                       "StripByteCounts": {"$value_sha256": "bb"},
                                                       "RowsPerStrip": 5520}}]}}
    files.update(extra_files or {})
    for number, entry in enumerate(files.values()):              # what every fingerprint entry carries
        entry.setdefault("sha256", f"{number:064x}")
        entry.setdefault("bytes", 100 + number)
        entry.setdefault("birth_ns", number)
    return {"schema": "aester-workflow-fingerprint-1", "created_utc": created,
            "roots": {"{WF}": "C:\\b\\runs\\L20\\c\\workflow-00112233445566778899aabb",
                      "{DELIV}": "C:\\b\\runs\\L20\\d\\aester-00112233445566778899"},
            "files": files, "dirs": {}}


def _summary(inherited=None, **changes):
    summary = {"run_id": "R", "state": "succeeded",
               "inherited_environment": inherited if inherited is not None else {"MSYSTEM": "MINGW64"},
               "thread_audit": {"processes": [{"pid": 7, "argv": ["worker.py"], "frame_pool_max": 4,
                                               "blas_capable_max": 6, "python_threads_max": 9, "samples": 30}],
                                "frame_pool_max": 4},
               "sampler": {"per_exe": {"python.exe": {"max_threads": 40}, "siril-cli.exe": {"max_threads": 70}}},
               "foreign_cpu": {"status": "ok", "foreign_cores_mean": .2, "over_one_core": False},
               "hook_events": {"views": {}, "openblas": [{"corename": "SkylakeX", "num_threads": 24}]}}
    summary.update(changes)
    return summary


def _job(mode="L", notes=None, descendants=2, attempts=(), state="succeeded", settings=True):
    operations = {f"script:{name}": {"children": {"processes": 1, "descendant_processes": descendants}}
                  for name in ("01_convert", "02_export", "02_stack")}
    job = {"state": state, "error": None,
           "admission_options": {"transient_mode": "review", **({"stack_settings": {
               "version": "siril-stack-settings-1", "engine_version": "1.4.4"}} if settings else {})},
           "diagnostics": {"timings": {"notes": notes if notes is not None else _job_notes(mode),
                                       "resources": {"operations": operations}},
                           "timing_attempts": list(attempts)}}
    return job


def _workflow(notes=None, nox=8, stars=True):
    return {"state": "succeeded", "request": {"stars_enabled": stars, "backend_options": {"intra_op_threads": nox}},
            "timings": {"notes": notes if notes is not None else _workflow_notes()}, "timing_attempts": [],
            "review_deliveries": [{"timings": {"notes": {"frame_pool.trail_pair": [_entry(4, units=44)]}}}]}


def _write(folder: Path, *, plan, summary, job, workflow, manifest):
    """A bench result folder; workflow None: a --job run (records/job.json only)."""
    (folder / "records").mkdir(parents=True)
    for name, value in (("plan.json", plan), ("summary.json", summary), ("manifest.json", manifest),
                        ("records/job.json", job), ("records/workflow.json", workflow)):
        if value is not None:
            (folder / name).write_text(json.dumps(value), encoding="utf-8")


def _report(**changes):
    report = {"pass": True, "a": {"created_utc": "A"}, "b": {"created_utc": "B"}, "allowed": {}, "allow_patterns": [],
              "allow_fields": [], "tolerances": [], "tolerated": {}, "strict_ids": {"on": True},
              "only_in_listed": {"a": [], "b": []}}
    report.update(changes)
    return report


@pytest.fixture
def world(tmp_path):
    counter = [0]

    def build(mode="L", *, run=None, ref=None, report=None):
        """run / ref: dicts overriding plan, summary, job, workflow, manifest of that side."""
        counter[0] += 1
        root = tmp_path / f"w{counter[0]}"
        sides = {}
        for name, created, overrides in (("reference", "A", ref or {}), ("run", "B", run or {})):
            parts = {"plan": _plan(mode), "summary": _summary(), "job": _job(mode), "workflow": _workflow(),
                     "manifest": _manifest(created, mode=mode)}
            parts.update(overrides)
            parts["manifest"] = dict(parts["manifest"], created_utc=created)
            _write(root / name, **parts)
            sides[name] = root / name
        compare = root / "compare.json"
        compare.write_text(json.dumps(report if report is not None else _report()), encoding="utf-8")
        return sides["run"], sides["reference"], compare

    def check(paths, **options):
        run, ref, compare = paths
        return check_p1c.check(run, ref, compare, **options)

    return build, check


def _failed(result, section):
    assert result["passed"] is False and result[section]["passed"] is False, result
    others = ("plan", "parallel", "notes", "threads", "contract", "siril", "compare", "bytes")
    assert all(result[name]["passed"] for name in others if name != section), \
        {name: result[name]["problems"] for name in others if not result[name]["passed"]}
    return result[section]["problems"]


# --------------------------------------------------------------------------- a good run

@pytest.mark.parametrize("mode", ["L", "F", "R"])
def test_a_good_default_run_passes(world, mode):
    build, check = world
    run = {"job": _job(mode, settings=mode != "F")}
    result = check(build(mode, run=run, ref=dict(run)))
    assert result["passed"], {k: result[k]["problems"] for k in result if isinstance(result[k], dict)
                              and result[k].get("problems")}
    assert result["parallel"]["mode"] == "default" and result["coverage"]["parallel"] == "default"
    sites = result["parallel"]["sites"]
    assert sites["median"]["reached"] and sites["median"]["planned_max"] == 4
    assert sites["star_reduction"]["calls"] == 2 and sites["trail_pair"]["receipts"] == ["workflow.review_deliveries[0]"]
    assert sites["transient_scan"]["max_workers"] == 4 and not sites["transient_scan"]["reached"]   # 4 of 8 here
    assert result["bytes"]["json_pairs"] == 2 and result["bytes"]["tiff_pairs"] == 1
    assert result["info"]["openblas"] == [("SkylakeX", 24)] and result["info"]["warnings"] == []
    assert result["info"]["governor_shadow"]["job:py.median"] == {"recommended": 4, "actual": 4, "bound_by": "cap",
                                                                 "pool_workers": [4]}


# --------------------------------------------------------------------------- plan

def test_plan_parity_and_the_expected_variable_set(world):
    build, check = world
    plan = _plan()
    plan["submit"]["backend_options"]["intra_op_threads"] = 8
    assert any("submit.backend_options" in p for p in _failed(check(build(run={"plan": plan})), "plan"))
    leaked = build(run={"summary": _summary({"MSYSTEM": "MINGW64", "AESTER_FINITE_WRITE": "reference"})})
    problems = _failed(check(leaked), "plan")
    assert any("AESTER_FINITE_WRITE" in p and "not one the bench sets" in p for p in problems)
    assert check(leaked, expect_env=["AESTER_FINITE_WRITE=reference"])["passed"]
    # W5: a shadow ledger from the shell is named; the run's value must then be it
    shadow = build(run={"summary": _summary({"MSYSTEM": "MINGW64", "AESTER_HASH_LEDGER": "shadow"})})
    assert any("AESTER_HASH_LEDGER" in p for p in _failed(check(shadow), "plan"))
    assert check(shadow, expect_env=["AESTER_HASH_LEDGER=shadow"])["passed"]
    # an inherited 'on' is the default: the same setting as none (P1b precedent)
    assert check(build(ref={"summary": _summary({"MSYSTEM": "MINGW64", "AESTER_HASH_LEDGER": "on"})}))["passed"]
    trails = _plan()
    trails["submit"].update(trails_enabled=False)
    del trails["submit"]["trail_default"]
    assert any("trail_default" in p for p in _failed(check(build(run={"plan": trails})), "plan"))
    # paths are not parameters
    moved = _plan()
    moved["submit"]["delivery_root"] = "D:\\other\\d"
    moved["environment"]["AESTER_CACHE_ROOT"] = "D:\\other\\c"
    assert check(build(run={"plan": moved}))["passed"]


def test_no_fixed_parameters_unless_a_contract_comparison(world):
    build, check = world
    for pins in ({"ort_intra_op_threads": 8}, {"siril": PIN}):
        plan, ref = _plan(), _plan()
        plan["pins"].update(pins)
        ref["pins"].update(pins)
        paths = build(run={"plan": plan}, ref={"plan": ref})
        assert any("--allow-pins" in p for p in _failed(check(paths), "plan"))
        assert check(paths, allow_pins=True)["passed"]
    # a profile's own fixed parameters are the tier, not extra pins
    plan = _plan(profile="6900hx-proxy")
    plan["pins"].update(siril={"mem": 9}, ort_intra_op_threads=8, render_workers=3, affinity_logical=16)
    plan["parallel"]["tier"] = ["render_workers"]
    plan["environment"]["AESTER_RENDER_WORKERS"] = "3"
    plan["parallel"]["environment"] = dict(plan["parallel"]["environment"], AESTER_RENDER_WORKERS="3")
    assert check(build(run={"plan": plan}, ref={"plan": copy.deepcopy(plan)}))["passed"]
    custom = _plan()
    custom["parallel"].update(effective="custom", knobs=["frame_workers"])
    custom["environment"]["AESTER_FRAME_WORKERS"] = "4"
    paths = build(run={"plan": custom})
    problems = _failed(check(paths, expect_parallel="custom"), "plan")
    assert any("single parallelism knobs" in p for p in problems)


def test_the_parallel_mode_must_be_the_references_or_named(world):
    build, check = world
    serial = {"plan": _plan(parallel="serial"), "job": _job(notes=_job_notes(workers=1)),
              "workflow": _workflow(_workflow_notes(workers=1)), "manifest": _manifest("B", scan_workers=1)}
    serial["workflow"]["review_deliveries"][0]["timings"]["notes"]["frame_pool.trail_pair"] = [_entry(1, units=44)]
    serial["job"]["diagnostics"]["timings"]["notes"]["frame_pool.coverage"] = _entry(1, units=20)
    paths = build(run=serial)
    assert any("parallel mode: run 'serial'" in p for p in _failed(check(paths), "plan"))
    assert check(paths, expect_parallel="serial")["passed"]
    assert check_p1c.parallel_mode({"pins": {"frame_workers": None, "render_workers": None}}) == "default"
    assert check_p1c.parallel_mode({"pins": {"frame_workers": "4"}}) == "custom"            # a P1b-era plan
    assert check_p1c.parallel_mode({"profile": "6900hx-proxy", "pins": {"render_workers": 3}}) == "default"


# --------------------------------------------------------------------------- parallel

def _serial_run():
    run = {"plan": _plan(parallel="serial"), "job": _job(notes=_job_notes(workers=1)),
           "workflow": _workflow(_workflow_notes(workers=1)), "manifest": _manifest("B", scan_workers=1)}
    run["workflow"]["review_deliveries"][0]["timings"]["notes"]["frame_pool.trail_pair"] = [_entry(1, units=44)]
    run["job"]["diagnostics"]["timings"]["notes"]["frame_pool.coverage"] = _entry(1, units=20)
    return run


def test_serial_runs_have_every_pool_serial_for_reason_workers(world):
    build, check = world
    assert check(build(run=_serial_run()), expect_parallel="serial")["passed"]
    run = _serial_run()
    run["job"]["diagnostics"]["timings"]["notes"]["frame_pool.sky_zero"] = [_entry(4, units=20)]
    assert any("frame_pool.sky_zero[0]" in p for p in _failed(check(build(run=run), expect_parallel="serial"),
                                                               "parallel"))
    run = _serial_run()
    run["job"]["diagnostics"]["timings"]["notes"]["frame_pool.median"] = [_entry(1, reason="route")]
    paths = build(run=run)
    assert _failed(check(paths, expect_parallel="serial"), "parallel")
    assert check(paths, expect_parallel="serial", allow_serial_reasons=["route"])["passed"]
    run = _serial_run()
    run["manifest"] = _manifest("B", scan_workers=4)
    assert any("trail scan used [4]" in p for p in _failed(check(build(run=run), expect_parallel="serial"), "parallel"))
    run = _serial_run()
    run["job"]["diagnostics"]["timings"]["notes"]["overlap.o1"] = _o1_note()
    assert any("O-1 ran" in p for p in _failed(check(build(run=run), expect_parallel="serial"), "parallel"))
    run = _serial_run()
    run["plan"]["environment"]["AESTER_OVERLAP"] = "auto"
    assert any("AESTER_OVERLAP" in p for p in _failed(check(build(run=run), expect_parallel="serial"), "parallel"))


def test_default_runs_allow_only_pre_registered_serial_reasons(world):
    build, check = world
    for reason, ok in (("workers", False), ("nested", False), ("budget", True), ("units", True), ("route", True)):
        notes = _job_notes()
        notes["frame_pool.median"] = [_entry(4), _entry(1, reason=reason)]
        result = check(build(run={"job": _job(notes=notes)}))
        assert result["passed"] is ok, result["parallel"]["problems"]
    notes = _job_notes()
    notes["frame_pool.median"] = [_entry(1, units=1)]                     # one unit: nothing to parallelise
    assert check(build(run={"job": _job(notes=notes)}))["passed"]
    notes = _job_notes()
    notes["frame_pool.median"] = [_entry(1, reason="exclusive")]
    paths = build(run={"job": _job(notes=notes)})
    assert _failed(check(paths), "parallel")
    assert check(paths, allow_serial_reasons=["exclusive"])["passed"]


def test_max_runs_use_the_table_and_the_explicit_scan_workers(world):
    build, check = world

    def max_run(scan=8, **env):
        plan = _plan(parallel="max")
        plan["environment"].update(env)
        notes = _job_notes(workers=16)
        return {"plan": plan, "job": _job(notes=notes), "manifest": _manifest("B", scan_workers=scan)}
    result = check(build(run=max_run()), expect_parallel="max")
    assert result["passed"], result["parallel"]["problems"]
    assert result["parallel"]["sites"]["median"]["planned_max"] == 16 and result["parallel"]["sites"]["median"]["reached"]
    assert result["parallel"]["sites"]["transient_scan"] == {**result["parallel"]["sites"]["transient_scan"],
                                                            "planned_max": 8, "reached": True}
    assert any("trail scan used [4]" in p for p in _failed(check(build(run=max_run(scan=4)), expect_parallel="max"),
                                                           "parallel"))
    assert any("AESTER_HASH_WORKERS" in p for p in _failed(check(build(run=max_run(AESTER_HASH_WORKERS="4")),
                                                                 expect_parallel="max"), "parallel"))


def test_a_profiles_tier_knob_stands_under_default(world):
    build, check = world
    plan = _plan(parallel="default", profile="6900hx-proxy-v2")
    plan["pins"].update(siril={"mem": 9}, render_workers=3, affinity_logical=16)
    plan["parallel"].update(tier=["render_workers"], environment=dict(bench.PARALLEL_TABLE["default"],
                                                                      AESTER_RENDER_WORKERS="3"))
    plan["environment"]["AESTER_RENDER_WORKERS"] = "3"
    plan["environment"]["AESTER_OVERLAP"] = "auto"
    assert check(build(run={"plan": plan}, ref={"plan": copy.deepcopy(plan)}))["passed"]
    plan = copy.deepcopy(plan)
    plan["environment"]["AESTER_RENDER_WORKERS"] = "8"
    assert any("AESTER_RENDER_WORKERS" in p for p in _failed(check(build(run={"plan": plan}, ref={"plan": plan})),
                                                             "parallel"))


def test_a_plan_from_before_p1c_reads_as_default_with_its_profiles_render_workers(world):
    build, check = world
    plan = _plan(profile="6900hx-proxy")
    del plan["parallel"]
    plan["pins"].update(siril={"mem": 9}, ort_intra_op_threads=8, render_workers=3, affinity_logical=16)
    plan["environment"]["AESTER_RENDER_WORKERS"] = "3"
    result = check(build(run={"plan": plan}, ref={"plan": copy.deepcopy(plan)}))
    assert result["passed"] and result["parallel"]["mode"] == "default"
    plan = copy.deepcopy(plan)
    plan["pins"]["render_workers"] = 4                                    # not the profile's: a single knob
    plan["environment"]["AESTER_RENDER_WORKERS"] = "4"
    assert check_p1c.parallel_mode(plan) == "custom"


# --------------------------------------------------------------------------- notes

def test_pool_notes_accumulate_per_call_with_their_fields(world):
    build, check = world
    notes = _job_notes()
    notes["frame_pool.median"] = _entry(4)                                 # a single dict: overwritten calls
    assert any("not a list of calls" in p for p in _failed(check(build(run={"job": _job(notes=notes)})), "notes"))
    notes = _job_notes()
    del notes["frame_pool.median"][0]["unit_bytes"]
    assert any("lacks unit_bytes" in p for p in _failed(check(build(run={"job": _job(notes=notes)})), "notes"))
    notes = _job_notes()
    del notes["frame_pool.median"][0]["fp_env_applied"]
    paths = build(run={"job": _job(notes=notes)})
    assert any("lacks fp_env" in p for p in _failed(check(paths), "notes"))
    assert check(paths, optional_note_fields=["fp_env"])["passed"]
    notes = _job_notes()
    notes["frame_pool.median"][0]["cpu_user"] = notes["frame_pool.median"][0].pop("cpu_user_s")
    paths = build(run={"job": _job(notes=notes)})
    assert _failed(check(paths), "notes")
    assert check(paths, note_renames=["cpu_user=cpu_user|cpu_user_s"])["passed"]
    serial = _job_notes()
    serial["frame_pool.median"] = [_entry(1, units=1)]                     # no fp_env on a serial entry: fine
    assert check(build(run={"job": _job(notes=serial)}))["passed"]
    compact = _job_notes()
    compact["frame_pool.median"] = [json.dumps(_entry(4))]                 # one-line JSON entries are read too
    assert check(build(run={"job": _job(notes=compact)}))["passed"]
    with pytest.raises(check_p1c.UsageError):
        check_p1c.note_fields(["speed=x"])


def test_ledger_governor_and_placement(world):
    build, check = world
    notes = _job_notes()
    notes["hash_ledger"]["version"] = "aester-hashing-4"
    assert any("the reference's" in p for p in _failed(check(build(run={"job": _job(notes=notes)})), "notes"))
    notes = _job_notes()
    del notes["governor"]
    assert any("governor note absent" in p for p in _failed(check(build(run={"job": _job(notes=notes)})), "notes"))
    off_plan, off_notes, off_wf = _plan(), _job_notes(), _workflow_notes()
    off_plan["environment"]["AESTER_GOVERNOR"] = "off"
    del off_notes["governor"], off_wf["governor"]
    paths = build(run={"plan": off_plan, "job": _job(notes=off_notes), "workflow": _workflow(off_wf)})
    assert check(paths, expect_env=["AESTER_GOVERNOR=off"])["passed"]
    job = _job()
    job["overlap"] = {"leak": 1}
    job["diagnostics"]["siril_stack_settings"] = 1
    problems = _failed(check(build(run={"job": job})), "notes")
    assert any("/overlap outside" in p for p in problems) and any("siril_stack_settings outside" in p for p in problems)
    workflow = _workflow()
    workflow["checkpoints"] = {"x": {"pool_stats": {}}}
    assert any("pool_stats" in p for p in _failed(check(build(run={"workflow": workflow})), "notes"))
    layout = _job_notes(l_siril_layout="combined")
    assert any("l_siril_layout" in p for p in _failed(check(build(run={"job": _job(notes=layout)})), "notes"))


# --------------------------------------------------------------------------- threads

def test_thread_audit(world):
    build, check = world
    assert any("no thread audit" in p for p in _failed(check(build(run={"summary": _summary(thread_audit=None)})),
                                                       "threads"))
    many = _summary()
    many["thread_audit"]["processes"].append({"pid": 9, "argv": ["workflow.py"], "frame_pool_max": 17,
                                              "blas_capable_max": 30})
    result = check(build(run={"summary": many}))
    assert any("17 frame-pool threads" in p for p in _failed(result, "threads"))
    assert any("BLAS-capable" in i for i in result["threads"]["info"])
    assert check(build())["threads"]["os_threads_max_per_exe"] == {"python.exe": 40, "siril-cli.exe": 70}


# --------------------------------------------------------------------------- contract

def test_d4_blocks_read_past_the_pin_lines():
    pinned = ("requires 1.4.4 1.5.0\nsetcpu 16\nset core.mem_mode=1\nset core.mem_amount=16\n"
              "setcpu 20\nset core.mem_mode=1\nset core.mem_amount=153\ncd work\nstack r_light\n")
    found = check_p1c.d4_blocks(pinned, PIN)
    assert found["blocks"] == [{"line": 4, "setcpu": 20, "mem_amount": 153}] and found["stray"] == []
    assert check_p1c.d4_blocks(pinned, None)["blocks"][0]["setcpu"] == 16                 # unpinned: two blocks
    assert len(check_p1c.d4_blocks(pinned, None)["blocks"]) == 2
    hx = "requires 1.4.4 1.5.0\nset core.mem_mode=1\nset core.mem_amount=9\nsetcpu 10\nset core.mem_mode=1\n" \
         "set core.mem_amount=9\nstack r\n"
    assert check_p1c.d4_blocks(hx, {"mem": 9})["blocks"] == [{"line": 3, "setcpu": 10, "mem_amount": 9}]
    assert check_p1c.d4_blocks("requires 1\nsetcpu 4\nstack r\n", None)["stray"] == [1]
    offline = "requires 1\noffline\nsetcpu 2\nset core.mem_mode=1\nset core.mem_amount=4\nstack q\n"
    assert check_p1c.d4_blocks(offline, None)["blocks"][0]["line"] == 2


def test_the_two_contracts(world):
    build, check = world
    assert any("stack_settings" in p for p in _failed(check(build(run={"job": _job(settings=False)})), "contract"))
    two = D4_STACK.replace("stack r_light", "setcpu 2\nset core.mem_mode=1\nset core.mem_amount=4\nstack r_light")
    assert any("2 D4 blocks" in p for p in _failed(check(build(run={"manifest": _manifest("B", ssf=two)})), "contract"))
    after = "requires 1.4.4 1.5.0\ncd work\nstack r_light\nsetcpu 20\nset core.mem_mode=1\nset core.mem_amount=153\n"
    assert any("follows the stack" in p for p in _failed(check(build(run={"manifest": _manifest("B", ssf=after)})),
                                                         "contract"))
    none = "requires 1.4.4 1.5.0\ncd work\nstack r_light\n"
    assert any("0 D4 blocks" in p for p in _failed(check(build(run={"manifest": _manifest("B", ssf=none)})),
                                                   "contract"))
    notes = _job_notes()
    del notes["siril_stack_settings"]
    assert any("siril_stack_settings" in p for p in _failed(check(build(run={"job": _job(notes=notes)})), "contract"))
    # F: none of the three
    f_run = {"job": _job("F", settings=True)}
    assert any("an F job has" in p for p in _failed(check(build("F", run=f_run)), "contract"))
    f_ssf = {"manifest": _manifest("B", mode="F", ssf="requires 1\nsetcpu 8\nstack q\n"), "job": _job("F", settings=False)}
    assert any("beyond the pin" in p for p in _failed(check(build("F", run=f_ssf, ref={"job": _job("F", settings=False)})),
                                                      "contract"))
    # R: the range trial script carries it
    r_none = {"manifest": _manifest("B", mode="R", ssf="requires 1.4.4 1.5.0\nseqclean r_light -stat\nstack r_light\n")}
    assert any("0 D4 blocks" in p for p in _failed(check(build("R", run=r_none)), "contract"))
    # EFF-NOX: the frozen request carries 8 whenever nox runs
    assert any("intra_op_threads is 4" in p for p in _failed(check(build(run={"workflow": _workflow(nox=4)})),
                                                             "contract"))
    no_stars = check(build(run={"workflow": _workflow(nox=None, stars=False)}))
    assert no_stars["contract"]["passed"] and no_stars["contract"]["info"]


def test_code_from_before_the_contracts(world):
    """A1b / KN-2 / X2: code before K2 has no D4 at all; code before K1 froze no thread count."""
    build, check = world
    notes = _job_notes()
    del notes["siril_stack_settings"]
    old = {"job": _job(notes=notes, settings=False), "workflow": _workflow(nox=None),
           "manifest": _manifest("B", ssf="requires 1.4.4 1.5.0\ncd work\nstack r_light\n")}
    paths = build(run=old, ref=copy.deepcopy(old))
    assert not check(paths)["contract"]["passed"]
    result = check(paths, before_contracts=["d4", "nox"])
    assert result["passed"] and any("before EFF-NOX" in i for i in result["contract"]["info"])
    assert any("before EFF-D4" in p for p in _failed(check(build(), before_contracts=["d4"]), "contract"))
    with pytest.raises(check_p1c.UsageError):
        check(paths, before_contracts=["k3"])
    # a P1b-era reference has no P1c pins: their defaults are what it ran with
    plan = _plan()
    for key in ("trails", "memory_view_gib", "commit_view_gib", "python_openblas", "parallel", "overlap"):
        del plan["pins"][key]
    del plan["parallel"]
    for key in (bench.HOOK_EVENTS_ENV, bench.THREAD_LOG_ENV, bench.BLAS_LOG_ENV, bench.MEMORY_VIEW_ENV,
                bench.COMMIT_VIEW_ENV, bench.PYTHON_ONLY_ENV, "STARSTACK_TRANSIENT_WORKERS", "AESTER_HASH_WORKERS",
                "AESTER_OVERLAP", "PYTHONPATH", bench.HOOK_LOG_ENV):
        del plan["environment"][key]                                    # the P1b bench set none of these
    assert check(build(ref={"plan": plan}))["plan"]["passed"]
    trails_off = _plan()
    trails_off["pins"]["trails"] = "off"
    assert any("pins.trails is new" in p for p in check(build(run={"plan": trails_off}, ref={"plan": plan}))
               ["plan"]["problems"])


# --------------------------------------------------------------------------- Siril, compare

def test_siril_section(world):
    build, check = world
    assert any("3 processes" in p for p in _failed(check(build(run={"job": _job(descendants=3)})), "siril"))
    assert any("2 attempts" in p for p in _failed(check(build(run={"job": _job(attempts=[{}])})), "siril"))
    resumed = _summary(resume={"segments": ["R1", "R2"]})
    assert check(build(run={"job": _job(attempts=[{}]), "summary": resumed}))["passed"]
    failed = check(build(run={"job": _job(state="failed")}))
    assert not failed["passed"] and any("job state" in p for p in failed["siril"]["problems"])


def test_a_run_resumed_after_the_stack_checkpoint_keeps_its_earlier_attempt_s_siril_and_d4(world):
    """事后 10-07 (window D review): C2-hx2 and C7 cancel after checkpoint('stack'), so the resumed attempt launches
    no Siril and writes no D4 note; both come from the attempt that stacked."""
    build, check = world
    stacked = _job()["diagnostics"]["timings"]
    resumed_job = _job(notes={k: v for k, v in _job_notes("L").items() if k != "siril_stack_settings"},
                       attempts=[{"notes": stacked["notes"], "resources": stacked["resources"]}])
    resumed_job["diagnostics"]["timings"]["resources"]["operations"] = {}
    resumed = _summary(resume={"segments": ["R1", "R2"]})
    result = check(build(run={"job": resumed_job, "summary": resumed}))
    assert result["passed"], {k: result[k]["problems"] for k in ("siril", "contract")}
    assert sorted(result["siril"]["launches"]) == ["attempt1:script:01_convert", "attempt1:script:02_export",
                                                  "attempt1:script:02_stack"]
    assert result["contract"]["note_source"] == "attempt1"
    # nothing in any attempt: both sections still fail
    bare = _job(notes={k: v for k, v in _job_notes("L").items() if k != "siril_stack_settings"}, attempts=[{}])
    bare["diagnostics"]["timings"]["resources"]["operations"] = {}
    result = check(build(run={"job": bare, "summary": resumed}))
    assert any("no Siril operations" in p for p in result["siril"]["problems"])
    assert any("siril_stack_settings is missing" in p for p in result["contract"]["problems"])
    # an earlier attempt's launch is judged like the final one's
    noisy = copy.deepcopy(resumed_job)
    noisy["diagnostics"]["timing_attempts"][0]["resources"]["operations"]["script:02_stack"]["children"][
        "descendant_processes"] = 3
    assert any("attempt1:script:02_stack: 3 processes" in p
               for p in _failed(check(build(run={"job": noisy, "summary": resumed})), "siril"))


def test_compare_section(world):
    build, check = world
    assert _failed(check(build(report=_report(**{"pass": False}))), "compare")
    assert any("allowed" in p for p in _failed(check(build(report=_report(allowed={"x": {}}))), "compare"))
    assert any("allow_patterns" in p for p in _failed(check(build(report=_report(allow_patterns=["*.ssf"]))),
                                                      "compare"))
    no_strict = build(report=_report(strict_ids={"on": False}))
    assert any("strict ids" in p for p in _failed(check(no_strict), "compare"))
    assert check(no_strict, allow_no_strict_ids=True)["passed"]
    listed = build(report=_report(only_in_listed={"a": [], "b": ["{WF}/x.log"]}))
    assert _failed(check(listed), "compare")
    assert check(listed, extra_listed=["b:{WF}/x.log"])["passed"]
    swapped = build(report=_report(a={"created_utc": "B"}, b={"created_utc": "A"}))
    assert any("a = reference" in p for p in _failed(check(swapped), "compare"))


def _worker_files(indices, extensions=("json", "log"), folder="<scan-runs-hex32#1>"):
    return [f"{{WF}}/stack/job-<job#1>/work/transients/scan-runs/{folder}/worker-{k:03d}.{e}"
            for k in indices for e in extensions]


def test_scan_worker_files_follow_the_worker_counts(world):
    """K's planner ran 4 scan workers, a serial run 1: K alone lists workers 1-3 (W1, 10-07)."""
    build, check = world
    serial = {"manifest": _manifest("B", scan_workers=1)}
    result = check(build(run=serial, report=_report(only_in_listed={"a": _worker_files(range(1, 4)), "b": []})))
    assert result["passed"], result
    assert result["compare"]["scan_worker_files"] == {"a": sorted(_worker_files(range(1, 4))), "b": []}
    wider = {"manifest": _manifest("B", scan_workers=6)}                       # max: the run lists workers 4-5
    assert check(build(run=wider, report=_report(only_in_listed={"a": [], "b": _worker_files(range(4, 6))})))["passed"]
    for only in ({"a": _worker_files(range(1, 3)), "b": []},                  # worker 3 missing
                 {"a": _worker_files(range(1, 4), ("json",)), "b": []},       # logs missing
                 {"a": [], "b": _worker_files(range(1, 4))},                  # the side with fewer workers
                 {"a": _worker_files([1]) + _worker_files([2, 3], folder="<scan-runs-hex32#2>"), "b": []},
                 {"a": _worker_files(range(1, 4)) + ["{WF}/x.log"], "b": []}):  # anything else stays a problem
        assert _failed(check(build(run=serial, report=_report(only_in_listed=only))), "compare")
    same = build(report=_report(only_in_listed={"a": _worker_files(range(1, 4)), "b": []}))   # 4 against 4
    assert _failed(check(same), "compare")


# --------------------------------------------------------------------------- bytes

def test_json_key_order_outside_the_timing_receipts(world):
    build, check = world
    swapped = _manifest("B", docs={"request": {"b": 2, "a": 1}, "state": "succeeded",
                                   "timings": {"notes": {"frame_pool.x": [1]}}})
    problems = _failed(check(build(run={"manifest": swapped})), "bytes")
    assert any("{WF}/workflow.json /:" in p for p in problems) and any("/request" in p for p in problems)
    timing_only = _manifest("B", docs={"state": "succeeded", "request": {"a": 1, "b": 2},
                                       "timings": {"notes": {"overlap.o1": {}, "frame_pool.x": [1]}, "z": 1}})
    assert check(build(run={"manifest": timing_only}))["passed"]          # new keys inside timings are allowed
    # keys named by run ids are compared canonically
    files = {"{WF}/stack/job-0123456789abcdef01234567/job.json": {
        "class": "json", "sha256": "1" * 64,
        "content": {"doc": {"inputs": {"C:\\b\\runs\\L20\\c\\workflow-00112233445566778899aabb\\a": 1, "z": 2}}}}}
    paths = build(run={"manifest": _manifest("B", extra_files=files)}, ref={"manifest": _manifest("A", extra_files=files)})
    assert check(paths)["passed"]


def test_tiff_strip_layout(world):
    build, check = world
    problems = _failed(check(build(run={"manifest": _manifest("B", tiff_layout="zz")})), "bytes")
    assert problems == ["{DELIV}/delivery-0123456789ab/final_srgb16.tif page 0: StripOffsets differs"]   # no dirs here
    page = {"tags": [], "dropped": {"StripOffsets": {"$value_sha256": "aa"}, "StripByteCounts": {"$value_sha256": "bb"},
                                    "RowsPerStrip": 5520}}
    two_pages = {"{DELIV}/delivery-0123456789ab/final_srgb16.tif": {"class": "tiff", "content": {"pages": [page, page]}}}
    problems = _failed(check(build(run={"manifest": _manifest("B", extra_files=two_pages)})), "bytes")
    assert problems == ["{DELIV}/delivery-0123456789ab/final_srgb16.tif: 1 vs 2 pages"]
    rows = copy.deepcopy(page)
    rows["dropped"]["RowsPerStrip"] = 64
    one_page = {"{DELIV}/delivery-0123456789ab/final_srgb16.tif": {"class": "tiff", "content": {"pages": [rows]}}}
    assert any("RowsPerStrip" in p for p in _failed(check(build(run={"manifest": _manifest("B", extra_files=one_page)})),
                                                    "bytes"))


# --------------------------------------------------------------------------- coverage

def _coverage_result(mode, parallel, passed=True, **sites):
    return {"mode": mode, "parallel": parallel, "passed": passed, "run_id": f"{mode}-{parallel}",
            "sites": {name: {"max_workers": w, "max_units": 87, "reached": reached} for name, (w, reached) in sites.items()}}


def test_coverage_requires_each_site_at_its_planned_maximum_once():
    full = {name: (16, True) for name, entry in check_p1c.SITES.items() if "L" in entry["modes"]}
    results = [_coverage_result("L", "max", **full), _coverage_result("L", "default", **{k: (4, True) for k in full})]
    only_l = ["L:default", "L:max"]
    result = check_p1c.coverage(results, expect_combos=only_l)
    assert result["passed"], result["problems"]
    assert result["combinations"] == [["L", "default"], ["L", "max"]]
    silent = dict(full, median=(1, False))
    result = check_p1c.coverage([_coverage_result("L", "max", **silent), results[1]], expect_combos=only_l)
    assert any(p.startswith("median/L/max: ran in 1 run(s) but never") for p in result["problems"])
    # a failed run does not count, and its combination then has no accepted run at all: a problem, not a silence
    result = check_p1c.coverage([_coverage_result("L", "max", passed=False, **full), results[1]], expect_combos=only_l)
    assert result["combinations"] == [["L", "default"]] and result["ignored_runs"] == 1
    assert result["problems"] == ["L/max: no accepted run (1 failed), so its sites were never shown at their planned "
                                  "maxima"]
    # a conditional site that never ran is information only
    partial = {k: v for k, v in full.items() if k not in ("trail_pair", "median")}
    result = check_p1c.coverage([_coverage_result("L", "default", **{k: (4, True) for k in partial})],
                                expect_combos=["L:default"])
    assert any(p.startswith("median/L/default: never ran") for p in result["problems"])
    assert any(i.startswith("trail_pair/L/default: never ran") for i in result["info"])
    # a measured cap from the command line
    strict = _coverage_result("F", "default", strict_support=(6, False))
    assert any("strict_support/F/default" in p for p in check_p1c.coverage([strict], expect_combos=["F:default"])
               ["problems"])
    assert not any("strict_support" in p for p in check_p1c.coverage([strict], ["strict_support=6"],
                                                                     expect_combos=["F:default"])["problems"])
    other = _coverage_result("L", "default", starless_scope_advice=(8, True))
    assert any("starless_scope_advice" in i for i in check_p1c.coverage([other])["info"])


def test_coverage_expects_the_w_tables_combinations_by_default():
    """L/F/R default, L/F max and the strict L job (SL) default and max: a combination whose runs all failed, or that
    never ran, is a problem; the strict L job's own sites are required there."""
    result = check_p1c.coverage([_coverage_result("L", "default", median=(4, True))])
    missing = {p.split(":")[0] for p in result["problems"] if "no accepted run" in p}
    assert missing == {"L/max", "F/default", "F/max", "R/default", "SL/default", "SL/max"}
    assert result["expected_combinations"] == [list(c) for c in check_p1c.EXPECTED_COMBOS]
    sl = _coverage_result("SL", "max", median=(16, True), sky_zero=(16, True))
    result = check_p1c.coverage([sl], expect_combos=["SL:max"])
    assert any(p.startswith("strict_support/SL/max: never ran") for p in result["problems"])
    assert not any("milkyway_render" in p or "coverage/SL" in p for p in result["problems"])   # no workflow there
    assert any(i.startswith("native_mask/SL/max: never ran") for i in result["info"])          # conditional
    with pytest.raises(check_p1c.UsageError):
        check_p1c.coverage([sl], expect_combos=["SL:serial"])
    with pytest.raises(check_p1c.UsageError):
        check_p1c.coverage([sl], expect_combos=["X:max"])


# --------------------------------------------------------------------------- command line

def test_cli_exit_codes_json_and_coverage(world, tmp_path, capsys):
    build, _ = world
    run, ref, compare = build()
    common = ["check", "--run", str(run), "--reference", str(ref), "--compare", str(compare)]
    out = tmp_path / "result.json"
    assert check_p1c.main([*common, "--json", str(out)]) == 0
    stored = json.loads(out.read_text(encoding="utf-8"))
    assert stored["passed"] and stored["schema"] == check_p1c.SCHEMA and stored["coverage"]["mode"] == "L"
    assert check_p1c.main([*common, "--expect-parallel", "max"]) == 1
    assert check_p1c.main([*common, "--expect-env", "NOEQUALS"]) == 2
    assert check_p1c.main([*common, "--expect-parallel", "fast"]) == 2
    assert check_p1c.main([*common, "--site-max", "nowhere=4"]) == 2
    assert check_p1c.main(["coverage", str(out), "--json", str(tmp_path / "coverage.json")]) == 1   # one combo only
    coverage = json.loads((tmp_path / "coverage.json").read_text(encoding="utf-8"))
    assert coverage["schema"] == check_p1c.COVERAGE_SCHEMA and coverage["combinations"] == [["L", "default"]]
    assert check_p1c.main(["coverage", str(out), "--expect-combo", "L:serial"]) == 2
    bad = tmp_path / "bad.json"
    bad.write_text("{}", encoding="utf-8")
    assert check_p1c.main(["coverage", str(bad)]) == 2
    capsys.readouterr()


def test_no_native_mask_pool_on_the_reference_route(world):
    build, check = world
    notes = _job_notes("F")
    notes["frame_pool.native_mask"] = [_entry(4, units=87)]

    def f_run(implementation):
        record = {f"{JOB}/work/interpolation_support.json": {
            "class": "json", "content": {"doc": {"implementation": implementation, "state": "complete"}}}}
        return {"job": _job("F", notes=copy.deepcopy(notes), settings=False),
                "manifest": _manifest("B", mode="F", extra_files=record)}
    ref = {"job": _job("F", settings=False)}
    quick = check(build("F", run=f_run(check_p1c.QUICK_SUPPORT_IMPLEMENTATION), ref=ref))
    assert quick["passed"] and quick["parallel"]["support_routes"] == [check_p1c.QUICK_SUPPORT_IMPLEMENTATION]
    problems = _failed(check(build("F", run=f_run("numpy_reference_1"), ref=ref)), "parallel")
    assert any("reference route" in p for p in problems)


def _p1b_results() -> Path | None:
    """The bench results of P1b (untracked, in the main checkout; a git worktree under .claude/worktrees reads the
    main checkout's copy, read-only)."""
    tail = Path("outputs") / "perf-baseline-20260928" / "results"
    candidates = [ROOT / tail]
    if ROOT.parent.name == "worktrees" and ROOT.parents[1].name == ".claude":
        candidates.append(ROOT.parents[2] / tail)
    names = ("20261001T175138Z-L20-P1B-L20-max", "20261001T172303Z-L20-P1B-L20")
    return next((c for c in candidates if all((c / n / "manifest.json").is_file() for n in names)), None)


@pytest.mark.skipif(_p1b_results() is None, reason="the P1b bench results (untracked outputs/) are not here")
def test_two_real_p1b_runs_pass_the_byte_level_checks(tmp_path):
    """Real records and manifests (P1B-L20 with --frame-workers max against P1B-L20, bit-identical): the byte-level
    supplements pass on every JSON and TIFF pair; the P1c-only sections fail as they must on a P1b-era run."""
    results = _p1b_results()
    run, ref = results / "20261001T175138Z-L20-P1B-L20-max", results / "20261001T172303Z-L20-P1B-L20"
    created = {side: json.loads((path / "manifest.json").read_text(encoding="utf-8"))["created_utc"]
               for side, path in (("a", ref), ("b", run))}
    compare = tmp_path / "compare.json"           # a stand-in report: the comparison itself is not re-run here
    compare.write_text(json.dumps(_report(a={"created_utc": created["a"]}, b={"created_utc": created["b"]})),
                       encoding="utf-8")
    result = check_p1c.check(run, ref, compare, allow_pins=True, expect_parallel="custom",
                             expect_pin=['frame_workers="max"'])
    assert result["bytes"]["passed"] and result["bytes"]["json_pairs"] >= 40 and result["bytes"]["tiff_pairs"] >= 20
    assert result["plan"]["passed"] and result["siril"]["passed"] and result["notes"]["passed"]
    assert result["parallel"]["sites"]["coverage"]["max_workers"] == 16
    assert not result["threads"]["passed"] and not result["contract"]["passed"]     # no audit, no D4 yet
    assert not result["passed"]
    # as A1b checks code from before K1/K2: no D4 anywhere is then what passes
    before = check_p1c.check(run, ref, compare, allow_pins=True, expect_parallel="custom",
                             expect_pin=['frame_workers="max"'], before_contracts=["d4", "nox"])
    sections = ("plan", "parallel", "notes", "threads", "contract", "siril", "compare", "bytes")
    assert [name for name in sections if not before[name]["passed"]] == ["threads"]


def test_the_checker_reads_what_the_bench_records():
    """Names shared with the bench (PARALLEL_TABLE, environment variables, hook keys) come from it."""
    assert set(check_p1c.PARALLEL_VARIABLES) == set(bench.PARALLEL_VARIABLES)
    assert {bench.HOOK_LOG_ENV, bench.HOOK_SHIM_ENV, bench.HOOK_EVENTS_ENV, bench.THREAD_LOG_ENV} <= check_p1c.PATH_KEYS
    assert check_p1c.RELEVANT_ENV.pattern == bench.RELEVANT_ENV.pattern
    assert check_p1c.MAX_FRAME_POOL_THREADS == bench.MAX_FRAME_WORKERS
    from starstack import frame_pool, registered_support
    assert check_p1c.MAX_FRAME_POOL_THREADS == frame_pool.MAX_FRAME_WORKERS
    assert (check_p1c.SUPPORT_RECORD, check_p1c.QUICK_SUPPORT_IMPLEMENTATION) == (
        registered_support.RECORD, registered_support.SUPPORT_IMPLEMENTATION)


# --------------------------------------------------------------------------- small fixes

def test_overlap_notes_and_default_knobs():
    assert check_p1c.overlap_on([]) is None and check_p1c.overlap_on(None) is None
    assert check_p1c.overlap_on([{"eligible": False}, {"eligible": True}]) is True
    # a P1b-era plan that wrote a knob at its documented default is still the default
    assert check_p1c.parallel_mode({"pins": {"frame_workers": "auto", "overlap": "on"}}) == "default"
    assert check_p1c.parallel_mode({"pins": {"overlap": "off"}}) == "custom"


def test_an_empty_overlap_note_is_a_problem_not_a_crash(world):
    build, check = world
    notes = _job_notes()
    notes["overlap.o1"] = []
    assert any("overlap.o1 note is empty" in p for p in _failed(check(build(run={"job": _job(notes=notes)})),
                                                                 "parallel"))


@pytest.mark.parametrize("note, expected", [
    (_o1_note(), []),
    (_o1_note(False, "overlap_off"), []),
    (_o1_note(False, "slower_than_serial", n_o1=1, n_serial=4), []),       # a plan may come with an ineligible note
    (_o1_note(False, "overlap_error:OSError"), []),
    (_o1_note(outcome="discarded:the main thread failed: InjectedA", join_wait_s=None, window_s={}), []),
    (_o1_note(outcome="recorded", failure="ScanWorkerError"), []),
    (_o1_note(False, "overlap off"), ["unknown reason 'overlap off'"]),
    (_o1_note(False, None), ["unknown reason None"]),
    (_o1_note(reason="memory_tight"), ["eligible with the reason"]),
    (_o1_note(schema="overlap-o1-0"), ["schema 'overlap-o1-0'"]),
    (_o1_note(eligible="yes"), ["eligible is 'yes'"]),
    ({k: v for k, v in _o1_note().items() if k != "n_o1"}, ["eligible without n_o1"]),
    (_o1_note(headroom_bytes={"physical": 1}), ["headroom_bytes lacks"]),
    ({k: v for k, v in _o1_note().items() if k != "outcome"}, ["without an outcome"]),
    ({k: v for k, v in _o1_note().items() if k != "window_s"}, ["joined without join_wait_s / window_s"]),
    ([_o1_note(False, "stack_checkpointed"), _o1_note()], []),
])
def test_the_overlap_note_has_core_s_shape(note, expected):
    problems = check_p1c.overlap_problems(note)
    assert len(problems) == len(expected) and all(e in p for e, p in zip(expected, problems)), problems


def test_a_malformed_overlap_note_fails_the_parallel_section(world):
    build, check = world
    notes = _job_notes()
    notes["overlap.o1"] = _o1_note(False, "overlap off")
    assert any("unknown reason" in p for p in _failed(check(build(run={"job": _job(notes=notes)})), "parallel"))


def test_memory_views_must_keep_the_in_process_reservations(world):
    """R20 under the 6900HX proxy: a view wrapped around the public function without deducting frame_pool's
    reservations (or a view record that predates saying so) fails the plan section."""
    build, check = world

    def viewed(applied):
        plan = _plan()
        plan["pins"]["memory_view_gib"] = 9.0
        views = {"processes": [{"pid": 7, "argv": ["worker.py"], "applied": {"available_memory": applied}}]}
        return {"plan": plan, "summary": _summary(hook_events={"views": views, "openblas": []})}
    base = {"status": "wrapped", "view_bytes": 9 * 1024 ** 3}
    for applied, ok in ((dict(base, function="_memory_status[0]", reservations="below the view"), True),
                        (dict(base, function="available_memory", reservations="deducted from the view",
                              frame_pool_reserved=True), True),
                        (dict(base, function="available_memory", reservations="none", frame_pool_reserved=False), True),
                        (dict(base, function="available_memory"), False),
                        (dict(base, function="available_memory", reservations="none", frame_pool_reserved=True), False),
                        ({"status": "absent"}, True)):
        run = viewed(applied)
        result = check(build(run=run, ref={"plan": copy.deepcopy(run["plan"])}))
        if ok:
            assert result["passed"], result["plan"]["problems"]
        else:
            assert any("hides the code root's in-process reservations" in p for p in _failed(result, "plan"))
    # no view requested: the record is not judged
    run = viewed(dict(base, function="available_memory"))
    run["plan"]["pins"]["memory_view_gib"] = None
    assert check(build(run=run))["passed"]


# --------------------------------------------------------------------------- --job runs (K-SL20, W19-W21)

JOB_DIR = "{JOB}"


def _job_run(parallel=None, *, ssf=None, workers=4):
    """A bench --job strict-legacy result: plan job/slot/submit of core.submit_job, job notes with strict support
    instead of coverage counting, no workflow record, a manifest of the job folder as {JOB}."""
    plan = _plan("L", parallel, job="strict-legacy", slot="SL20",
                 submit={"siril_path": "C:\\repo\\tools\\siril-cli.exe", "sky_mask_path": "C:\\b\\assets\\sky.png",
                         "idempotency_key": "aester-bench-SL20", "support_mode": "strict_siril_1_4_4",
                         "stack_range_mode": "legacy"})
    notes = _job_notes("L", workers=workers)
    del notes["frame_pool.coverage"]
    notes["frame_pool.strict_support"] = [_entry(workers, units=20)]
    notes["frame_pool.native_mask"] = [_entry(workers, units=87)]
    files = {f"{JOB_DIR}/02_stack.ssf": {"class": "text", "content": {"text": ssf if ssf is not None else D4_STACK}},
             f"{JOB_DIR}/work/interpolation_support.json": {
                 "class": "json", "content": {"doc": {"implementation": check_p1c.QUICK_SUPPORT_IMPLEMENTATION}}},
             f"{JOB_DIR}/job.json": {"class": "json", "content": {"doc": {"state": "succeeded", "b": 1, "a": 2}}}}
    for number, entry in enumerate(files.values()):
        entry.update(sha256=f"{number + 7:064x}", bytes=10 + number, birth_ns=number)
    manifest = {"schema": "aester-workflow-fingerprint-1", "created_utc": "B", "workflow_dir": None,
                "job_dir": "C:\\b\\runs\\SL20\\c\\job-0123456789abcdef01234567",
                "roots": {"{JOB}": "C:\\b\\runs\\SL20\\c\\job-0123456789abcdef01234567", "{CACHE}": "C:\\b\\runs\\SL20\\c"},
                "files": files, "dirs": {}}
    return {"plan": plan, "job": _job(notes=notes), "workflow": None, "manifest": manifest}


def test_a_strict_l_job_run_is_checked_and_counted_as_sl(world):
    build, check = world
    for parallel in (None, "max"):
        run = _job_run(parallel, workers=16 if parallel == "max" else 4)
        result = check(build(run=run, ref=copy.deepcopy(run)), expect_parallel=parallel)
        assert result["passed"], {k: result[k]["problems"] for k in result["sections"] if result[k]["problems"]}
        assert result["job"] == "strict-legacy" and result["coverage"]["mode"] == "SL"
        assert result["coverage"]["parallel"] == (parallel or "default")
        assert result["contract"]["scripts"][f"{JOB_DIR}/02_stack.ssf"]["blocks"][0]["setcpu"] == 20
        assert result["parallel"]["support_routes"] == [check_p1c.QUICK_SUPPORT_IMPLEMENTATION]
        assert any("strict-legacy job run" in i for i in result["contract"]["info"])
        assert result["notes"]["pool_notes"] == ["frame_pool.median", "frame_pool.native_mask", "frame_pool.sky_zero",
                                                 "frame_pool.strict_support"]
    # the D4 block of the strict L stack is required like L's
    none = _job_run(ssf="requires 1.4.4 1.5.0\ncd work\nstack r_light\n")
    assert any("0 D4 blocks" in p for p in _failed(check(build(run=none, ref=_job_run())), "contract"))
    # a job run is not a workflow run's peer, nor the other way round
    problems = _failed(check(build(run=_job_run(), ref={})), "plan")
    assert any(p.startswith("job: reference None, run 'strict-legacy'") for p in problems)
    with pytest.raises(check_p1c.UsageError):
        bad = _job_run()
        bad["plan"]["job"] = "strict-range"
        check(build(run=bad, ref=_job_run()))


# --------------------------------------------------------------------------- cancel and resume (prereg 6.4)

RESUMED_JOB = f"{JOB}/products/abcdef0123456789abcdef0123456789/transients/candidates.json"
# the delivery TIFF of _manifest, its layout unchanged, with other pixels
OTHER_PIXELS = {"{DELIV}/delivery-0123456789ab/final_srgb16.tif": {
    "class": "tiff", "content": {"pages": [{"tags": [], "pixels": {"digest": "zz"},
                                            "dropped": {"StripOffsets": {"$value_sha256": "aa"},
                                                        "StripByteCounts": {"$value_sha256": "bb"},
                                                        "RowsPerStrip": 5520}}]}}}


def _resume_parts(created="B", *, attempt=2, latency=1.5, quiet=.6, segments=("R1", "R2"), valid=True, extra=None,
                  files=None, attempts=None, dirs=None, scan_workers=4):
    """``attempts``: the job's timing_attempts (the cancelled attempt's timings; default one without notes);
    ``dirs``: the manifest's directories (canonical names, as the fingerprint records every walked directory)."""
    docs = {"state": "succeeded", "request": {"a": 1, "b": 2}, "attempt": attempt,
            "timings": {"notes": {"frame_pool.x": [1]}}, **(extra or {})}
    cancel = {"valid": valid, "invalid_reasons": [] if valid else ["boom"], "spec": {"text": "job:stage=stack+2"},
              "latency": {"to_cancelled_s": latency, "cancelled_to_no_descendants_s": quiet}}
    parts = {"manifest": _manifest(created, docs=docs, extra_files=files, scan_workers=scan_workers)}
    if dirs:
        parts["manifest"]["dirs"] = {name: number for number, name in enumerate(dirs)}
    if segments is not None:
        parts["summary"] = _summary(resume={"segments": list(segments), "first_cancel": cancel})
        parts["job"] = _job(attempts=[{}] if attempts is None else attempts)
    return parts


def _o1_attempt(eligible: bool) -> dict:
    """An earlier attempt's timings carrying core's overlap.o1 note (eligible: B ran beside A in that attempt)."""
    note = {"schema": check_p1c.OVERLAP_SCHEMA, "eligible": eligible, "reason": None if eligible else "trails_off"}
    return {"notes": {check_p1c.OVERLAP_NOTE: note}}


def _resume_report(**changes):
    report = _report(**{"pass": False, "different": {"{WF}/workflow.json": {
        "verdict": "DIFFERENT", "class": "json", "json": [{"path": "/json/attempt", "a": 1, "b": 2}]}},
        "only_in_b": [RESUMED_JOB], "dirs_only_in_b": [f"{JOB}/products/abcdef0123456789abcdef0123456789/"]})
    report.update(changes)
    return report


def _allowance(world, tmp_path, **k_changes):
    build, _ = world
    k_run, k_ref, k_compare = build(run=_resume_parts(**k_changes), ref=_resume_parts("A", attempt=1, segments=None),
                                    report=_resume_report())
    result = check_p1c.resume_allowance(k_run, k_ref, k_compare, "C1")
    assert result["passed"], result["problems"]
    path = tmp_path / "R_RESUME-C1.json"
    path.write_text(json.dumps(result["allowance"]), encoding="utf-8")
    return path, result["allowance"]


def test_resume_allowance_freezes_the_k_calibration(world, tmp_path):
    path, allowance = _allowance(world, tmp_path)
    assert allowance["schema"] == check_p1c.RESUME_SCHEMA and allowance["cancel_point"] == "C1"
    assert allowance["different"] == {"{WF}/workflow.json": {"class": "json", "paths": ["/json/attempt"]}}
    assert allowance["only_in_b"] == [RESUMED_JOB] and allowance["k_latency"]["to_cancelled_s"] == 1.5
    assert allowance["cancel_when"] == "job:stage=stack+2"
    build, _ = world
    # what criterion 4 refuses cannot be frozen into R_resume
    paths = build(run=_resume_parts(files=OTHER_PIXELS), ref=_resume_parts("A", attempt=1, segments=None),
                  report=_resume_report())
    assert any("data digests differ in K itself" in p for p in check_p1c.resume_allowance(*paths, "C1")["problems"])
    paths = build(run=_resume_parts(), ref=_resume_parts("A", attempt=1, segments=None),
                  report=_resume_report(strict_ids={"on": True, "differ": [{"kind": "job"}]}))
    assert any("deterministic ids differ" in p for p in check_p1c.resume_allowance(*paths, "C1")["problems"])
    paths = build(run=_resume_parts(segments=("R1",)), ref=_resume_parts("A", attempt=1, segments=None),
                  report=_resume_report())
    assert any("1 segment(s)" in p for p in check_p1c.resume_allowance(*paths, "C1")["problems"])
    with pytest.raises(check_p1c.UsageError):
        check_p1c.resume_allowance(*paths, "C 1")
    # C7 on K (prereg 6.4): K's serial scan cancelled while it ran keeps its scan output, and the resume adds its own:
    # recorded as k_leftovers (the residue O-1's marker protocol removes); a .partial-* entry still refuses
    scan_output = {f"{JOB}/work/transients/candidates.json": {"class": "json", "content": {"doc": {"x": 1}}}}
    k_extra = {**scan_output, f"{JOB}/work/transients/scan-runs/0123456789abcdef0123456789abcdef/r.json": {
        "class": "json", "content": {"doc": {"r": 1}}}}
    paths = build(run=_resume_parts(files=k_extra), ref=_resume_parts("A", attempt=1, segments=None,
                                                                     files=dict(scan_output)),
                  report=_resume_report())
    result = check_p1c.resume_allowance(*paths, "C7")
    assert result["passed"], result["problems"]
    assert len(result["allowance"]["k_leftovers"]) >= 1 and all(
        "work/transients/scan-runs" in name for name in result["allowance"]["k_leftovers"])
    partial = {"{DELIV}/.partial-delivery-0123456789ab-01234567/final.jpg": {"class": "bytes", "content": {}}}
    paths = build(run=_resume_parts(files=partial), ref=_resume_parts("A", attempt=1, segments=None),
                  report=_resume_report())
    assert any("leftovers in K's resumed tree" in p for p in check_p1c.resume_allowance(*paths, "C1")["problems"])
    out = tmp_path / "frozen.json"
    k = build(run=_resume_parts(), ref=_resume_parts("A", attempt=1, segments=None), report=_resume_report())
    args = ["resume-allowance", "--run", str(k[0]), "--reference", str(k[1]), "--compare", str(k[2]),
            "--cancel-point", "C1", "--out", str(out)]
    assert check_p1c.main(args) == 0 and json.loads(out.read_text(encoding="utf-8"))["cancel_point"] == "C1"
    assert check_p1c.main(args) == 2                                     # frozen once


def test_k_s_scan_residue_is_the_serial_path_s_and_o1_s_marker_never_passes(world, tmp_path):
    """C7 (prereg 6.4): K's serial scan cancelled while it ran leaves a scan-runs folder that its resume does not
    remove; a P1c run where O-1 did not run shows the same residue and passes, one with any other residue under
    work/transients (O-1's marker, another folder) fails."""
    build, check = world
    scan_output = {f"{JOB}/work/transients/candidates.json": {"class": "json", "content": {"doc": {"x": 1}}}}
    residue = f"{JOB}/work/transients/scan-runs/0123456789abcdef0123456789abcdef/r.json"
    k_files = {**scan_output, residue: {"class": "json", "content": {"doc": {"r": 1}}}}
    reference = _resume_parts("A", attempt=1, segments=None, files=dict(scan_output))
    k_run, k_ref, k_compare = build(run=_resume_parts(files=dict(k_files)), ref=reference, report=_resume_report())
    calibrated = check_p1c.resume_allowance(k_run, k_ref, k_compare, "C7")
    assert calibrated["passed"] and calibrated["allowance"]["k_leftovers"], calibrated
    path = tmp_path / "R_RESUME-C7.json"
    path.write_text(json.dumps(calibrated["allowance"]), encoding="utf-8")
    same = check(build(run=_resume_parts(files=dict(k_files)), ref=reference, report=_resume_report()),
                 resume_allowance_path=path)
    assert not any("leftovers" in p for p in same["compare"]["problems"]), same["compare"]["problems"]
    for extra in (f"{JOB}/work/transients.o1-pending", f"{JOB}/work/transients/other.npy"):
        other = check(build(run=_resume_parts(files={**k_files, extra: {"class": "bytes", "content": {}}}),
                            ref=reference, report=_resume_report()), resume_allowance_path=path)
        assert any("leftovers in the final tree" in p and extra.split("/")[-1] in p
                   for p in other["compare"]["problems"]), extra


def test_k_s_residue_is_excused_only_when_the_cancelled_attempt_did_not_run_o1(world, tmp_path):
    """The same residue as K's passes when the cancelled attempt had no overlap.o1 note (code before O-1, or O-1 not
    reached) or an ineligible one; when O-1 ran in it (an eligible note in job.json's timing_attempts), the marker
    protocol leaves nothing, so the same residue fails (prereg 6.4 criterion 4 at C7)."""
    build, check = world
    scan_output = {f"{JOB}/work/transients/candidates.json": {"class": "json", "content": {"doc": {"x": 1}}}}
    residue = f"{JOB}/work/transients/scan-runs/0123456789abcdef0123456789abcdef/r.json"
    k_files = {**scan_output, residue: {"class": "json", "content": {"doc": {"r": 1}}}}
    reference = _resume_parts("A", attempt=1, segments=None, files=dict(scan_output))
    calibrated = check_p1c.resume_allowance(*build(run=_resume_parts(files=dict(k_files)), ref=reference,
                                                   report=_resume_report()), "C7")
    assert calibrated["passed"] and calibrated["allowance"]["k_leftovers"], calibrated
    path = tmp_path / "R_RESUME-C7.json"
    path.write_text(json.dumps(calibrated["allowance"]), encoding="utf-8")
    for attempts, ran in (([{}], None), ([_o1_attempt(False)], False), ([_o1_attempt(True)], True),
                          ([{}, _o1_attempt(True)], True)):
        result = check(build(run=_resume_parts(files=dict(k_files), attempts=attempts), ref=reference,
                             report=_resume_report()), resume_allowance_path=path)
        left = [p for p in result["compare"]["problems"] if "leftovers in the final tree" in p]
        assert result["compare"]["overlap_o1_cancelled_attempt"] is ran, attempts
        if ran:
            assert left and "O-1 ran in the cancelled attempt" in left[0], result["compare"]["problems"]
        else:
            assert not left, left


def test_a_scan_run_s_worker_files_follow_the_worker_count_and_are_not_leftovers(world, tmp_path):
    """An overlapped reference ran 6 scan processes, the resumed serial scan 7: the seventh worker's ledger and log
    are the listed-only check's concern, not leftovers; a scan-runs folder the reference lacks still is (by its
    directory entry), and so are K's residue folders' timing-dependent entries only when K left that folder."""
    build, check = world
    path, _ = _allowance(world, tmp_path)
    run_dir = f"{JOB}/work/transients/scan-runs/0123456789abcdef0123456789abcdef"

    def scan(workers, folder=run_dir):
        files = {f"{JOB}/work/transients/candidates.json": {"class": "json", "content": {"doc": {"x": 1}}},
                 f"{folder}/result-000001.json": {"class": "json", "content": {"doc": {"r": 1}}}}
        for number in range(workers):
            files[f"{folder}/worker-{number:03d}.json"] = {"class": "json", "content": {"doc": {"w": number}}}
            files[f"{folder}/worker-{number:03d}.log"] = {"class": "log", "content": {}}
        return files
    reference = _resume_parts("A", attempt=1, segments=None, files=scan(6), dirs=[run_dir])
    seven = check(build(run=_resume_parts(files=scan(7), dirs=[run_dir]), ref=reference, report=_resume_report()),
                  resume_allowance_path=path)
    assert not any("leftovers" in p for p in seven["compare"]["problems"]), seven["compare"]["problems"]
    other = f"{JOB}/work/transients/scan-runs/fedcba9876543210fedcba9876543210"
    second = {**scan(7), **{name: entry for name, entry in scan(7, other).items() if "/scan-runs/" in name}}
    extra = check(build(run=_resume_parts(files=second, dirs=[run_dir, other]), ref=reference,
                        report=_resume_report()), resume_allowance_path=path)
    assert any("leftovers in the final tree" in p for p in extra["compare"]["problems"]), extra["compare"]["problems"]
    # a folder holding worker files only (a scan cancelled before any result) is still left over by its directory
    bare = {**scan(7), **{name: entry for name, entry in scan(1, other).items() if "/worker-" in name}}
    lonely = check(build(run=_resume_parts(files=bare, dirs=[run_dir, other]), ref=reference,
                         report=_resume_report()), resume_allowance_path=path)
    assert any("leftovers in the final tree" in p for p in lonely["compare"]["problems"]), lonely["compare"]
    # K's residue (C7): a cancelled serial scan's folder whose .pending names depend on which frames had finished
    k_files = {**scan(6), f"{run_dir}/result-000003.pending": {"class": "bytes", "content": {}}}
    k_path = tmp_path / "R_RESUME-C7-pending.json"
    calibrated = check_p1c.resume_allowance(*build(run=_resume_parts(files=k_files, dirs=[run_dir]), ref=reference,
                                                   report=_resume_report()), "C7")
    assert calibrated["passed"] and calibrated["allowance"]["k_leftovers"], calibrated
    k_path.write_text(json.dumps(calibrated["allowance"]), encoding="utf-8")
    p1c_files = {**scan(7), f"{run_dir}/result-000004.pending": {"class": "bytes", "content": {}}}
    pending = check(build(run=_resume_parts(files=p1c_files, dirs=[run_dir]), ref=reference, report=_resume_report()),
                    resume_allowance_path=k_path)
    assert not any("leftovers" in p for p in pending["compare"]["problems"]), pending["compare"]["problems"]
    ran = check(build(run=_resume_parts(files=p1c_files, dirs=[run_dir], attempts=[_o1_attempt(True)]),
                      ref=reference, report=_resume_report()), resume_allowance_path=k_path)
    assert any("leftovers in the final tree" in p and "result-000004.pending" in p
               for p in ran["compare"]["problems"]), ran["compare"]["problems"]


def test_a_resumed_run_passes_inside_r_resume_only(world, tmp_path, capsys):
    build, check = world
    path, _ = _allowance(world, tmp_path)
    ref = _resume_parts("A", attempt=1, segments=None)
    ok = check(build(run=_resume_parts(latency=2.4), ref=ref, report=_resume_report()), resume_allowance_path=path)
    assert ok["passed"], {k: ok[k]["problems"] for k in ok["sections"] if ok[k]["problems"]}
    assert ok["sections"][-1] == "cancel" and ok["cancel"]["limit_s"] == 3.0 and ok["compare"]["mode"] == "resume"
    # without the allowance the same run fails its comparison, with a hint
    plain = check(build(run=_resume_parts(), ref=ref, report=_resume_report()))
    assert any("--resume-allowance" in p for p in _failed(plain, "compare"))
    # a difference path outside R_resume
    extra = build(run=_resume_parts(extra={"resumed_from": "x"}), ref=ref, report=_resume_report())
    assert any("difference paths outside R_resume ['/json/resumed_from']" in p
               for p in _failed(check(extra, resume_allowance_path=path), "compare"))
    # a file on one side that K did not have
    more = build(run=_resume_parts(), ref=ref, report=_resume_report(only_in_b=[RESUMED_JOB, "{WF}/x.json"]))
    assert any("only_in_b outside R_resume" in p for p in _failed(check(more, resume_allowance_path=path), "compare"))
    # data digests, whatever the report says; leftovers, whatever R_resume says
    data = build(run=_resume_parts(files=OTHER_PIXELS), ref=ref, report=_resume_report())
    assert any("data digests differ" in p for p in _failed(check(data, resume_allowance_path=path), "compare"))
    for leftover in (f"{JOB}/work/transients/f.npy", f"{JOB}/work/transients.o1-pending",
                     "{DELIV}/.partial-delivery-0123456789ab-01234567/final.jpg"):
        left = build(run=_resume_parts(files={leftover: {"class": "bytes", "content": {}}}), ref=ref,
                     report=_resume_report())
        assert any("leftovers in the final tree" in p
                   for p in _failed(check(left, resume_allowance_path=path), "compare")), leftover
    # a trails-on job's own scan output, which the uninterrupted reference has too, is not left over (O-1's marker
    # protocol removes only what an interrupted attempt added)
    scan_output = {f"{JOB}/work/transients/candidates.json": {"class": "json", "content": {"doc": {"x": 1}}},
                   f"{JOB}/work/transients/crops/light_00001-1.png": {"class": "bytes", "content": {}}}
    trails = build(run=_resume_parts(files=dict(scan_output)), ref=_resume_parts("A", attempt=1, segments=None,
                                                                                 files=dict(scan_output)),
                   report=_resume_report())
    assert not any("leftovers" in p for p in check(trails, resume_allowance_path=path)["compare"]["problems"])
    second = {**scan_output, f"{JOB}/work/transients/f.npy": {"class": "bytes", "content": {}}}
    extra_scan = build(run=_resume_parts(files=second), ref=_resume_parts("A", attempt=1, segments=None,
                                                                          files=dict(scan_output)),
                       report=_resume_report())
    assert any("leftovers in the final tree" in p and "work/transients/f.npy" in p
               for p in _failed(check(extra_scan, resume_allowance_path=path), "compare"))
    # strict ids never pass
    ids = build(run=_resume_parts(), ref=ref, report=_resume_report(strict_ids={"on": True, "differ": [{"k": 1}]}))
    assert any("deterministic ids" in p for p in _failed(check(ids, resume_allowance_path=path), "compare"))
    # criteria 1-3
    slow = build(run=_resume_parts(latency=3.4), ref=ref, report=_resume_report())
    assert any("> 3.0 s" in p for p in _failed(check(slow, resume_allowance_path=path), "cancel"))
    assert check(slow, resume_allowance_path=path, cross_version=True)["passed"]          # X1/X2: criterion 4 only
    assert check(slow, resume_allowance_path=path, cancel_limit_s=30)["passed"]           # C7's O-1 window
    lingering = build(run=_resume_parts(quiet=6.0), ref=ref, report=_resume_report())
    assert any("descendants alive 6.0 s" in p for p in _failed(check(lingering, resume_allowance_path=path), "cancel"))
    twice = build(run=_resume_parts(segments=("R1", "R2", "R3")), ref=ref, report=_resume_report())
    assert any("3 segment(s)" in p for p in _failed(check(twice, resume_allowance_path=path), "cancel"))
    invalid = build(run=_resume_parts(valid=False), ref=ref, report=_resume_report())
    assert any("cancel was invalid" in p for p in _failed(check(invalid, resume_allowance_path=path), "cancel"))
    ok_paths = build(run=_resume_parts(), ref=ref, report=_resume_report())
    with pytest.raises(check_p1c.UsageError):
        check(ok_paths, cross_version=True)
    bad = tmp_path / "bad.json"
    bad.write_text("{}", encoding="utf-8")
    with pytest.raises(check_p1c.UsageError):
        check(ok_paths, resume_allowance_path=bad)
    common = ["check", "--run", str(ok_paths[0]), "--reference", str(ok_paths[1]), "--compare", str(ok_paths[2])]
    assert check_p1c.main([*common, "--resume-allowance", str(path)]) == 0
    assert '"cancel": true' in capsys.readouterr().out


def test_the_cancel_limit_follows_the_host(tmp_path):
    allowance = {"k_latency": {"to_cancelled_s": 2.6}}

    def run(profile, latency):
        return {"folder": str(tmp_path), "plan": {"profile": profile},
                "summary": {"resume": {"segments": ["R1", "R2"], "first_cancel": {
                    "valid": True, "latency": {"to_cancelled_s": latency, "cancelled_to_no_descendants_s": .2}}}}}
    assert check_p1c.check_cancel(run(None, 3.5), allowance, False, None)["passed"]        # max(3, 2.6 + 1) = 3.6
    assert not check_p1c.check_cancel(run(None, 3.7), allowance, False, None)["passed"]
    hx2 = check_p1c.check_cancel(run("6900hx-proxy-v2", 4.9), allowance, False, None)
    assert hx2["passed"] and hx2["limit_s"] == 5.0
    assert not check_p1c.check_cancel(run("6900hx-proxy-v2", 5.1), allowance, False, None)["passed"]
    assert not check_p1c.check_cancel(run(None, 1.0), {}, False, None)["passed"]          # nothing calibrated



def test_a_resumed_run_s_scan_worker_files_follow_the_worker_count(world, tmp_path):
    """事后 10-07: P1c's resumed attempt scans with 8 workers against the K reference's 4; K's own calibration
    compared 4 with 4, so R_resume holds no worker files."""
    path, _ = _allowance(world, tmp_path)
    build, check = world
    reference = _resume_parts("A", attempt=1, segments=None)
    extra = {"a": [], "b": _worker_files(range(4, 8))}
    ok = check(build(run=_resume_parts(scan_workers=8), ref=reference, report=_resume_report(only_in_listed=extra)),
               resume_allowance_path=path, cancel_point="C1")
    assert ok["passed"], {k: ok[k]["problems"] for k in ("compare", "cancel")}
    assert ok["compare"]["scan_worker_files"] == {"a": [], "b": sorted(_worker_files(range(4, 8)))}
    same = check(build(run=_resume_parts(), ref=reference, report=_resume_report(only_in_listed=extra)),
                 resume_allowance_path=path)                                    # 4 against 4: nothing explained
    assert any("listed-only files of side b outside R_resume" in p for p in _failed(same, "compare"))
    more = {"a": [], "b": _worker_files(range(4, 8)) + ["{WF}/x.log"]}
    other = check(build(run=_resume_parts(scan_workers=8), ref=reference, report=_resume_report(only_in_listed=more)),
                  resume_allowance_path=path)
    assert any("x.log" in p for p in _failed(other, "compare"))


def test_the_allowance_must_belong_to_the_run(world, tmp_path):
    """事后 10-07 (window D review): the cancel point, the profile, the trigger text and the reference of the
    calibration are those of the run, unless a registered exception is named."""
    allowance = {"cancel_point": "C2", "profile": None, "cancel_when": "job:progress=60+6",
                 "k_reference": "C:/b/results/20261005T172212Z-L20-K-L20"}

    def run(profile=None, text="job:progress=60+6"):
        return {"plan": {"profile": profile}, "folder": "x",
                "summary": {"resume": {"first_cancel": {"spec": {"text": text}}}}}
    ref = {"folder": "outputs/perf-baseline-20260928/results/20261005T172212Z-L20-K-L20"}
    bind = check_p1c.allowance_binding
    assert bind(run(), allowance, reference=ref, cancel_point="C2") == ([], [])
    assert bind(run(), allowance, reference=ref, cancel_point="C2-hx2")[0]
    assert bind(run("6900hx-proxy-v2"), allowance, reference=ref, cancel_point="C2")[0]
    assert bind(run(text="job:progress=60+4"), allowance, reference=ref, cancel_point="C2")[0]
    problems, info = bind(run(text="job:progress=60+4"), allowance, reference=ref, cancel_point="C2",
                          cancel_when_changed=True)
    assert not problems and "--cancel-when-changed" in info[0]
    other = {"folder": "results/20261005T174418Z-L20-X2b-ref"}
    assert bind(run(), allowance, reference=other, cancel_point="C2")[0]
    assert bind(run(), allowance, reference=other, cancel_point="C2", allowance_reference_differs=True)[0] == []
    assert "no --cancel-point" in bind(run(), allowance, reference=ref)[1][0]
    # through check: the cancel section fails on another point's allowance
    path, _ = _allowance(world, tmp_path)
    build, check = world
    paths = build(run=_resume_parts(), ref=_resume_parts("A", attempt=1, segments=None), report=_resume_report())
    assert check(paths, resume_allowance_path=path, cancel_point="C1")["passed"]
    assert any("not 'C2'" in p for p in _failed(check(paths, resume_allowance_path=path, cancel_point="C2"), "cancel"))
    with pytest.raises(check_p1c.UsageError):
        check(paths, cancel_point="C1")

# --------------------------------------------------------------------------- KD-3 (prereg 4.2)

PINNED_STACK = ("requires 1.4.4 1.5.0\nsetcpu 16\nset core.mem_mode=1\nset core.mem_amount=16\ncd work\n"
                "stack r_light rej 3 3\n")
PINNED_CONVERT = "requires 1.4.4 1.5.0\nsetcpu 16\nset core.mem_mode=1\nset core.mem_amount=16\ncd work\nconvert x\n"


def _kd3_side(created, *, d4, convert=None, signature=None, extra=None):
    contract = {"version": "siril-stack-settings-1", "engine_version": "1.4.4"}
    job = {"job_dir": "C:\\b\\runs\\L20\\c\\workflow-00112233445566778899aabb\\stack\\job-0123456789abcdef01234567",
           "signature": signature or ("b" * 64 if d4 else "a" * 64),
           "admission_options": {"transient_mode": "review", **({"stack_settings": contract} if d4 else {})},
           **({"stack_settings": contract} if d4 else {}), **(extra or {})}
    files = {f"{JOB}/job.json": {"class": "json", "content": {"doc": job}},
             f"{JOB}/02_stack.ssf": {"class": "text", "content": {"text": D4_STACK if d4 else PINNED_STACK}},
             f"{JOB}/01_convert.ssf": {"class": "text", "content": {"text": convert or (
                 "requires 1.4.4 1.5.0\ncd work\nconvert x\n" if d4 else PINNED_CONVERT)}},
             f"{JOB}/work/siril.ini": {"class": "text", "content": {"text": "mem=16" if not d4 else "mem=0.9"}}}
    if d4:
        files[f"{JOB}/work/siril-stack.ini"] = {"class": "text", "content": {"text": "mem_amount=153"}}
    return {"manifest": _manifest(created, extra_files=files)}


def _kd3_report(**changes):
    different = {f"{JOB}/job.json": {"verdict": "DIFFERENT", "class": "json"},
                 f"{JOB}/02_stack.ssf": {"verdict": "DIFFERENT", "class": "text", "text": {}},
                 f"{JOB}/01_convert.ssf": {"verdict": "DIFFERENT", "class": "text", "text": {}},
                 f"{JOB}/work/siril.ini": {"verdict": "DIFFERENT", "class": "text", "text": {}}}
    report = _report(**{"pass": False, "strict_ids": {"on": False}, "different": different,
                        "only_in_b": [f"{JOB}/work/siril-stack.ini"]})
    report.update(changes)
    return report


def test_kd3_accepts_the_contract_field_in_the_stack_snapshot(world):
    """Window A's real KD-3 pairs: the workflow's stack_snapshot-<id>.json copies the admission options."""
    build, _ = world
    snapshot = "{WF}/stack_snapshot-<stack_snapshot#1>.json"

    def side(created, d4, extra=None):
        built = _kd3_side(created, d4=d4)
        contract = {"version": "siril-stack-settings-1", "engine_version": "1.4.4"}
        doc = {"admission_options": {"transient_mode": "review", **({"stack_settings": contract} if d4 else {})},
               **({"stack_settings": contract} if d4 else {}), **(extra or {})}
        built["manifest"]["files"][snapshot] = {"class": "json", "content": {"doc": doc}, "sha256": "f" * 64,
                                                "bytes": 99, "birth_ns": 99}
        return built

    report = _kd3_report()
    report["different"] = {**report["different"], snapshot: {"verdict": "DIFFERENT", "class": "json"}}
    result = check_p1c.kd3(*build(run=side("B", True), ref=side("A", False), report=report))
    assert result["passed"], result["problems"]
    assert result["json_differences"][snapshot] == ["/json/admission_options/stack_settings", "/json/stack_settings"]
    # anything else in the snapshot still fails
    result = check_p1c.kd3(*build(run=side("B", True, {"worker": 1}), ref=side("A", False), report=report))
    assert any("stack_snapshot" in p and "beyond the contract field" in p for p in result["problems"])


def test_kd3_asserts_the_three_contract_differences(world, tmp_path):
    build, _ = world
    paths = build(run=_kd3_side("B", d4=True), ref=_kd3_side("A", d4=False), report=_kd3_report())
    result = check_p1c.kd3(*paths)
    assert result["passed"], result["problems"]
    assert [d["kind"] for d in result["id_differences"]] == ["job-signature"]
    assert result["json_differences"][f"{JOB}/job.json"] == ["/json/admission_options/stack_settings",
                                                              "/json/stack_settings"]
    out = tmp_path / "kd3.json"
    assert check_p1c.main(["kd3", "--run", str(paths[0]), "--reference", str(paths[1]), "--compare", str(paths[2]),
                           "--json", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["schema"] == check_p1c.KD3_SCHEMA
    # a JSON difference beyond the contract field
    paths = build(run=_kd3_side("B", d4=True, extra={"seqclean": 1}), ref=_kd3_side("A", d4=False),
                  report=_kd3_report())
    assert any("beyond the contract field: ['/json/seqclean']" in p for p in check_p1c.kd3(*paths)["problems"])
    # another script that differs beyond its pin lines
    paths = build(run=_kd3_side("B", d4=True, convert="requires 1.4.4 1.5.0\ncd work\nconvert y\n"),
                  ref=_kd3_side("A", d4=False), report=_kd3_report())
    assert any("01_convert.ssf: differs beyond" in p for p in check_p1c.kd3(*paths)["problems"])
    # no job signature difference: the contract did not change the admission
    paths = build(run=_kd3_side("B", d4=True, signature="a" * 64), ref=_kd3_side("A", d4=False), report=_kd3_report())
    assert any("exactly the job signature" in p for p in check_p1c.kd3(*paths)["problems"])
    # pixels, or a new file that is not a Siril configuration
    paths = build(run=_kd3_side("B", d4=True), ref=_kd3_side("A", d4=False),
                  report=_kd3_report(different={**_kd3_report()["different"], "{DELIV}/x.tif": {
                      "verdict": "DIFFERENT", "class": "tiff", "pixels": []}}, only_in_b=["{WF}/extra.json"]))
    problems = check_p1c.kd3(*paths)["problems"]
    assert any("{DELIV}/x.tif" in p for p in problems) and any("extra.json" in p for p in problems)
