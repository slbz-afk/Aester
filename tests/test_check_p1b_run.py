"""scripts/check_p1b_run.py (the P1b run check, CROSS_CHECK X7/X9) on synthetic bench result folders."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from starstack import core, siril  # noqa: E402

JOB = "{WF}/stack/job-0123456789abcdef01234567"
PIN = {"cpu": 16, "mem": 16}


def _module():
    spec = importlib.util.spec_from_file_location("check_p1b_run", ROOT / "scripts" / "check_p1b_run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _crlf(commands, pin):
    return siril.script_text(commands, pin).replace("\n", "\r\n")      # manifests store the CRLF the run wrote


def _plan(mode="L", **changes):
    plan = {"mode": mode, "frames": 20, "slot": f"{mode}20", "profile": None, "simulation": None,
            "pins": {"siril": PIN, "ort_intra_op_threads": 8, "affinity_logical": None, "mask_sha": False,
                     "render_workers": None},
            "submit": {"backend_options": {"tile": 512, "intra_op_threads": 8},
                       "stack_range_mode": "legacy" if mode == "L" else ("finite_validity_v1" if mode == "F" else "range")},
            "environment": {"AESTER_CACHE_ROOT": "C:\\b\\runs\\L20\\c", "AESTER_COMPONENT_ROOT": "C:\\b\\runs\\L20\\k",
                            "AESTER_SIRIL_PIN": "cpu=16,mem=16", "OPENSSL_ia32cap": None,
                            "AESTER_RENDER_WORKERS": None, "MSYSTEM": "MINGW64", "XDG_DATA_HOME": "C:\\b\\runs\\L20\\x"},
            "siril_launch": {"MSYSTEM": "MINGW64"}}
    plan.update(changes)
    return plan


def _notes(governor=True, version="aester-hashing-3", coverage=True):
    notes = {"hash_ledger": {"version": version, "miss_bytes": {"no_entry": 0}, "largest_misses": []}}
    if governor:
        notes["governor"] = {"mode": "shadow-full"}
    if coverage:
        notes["frame_pool.coverage"] = {"workers": 8, "serial": False, "reason": None}
    return notes


def _write(folder: Path, *, plan, created, texts, inherited=None, job_notes=None, workflow_notes=None,
           descendants=2, job_state="succeeded", attempts=()):
    (folder / "records").mkdir(parents=True)
    (folder / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    (folder / "summary.json").write_text(json.dumps({"state": "succeeded", "inherited_environment":
                                                     inherited if inherited is not None else {"MSYSTEM": "MINGW64"}}),
                                         encoding="utf-8")
    files = {f"{JOB}/{name}": {"class": "text", "content": {"text": text}} for name, text in texts.items()}
    files[f"{JOB}/02_stack.log"] = {"class": "log"}
    (folder / "manifest.json").write_text(json.dumps({"created_utc": created, "files": files}), encoding="utf-8")
    operations = {f"script:{name}": {"children": {"processes": 1, "descendant_processes": descendants}}
                  for name in ("01_convert", "02_probe", "02_stack")}
    job = {"state": job_state, "error": None if job_state == "succeeded" else "Siril 02_stack 失败（3）",
           "diagnostics": {"timings": {"notes": job_notes if job_notes is not None else _notes(),
                                       "resources": {"operations": operations}},
                           "timing_attempts": list(attempts)}}
    (folder / "records" / "job.json").write_text(json.dumps(job), encoding="utf-8")
    workflow = {"state": "succeeded", "timings": {"notes": workflow_notes if workflow_notes is not None
                                                  else _notes(coverage=False)}, "timing_attempts": []}
    (folder / "records" / "workflow.json").write_text(json.dumps(workflow), encoding="utf-8")


def _report(**changes):
    report = {"pass": True, "a": {"created_utc": "A"}, "b": {"created_utc": "B"},
              "allowed": {f"{'{WF}'}/stack/job-<job#1>/02_export.ssf": {"verdict": "ALLOWED_ONLY_IN_B",
                                                                        "allowed_by": "*/stack/job-*/02_export.ssf"},
                          f"{'{WF}'}/stack/job-<job#1>/02_stack.ssf": {"verdict": "ALLOWED_DIFFERENT",
                                                                       "allowed_by": "*/stack/job-*/02_stack.ssf"}},
              "allow_patterns": ["*/stack/job-*/02_export.ssf", "*/stack/job-*/02_stack.ssf"],
              "allow_fields": [], "tolerances": [],
              "only_in_listed": {"a": [], "b": ["{WF}/stack/job-<job#1>/02_export.log"]}}
    report.update(changes)
    return report


BASE_TEXTS = {"02_stack.ssf": _crlf(core.stack_script(core.DISTORTION_WCS, own_registration=True), PIN)}
RUN_TEXTS = {"02_export.ssf": _crlf(core.export_script(core.DISTORTION_WCS, own_registration=True), PIN),
             "02_stack.ssf": _crlf(core.LEGACY_STACK_SCRIPT, PIN)}


@pytest.fixture
def world(tmp_path):
    module = _module()

    def build(*, base_plan=None, run_plan=None, report=None, base_texts=None, run_texts=None, run_kwargs=None,
              base_kwargs=None, mode="L"):
        root = tmp_path / f"w{len(list(tmp_path.iterdir()))}"
        base, run = root / "baseline", root / "run"
        _write(base, plan=base_plan or _plan(mode), created="A", texts=base_texts or BASE_TEXTS, **(base_kwargs or {}))
        plan = run_plan or _plan(mode)
        plan.setdefault("environment", {})
        _write(run, plan=plan, created="B", texts=run_texts or RUN_TEXTS, **(run_kwargs or {}))
        compare = root / "compare.json"
        compare.write_text(json.dumps(report if report is not None else _report()), encoding="utf-8")
        return base, run, compare

    def check(paths, **options):
        base, run, compare = paths
        options.setdefault("coverage", "workers=8,serial=false,reason=none")
        options.setdefault("expect_env", [])
        return module.check(run, base, compare, **options)

    return module, build, check


def _failed(result, section):
    assert result["passed"] is False and result[section]["passed"] is False, result
    assert all(result[name]["passed"] for name in ("plan", "compare", "notes", "siril", "scripts")
               if name in result and name != section), result
    return result[section]["problems"]


def test_a_good_l_run_passes(world):
    module, build, check = world
    result = check(build())
    assert result["passed"], result
    assert set(result) >= {"plan", "compare", "scripts", "notes", "siril"}


def test_plan_parity_catches_a_missing_ort_pin(world):
    module, build, check = world
    run_plan = _plan()
    run_plan["submit"]["backend_options"]["intra_op_threads"] = None          # the S4B-L20 precedent
    problems = _failed(check(build(run_plan=run_plan)), "plan")
    assert any("backend_options" in p for p in problems)


def test_knobs_must_equal_the_baseline_or_be_named(world):
    module, build, check = world
    # an inherited diagnostics switch that the plan does not clear leaks into the run
    leaked = build(run_kwargs={"inherited": {"MSYSTEM": "MINGW64", "AESTER_L_SIRIL_LAYOUT": "combined"}})
    assert any("AESTER_L_SIRIL_LAYOUT" in p for p in _failed(check(leaked), "plan"))
    cleared_plan = _plan()
    cleared_plan["environment"]["AESTER_L_SIRIL_LAYOUT"] = None                # what the bench now does
    cleared = build(run_plan=cleared_plan, run_kwargs={"inherited": {"MSYSTEM": "MINGW64",
                                                                     "AESTER_L_SIRIL_LAYOUT": "combined"}})
    assert check(cleared)["passed"]
    # a knob meant to differ (B4: governor off, ledger off) must be named, and the governor note follows it
    off_plan = _plan()
    off_plan["environment"].update(AESTER_GOVERNOR="off", AESTER_HASH_LEDGER="off")
    off_notes = _notes(governor=False)
    paths = build(run_plan=off_plan, run_kwargs={"job_notes": off_notes, "workflow_notes": _notes(False, coverage=False)})
    assert any("AESTER_GOVERNOR" in p for p in _failed(check(paths), "plan"))
    assert check(paths, expect_env=["AESTER_GOVERNOR=off", "AESTER_HASH_LEDGER=off"])["passed"]
    # and an expected knob the run does not have fails
    assert _failed(check(build(), expect_env=["AESTER_HASH_LEDGER=shadow"]), "plan")
    # an absent knob is its default: the P1a baselines inherited AESTER_HASH_LEDGER=on, the S-4b runs had none
    defaulted = build(base_kwargs={"inherited": {"MSYSTEM": "MINGW64", "AESTER_HASH_LEDGER": "on"}})
    assert check(defaulted)["passed"]
    assert module.knob_value("AESTER_L_SIRIL_LAYOUT", "") == "split" and module.knob_value("OTHER", "") == ""


def test_new_pins_must_be_named(world):
    module, build, check = world
    run_plan = _plan()
    run_plan["pins"]["frame_workers"] = "auto"
    paths = build(run_plan=run_plan)
    assert any("frame_workers" in p for p in _failed(check(paths), "plan"))
    assert check(paths, expect_pin=['frame_workers="auto"'])["passed"]
    assert _failed(check(paths, expect_pin=["frame_workers=1"]), "plan")


def test_a_desktop_sentinel_run_skips_msystem(world):
    module, build, check = world
    run_plan = _plan()
    run_plan["environment"].update(MSYSTEM=None, XDG_DATA_HOME="C:\\scratch\\desktop")
    run_plan["siril_launch"] = {"mode": "desktop-sentinel"}
    assert check(build(run_plan=run_plan))["passed"]
    run_plan = copy.deepcopy(run_plan)
    run_plan["siril_launch"] = {}
    assert any("MSYSTEM" in p for p in _failed(check(build(run_plan=run_plan)), "plan"))


def test_the_bench_session_bus_setting_is_hygiene_not_parity(world):
    # P1b review SIRIL-2: every bench run now disables Siril's session bus; P1a baselines have no value for it
    module, build, check = world
    run_plan = _plan()
    run_plan["environment"]["DBUS_SESSION_BUS_ADDRESS"] = "disabled:"
    assert check(build(run_plan=run_plan))["passed"]
    assert "DBUS_SESSION_BUS_ADDRESS" in module.HYGIENE_KEYS


def test_the_l_allowance_is_exact(world):
    module, build, check = world
    extra = _report()
    extra["allowed"]["{WF}/stack/job-<job#1>/siril.ini"] = {"verdict": "ALLOWED_DIFFERENT"}
    assert _failed(check(build(report=extra)), "compare")
    wrong = _report()
    wrong["allowed"]["{WF}/stack/job-<job#1>/02_stack.ssf"]["verdict"] = "ALLOWED_ONLY_IN_B"
    assert _failed(check(build(report=wrong)), "compare")
    broad = _report(allow_patterns=["*.ssf"])
    assert _failed(check(build(report=broad)), "compare")
    listed = _report(only_in_listed={"a": [], "b": ["{WF}/stack/job-<job#1>/02_export.log", "{WF}/x.log"]})
    assert _failed(check(build(report=listed)), "compare")
    assert check(build(report=listed), extra_listed=["b:{WF}/x.log"])["passed"]
    swapped = _report(a={"created_utc": "B"}, b={"created_utc": "A"})
    assert any("a = baseline" in p for p in _failed(check(build(report=swapped)), "compare"))
    failing = _report(**{"pass": False})
    assert _failed(check(build(report=failing)), "compare")


def test_the_scripts_are_path_3a_before_and_the_split_after(world):
    module, build, check = world
    combined_run = {"02_export.ssf": RUN_TEXTS["02_export.ssf"], "02_stack.ssf": BASE_TEXTS["02_stack.ssf"]}
    assert _failed(check(build(run_texts=combined_run)), "scripts")
    fallback_base = {"02_stack.ssf": _crlf(core.stack_script(None), PIN)}             # path 3c, not 3a
    assert _failed(check(build(base_texts=fallback_base)), "scripts")
    unpinned = {name: text.replace("setcpu 16\r\nset core.mem_mode=1\r\nset core.mem_amount=16\r\n", "")
                for name, text in RUN_TEXTS.items()}
    assert _failed(check(build(run_texts=unpinned)), "scripts")                       # the run's pin is in its script
    # a P1b baseline (split on both sides): no allowance, the same split texts
    split_report = _report(allowed={}, allow_patterns=[], only_in_listed={"a": [], "b": []})
    assert check(build(base_texts=RUN_TEXTS, report=split_report), baseline_layout="split")["passed"]


def test_f_and_r_allow_nothing(world):
    module, build, check = world
    plain = _report(allowed={}, allow_patterns=[], only_in_listed={"a": [], "b": []})
    f_notes = {"job_notes": _notes(coverage=False)}
    f_paths = build(mode="F", report=plain, run_kwargs=f_notes, base_texts={}, run_texts={})
    result = check(f_paths)
    assert result["passed"] and "scripts" not in result
    assert _failed(check(build(mode="F", run_kwargs=f_notes, base_texts={}, run_texts={})), "compare")
    assert _failed(check(build(mode="F", report=plain, base_texts={}, run_texts={})), "notes")   # a coverage note in F
    r_paths = build(mode="R", report=plain, base_texts={}, run_texts={})
    assert check(r_paths)["passed"]
    with pytest.raises(module.UsageError):
        check(r_paths, coverage=None)
    assert _failed(check(r_paths, coverage="workers=8,serial=true"), "notes")


def test_note_assertions(world):
    module, build, check = world
    layout = _notes()
    layout["l_siril_layout"] = "combined"
    assert any("l_siril_layout" in p for p in _failed(check(build(run_kwargs={"job_notes": layout})), "notes"))
    old = build(run_kwargs={"job_notes": _notes(version="aester-hashing-2")})
    assert any("version" in p for p in _failed(check(old), "notes"))
    no_misses = _notes()
    del no_misses["hash_ledger"]["largest_misses"]
    assert _failed(check(build(run_kwargs={"workflow_notes": no_misses})), "notes")
    no_governor = build(run_kwargs={"workflow_notes": _notes(governor=False, coverage=False)})
    assert any("governor" in p for p in _failed(check(no_governor), "notes"))
    paths = build()
    job = json.loads((paths[1] / "records" / "job.json").read_text(encoding="utf-8"))
    job["governor"] = {"leak": True}
    (paths[1] / "records" / "job.json").write_text(json.dumps(job), encoding="utf-8")
    assert any("/governor outside" in p for p in _failed(check(paths), "notes"))
    assert module.keys_outside_timings({"a": {"timings": {"governor": 1}}, "b": [{"frame_pool.coverage": 1}]}) == [
        "/b[0]/frame_pool.coverage"]


def test_siril_section(world):
    module, build, check = world
    assert any("3 processes" in p for p in _failed(check(build(run_kwargs={"descendants": 3})), "siril"))
    assert any("attempts" in p for p in _failed(check(build(run_kwargs={"attempts": [{"state": "failed"}]})), "siril"))
    result = check(build(run_kwargs={"job_state": "failed"}))
    assert result["passed"] is False and any("失败（3）" in p for p in result["siril"]["problems"])


def test_cli_exit_codes_and_json(world, tmp_path, capsys):
    module, build, _ = world
    base, run, compare = build()
    out = tmp_path / "result.json"
    common = ["--run", str(run), "--baseline", str(base), "--compare", str(compare)]
    assert module.main([*common, "--expect-coverage", "workers=8,serial=false,reason=none", "--json", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["passed"] is True
    assert module.main([*common, "--expect-coverage", "workers=4,serial=false"]) == 1
    assert module.main(common) == 2                                           # L needs the coverage expectation
    assert module.main([*common, "--expect-coverage", "threads=8"]) == 2
    assert module.main([*common, "--expect-coverage", "workers=8,serial=false,reason=none",
                        "--expect-env", "NOEQUALS"]) == 2
    capsys.readouterr()
