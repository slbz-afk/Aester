"""scripts/bench_report.py governor join (efficiency P1b-2 A3-A6) on a synthetic run made by the real governor."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

from starstack import profiling, resources

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("bench_report_governor", REPO / "scripts" / "bench_report.py")
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)
LAPTOP = "total=15.4G,avail=9G,lps=16,cores=8,siril_pool=16,commit=24G,commit_avail=14G,io=nvme"
GB = 10 ** 9
LAUNCHES = (("01_convert", ["cd stage", "convertraw light -debayer -out=../work"], 50.0e9, 40e9),
            ("02_probe", ["cd work", "register light -2pass -transf=homography"], 30.0e9, 22e9),
            ("02_export", ["cd work", "seqapplyreg light -filter-included"], 95.0e9, 90e9),
            ("02_stack", ["cd work", "seqclean r_light -stat",
                          "stack r_light rej w 3 3 -norm=addscale -32b -filter-included -out=sky_master.fit"], 47e9, 45e9))


# P1c pool sites (tests/test_p1c_performance_resources.py): with LAUNCHES, enough to trim the pool entries (step 4b)
MANY_POOLS = ("py.median", "py.sky_zero", "py.native_mask", "py.strict_exports", "py.finite_write",
              "py.finite_inventory", "py.trail_crops", "py.addback_boxes", "py.transient_scan", "py.coverage",
              "py.milkyway_render", "py.star_reduction", "py.trail_pair")


def make_run(tmp_path, monkeypatch, launches=LAUNCHES, pools=()):
    monkeypatch.setenv(resources.MODE_ENV, "shadow-full")
    monkeypatch.setenv(resources.FACTS_ENV, LAPTOP)
    monkeypatch.setattr(resources, "_scope", None)
    timer = profiling.StageTimer(peak_sampler_hz=10)
    resources.shadow_start(resources.JOB_SCOPE, timer=timer, job={"inputs": [{"path": "C:\\x.dng"}] * 48,
                                                                  "disk_estimate": {"frame_pixels": 45705600}})
    for name, commands, commit, ws in launches:
        resources.shadow_siril(name, tmp_path, commands, "siril-cli.exe")
        with timer.operation("script:" + name):
            profiling.note_child_accounting({"peak_job_memory_bytes": int(commit),
                                             "child_peak_working_set_bytes": int(ws)})
    for stage in pools:
        resources.shadow_pool(stage, actual=4, items=48, width=8280)
    timings = timer.finish()
    resources._scope = None
    job = {"diagnostics": {"timings": timings}}
    summary = {"run_id": "20261001T000000Z-L48-test", "plan": {"slot": "L48", "pins": {}}, "state": "succeeded"}
    return {"folder": tmp_path, "summary": summary, "workflow": None, "job": job, "timeline": []}


@pytest.fixture
def run(tmp_path, monkeypatch):
    return make_run(tmp_path, monkeypatch)


def job_note(run):
    return run["job"]["diagnostics"]["timings"]["notes"]["governor"]


def with_whatif(note, name, **changes):
    """A copy of ``note`` whose decision ``name`` has its what-if view changed (either stored form)."""
    note = copy.deepcopy(note)
    decision = next(d for d in note["decisions"] if d["name"] == name)
    decision["whatif"] = {**report.note_view(decision["whatif"]), **changes}
    return note


def test_a4_joins_each_launch_with_its_measured_peak(run):
    view = report.governor(run)
    rows = {row["name"]: row for row in view["launches"]}
    assert set(rows) == {name for name, *_ in LAUNCHES}
    assert rows["01_convert"]["category"] == "a" and rows["02_export"]["category"] == "b"
    assert rows["02_stack"]["parts"] == "stack_rej"
    assert rows["01_convert"]["measured_gb"] == 50.0 and rows["01_convert"]["measured_ws_gb"] == 40.0
    for row in rows.values():
        assert row["verdict"] in ("pass", "miss") and row["ws_verdict"] in ("pass", "miss")
        if row["category"] == "a":
            assert row["central_error"] == pytest.approx(row["central_gb"] / row["measured_gb"] - 1, abs=1e-3)
        else:
            assert row["high_ratio"] == pytest.approx(row["high_gb"] / row["measured_gb"], abs=1e-3)


def test_a6_checks_the_whatif_rule(run):
    rows = {row["name"]: row for row in report.governor(run)["whatif"]}
    assert rows["01_convert"]["route"] == "A" and rows["01_convert"]["verdict"] == "pass"
    assert rows["02_stack"]["route"] == "today" and rows["02_stack"]["planned_action"] == "needs_U6"
    assert all(row["verdict"] == "pass" for row in rows.values()), rows


def test_a3_a5_and_the_markdown_section(run):
    view = report.governor(run)
    assert view["note_bytes"]["job"] <= resources.JOB_NOTE_CAP_BYTES and view["errors"] == []
    assert view["operation_seconds"]["job"] is not None and view["sampler"]["job"]["phases_without_local_peak"] == []
    group = {"slot": "L48", "variant": "test", "runs": [{"run_id": run["summary"]["run_id"], "governor": view}],
             "stages": [], "phases": [], "operations": []}
    text = "\n".join(report.governor_markdown([group]))
    assert "### A4" in text and "### A6" in text and "01_convert" in text and "needs_U6" in text


def test_runs_without_a_governor_note_have_no_section():
    old = {"summary": {"run_id": "x", "plan": {}}, "workflow": None, "job": {"diagnostics": {"timings": {}}}}
    assert report.governor(old) is None
    assert report.governor_markdown([{"runs": [{"run_id": "x", "governor": None}]}]) == []


# --------------------------------------------------------------------------- review fixes GOV-4 / GOV-5 / GOV-6

def test_a6_mask_rule_fires_on_sibling_packing_and_on_more_than_64_lps(run):
    """GOV-4: the what-if CPU is in the note (whatif_facts.cpu), so the mask rule can fail."""
    note = job_note(run)
    assert note["whatif_facts"]["cpu"] == {"logical_total": 16, "physical_cores": 8}
    assert report.whatif_cpu(note) == (16, 8)
    rows = {row["name"]: row for row in report.a6_whatif(with_whatif(note, "01_convert", mask="0x3"))}
    assert "mask packs SMT siblings" in rows["01_convert"]["verdict"]          # LPs 0 and 1: both on core 0
    assert rows["02_probe"]["verdict"] == "pass"
    wide = hex((1 << 64) | 0x5555)
    rows = {row["name"]: row for row in report.a6_whatif(with_whatif(note, "01_convert", mask=wide))}
    assert "exceeds 64" in rows["01_convert"]["verdict"]
    spread = {row["name"]: row for row in report.a6_whatif(with_whatif(note, "01_convert", mask="0x5"))}
    assert spread["01_convert"]["verdict"] == "pass"                          # LPs 0 and 2: two cores


def test_a6_mask_rule_reads_the_whatif_text_of_older_notes(run):
    note = with_whatif(job_note(run), "01_convert", mask="0x3")
    note["whatif_facts"] = {"level": "whatif", "text": LAPTOP}                 # no cpu block (written before the fix)
    assert report.whatif_cpu(note) == (16, 8)
    rows = {row["name"]: row for row in report.a6_whatif(note)}
    assert "mask packs SMT siblings" in rows["01_convert"]["verdict"]
    note["whatif_facts"] = {"omitted": "size cap"}                            # nothing to read: no mask verdict
    assert report.whatif_cpu(note) == (None, None)
    assert {row["name"]: row for row in report.a6_whatif(note)}["01_convert"]["verdict"] == "pass"


def test_a6_reports_no_data_for_a_whatif_trimmed_before_the_fix(run):
    """GOV-5: an old trimmed what-if (route, t, c_gb, w_gb, over, action) is 'no data', never a failure."""
    note = copy.deepcopy(job_note(run))
    for decision in note["decisions"]:
        view = report.note_view(decision.get("whatif"))
        if view:
            decision["whatif"] = {k: view.get(k) for k in ("route", "t", "c_gb", "w_gb", "over", "action")}
    rows = report.a6_whatif(note)
    assert rows and all(row["verdict"].startswith("no data (missing ") for row in rows), rows
    assert "neutral" in rows[0]["verdict"] and "ram_gb" in rows[0]["verdict"]


def test_a_note_trimmed_to_its_cap_keeps_what_a6_needs(tmp_path, monkeypatch):
    """GOV-5: with a cap small enough to trim the what-if and rec detail, every launch is kept and A6 sees no false
    failure (ram_gb, commit_gb, neutral and mask survive the trim)."""
    full = job_note(make_run(tmp_path / "full", monkeypatch))       # compacted, every view complete
    assert full["compact"] is True and full["dropped"] == 0
    cap = full["bytes"] - 200                                         # step 4 (about 500 bytes here) must run
    monkeypatch.setattr(resources, "JOB_NOTE_CAP_BYTES", cap)
    trimmed = make_run(tmp_path / "trimmed", monkeypatch)
    note = job_note(trimmed)
    assert note["compact"] is True and note["dropped"] == 0 and note["bytes"] <= cap
    assert [d["name"] for d in note["decisions"]] == [name for name, *_ in LAUNCHES]
    for decision in note["decisions"]:
        whatif = report.note_view(decision["whatif"])
        assert "omp" not in whatif and set(resources.WHATIF_TRIMMED_KEYS) - {"mask"} <= set(whatif)   # step 4 ran
        assert set(report.note_view(decision["rec"])) <= set(resources.REC_TRIMMED_KEYS)
    rows = {row["name"]: row for row in report.governor(trimmed)["whatif"]}
    assert all(row["verdict"] == "pass" for row in rows.values()), rows
    assert rows["01_convert"]["mask"] and rows["02_stack"]["planned_action"] == "needs_U6"


def test_a4d_reads_a_trimmed_pool_entry_with_its_dropped_defaults(tmp_path, monkeypatch):
    """P1c-0 step 4b drops calls 1 / over_budget 0 / planned_action "run" from the pool entries of a note over its cap;
    the A4 (d) table shows those values again, never "None"."""
    assert report.POOL_TRIM_DEFAULTS == resources.POOL_TRIM_DEFAULTS
    run = make_run(tmp_path, monkeypatch, pools=MANY_POOLS)
    note = job_note(run)
    assert note.get("pools_trimmed") is True and note["dropped"] == 0 and set(MANY_POOLS) <= set(note["pools"])
    stored = {stage: resources.note_view(entry) for stage, entry in note["pools"].items()}
    assert any("calls" not in entry for entry in stored.values())             # step 4b ran on this note
    assert any("planned_action" not in entry for entry in stored.values())
    view = report.governor(run)
    for stage in note["pools"]:
        entry, kept = view["pools"][f"job:{stage}"], stored[stage]
        assert entry == resources.pool_view(note["pools"][stage], trimmed=True)
        assert entry["calls"] == kept.get("calls", 1) and entry["over_budget"] == kept.get("over_budget", 0)
        assert entry["planned_action"] == kept.get("planned_action", "run")
        assert {"actual", "recommended", "whatif"} <= set(entry)
    group = {"slot": "F48", "variant": "test", "runs": [{"run_id": "r", "governor": view}],
             "stages": [], "phases": [], "operations": []}
    lines = report.governor_markdown([group])
    start = lines.index("### A4 (d): Python pools, actual vs recommended (predictions are [est] rows; not gated)")
    rows = [line for line in lines[start:] if line.startswith("| r | job:")]
    assert len(rows) == len(note["pools"]) and not any("None" in line for line in rows), rows
    cells = {line.split(" | ")[1]: line.strip("| ").split(" | ") for line in rows}
    for stage in note["pools"]:
        assert cells[f"job:{stage}"][2] == str(view["pools"][f"job:{stage}"]["calls"])
        assert cells[f"job:{stage}"][-1] == view["pools"][f"job:{stage}"]["planned_action"]


def test_pool_view_fills_defaults_only_in_a_trimmed_note():
    entry = {"actual": 4, "recommended": 2, "whatif": "2 run"}
    assert resources.pool_view(entry) == entry and resources.pool_view(json.dumps(entry)) == entry
    assert resources.pool_view(entry, trimmed=True) == {**entry, "calls": 1, "over_budget": 0, "planned_action": "run"}
    kept = {**entry, "calls": 3, "planned_action": "needs_U6"}
    assert resources.pool_view(kept, trimmed=True) == {**kept, "over_budget": 0}
    assert resources.pool_view("not json", trimmed=True) == {} and "calls" not in entry          # a copy, never the entry
    for view in (resources.pool_view, report.pool_view):
        assert view(json.dumps(kept), True) == resources.pool_view(kept, trimmed=True)


def test_note_bytes_are_the_size_as_stored_in_workflow_json(run):
    """GOV-6: ``bytes`` is the note's own size at its deepest workflow.json copy (indent 2 at depth 9, CRLF)."""
    note = job_note(run)

    def stored(value, depth):
        for level in range(depth):
            value = {f"level{level}": value}
        return len(json.dumps(value, ensure_ascii=False, indent=2).replace("\n", "\r\n").encode("utf-8"))

    depth = resources.NOTE_DEPTH[resources.JOB_SCOPE]
    assert stored(note, depth) - stored(None, depth) + len("null") == note["bytes"] <= resources.JOB_NOTE_CAP_BYTES
    group = {"slot": "L48", "variant": "test", "runs": [{"run_id": "r", "governor": report.governor(run)}],
             "stages": [], "phases": [], "operations": []}
    assert "as stored at the note's deepest copy" in "\n".join(report.governor_markdown([group]))
