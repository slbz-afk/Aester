"""scripts/bench_report.py on synthetic result folders: stage splitting, phase nesting, repeat grouping."""
import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "bench_report.py"
spec = importlib.util.spec_from_file_location("bench_report_tested", SCRIPT)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)
GB = 10 ** 9


def timings(phases, resources=None, host=None):
    return {"version": "monotonic-stages-2", "total_seconds": sum(phases.values()), "phases_seconds": phases,
            "operations_seconds": {}, "resources": {"phases": resources or {}, "total": {"hashed_bytes": 2 * GB}},
            "host": host or {}}


def make_run(root, run_id, final, label="R1a", state="succeeded"):
    folder = root / "results" / run_id
    (folder / "records").mkdir(parents=True)
    summary = {"run_id": run_id, "state": state,
               "plan": {"slot": "L20", "mode": "L", "frames": 20, "label": label, "pins": {"siril": None},
                        "code_src": "src"},
               "wall": {"submit_to_base_s": final - 10, "submit_to_final_s": final, "addback_s": 10},
               "sampler": {"tree_cpu_s": 100.0, "peak_tree_private_bytes": 3 * GB, "min_sys_avail_bytes": 9 * GB},
               "job_accounting": {"peak_job_commit_bytes": 4 * GB}, "fingerprint": {"status": "ok"},
               "trail_review": {"kept_count": 1}}
    (folder / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    workflow = {"timings": timings({"stack_wait": 50.0, "publish.stage": 5.0},
                                   {"stack_wait": {"cpu_seconds": 1.0, "hashed_bytes": GB}},
                                   host={"sha256_rate_gbps": 2.0, "sha_ext_guess": True})}
    job = {"diagnostics": {"timings": {**timings({"convert": 30.0, "registration_stack": 19.0},
                                                 {"registration_stack": {"cpu_seconds": 0.0,
                                                                         "children": {"cpu_seconds": 400.0,
                                                                                      "processes": 1}}}),
                                       "operations_seconds": {"script:02_stack": 19.0},
                                       "operation_counts": {"script:02_stack": 1}}}}
    (folder / "records" / "workflow.json").write_text(json.dumps(workflow), encoding="utf-8")
    (folder / "records" / "job.json").write_text(json.dumps(job), encoding="utf-8")
    rows = [(0.0, None, 0, 0), (1.0, "running/stack", 1, 1), (41.0, "running/stack", 201, 11),
            (51.0, "running/publish", 211, 21), (60.0, "succeeded/complete", 220, 30)]
    with (folder / "timeline.jsonl").open("w", encoding="utf-8") as fh:
        for t, stage, cpu, read in rows:
            fh.write(json.dumps({"t": t, "stage": stage, "tree_cpu_s": cpu, "io_read": read * GB, "io_write": 0,
                                 "tree_private": (2 if stage == "running/stack" else 1) * GB, "tree_ws": GB,
                                 "sys_avail": 10 * GB}) + "\n")
    return folder


def test_stage_split_nesting_and_grouping(tmp_path):
    a = make_run(tmp_path, "20260928T000000Z-L20-R1a", 60.0)
    b = make_run(tmp_path, "20260928T001000Z-L20-R1b", 70.0, label="R1b")
    failed = make_run(tmp_path, "20260928T002000Z-L20-R1c", 80.0, label="R1c", state="failed")
    groups = report.build([a, b, failed])
    assert len(groups) == 1 and len(groups[0]["runs"]) == 2          # failed run skipped, repeats grouped
    stages = {r["stage"]: r for r in groups[0]["stages"]}
    mean = lambda value: value["mean"]                               # two runs: mean (min-max)
    assert mean(stages["stack"]["wall_s"]) == 50.0 and mean(stages["stack"]["cpu_s"]) == 210.0
    assert mean(stages["stack"]["peak_private_gb"]) == 2.0 and mean(stages["publish"]["io_read_gb"]) == 9.0
    names = [(r["scope"], r["name"], r["depth"]) for r in groups[0]["phases"]]
    assert names == [("workflow", "stack_wait", 0), ("job", "convert", 1), ("job", "registration_stack", 1),
                     ("workflow", "publish.stage", 1)]
    job_stack = groups[0]["phases"][2]
    assert job_stack["child_cpu_s"] == {"mean": 400.0, "min": 400.0, "max": 400.0}
    finals = [o["submit_to_final_s"] for o in groups[0]["runs"]]
    assert finals == [60.0, 70.0]
    text = report.markdown(groups)
    assert "L20 · product defaults (2 runs; mean (min–max))" in text and "job:registration_stack" in text


def test_label_resolution_and_outputs(tmp_path):
    make_run(tmp_path, "20260928T000000Z-L20-R1a", 60.0)
    out, js = tmp_path / "t.md", tmp_path / "t.json"
    assert report.main(["--bench-root", str(tmp_path), "--label", "L20-R1a", "-o", str(out), "--json", str(js)]) == 0
    assert "| 20260928T000000Z-L20-R1a | L20 |" in out.read_text(encoding="utf-8")
    assert json.loads(js.read_text(encoding="utf-8"))[0]["slot"] == "L20"


def test_p1_hashing_columns_include_saved_reads_and_the_submitter(tmp_path):
    """Efficiency P1 H4: real reads of workers and submitter, saved reads by kind, the ledger notes."""
    folder = make_run(tmp_path, "20260929T000000Z-L20-P1a", 60.0, label="P1a")
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    summary["submitter_hashing"] = {"hashed_bytes": GB // 2, "hashed_files": 20, "hash_saved_bytes": GB,
                                    "hash_saved_ledger_bytes": GB, "hashed_memory_bytes": 0}
    (folder / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    workflow = json.loads((folder / "records" / "workflow.json").read_text(encoding="utf-8"))
    total = workflow["timings"]["resources"]["total"]
    total.update(hash_saved_bytes=10 * GB, hash_saved_ledger_bytes=6 * GB, hash_saved_handover_bytes=3 * GB,
                 hash_saved_write_bytes=GB, hash_saved_batch_bytes=0, hashed_memory_bytes=GB)
    workflow["timings"]["resources"]["phases"]["stack_wait"].update(hash_saved_bytes=3 * GB, hashed_memory_bytes=0)
    workflow["timings"]["notes"] = {"hash_ledger": {"mode": "shadow", "shadow": {"checks": 40, "mismatches": 0,
                                                                                 "concurrent": 1}}}
    (folder / "records" / "workflow.json").write_text(json.dumps(workflow), encoding="utf-8")
    job = json.loads((folder / "records" / "job.json").read_text(encoding="utf-8"))
    job["diagnostics"]["timings"]["resources"]["total"].update(hash_saved_bytes=2 * GB, hash_saved_write_bytes=2 * GB)
    job["diagnostics"]["timings"]["notes"] = {"hash_ledger": {"mode": "shadow", "shadow": {"checks": 7, "mismatches": 0,
                                                                                           "concurrent": 0}}}
    (folder / "records" / "job.json").write_text(json.dumps(job), encoding="utf-8")
    old = make_run(tmp_path, "20260928T000000Z-L20-R1a", 60.0)          # a P0 run: no P1 fields
    groups = report.build([folder, old], group_by="run")
    runs = {g["runs"][0]["run_id"]: g["runs"][0]["hashing"] for g in groups}
    new, before = runs["20260929T000000Z-L20-P1a"], runs["20260928T000000Z-L20-R1a"]
    assert new["workers_real_gb"] == 4.0 and new["submitter_real_gb"] == 0.5 and new["total_real_gb"] == 4.5
    assert new["saved_gb"] == 13.0 and new["saved_ledger_gb"] == 7.0 and new["saved_handover_gb"] == 3.0
    assert new["saved_write_gb"] == 3.0 and new["saved_batch_gb"] == 0.0 and new["memory_gb"] == 1.0
    assert new["ledger_modes"] == {"workflow": "shadow", "job": "shadow"}
    assert (new["shadow_checks"], new["shadow_mismatches"], new["shadow_concurrent"]) == (47, 0, 1)
    assert before["submitter_included"] is False and before["total_real_gb"] == 4.0 and before["saved_gb"] is None
    assert before["ledger_modes"] is None and before["shadow_checks"] is None
    phases = {(r["scope"], r["name"]): r for r in groups[[g["runs"][0]["run_id"] for g in groups].index(
        "20260929T000000Z-L20-P1a")]["phases"]}
    assert phases[("workflow", "stack_wait")]["saved_gb"] == 3.0
    text = report.markdown(groups)
    assert "## Hashing (efficiency P1, H4)" in text and "| 20260929T000000Z-L20-P1a | 4.0 | 0.5 | 4.5 | 13.0 |" in text
    assert "| phase | wall s | own CPU s | child CPU s | read GB | write GB | child read GB | child write GB | " \
           "hashed GB | saved GB | memory GB |" in text


def test_o1_rows_show_the_scan_the_join_wait_and_the_window(tmp_path):
    """Efficiency P1c O-1 (prereg §3.6): the overlap.o1 note, the window native_remap_support + diagnostics +
    transient_scan, the transient_detect operation (the scan) and the join wait (transient_join); runs before O-1
    (no note) show no row."""
    on = make_run(tmp_path, "20261006T000000Z-L48-O1on", 60.0, label="O1on")
    off = make_run(tmp_path, "20261006T001000Z-L48-O1off", 70.0, label="O1off")
    old = make_run(tmp_path, "20260928T000000Z-L20-R1a", 60.0)
    for folder, note, phases, operations in (
            (on, {"schema": "overlap-o1-1", "eligible": True, "reason": None, "outcome": "joined", "n_o1": 8,
                  "n_serial": 8, "cov_plan": 8, "crop_cap": 8, "b_done_at_join": True, "join_wait_s": 0.0},
             {"native_remap_support": 37.5, "diagnostics": 0.1, "transient_scan": 0.4},
             {"transient_detect": 24.2, "transient_join": 0.38}),
            (off, {"schema": "overlap-o1-1", "eligible": False, "reason": "overlap_off"},
             {"native_remap_support": 37.0, "diagnostics": 0.1, "transient_scan": 25.0},
             {"transient_detect": 24.6})):
        job = json.loads((folder / "records" / "job.json").read_text(encoding="utf-8"))
        t = job["diagnostics"]["timings"]
        t["phases_seconds"].update(phases)
        t["operations_seconds"].update(operations)
        t["resources"]["operations"] = {"transient_detect": {"children": {"cpu_seconds": 172.0}}}
        t["notes"] = {"overlap.o1": note}
        (folder / "records" / "job.json").write_text(json.dumps(job), encoding="utf-8")
    groups = report.build([on, off, old], group_by="run")
    views = {g["runs"][0]["run_id"]: g["runs"][0]["overlap"] for g in groups}
    assert views["20260928T000000Z-L20-R1a"] is None
    view = views["20261006T000000Z-L48-O1on"]
    assert view["eligible"] is True and view["outcome"] == "joined" and (view["n_o1"], view["n_serial"]) == (8, 8)
    assert view["window_s"] == 38.0 and view["transient_detect_s"] == 24.2 and view["join_wait_s"] == 0.38
    assert view["transient_detect_child_cpu_s"] == 172.0
    serial = views["20261006T001000Z-L48-O1off"]
    assert serial["reason"] == "overlap_off" and serial["window_s"] == 62.1 and serial["join_wait_s"] is None
    text = report.markdown(groups)
    assert "## O-1: trail scan beside the post-stack work" in text
    assert "| 20261006T000000Z-L48-O1on | True | – | joined | 8 / 8 | 8 | 8 | 37.5 | 0.1 | 0.4 | 38.0 | 24.2 |" in text
    assert "| 20261006T001000Z-L48-O1off | False | overlap_off | – |" in text
    assert "| transient_detect |" in text and "| transient_join |" in text
