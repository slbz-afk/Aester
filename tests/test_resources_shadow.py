"""Governor shadow mode (efficiency P1b-2 §3.9, §4, T-S1..T-S9; cross-check X6, X11, X15; prereg §8 U-G4).

Shadow mode only reads: it never changes an environment, an affinity, a priority, a script line or a run_script
argument, never fails a job, and writes its note only under the scope timer's ``timings``.
"""
import ast
import copy
import ctypes
import dataclasses
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pytest

from starstack import core, profiling, resources, siril
from starstack import workflow as wf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_core_selected_inputs import admitted  # noqa: E402,F401  (fixture)
from test_workflow import setup, stub_model  # noqa: E402,F401  (fixtures)
from test_trail_default_and_extras import build  # noqa: E402
import verify_workflow_equivalence as wfeq  # noqa: E402

# Controlled engines here model Siril before the EFF-D4 stack-settings contract (efficiency P1c §4.2): jobs are
# admitted without it (tests/conftest.py pre_contract_jobs); contract jobs: tests/test_stack_settings.py.
pytestmark = pytest.mark.usefixtures("pre_contract_jobs")

WINDOWS = os.name == "nt"
LAPTOP = "total=15.4G,avail=9G,lps=16,cores=8,siril_pool=16,commit=24G,commit_avail=14G,io=nvme"
INVENTORY = re.compile(r'"(total_physical|available_physical|commit_limit|commit_available|pagefile|power|volumes|'
                       r'file_system|bus_type|seek_penalty|io_class|root_exe|detail|facts)"')


@pytest.fixture(autouse=True)
def clean_governor(monkeypatch):
    for name in (resources.MODE_ENV, resources.FACTS_ENV, siril.SIRIL_PIN_ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(resources, "_scope", None)
    yield
    resources._scope = None


def _walk(value, path=()):
    if isinstance(value, dict):
        for key, item in value.items():
            yield path + (key,), item
            yield from _walk(item, path + (key,))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk(item, path + (index,))


def governor_paths(document):
    return [path for path, _ in _walk(document) if path and path[-1] == "governor"]


def stored_bytes(note, depth):
    """The note's bytes as records.atomic_json stores it ``depth`` containers deep (indent 2, CRLF line ends as written
    on Windows), measured by actually nesting it -- independent of resources._note_size."""
    def text(value):
        for level in range(depth):
            value = {f"level{level}": value}
        return json.dumps(value, ensure_ascii=False, indent=2).replace("\n", "\r\n").encode("utf-8")
    return len(text(note)) - len(text(None)) + len("null")


# U-G4 (prereg section 8): distinctive raw system-memory values at every reader (the governor's snapshot and the peak
# sampler); none of them may appear anywhere in a receipt below shadow-full.
SYSTEM_MEMORY = {"total_physical": 34_359_738_383, "available_physical": 23_456_789_017,
                 "commit_limit": 51_539_607_557, "commit_available": 41_234_567_891}
SYSTEM_MEMORY_KEYS = {"total_physical", "available_physical", "commit_limit", "commit_available",
                      "min_system_available_bytes"}


@pytest.fixture
def system_memory(monkeypatch):
    real_snapshot, real_peak = resources.memory_snapshot, profiling.peak_reading

    def snapshot():
        return dataclasses.replace(real_snapshot(), **SYSTEM_MEMORY)

    def peak():                       # the sampler thread's reading (private, working set, system available)
        private, working = (real_peak() or (None, None, None))[:2]
        return private, working, SYSTEM_MEMORY["available_physical"]

    monkeypatch.setattr(resources, "memory_snapshot", snapshot)
    monkeypatch.setattr(profiling, "peak_reading", peak)
    return SYSTEM_MEMORY


def system_memory_leaks(document) -> list:
    """Raw system-memory values and keys anywhere in ``document`` (whole receipts, timings.resources.* included)."""
    text = json.dumps(document, ensure_ascii=False)
    leaks = [name for name, value in SYSTEM_MEMORY.items() if str(value) in text]
    leaks += [".".join(map(str, path)) for path, _ in _walk(document) if path and path[-1] in SYSTEM_MEMORY_KEYS]
    return leaks


def assert_only_under_timings(document, where):
    for path in governor_paths(document):
        assert any(part in ("timings", "timing_attempts") for part in path[:-1]), (where, path)
        assert "color_export" not in path, (where, path)


def run_job(admitted, out, mode, monkeypatch, facts=None):
    source, paths, root = admitted
    if mode is None:
        monkeypatch.delenv(resources.MODE_ENV, raising=False)
    else:
        monkeypatch.setenv(resources.MODE_ENV, mode)
    if facts:
        monkeypatch.setenv(resources.FACTS_ENV, facts)
    else:
        monkeypatch.delenv(resources.FACTS_ENV, raising=False)
    calls = []
    original = core.run_script

    def recording(executable, directory, name, commands, check, *rest, **kwargs):
        assert not rest and not kwargs                      # still exactly five positional arguments
        calls.append((Path(directory).name, name, list(commands), siril.siril_environment(directory, None)))
        return original(executable, directory, name, commands, check)

    monkeypatch.setattr(core, "run_script", recording)
    job = core.submit_job(None, str(root / out), input_files=paths[:3], transient_mode="off")
    core.run_job(job["job_dir"])
    monkeypatch.setattr(core, "run_script", original)
    result = core.load_job(job["job_dir"])
    assert result["state"] == "succeeded", result.get("error")
    return result, calls


def normalised(record):
    """A job record without timings and without what differs between two job folders of the same job."""
    job_dir = Path(record["job_dir"])
    text = json.dumps(record, sort_keys=True)
    # the folder names of two jobs of the same request (in any escaping), and their random product ids
    text = text.replace(job_dir.parent.name, "<OUT>").replace(job_dir.name, "<JOB>")
    text = re.sub(r"products(\\\\+)[0-9a-f]{12}", r"products\1<ID>", text)
    record = json.loads(text)
    drop = {"timings", "timing_attempts", "created_at", "updated_at", "worker_pid", "free_bytes_at_admission"}

    def clean(value):
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items() if k not in drop}
        if isinstance(value, list):
            return [clean(v) for v in value]
        return value

    return clean(record)


# --------------------------------------------------------------------------- T-S1 / T-S3 / T-S4: a whole job

def test_ts1_job_records_scripts_and_siril_environments_are_identical_in_every_mode(admitted, monkeypatch):
    results = {}
    for mode in ("off", "shadow", "shadow-full", None):
        results[mode] = run_job(admitted, f"m-{mode}", mode, monkeypatch,
                                facts=LAPTOP if mode == "shadow-full" else None)
    reference_record, reference_calls = results["off"]
    for mode, (record, calls) in results.items():
        assert normalised(record) == normalised(reference_record), mode
        assert [(n, c) for _, n, c, _ in calls] == [(n, c) for _, n, c, _ in reference_calls], mode
        for (_, _, _, env), (_, _, _, ref_env) in zip(calls, reference_calls):
            knobs = ("XDG_DATA_HOME", resources.MODE_ENV, resources.FACTS_ENV)      # the folder and the knob itself
            env = {k: v for k, v in env.items() if k not in knobs}
            ref_env = {k: v for k, v in ref_env.items() if k not in knobs}
            assert env == ref_env, mode
        notes = record["diagnostics"]["timings"].get("notes") or {}
        if mode == "off":
            assert "governor" not in notes
        else:
            note = notes["governor"]
            assert note["mode"] == (mode or "shadow") and note["errors"] == []
            # P1b-3 split layout: L exports and stacks in two Siril runs (X16)
            assert [d["name"] for d in note["decisions"] if d["kind"] == "siril"] == ["01_convert", "02_probe",
                                                                                      "02_export", "02_stack"]
            # X15: the copied job note, as stored at its deepest workflow.json copy (and at the current one, depth 8)
            depth = resources.NOTE_DEPTH[resources.JOB_SCOPE]
            assert stored_bytes(note, depth) == note["bytes"] <= resources.JOB_NOTE_CAP_BYTES
            assert stored_bytes(note, 8) < note["bytes"]
        assert_only_under_timings(record, mode)


def test_shadow_is_the_default_and_writes_no_machine_inventory(admitted, monkeypatch):
    """U-G4 (prereg §8): the product default writes recommendations and rounded budgets only."""
    record, _ = run_job(admitted, "default", None, monkeypatch)
    note = record["diagnostics"]["timings"]["notes"]["governor"]
    text = json.dumps(note)
    assert note["mode"] == "shadow" and note["level"] == "light" and "privacy" in note
    assert not INVENTORY.search(text), INVENTORY.search(text)
    budget = note["budget"]
    assert set(budget) == {"preset", "ram_total_gb", "commit_total_gb", "cpu_lps", "python_cpu_cores"}
    assert all(isinstance(budget[k], int) for k in ("ram_total_gb", "commit_total_gb"))      # whole GB
    decision = note["decisions"][0]
    assert isinstance(decision["ram_gb"], (int, float)) and decision["ram_gb"] == round(decision["ram_gb"])
    assert "src" not in decision["today"] and "res_ws_gb" in decision


@pytest.mark.parametrize("mode", [None, "shadow", "off", "bogus"])
def test_u_g4_no_raw_system_memory_anywhere_in_the_job_receipt(admitted, monkeypatch, system_memory, mode):
    """The whole job.json (timings.resources.* from the peak sampler, timing attempts, every note) carries no raw
    system RAM / commit value under the default 'shadow', 'off' or a malformed knob (off with a note)."""
    record, _ = run_job(admitted, f"priv-{mode}", mode, monkeypatch)
    timings = record["diagnostics"]["timings"]
    assert timings["resources"]["peak_sampler"]["hz"] == 10                      # the sampler did run
    assert all("local_peak_private_bytes" in view for view in timings["resources"]["phases"].values())
    assert system_memory_leaks(record) == []
    on_disk = json.loads(Path(record["job_dir"], "job.json").read_text(encoding="utf-8"))
    assert system_memory_leaks(on_disk) == []


def test_u_g4_shadow_full_writes_the_raw_values_the_scan_looks_for(admitted, monkeypatch, system_memory):
    """Positive control for the scans above: at shadow-full the sampler's minimum and the facts are written."""
    record, _ = run_job(admitted, "priv-full", "shadow-full", monkeypatch)
    phases = record["diagnostics"]["timings"]["resources"]["phases"]
    minima = {view["min_system_available_bytes"] for view in phases.values()}    # None: a phase no tick reached
    assert system_memory["available_physical"] in minima and minima <= {None, system_memory["available_physical"]}
    assert set(system_memory_leaks(record)) >= set(SYSTEM_MEMORY)


@pytest.mark.parametrize("mode", [None, "shadow", "off", "shadow-full"])
def test_u_g4_no_raw_system_memory_anywhere_in_the_workflow_receipts(setup, monkeypatch, system_memory, mode):
    """workflow.json, its stack snapshots and every other JSON record of the workflow and its stack job: no raw system
    RAM / commit value below shadow-full (the workflow's own timer runs the peak sampler too)."""
    if mode is None:
        monkeypatch.delenv(resources.MODE_ENV, raising=False)
    else:
        monkeypatch.setenv(resources.MODE_ENV, mode)
    record, _ = build(setup, monkeypatch, stars=True, save_starless=True)
    done = wf.run_workflow(record["workflow_dir"])
    assert done["state"] == "succeeded", done.get("error")
    assert done["timings"]["resources"]["peak_sampler"]["hz"] == 10
    documents = {}
    for folder in (Path(done["workflow_dir"]), Path(done["stack_job_dir"])):
        for path in sorted(folder.rglob("*.json")):
            documents[str(path)] = json.loads(path.read_text(encoding="utf-8"))
    assert any(Path(name).name == "workflow.json" for name in documents)
    leaks = {name: system_memory_leaks(document) for name, document in documents.items()}
    if mode == "shadow-full":
        workflow_leaks = leaks[str(Path(done["workflow_dir"]) / "workflow.json")]
        assert any(leak.endswith("min_system_available_bytes") for leak in workflow_leaks)
    else:
        assert {name: found for name, found in leaks.items() if found} == {}


def test_shadow_full_writes_the_facts_and_the_whatif_laptop(admitted, monkeypatch):
    record, _ = run_job(admitted, "full", "shadow-full", monkeypatch, facts=LAPTOP)
    note = record["diagnostics"]["timings"]["notes"]["governor"]
    assert note["mode"] == "shadow-full" and note["level"] == "full"
    assert note["whatif_budget"]["ram_total"] > 0
    siril_decisions = [d for d in note["decisions"] if d["kind"] == "siril"]
    assert all("whatif" in d for d in siril_decisions)
    export = next(d for d in siril_decisions if d["name"] == "02_export")
    stack = next(d for d in siril_decisions if d["name"] == "02_stack")
    view = resources.note_view      # a note over its stored-size cap writes the views as one-line JSON (GOV-6)
    # P1b-3 split layout (X16): the export run is cappable (route A), the stock rejection stack never changes (U6)
    assert export["parts"] == "probe+export" and view(export["rec"])["route"] == "A"
    assert stack["parts"] == "stack_rej" and view(stack["rec"])["route"] == "today"
    assert view(stack["today"])["src"] and view(stack["whatif"])["route"] == "today"
    assert note["bytes"] <= resources.JOB_NOTE_CAP_BYTES and note["dropped"] == 0
    assert "py.hash" in note["pools"]
    if WINDOWS and note.get("facts", {}).get("omitted") is None:
        assert note["facts"]["cpu"]["logical_total"] == os.cpu_count()


def test_ts4_governor_failures_never_fail_the_job(admitted, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("governor exploded")

    monkeypatch.setattr(resources, "siril_today", boom)
    monkeypatch.setattr(resources, "pool_workers", boom)
    record, _ = run_job(admitted, "boom", "shadow-full", monkeypatch)
    note = record["diagnostics"]["timings"]["notes"]["governor"]
    assert any("governor exploded" in error for error in note["errors"])
    monkeypatch.setattr(resources, "machine_facts", boom)
    record, _ = run_job(admitted, "boom2", "shadow", monkeypatch)
    note = record["diagnostics"]["timings"]["notes"]["governor"]
    assert note["errors"] and "governor exploded" in note["errors"][0]


@pytest.mark.parametrize("value", ["full", "on", "SHADOW-FULLY"])
def test_a_malformed_mode_is_off_with_a_note_and_the_job_runs(admitted, monkeypatch, value):
    record, _ = run_job(admitted, "bad", value, monkeypatch)
    note = record["diagnostics"]["timings"]["notes"]["governor"]
    assert note["mode"] == "off" and "AESTER_GOVERNOR" in note["errors"][0] and "decisions" not in note


def test_a_malformed_whatif_value_is_an_error_entry_only(admitted, monkeypatch):
    record, _ = run_job(admitted, "badfacts", "shadow-full", monkeypatch, facts="total=lots")
    note = record["diagnostics"]["timings"]["notes"]["governor"]
    assert any(error.startswith("whatif") for error in note["errors"]) and "whatif_budget" not in note
    assert [d["name"] for d in note["decisions"] if d["kind"] == "siril"]


def test_job_receipt_has_local_peaks_a_governor_operation_and_windows(admitted, monkeypatch):
    record, _ = run_job(admitted, "peaks", "shadow", monkeypatch)
    timings = record["diagnostics"]["timings"]
    phases = timings["resources"]["phases"]
    assert phases and all("local_peak_private_bytes" in view for view in phases.values())
    if WINDOWS:
        assert all(view["local_peak_private_bytes"] for view in phases.values())
    assert timings["resources"]["version"] == "process-resources-2"
    assert timings["operations_seconds"]["governor"] < 0.5 * max(1, timings["operation_counts"]["governor"])
    assert timings["operation_windows"]["governor"] and timings["resources"]["peak_sampler"]["hz"] == 10


# --------------------------------------------------------------------------- T-S2: worker counts unchanged

def test_ts2_render_workers_and_photo_are_identical_with_and_without_a_scope(tmp_path, monkeypatch):
    from test_photo_render import scene as scene_fixture  # noqa: F401
    import color_fixtures as cf
    from starstack import imaging, master_export as me, photo_render as pr
    camera, sky, _, _ = cf.synthetic_camera_master()
    dng = cf.write_reference_dng(tmp_path / "ref.dng")
    src = tmp_path / "master_linear.fit"
    imaging.write_linear(src, np.flipud(camera), "synthetic camera master")
    rec = me.export_master(src, me.colour_basis_from_dng(dng), tmp_path / "M")
    data, meta, _ = me.read_master(rec["master"])
    params = pr.default_params(meta, data, sky=sky)
    outputs = {}
    for mode in ("off", "shadow-full"):
        monkeypatch.setenv(resources.MODE_ENV, mode)
        timer = profiling.StageTimer()
        resources.shadow_start("test", timer=timer)
        outputs[mode] = pr.render_photo(rec["master"], tmp_path / mode, params)
        report = timer.finish()
        if mode == "shadow-full":
            pool = report["notes"]["governor"]["pools"]["py.render_rows"]
            assert pool["actual"] == outputs[mode]["render_workers"]
    a, b = outputs["off"], outputs["shadow-full"]
    assert a["render_workers"] == b["render_workers"]
    assert a["tiff_sha256"] == b["tiff_sha256"] and a["jpeg_sha256"] == b["jpeg_sha256"]


def test_ts2_transient_scan_workers_and_record_are_identical(tmp_path, monkeypatch):
    from test_transient_trails import make_dataset
    from starstack import transient_stage as ts
    records = {}
    for mode in ("off", "shadow-full"):
        monkeypatch.setenv(resources.MODE_ENV, mode)
        work = tmp_path / mode
        work.mkdir()
        data = make_dataset(work)
        for path in data["frames"][4:12]:
            path.unlink()
        inputs = [{"path": str(work / f"in_{i:05d}.fit"), "size": 1, "sha256": "0" * 64} for i in range(1, 14)]
        transforms = {f"light_{i:05d}": np.eye(3).tolist() for i in range(1, 14)}
        timer = profiling.StageTimer()
        resources.shadow_start("test", timer=timer)
        records[mode] = ts.detect_job(work, own_registration=False, transforms=transforms, model=None, inputs=inputs,
                                      reference_stem="light_00002", log=lambda *_: None, workers=2)
        report = timer.finish()
        if mode == "shadow-full":
            assert report["notes"]["governor"]["pools"]["py.transient_scan"]["actual"] == 2
    a, b = records["off"], records["shadow-full"]
    assert a["workers"] == b["workers"] == 2
    assert [c["id"] for c in a["candidates"]] == [c["id"] for c in b["candidates"]]
    assert a["frame_order"] == b["frame_order"]


# --------------------------------------------------------------------------- T-S3 / T-S5: workflow copies

def test_ts3_governor_notes_stay_under_timings_in_workflow_and_snapshot_copies(setup, monkeypatch):
    monkeypatch.setenv(resources.MODE_ENV, "shadow-full")
    big = {"version": resources.VERSION, "mode": "shadow-full", "decisions": [{"kind": "siril", "name": "01_convert"}]}
    record, state = build(setup, monkeypatch, stars=True, save_starless=True)
    original = wf.core.submit_job          # build()'s stub, which the workflow calls

    def submit(*args, **kwargs):            # the stack job's own note, as its worker writes it
        job = original(*args, **kwargs)
        job.setdefault("diagnostics", {}).setdefault("timings", {}).setdefault("notes", {})["governor"] = big
        wf.atomic_json(Path(job["job_dir"]) / "job.json", job)
        return job

    monkeypatch.setattr(wf.core, "submit_job", submit)
    done = wf.run_workflow(record["workflow_dir"])
    assert done["state"] == "succeeded", done.get("error")
    workflow_note = done["timings"]["notes"]["governor"]
    assert workflow_note["scope"] == "workflow" and workflow_note["errors"] == []
    assert any(d["kind"] == "nox" for d in workflow_note["decisions"])
    assert workflow_note["bytes"] <= resources.NOTE_CAP_BYTES
    assert_only_under_timings(done, "workflow.json")
    for snapshot in Path(done["workflow_dir"]).glob("stack_snapshot-*.json"):
        assert_only_under_timings(json.loads(snapshot.read_text(encoding="utf-8")), snapshot.name)
    stack_job = done["checkpoints"]["stack"]["data"]["job"]
    assert governor_paths(stack_job), "the job note travels with the checkpoint copy (under timings)"
    # X15 / GOV-6: the job note's copy as stored in the real workflow.json file is what resources measures at depth 8
    # (the current attempt; NOTE_DEPTH takes the one-deeper timing_attempts copy)
    path = Path(done["workflow_dir"]) / "workflow.json"
    raw = path.read_bytes()
    document = json.loads(raw.decode("utf-8"))
    newline = "\r\n" if b"\r\n" in raw else "\n"

    def dump(value):
        return json.dumps(value, ensure_ascii=False, indent=2).replace("\n", newline).encode("utf-8")

    assert dump(document) == raw                     # the file is exactly this serialization
    copy_note = document["checkpoints"]["stack"]["data"]["job"]["diagnostics"]["timings"]["notes"]["governor"]
    document["checkpoints"]["stack"]["data"]["job"]["diagnostics"]["timings"]["notes"]["governor"] = None
    in_file = len(raw) - len(dump(document)) + len("null")
    assert in_file <= resources._note_size(copy_note, 8) <= resources._note_size(copy_note, 9)
    if newline == "\r\n":
        assert in_file == resources._note_size(copy_note, 8) == stored_bytes(copy_note, 8)
    assert_only_under_timings(json.loads(Path(done["stack_job_dir"], "job.json").read_text(encoding="utf-8")), "job")


def _rewrite(path, mutate):
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    mutate(document)
    Path(path).write_text(json.dumps(document, indent=1), encoding="utf-8")


def test_ts5_runs_differing_only_in_governor_notes_pass_the_verifier(tmp_path):
    trees = {label: wfeq.build_synthetic_workflow(tmp_path / label, seed=seed, inputs=tmp_path / "inputs")
             for label, seed in (("a", 1), ("b", 2))}
    note = {"version": resources.VERSION, "mode": "shadow-full", "decisions": [{"kind": "siril", "name": "01_convert",
                                                                                 "today": {"threads": 64}}]}
    for label, tree in trees.items():
        paths = tree["paths"]
        job_path = paths["job_record"]

        def job_mutation(document):
            if label == "b":
                document["diagnostics"]["timings"]["notes"] = {"governor": note}

        _rewrite(job_path, job_mutation)
        job = json.loads(job_path.read_text(encoding="utf-8"))
        snapshots = list(Path(tree["workflow_dir"]).glob("stack_snapshot-*.json"))

        def snapshot_mutation(document):
            document["diagnostics"] = copy.deepcopy(job["diagnostics"])

        for snapshot in snapshots:
            _rewrite(snapshot, snapshot_mutation)

        def workflow_mutation(document):
            document["timings"] = {"version": "monotonic-stages-2", "notes": {"governor": note} if label == "b" else {}}
            document["timing_attempts"] = []
            stack = document["checkpoints"]["stack"]
            stack["data"]["job"] = copy.deepcopy(job)
            stack["data_sha256"] = wfeq._workflow_digest(stack["data"])
            stack["files"] = [wfeq._binding(Path(f["path"])) for f in stack["files"]]

        _rewrite(paths["workflow_record"], workflow_mutation)
    a = wfeq.fingerprint(trees["a"]["workflow_dir"], workers=2, block_bytes=1024)
    b = wfeq.fingerprint(trees["b"]["workflow_dir"], workers=2, block_bytes=1024)
    report = wfeq.compare(a, b)
    assert report["pass"], wfeq.format_report(report)
    data_a = json.loads(trees["a"]["paths"]["workflow_record"].read_text(encoding="utf-8"))["checkpoints"]["stack"]
    data_b = json.loads(trees["b"]["paths"]["workflow_record"].read_text(encoding="utf-8"))["checkpoints"]["stack"]
    assert data_a["data_sha256"] != data_b["data_sha256"]           # the digest differs; the verifier recomputes it


# --------------------------------------------------------------------------- T-S6..T-S9: scope mechanics

def _process_state():
    state = {"environ": dict(os.environ)}
    if WINDOWS:
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        k32.GetPriorityClass.argtypes = [ctypes.c_void_p]
        k32.GetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t),
                                               ctypes.POINTER(ctypes.c_size_t)]
        process, system = ctypes.c_size_t(), ctypes.c_size_t()
        k32.GetProcessAffinityMask(k32.GetCurrentProcess(), ctypes.byref(process), ctypes.byref(system))
        state.update(priority=k32.GetPriorityClass(k32.GetCurrentProcess()), affinity=process.value)
    return state


def test_ts6_shadow_changes_neither_environment_affinity_nor_priority(monkeypatch, tmp_path):
    monkeypatch.setenv(resources.MODE_ENV, "shadow-full")
    monkeypatch.setenv(resources.FACTS_ENV, LAPTOP)
    before = _process_state()
    timer = profiling.StageTimer(peak_sampler_hz=10)
    resources.shadow_start(resources.JOB_SCOPE, timer=timer, work_dir=tmp_path,
                           job={"inputs": [{"path": str(tmp_path / "x.dng")}] * 8,
                                "disk_estimate": {"frame_pixels": 1000}})
    resources.shadow_siril("01_convert", tmp_path, ["cd stage", "convertraw light -debayer -out=../work"], "siril")
    resources.shadow_pool("py.coverage", actual=8, items=8, frame_pixels=1000)
    resources.shadow_nox(actual_intra_op_threads=None)
    resources.machine_facts(level="full", paths=[("work", tmp_path)])
    resources.aester_resident(detail=True)
    timer.finish()
    after = _process_state()
    assert after == before
    assert list(tmp_path.iterdir()) == []                          # no file written


def test_ts7_a_second_start_resets_decisions_and_times_from_its_own_timer(monkeypatch, tmp_path):
    monkeypatch.setenv(resources.MODE_ENV, "shadow")
    now = [100.0]
    first = profiling.StageTimer(clock=lambda: now[0])
    resources.shadow_start("one", timer=first)
    resources.shadow_siril("a", tmp_path, ["cd work", "register light"], "siril")
    now[0] = 200.0
    second = profiling.StageTimer(clock=lambda: now[0])
    resources.shadow_start("two", timer=second)
    now[0] = 205.0
    resources.shadow_siril("b", tmp_path, ["cd work", "register light"], "siril")
    one, two = first.finish()["notes"]["governor"], second.finish()["notes"]["governor"]
    assert [d["name"] for d in one["decisions"]] == ["a"]
    assert [d["name"] for d in two["decisions"]] == ["b"] and two["decisions"][0]["at_s"] == 5.0
    assert resources.current_scope_name() is None                  # the scope's timer has finished
    resources.shadow_siril("late", tmp_path, ["register light"], "siril")       # no live scope: a no-op
    resources.shadow_pool("py.coverage", actual=1, items=1)


def test_no_scope_no_note_and_finished_timers_are_ignored(tmp_path):
    timer = profiling.StageTimer()
    resources.shadow_siril("x", tmp_path, ["register light"], "siril")
    assert "notes" not in timer.finish()


def test_ts8_note_stays_within_its_cap_keeping_siril_decisions(monkeypatch, tmp_path):
    monkeypatch.setenv(resources.MODE_ENV, "shadow-full")
    monkeypatch.setenv(resources.FACTS_ENV, LAPTOP)
    timer = profiling.StageTimer()
    resources.shadow_start(resources.JOB_SCOPE, timer=timer, job={"inputs": [{"path": "C:\\x.dng"}] * 48,
                                                                  "disk_estimate": {"frame_pixels": 45705600}})
    monkeypatch.setattr(resources, "aester_resident", lambda **_: {"ws": 10 ** 9, "commit": 2 * 10 ** 9, "members": 3})
    for index in range(5):
        resources.shadow_siril(f"s{index}", tmp_path, ["cd work", "seqapplyreg light -filter-included"], "siril")
    for index in range(495):
        resources.shadow_pool(("py.coverage", "py.render_rows", "py.transient_scan")[index % 3], actual=4, items=48,
                              width=8280)
    note = timer.finish()["notes"]["governor"]
    depth = resources.NOTE_DEPTH[resources.JOB_SCOPE]
    assert stored_bytes(note, depth) == note["bytes"] <= resources.JOB_NOTE_CAP_BYTES
    assert [d["name"] for d in note["decisions"]] == [f"s{i}" for i in range(5)]
    # GOV-5: a trimmed what-if keeps what bench_report's A6 rule needs (never a false failure)
    for decision in note["decisions"]:
        whatif = resources.note_view(decision["whatif"])
        assert set(whatif) >= {"route", "ram_gb", "commit_gb", "neutral", "over", "action", "c_gb", "w_gb"}
        assert whatif["route"] != "A" or whatif.get("mask")
    assert note["compact"] is True and note["dropped"] == 0          # lossless compaction first (GOV-6)
    assert sum(resources.note_view(entry)["calls"] for key, entry in note["pools"].items() if key != "py.hash") == 495
    # many Siril launches: the oldest are dropped and counted, the newest kept
    resources._scope = None
    timer = profiling.StageTimer()
    resources.shadow_start(resources.JOB_SCOPE, timer=timer)
    for index in range(500):
        resources.shadow_siril(f"t{index}", tmp_path, ["cd work", "seqapplyreg light -filter-included"], "siril")
    note = timer.finish()["notes"]["governor"]
    assert stored_bytes(note, depth) == note["bytes"] <= resources.JOB_NOTE_CAP_BYTES
    assert note["dropped"] > 0 and note["decisions"][-1]["name"] == "t499"
    assert note["dropped"] + len(note["decisions"]) == 500


def test_ts9_note_reaches_every_live_timer_and_attach_note_only_its_own():
    a, b = profiling.StageTimer(), profiling.StageTimer()
    profiling.note("hash_ledger", {"mode": "on"})
    a.attach_note("governor", {"x": 1, "bad": float("nan")})
    ra, rb = a.finish(), b.finish()
    assert ra["notes"]["hash_ledger"] == rb["notes"]["hash_ledger"] == {"mode": "on"}
    assert ra["notes"]["governor"] == {"x": 1, "bad": None} and "governor" not in rb["notes"]
    a.attach_note("late", 1)
    assert "late" not in a.finish()["notes"]


# --------------------------------------------------------------------------- X11: frozen closure unchanged

def _imports(path):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names.update(alias.name for alias in node.names)
            names.add(node.module or "")
        elif isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
    return names


def test_x11_profiling_and_process_guard_never_import_resources():
    for name in ("profiling", "process_guard"):
        names = _imports(ROOT / "src" / "starstack" / f"{name}.py")
        assert not any("resources" in n for n in names), name
    import verify_image_holdout as holdout
    manifest = holdout.source_manifest()
    assert not any(Path(name).stem == "resources" for name in manifest)


def test_resources_is_imported_only_from_its_call_sites():
    # imaging: the P1c C-1 median pool records its shadow_pool decision (prereg §6.2 (h)); it is not in the frozen
    # image-holdout closure (R22 keeps those modules' decisions in core)
    users = sorted(p.stem for p in (ROOT / "src" / "starstack").glob("*.py")
                   if p.stem != "resources" and "resources" in _imports(p))
    # P1c T-5 / T-5x pool sites record their shadow_pool decisions (prereg §6.2 (h)); performance.operation reads the
    # scope's timer (resources.scope_timer) where no timer is activated in the thread (the job worker)
    assert users == ["core", "imaging", "milkyway", "performance", "photo_render", "photographic_job", "region_blend",
                     "star_layer_export", "star_reduction", "transient_stage", "workflow", "workflow_edit"]
