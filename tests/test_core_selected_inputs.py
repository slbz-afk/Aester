"""Explicit selections, opt-in transient work and monotonic attempt receipts."""
from pathlib import Path
from types import SimpleNamespace
import json
import shutil

import numpy as np
import pytest

from starstack import core, imaging, transient_stage as ts, transient_revision
from starstack.profiling import StageTimer


@pytest.fixture
def admitted(tmp_path, monkeypatch, pre_contract_jobs):
    # pre_contract_jobs: this stand-in Siril models the scripts before the EFF-D4 stack-settings contract
    source = tmp_path / "sources"
    source.mkdir()
    paths = []
    for i in range(4):
        path = source / f"frame-{i}.fit"
        image = np.full((16, 24, 3), .02 + .001 * i, dtype=np.float32)
        image[3:6, 5:8] = .3
        imaging.write_linear(path, image, "Controlled numerical exposure")
        paths.append(str(path))
    engine = {"available": True, "executable": "test-engine", "version": "1.4.4", "message": "controlled"}
    monkeypatch.setattr(core, "doctor", lambda *_: engine)
    monkeypatch.setattr(core, "_spawn_worker", lambda *_: None)
    monkeypatch.setattr(core.shutil, "disk_usage", lambda *_: SimpleNamespace(free=10**12))

    def run(executable, directory, name, commands, check):
        check()
        work = directory / "work"
        if name == "01_convert":
            for i, path in enumerate(sorted((directory / "stage").glob("*")), 1):
                shutil.copyfile(path, work / f"light_{i:05d}.fit")
        elif name == "02_probe":
            pass
        elif name == "02_export":   # P1b-3: L's export process (this stand-in stacks the light frames directly)
            assert list(commands) == core.export_script(None), commands
        elif name == "02_stack":
            assert list(commands) == list(core.LEGACY_STACK_SCRIPT), commands
            pixels = np.mean([imaging.read_rgb(p) for p in sorted(work.glob("light_*.fit"))], axis=0)
            imaging.write_linear(work / "sky_master.fit", pixels.astype(np.float32), "controlled stack")
        else:
            raise AssertionError(name)
    monkeypatch.setattr(core, "run_script", run)
    return source, paths, tmp_path / "out"


def test_explicit_order_and_mixed_parents_do_not_scan_siblings(admitted, tmp_path):
    source, paths, _ = admitted
    other = tmp_path / "second"
    other.mkdir()
    moved = other / "same.fit"
    shutil.copyfile(paths[1], moved)
    selection = [paths[3], str(moved), paths[0]]
    (other / "broken.nef").write_bytes(b"unsupported sibling ignored")
    result = core.inspect_sources(None, input_files=selection)
    assert result["files"] == selection
    assert result["selection_mode"] == "files"
    assert result["input_dir"] == str(source)
    assert result["input_roots"] == [str(source), str(other)]
    assert result["count"] == 3
    assert core.inspect_sources(None, input_files=selection[:1], min_count=1)["count"] == 1


@pytest.mark.parametrize("selection", [[], "not-a-list", {}, [None], [False], [""]])
def test_rejects_invalid_selection_container(admitted, selection):
    with pytest.raises(ValueError):
        core.inspect_sources(None, input_files=selection)


def test_rejects_duplicate_aliases_unsupported_and_mixed_kinds(admitted, tmp_path):
    source, paths, _ = admitted
    for selected in ([paths[0], paths[1], str(source / "." / "frame-0.fit")],
                     [paths[0], paths[1], str(source / "missing.fit")]):
        with pytest.raises(ValueError):
            core.inspect_sources(None, input_files=selected)
    raw = tmp_path / "a.nef"
    raw.write_bytes(b"RAW admission only")
    with pytest.raises(ValueError, match="混合"):
        core.inspect_sources(None, input_files=[paths[0], paths[1], str(raw)])


@pytest.mark.parametrize("minimum", [0, 301, True, 1.5, "1"])
def test_preview_minimum_strict(admitted, minimum):
    with pytest.raises(ValueError):
        core.inspect_sources(None, input_files=admitted[1], min_count=minimum)


def test_output_cannot_be_nested_in_any_selected_parent(admitted, tmp_path):
    source, paths, _ = admitted
    other = tmp_path / "other"
    other.mkdir()
    path = other / "b.fit"
    shutil.copyfile(paths[0], path)
    with pytest.raises(ValueError, match="所有素材"):
        core.submit_job(None, str(other / "results"), input_files=[paths[1], paths[2], str(path)], transient_mode="off")
    assert not (other / "results").exists()


def test_idempotency_binds_order_and_mode(admitted):
    source, paths, out = admitted
    job = core.submit_job(None, str(out), input_files=paths[:3], transient_mode="off", idempotency_key="one")
    assert job["input_selection"]["files"] == paths[:3]
    assert core.submit_job(None, str(out), input_files=paths[:3], transient_mode="off", idempotency_key="one")["job_dir"] == job["job_dir"]
    for options in ({"input_files": paths[2::-1], "transient_mode": "off"},
                    {"input_files": paths[:3], "transient_mode": "review"}):
        with pytest.raises(ValueError, match="不同素材"):
            core.submit_job(None, str(out), idempotency_key="one", **options)


@pytest.mark.parametrize("mode", [True, 0, {}, "", "on", ["off"]])
def test_mode_strict_validation(admitted, mode):
    with pytest.raises(ValueError, match="transient_mode"):
        core.submit_job(str(admitted[0]), str(admitted[2]), transient_mode=mode)


def test_off_never_scans_or_exports_and_selected_list_stays_frozen(admitted, monkeypatch):
    source, paths, out = admitted
    selected = [paths[3], paths[0], paths[2]]
    job = core.submit_job(None, str(out), input_files=selected, transient_mode="off")
    monkeypatch.setattr(ts, "detect_job", lambda *a, **k: pytest.fail("off mode scanned"))
    monkeypatch.setattr(core, "_export_transients", lambda *a, **k: pytest.fail("off mode exported"))
    shutil.copyfile(paths[0], source / "new-after-admission.fit")
    core.run_job(job["job_dir"])
    result = core.load_job(job["job_dir"])
    assert result["state"] == "succeeded", result.get("error")
    assert [p["path"] for p in result["inputs"]] == selected
    assert result["diagnostics"]["input_count"] == 3
    assert "transients" not in result["checkpoints"]
    assert result["diagnostics"]["transients"]["status"] == "skipped"
    assert result["diagnostics"]["transients"]["reason"] == "not_requested"
    assert result["transient_revision"] is None
    assert not any(a.get("group") == "transients" for a in result["artifacts"])
    timing = result["diagnostics"]["timings"]
    assert timing["state"] == "succeeded"
    assert sum(timing["phases_seconds"].values()) == pytest.approx(timing["total_seconds"])
    assert timing["operation_counts"]["script:01_convert"] == 1


def test_review_publishes_candidates_but_does_not_export(admitted, monkeypatch):
    source, paths, out = admitted
    calls = []
    def detect(work, **kwargs):
        calls.append(kwargs)
        record = {"version": "test", "candidates": [], "frames_scanned": 3, "seconds": .25,
                  "summary": {"candidates": 0}}
        ts.tt.write_candidates(work / ts.TRANSIENTS_DIR, record)
        return record
    monkeypatch.setattr(ts, "detect_job", detect)
    monkeypatch.setattr(core, "_export_transients", lambda *a, **k: pytest.fail("review mode exported without confirmation"))
    job = core.submit_job(None, str(out), input_files=paths[:3], transient_mode="review")
    core.run_job(job["job_dir"])
    result = core.load_job(job["job_dir"])
    assert result["state"] == "succeeded", result.get("error")
    assert len(calls) == 1
    diag = result["diagnostics"]["transients"]
    assert diag["status"] == "needs_review" and diag["composite"] is None
    assert "尚未加回" in result["message"]
    assert not any(a.get("group") == "transients" for a in result["artifacts"])
    record = transient_revision.read_record(Path(job["job_dir"]) / "work", job=result)
    assert record["revision_id"] == result["transient_revision"]["id"]


@pytest.mark.parametrize("change", ["mode", "order", "options", "missing_options"])
def test_resume_rejects_mutated_new_request(admitted, change):
    _, paths, out = admitted
    job = core.submit_job(None, str(out), input_files=paths[:3], transient_mode="off")
    directory = Path(job["job_dir"])
    data = json.loads((directory / "job.json").read_text(encoding="utf-8"))
    data["state"] = "cancelled"
    if change == "mode":
        data["transient_mode"] = "legacy"
    elif change == "order":
        data["inputs"].reverse()
    elif change == "options":
        data["transient_mode"] = "legacy"
        data["admission_options"]["transient_mode"] = "legacy"
    else:
        del data["admission_options"]
    (directory / "job.json").write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        core.resume_job(str(directory))


def test_failures_and_resume_attempt_timings_are_kept(admitted, monkeypatch):
    _, paths, out = admitted
    original = core.run_script
    def fail(*a, **k):
        raise RuntimeError("test script failed")
    monkeypatch.setattr(core, "run_script", fail)
    job = core.submit_job(None, str(out), input_files=paths[:3], transient_mode="off")
    core.run_job(job["job_dir"])
    failed = core.load_job(job["job_dir"])
    assert failed["state"] == "failed"
    assert failed["diagnostics"]["timings"]["operation_counts"]["script:01_convert"] == 1
    monkeypatch.setattr(core, "run_script", original)
    core.resume_job(job["job_dir"])
    core.run_job(job["job_dir"])
    finished = core.load_job(job["job_dir"])
    assert finished["state"] == "succeeded"
    assert finished["diagnostics"]["timing_attempts"] == [failed["diagnostics"]["timings"]]


def test_timer_disjoint_phases_operations_and_failure():
    now = [0.0]
    timer = StageTimer(clock=lambda: now[0])
    now[0] = 2
    timer.phase("convert")
    with pytest.raises(RuntimeError):
        with timer.operation("script"):
            now[0] = 5
            raise RuntimeError("test")
    now[0] = 7
    report = timer.finish()
    assert report["phases_seconds"] == {"setup": 2, "convert": 5}
    assert report["operations_seconds"] == {"script": 3}
    now[0] = 10
    assert timer.finish() == report


def test_explicit_colour_uses_selected_first_frame_and_passes_recipe(tmp_path, monkeypatch):
    from starstack import color_export as ce
    selected = ["selected-first.nef", "geometry-reference.nef", "third.nef"]
    adjustment = {"warmth": 0.0, "tint": 0.0, "exposure_ev": 0.0}
    references, calls = [], []
    def basis(path):
        references.append(path)
        return {"source": path}
    def export(*args, **kwargs):
        calls.append(kwargs)
        return [], {"state": "succeeded"}
    monkeypatch.setattr(ce, "job_colour_basis", basis)
    monkeypatch.setattr(ce, "export_job_products", export)
    monkeypatch.setattr(core, "sequence_reference", lambda *a: {"reference_input": selected[1]})
    job = {"input_kind": "raw", "inputs": [{"path": p} for p in selected], "mask": None,
           "color_adjustments": adjustment}
    _, record = core._colour_products(tmp_path, job, tmp_path, None, tmp_path, tmp_path / "base.fit", lambda: None)
    assert references == [selected[0]]
    assert calls[0]["color_adjustments"] == adjustment
    assert record["reference_selection"] == "first_selected_input"
    del job["color_adjustments"]
    _, legacy = core._colour_products(tmp_path, job, tmp_path, None, tmp_path, tmp_path / "base.fit", lambda: None)
    assert references[-1] == selected[1]
    assert "color_adjustments" not in calls[-1]
    assert legacy["reference_selection"] == "geometric_reference"


def test_colour_controls_bound_at_admission_and_resume(admitted):
    source, paths, out = admitted
    # RAW admission does not decode; use stable opaque files to check identity
    # and request binding independently of the colour decoder.
    paths = []
    for i in range(3):
        path = source / f"raw-{i}.nef"
        path.write_bytes(f"raw fixture {i}".encode())
        paths.append(str(path))
    controls = {"warmth": 10.0, "tint": -2.0, "exposure_ev": .5}
    job = core.submit_job(None, str(out), input_files=paths[:3], transient_mode="off", color_adjustments=controls,
                          idempotency_key="colour")
    assert job["color_adjustments"] == controls
    core._verify_admission_options(job)
    with pytest.raises(ValueError, match="不同素材"):
        core.submit_job(None, str(out), input_files=paths[:3], transient_mode="off", color_adjustments={**controls, "warmth": 12},
                        idempotency_key="colour")
    file = Path(job["job_dir"]) / "job.json"
    saved = json.loads(file.read_text(encoding="utf-8"))
    saved.update(state="cancelled", color_adjustments={**controls, "warmth": 12})
    file.write_text(json.dumps(saved), encoding="utf-8")
    with pytest.raises(ValueError, match="选项已改变"):
        core.resume_job(job["job_dir"])


@pytest.mark.parametrize("controls", [{"warmth": 1}, {"tint": -.5}, {"exposure_ev": 2}])
def test_fits_nonzero_controls_fail_before_launch(admitted, controls):
    _, paths, out = admitted
    with pytest.raises(ValueError, match="FITS 暂不支持"):
        core.submit_job(None, str(out), input_files=paths[:3], color_adjustments=controls)
    assert not out.exists()


def test_fits_zero_controls_remain_accepted(admitted):
    _, paths, out = admitted
    job = core.submit_job(None, str(out), input_files=paths[:3], color_adjustments={})
    assert job["color_adjustments"] == {"warmth": 0., "tint": 0., "exposure_ev": 0.}
