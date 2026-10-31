"""Shared workflow orchestration: synthetic products, real layer operations.

These tests do not claim RAW decoding or real Siril/photographic acceptance.
"""
from pathlib import Path
import json
import hashlib
import os

import numpy as np
import pytest
import tifffile
from PIL import Image, ImageCms

from starstack import workflow as wf
from starstack.compose_evidence import write_weight


@pytest.fixture(autouse=True)
def _engine_path_limit_off(monkeypatch):
    # These tests admit F/R jobs under pytest's temporary folder but never run a stack engine.  The engine path limit
    # (efficiency P1a S-8) is tested in test_engine_path_admission.py; a deep --basetemp must not refuse these.
    from starstack import siril
    monkeypatch.setattr(siril, "ENGINE_PATH_LIMIT", 1 << 15)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    inputs = tmp_path / "raw"
    inputs.mkdir()
    for i in range(3):
        (inputs / f"{i}.NEF").write_bytes(bytes([i]) * 23)
    engine = tmp_path / "siril.exe"
    engine.write_bytes(b"fake-test-engine")
    model = tmp_path / "user-model.onnx"
    model.write_bytes(b"stub-weights")
    mask = tmp_path / "sky.png"
    a = np.full((144, 160), 255, np.uint8)
    a[-24:] = 0
    Image.fromarray(a).save(mask)
    monkeypatch.setattr(wf.core, "doctor", lambda p=None: {"available": True, "executable": str(engine), "version": "test"})
    monkeypatch.setattr(wf, "_spawn", lambda d: None)
    calls = {"submits": 0, "resumes": 0, "cancelled": []}

    def submit(input_dir, output_root, **kwargs):
        calls["submits"] += 1
        calls["kwargs"] = kwargs
        directory = Path(output_root) / ("job-" + hashlib.sha256(kwargs["idempotency_key"].encode()).hexdigest()[:24])
        directory.mkdir(parents=True, exist_ok=True)
        exp = directory / "products"
        exp.mkdir(exist_ok=True)
        yy, xx = np.mgrid[:144, :160]
        rgb = np.zeros((144, 160, 3), np.float64) + [4200, 4400, 4800]
        for y in range(12, 115, 18):
            for x in range(12, 155, 18):
                rgb += 30000 * np.exp(-((xx - x)**2 + (yy - y)**2) / 2)[..., None]
        rgb = np.rint(rgb).astype(np.uint16)
        icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
        tiff = exp / "natural_display_16bit.tif"
        tifffile.imwrite(tiff, rgb, photometric="rgb", iccprofile=icc,
            metadata={"roworder": "TOP-DOWN", "scene_linear": False,
                      "display_encoding_applied": True, "color_space": "encoded_sRGB"})
        jpeg = exp / "natural.jpg"
        Image.fromarray(np.rint(rgb / 257).astype(np.uint8)).save(jpeg)
        master = exp / "master_scene_linear.tif"
        tifffile.imwrite(master, rgb.astype(np.float32)/65535, photometric="rgb")
        recipe = exp / "master_recipe.json"
        recipe.write_text("{}")
        render = exp / "photo_render.json"
        render.write_text("{}")
        composite = exp / "master_linear.fit"
        composite.write_bytes(b"synthetic-composite-binding")
        desc = write_weight(exp, np.asarray(Image.open(mask), np.float32)/255, composite)
        colour = {"state": "succeeded", "contract": "M+D photo-default-1",
                  "master": {"path": str(master), "sha256": wf.core._sha(master), "recipe": str(recipe)},
                  "photo": {"tiff": str(tiff), "tiff_sha256": wf.core._sha(tiff), "jpeg": str(jpeg),
                            "jpeg_sha256": wf.core._sha(jpeg), "record": str(render)}}
        sources = wf.core.inspect_sources(str(inputs), **({"input_files": kwargs["input_files"]} if kwargs.get("input_files") is not None else {}))
        record = {"job_dir": str(directory), "state": "succeeded", "input_kind": sources["kind"],
                  "inputs": [{"path": p, "size": Path(p).stat().st_size, "sha256": wf.core._sha(Path(p))} for p in sources["files"]],
                  "engine": {"available": True, "executable": str(engine), "version": "test"},
                  "support_mode": kwargs["support_mode"], "stack_range_mode": kwargs["stack_range_mode"],
                  "mask": {"path": str(mask), "sha256": wf.core._sha(mask)} if kwargs["sky_mask_path"] else None,
                  "diagnostics": {"color_export": colour,
                                  "sky_validity": {"compose_weight": desc} if kwargs["sky_mask_path"] else {}},
                  "artifacts": [{"path": str(composite), "kind": "fits"}]}
        for key in ("transient_mode", "color_adjustments"):
            if key in kwargs:
                record[key] = kwargs[key]
        if kwargs.get("input_files") is not None:
            record["input_selection"] = {"mode": "files", "files": sources["files"], "input_roots": sources["input_roots"]}
        if kwargs.get("stack_engine_path"):
            record["stack_engine"] = {"executable": kwargs["stack_engine_path"],
                                      "sha256": wf.core._sha(Path(kwargs["stack_engine_path"]))}
        wf.atomic_json(directory / "job.json", record)
        return record

    monkeypatch.setattr(wf.core, "submit_job", submit)
    monkeypatch.setattr(wf.core, "load_job", lambda p: json.loads((Path(p) / "job.json").read_text()))
    monkeypatch.setattr(wf.core, "cancel_job", lambda p: calls["cancelled"].append(p))
    return {"inputs": inputs, "output": tmp_path / "out", "mask": mask,
            "engine": engine, "model": model, "calls": calls, "submit": submit}


def admit(s, **kwargs):
    return wf.submit_workflow(s["inputs"], s["output"], **kwargs)


def stub_model(monkeypatch):
    from starstack import starless_runtime as rt
    class Model:
        def __init__(self, path, **kw):
            self.identity = {"model_sha256": wf.core._sha(Path(path)), "backend": "test-half", **kw}
        def __call__(self, x):
            return np.ascontiguousarray(x * .8, np.float32)
    monkeypatch.setattr(rt, "OnnxModel", Model)


def test_no_model_is_explicit_skip_and_exports_master_photo(setup):
    r = admit(setup)
    done = wf.run_workflow(r["workflow_dir"])
    assert done["state"] == "succeeded"
    assert done["finishing"]["state"] == "skipped"
    assert done["finishing"]["reason"] == "not-requested"
    assert done["separation_result"] is None
    assert done["config"] == done["request"]["config"]
    files = {a["name"]: a for a in done["artifacts"]}
    assert files["final_srgb16.tif"]["sha256"] == done["checkpoints"]["stack"]["data"]["colour"]["photo"]["tiff_sha256"]
    assert "stack_before_reduction_srgb16.tif" not in files
    assert files["final.jpg"]["kind"] == "jpeg"
    assert all(Path(x["path"]).is_file() and x["label"] for x in done["artifacts"])
    assert setup["calls"]["submits"] == 1


@pytest.mark.parametrize("kw", [
    {"milkyway_offset": -.1}, {"background_strength": float("nan")}, {"background_strength": True},
    {"background_strength": 1.1}, {"backend_options": {"providers": ["CPU"]}},
    {"backend_options": {"layout": "auto"}}, {"backend_options": {"tile": 65, "stride": 32}},
    {"backend_options": {"batch": 0}}, {"backend_options": {"value_range": "unknown"}},
    {"backend_options": {"intra_op_threads": 0}}, {"stack_range_mode": "finite_validity_v1"},
    {"stack_range_mode": "unknown"}, {"idempotency_key": ""},
])
def test_invalid_options_before_spawn(setup, kw):
    with pytest.raises(ValueError):
        admit(setup, **kw)
    assert setup["calls"]["submits"] == 0


def test_model_requires_mask(setup):
    with pytest.raises(ValueError, match="天空蒙版"):
        admit(setup, star_model_path=setup["model"], stars_enabled=True)


def test_explicit_engine_selects_finite_mode(setup):
    r = admit(setup, sky_mask_path=setup["mask"], stack_engine_path=setup["engine"])
    assert r["config"]["stack_range_mode"] == "finite_validity_v1"
    done = wf.run_workflow(r["workflow_dir"])
    assert done["state"] == "succeeded"
    assert setup["calls"]["kwargs"]["support_mode"] == "strict_siril_1_4_4"
    assert setup["calls"]["kwargs"]["stack_engine_path"] == str(setup["engine"])


def test_legacy_positive_checkbox_preserved(setup):
    r = admit(setup, stack_range_mode="positive_power2_v1")
    wf.run_workflow(r["workflow_dir"])
    assert setup["calls"]["kwargs"]["stack_range_mode"] == "positive_power2_v1"


def test_idempotent_does_not_resubmit_and_rejects_changed_settings(setup):
    r = admit(setup, idempotency_key="same")
    assert admit(setup, idempotency_key="same")["workflow_dir"] == r["workflow_dir"]
    with pytest.raises(ValueError, match="不同"):
        admit(setup, idempotency_key="same", background_strength=.4)
    assert setup["calls"]["submits"] == 0


def test_reject_output_inside_originals(setup):
    with pytest.raises(ValueError, match="之外"):
        wf.submit_workflow(setup["inputs"], setup["inputs"] / "outputs")


@pytest.mark.parametrize("what", ["raw", "model", "mask", "config", "request", "new-raw"])
def test_binding_tampering_stops_before_stack(setup, what):
    r = admit(setup, sky_mask_path=setup["mask"], star_model_path=setup["model"], stars_enabled=True)
    d = Path(r["workflow_dir"])
    if what == "raw":
        (setup["inputs"] / "0.NEF").write_bytes(b"changed")
    elif what == "new-raw":
        (setup["inputs"] / "extra.NEF").write_bytes(b"new")
    elif what in ("mask", "model"):
        setup[what].write_bytes(b"changed")
    else:
        record = wf._read(d)
        record[what]["background_strength"] = .27
        wf.atomic_json(d / "workflow.json", record)
    done = wf.run_workflow(d)
    assert done["state"] == "failed" and not done["artifacts"]
    assert setup["calls"]["submits"] == 0


def test_cancel_before_start_then_resume(setup):
    r = admit(setup)
    d = Path(r["workflow_dir"])
    assert wf.cancel_workflow(d)["state"] == "cancel_requested"
    assert wf.run_workflow(d)["state"] == "cancelled"
    assert setup["calls"]["submits"] == 0
    assert wf.resume_workflow(d)["attempt"] == 2
    assert wf.run_workflow(d)["state"] == "succeeded"


def test_finished_workflow_cancel_is_noop(setup):
    r = admit(setup)
    done = wf.run_workflow(r["workflow_dir"])
    assert wf.cancel_workflow(r["workflow_dir"])["state"] == "succeeded"
    assert not (Path(r["workflow_dir"]) / "cancel.request").exists()
    assert done["artifacts"]


def test_model_real_layers_protect_ground_and_explicit_strength(setup, monkeypatch):
    stub_model(monkeypatch)
    r = admit(setup, sky_mask_path=setup["mask"], star_model_path=setup["model"], stars_enabled=True,
              background_strength=.7, backend_options={"tile": 64, "stride": 32})
    done = wf.run_workflow(r["workflow_dir"])
    assert done["state"] == "succeeded", done.get("error")
    assert done["finishing"]["background_strength_applied"] == .7
    source = tifffile.imread(done["checkpoints"]["stack"]["data"]["colour"]["photo"]["tiff"])
    final = tifffile.imread(Path(done["delivery_dir"]) / "final_srgb16.tif")
    assert np.array_equal(source[-24:], final[-24:])
    assert np.any(source[:100] != final[:100])
    assert Path(done["separation_result"]).is_file()
    assert done["finishing"]["layers"]["exact_integer_roundtrip"]


def test_model_failure_keeps_completed_stack_and_resume_reuses_it(setup, monkeypatch):
    from starstack import starless_runtime as rt
    class Bad:
        def __init__(self, *a, **k):
            raise RuntimeError("model unavailable")
    monkeypatch.setattr(rt, "OnnxModel", Bad)
    r = admit(setup, sky_mask_path=setup["mask"], star_model_path=setup["model"], stars_enabled=True, background_strength=.3)
    failed = wf.run_workflow(r["workflow_dir"])
    assert failed["state"] == "failed" and "model unavailable" in failed["error"]
    assert "stack" in failed["checkpoints"] and not failed["artifacts"]
    assert wf.core.load_job(failed["stack_job_dir"])["state"] == "succeeded"
    stub_model(monkeypatch)
    wf.resume_workflow(r["workflow_dir"])
    done = wf.run_workflow(r["workflow_dir"])
    assert done["state"] == "succeeded", done.get("error")
    assert setup["calls"]["submits"] == 1


def test_cancel_during_each_inference_batch_not_publish(setup, monkeypatch):
    from starstack import starless_runtime as rt
    r = admit(setup, sky_mask_path=setup["mask"], star_model_path=setup["model"], stars_enabled=True, background_strength=.3,
              backend_options={"tile": 64, "stride": 32})
    d = Path(r["workflow_dir"])
    calls = []
    class Cancel:
        def __init__(self, path, **k):
            self.identity = {"model_sha256": wf.core._sha(Path(path))}
        def __call__(self, x):
            calls.append(1)
            wf.cancel_workflow(d)
            return x
    monkeypatch.setattr(rt, "OnnxModel", Cancel)
    result = wf.run_workflow(d)
    assert result["state"] == "cancelled" and not result["artifacts"]
    assert len(calls) == 1
    assert wf.core.load_job(result["stack_job_dir"])["state"] == "succeeded"
    records = list((d / "finishing" / "separation").glob(".partial-*/result.json"))
    assert json.loads(records[0].read_text(encoding="utf-8"))["state"] == "cancelled"


def test_raw_colour_fallback_is_not_success(setup, monkeypatch):
    original = setup["submit"]
    def fallback(*a, **k):
        job = original(*a, **k)
        job["diagnostics"]["color_export"] = {"state": "fallback_legacy", "error": "no colour basis"}
        wf.atomic_json(Path(job["job_dir"]) / "job.json", job)
        return job
    monkeypatch.setattr(wf.core, "submit_job", fallback)
    r = admit(setup)
    result = wf.run_workflow(r["workflow_dir"])
    assert result["state"] == "failed" and "M+D" in result["error"]
    assert wf.core.load_job(result["stack_job_dir"])["state"] == "succeeded"


def test_resume_rejects_checkpoint_change(setup, monkeypatch):
    monkeypatch.setattr(wf, "_finish", lambda *a: (_ for _ in ()).throw(RuntimeError("stop")))
    r = admit(setup)
    result = wf.run_workflow(r["workflow_dir"])
    Path(result["checkpoints"]["stack"]["data"]["colour"]["photo"]["jpeg"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="改变"):
        wf.resume_workflow(r["workflow_dir"])


def test_resume_refuses_live_worker_lock(setup):
    r = admit(setup)
    d = Path(r["workflow_dir"])
    wf._update(d, state="failed")
    with wf.file_lock(d / "worker.lock"):
        with pytest.raises(ValueError, match="仍在运行"):
            wf.resume_workflow(d)


def test_failed_stack_does_not_start_finishing(setup, monkeypatch):
    original = setup["submit"]
    def fail(*a, **k):
        job = original(*a, **k)
        job.update(state="failed", error="stack range rejected")
        wf.atomic_json(Path(job["job_dir"]) / "job.json", job)
        return job
    monkeypatch.setattr(wf.core, "submit_job", fail)
    r = admit(setup)
    result = wf.run_workflow(r["workflow_dir"])
    assert result["state"] == "failed" and "stack range rejected" in result["error"]
    assert not result["checkpoints"] and not result["artifacts"]


def test_no_usable_density_keeps_base_but_exports_layers(setup, monkeypatch):
    stub_model(monkeypatch)
    mask = np.zeros((144, 160), np.uint8)
    mask[:70] = 255
    Image.fromarray(mask).save(setup["mask"])
    r = admit(setup, sky_mask_path=setup["mask"], star_model_path=setup["model"], stars_enabled=True,
              backend_options={"tile": 64, "stride": 32})
    result = wf.run_workflow(r["workflow_dir"])
    assert result["state"] == "succeeded", result.get("error")
    assert result["finishing"]["state"] == "reduction_skipped"
    assert result["finishing"]["background_strength_applied"] is None
    assert result["finishing"]["layers"]["state"] == "succeeded"


def test_fits_legacy_is_explicit_and_not_fake_camera_master(setup, monkeypatch):
    inspect = wf.core.inspect_sources
    monkeypatch.setattr(wf.core, "inspect_sources", lambda p: {**inspect(p), "kind": "fits"})
    original = setup["submit"]
    def legacy(*a, **k):
        job = original(*a, **k)
        job["input_kind"] = "fits"
        col = job["diagnostics"]["color_export"]
        master = Path(col["master"]["path"])
        new = master.with_name("master_linear_32bit.tif")
        master.rename(new)
        job["artifacts"] += [{"path": str(new)}, {"path": col["photo"]["tiff"]}, {"path": col["photo"]["jpeg"]}]
        job["diagnostics"]["color_export"] = {"state": "legacy"}
        wf.atomic_json(Path(job["job_dir"]) / "job.json", job)
        return job
    monkeypatch.setattr(wf.core, "submit_job", legacy)
    r = admit(setup)
    done = wf.run_workflow(r["workflow_dir"])
    assert done["state"] == "succeeded", done.get("error")
    assert done["color_contract"] == "legacy_input_RGB"
    assert (Path(done["delivery_dir"]) / "master_input_linear.tif").is_file()
    assert not (Path(done["delivery_dir"]) / "master_scene_linear.tif").exists()


def test_fits_legacy_display_adapter_preserves_codes_and_then_reduces(setup, monkeypatch):
    stub_model(monkeypatch)
    inspect = wf.core.inspect_sources
    monkeypatch.setattr(wf.core, "inspect_sources", lambda p: {**inspect(p), "kind": "fits"})
    original = setup["submit"]
    def legacy(*a, **k):
        job = original(*a, **k)
        job["input_kind"] = "fits"
        col = job["diagnostics"]["color_export"]
        photo = Path(col["photo"]["tiff"])
        with tifffile.TiffFile(photo) as tf:
            rgb, icc = tf.asarray(), tf.pages[0].tags[34675].value
        tifffile.imwrite(photo, rgb, photometric="rgb", iccprofile=icc,
                         metadata={"linear": False, "purpose": "uncalibrated display rendering"})
        master = Path(col["master"]["path"])
        new = master.with_name("master_linear_32bit.tif")
        master.rename(new)
        job["artifacts"] += [{"path": str(new)}, {"path": col["photo"]["tiff"]}, {"path": col["photo"]["jpeg"]}]
        job["diagnostics"]["color_export"] = {"state": "legacy"}
        wf.atomic_json(Path(job["job_dir"]) / "job.json", job)
        return job
    monkeypatch.setattr(wf.core, "submit_job", legacy)
    r = admit(setup, sky_mask_path=setup["mask"], star_model_path=setup["model"], stars_enabled=True, background_strength=.5,
              backend_options={"tile": 64, "stride": 32})
    done = wf.run_workflow(r["workflow_dir"])
    assert done["state"] == "succeeded", done.get("error")
    adapted = done["checkpoints"]["legacy_display"]["data"]
    assert np.array_equal(tifffile.imread(adapted["path"]), tifffile.imread(adapted["source"]))
    assert done["finishing"]["layers"]["exact_integer_roundtrip"]


def test_missing_actual_weight_rejects_model_editing(setup, monkeypatch):
    stub_model(monkeypatch)
    original = setup["submit"]
    def without_weight(*a, **k):
        job = original(*a, **k)
        job["diagnostics"]["sky_validity"] = {}
        wf.atomic_json(Path(job["job_dir"]) / "job.json", job)
        return job
    monkeypatch.setattr(wf.core, "submit_job", without_weight)
    r = admit(setup, sky_mask_path=setup["mask"], star_model_path=setup["model"], stars_enabled=True)
    done = wf.run_workflow(r["workflow_dir"])
    assert done["state"] == "failed" and "compose_weight" in done["error"]


def test_density_recommendation_is_applied_when_available(setup, monkeypatch):
    stub_model(monkeypatch)
    r = admit(setup, sky_mask_path=setup["mask"], star_model_path=setup["model"], stars_enabled=True,
              backend_options={"tile": 64, "stride": 32})
    done = wf.run_workflow(r["workflow_dir"])
    assert done["state"] == "succeeded", done.get("error")
    recommendation = done["finishing"]["advice"]["recommendation"]
    assert recommendation["available"]
    assert done["finishing"]["background_strength_applied"] == recommendation["background_strength"]
    assert done["finishing"]["reduction"]["recipe"]["milkyway_offset"] == 0


def test_stack_cancellation_waits_for_stack_terminal(setup, monkeypatch):
    original = setup["submit"]
    def pending(*a, **k):
        job = original(*a, **k)
        job.update(state="running", stage="stack", progress=50)
        wf.atomic_json(Path(job["job_dir"]) / "job.json", job)
        return job
    r = admit(setup)
    d = Path(r["workflow_dir"])
    monkeypatch.setattr(wf.core, "submit_job", pending)
    observations = []
    def pause(seconds):
        record = wf._read(d)
        observations.append(record["state"])
        if len(observations) == 1:
            wf.cancel_workflow(d)
        elif len(observations) == 2:
            job = wf.core.load_job(record["stack_job_dir"])
            job["state"] = "cancelled"
            wf.atomic_json(Path(job["job_dir"]) / "job.json", job)
    monkeypatch.setattr(wf.time, "sleep", pause)
    done = wf.run_workflow(d)
    assert done["state"] == "cancelled" and not done["artifacts"]
    assert observations == ["running", "cancel_requested"]
    assert setup["calls"]["cancelled"]


def test_cancel_during_delivery_never_publishes_success(setup, monkeypatch):
    original = wf.shutil.copyfile
    r = admit(setup)
    d = Path(r["workflow_dir"])
    def copy_then_cancel(source, target, **kw):
        result = original(source, target, **kw)
        wf.cancel_workflow(d)
        return result
    monkeypatch.setattr(wf.shutil, "copyfile", copy_then_cancel)
    done = wf.run_workflow(d)
    assert done["state"] == "cancelled" and not done["artifacts"]
    assert not list(d.glob("delivery-*"))


def test_idempotent_published_file_tampering_is_rejected(setup):
    r = admit(setup, idempotency_key="finished")
    done = wf.run_workflow(r["workflow_dir"])
    Path(done["artifacts"][0]["path"]).write_bytes(b"tamper")
    with pytest.raises(ValueError, match="改变"):
        admit(setup, idempotency_key="finished")


def test_other_stack_directory_tampering_is_rejected(setup):
    r = admit(setup)
    d = Path(r["workflow_dir"])
    record = wf._read(d)
    record["stack_job_dir"] = str(setup["inputs"])
    wf.atomic_json(d / "workflow.json", record)
    with pytest.raises(ValueError, match="范围"):
        wf.cancel_workflow(d)


def test_unlocked_stale_workflow_does_not_trust_reused_pid(setup):
    r = admit(setup)
    d = Path(r["workflow_dir"])
    record = wf._read(d)
    record.update(state="running", worker_pid=os.getpid(), updated_at="2020-01-01T00:00:00+00:00")
    wf.atomic_json(d / "workflow.json", record)
    assert wf.workflow_status(d)["state"] == "failed"
    assert wf.resume_workflow(d)["state"] == "queued"


def test_spawn_is_hidden_detached_and_bound_to_module(setup, monkeypatch):
    seen = {}
    def fake_popen(argv, **kwargs):
        seen.update(argv=argv, kwargs=kwargs)
        return type("Process", (), {"pid": 12345678})()
    # The test fixture disabled public spawning; call the original function body
    # via a fresh source module only for this subprocess-argument check.
    import importlib.util
    spec = importlib.util.spec_from_file_location("starstack._workflow_spawn_test", wf.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.subprocess, "Popen", fake_popen)
    r = admit(setup)
    module._spawn(Path(r["workflow_dir"]))
    assert seen["argv"][1:3] == ["-m", "starstack.workflow"]
    assert seen["kwargs"]["stdin"] == module.subprocess.DEVNULL
    if os.name == "nt":
        assert seen["kwargs"]["creationflags"] & module.subprocess.CREATE_NO_WINDOW


def test_finished_workflow_allows_later_trail_derivative_without_rebinding(setup):
    r = admit(setup, idempotency_key="with-later-review")
    done = wf.run_workflow(r["workflow_dir"])
    child = wf.core.load_job(done["stack_job_dir"])
    child["transient_revision"] = {"id": "later-reviewed"}
    child["transient_marks"] = [{"input": str(setup["inputs"] / "0.NEF"), "x0": 1, "x1": 4}]
    child["artifacts"].append({"path": str(Path(done["stack_job_dir"]) / "later-trails.tif"), "group": "transients"})
    wf.atomic_json(Path(done["stack_job_dir"]) / "job.json", child)
    wf.validate_stack_binding(done, child)
    again = admit(setup, idempotency_key="with-later-review")
    assert again["artifacts"] == done["artifacts"]
    assert again["state"] == "succeeded"
    paths = [Path(item["path"]).name for item in done["checkpoints"]["stack"]["files"]]
    assert "job.json" not in paths and any(p.startswith("stack_snapshot-") for p in paths)
    source = Path(done["checkpoints"]["stack"]["data"]["colour"]["photo"]["tiff"])
    source.write_bytes(b"modified base")
    with pytest.raises(ValueError, match="改变"):
        admit(setup, idempotency_key="with-later-review")


def test_incomplete_workflow_does_not_accept_new_trail_marks(setup, monkeypatch):
    monkeypatch.setattr(wf, "_finish", lambda *a: (_ for _ in ()).throw(RuntimeError("stop")))
    r = admit(setup)
    failed = wf.run_workflow(r["workflow_dir"])
    child = wf.core.load_job(failed["stack_job_dir"])
    child["transient_marks"] = [{"input": str(setup["inputs"] / "0.NEF")}]
    wf.atomic_json(Path(failed["stack_job_dir"]) / "job.json", child)
    with pytest.raises(ValueError, match="亮迹"):
        wf.resume_workflow(r["workflow_dir"])
