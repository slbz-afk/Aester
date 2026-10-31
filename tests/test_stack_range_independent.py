"""Independent range-wrapper tests; fake engine is a simple exact mean oracle.

The oracle intentionally does not model Siril rejection/normalization. These
tests check the new adapter's storage, units, selection and failure contract;
fixed-engine scientific comparisons are recorded separately under outputs/.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits

from starstack import stack_range as sr
from starstack.siril import Cancelled


@pytest.fixture(autouse=True)
def _engine_path_limit_off(monkeypatch):
    # These tests admit F/R jobs under pytest's temporary folder but never run a stack engine.  The engine path limit
    # (efficiency P1a S-8) is tested in test_engine_path_admission.py; a deep --basetemp must not refuse these.
    from starstack import siril
    monkeypatch.setattr(siril, "ENGINE_PATH_LIMIT", 1 << 15)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path, array, *, row="BOTTOM-UP", overwrite=False):
    header = fits.Header()
    header["ROWORDER"] = row
    header["STACKCNT"] = 3
    header["EXPTIME"] = 20.0
    fits.writeto(path, np.asarray(array, dtype=np.float32), header, overwrite=overwrite)


def _read(path):
    with fits.open(path, memmap=False) as hdus:
        return hdus[0].data.astype(np.float32), hdus[0].header.copy()


def _seq(name="light_", *, selected=(True, True, True), ref=1):
    lines = [f"S '{name}' 1 {len(selected)} {sum(selected)} 5 {ref} 6 0 0 0", "L 3"]
    lines += [f"I {i} {int(keep)}" for i, keep in enumerate(selected, 1)]
    for _ in selected:
        lines.append("R1 2 2 1 1 .03 40 H 1 0 0 0 1 0 0 0 1")
    return "\n".join(lines) + "\n"


def _oracle_mean(arrays):
    """Independent nonzero-sample mean, no range clipping or production helpers."""
    values = np.asarray(arrays, dtype=np.float64)
    count = np.count_nonzero(values, axis=0)
    total = np.sum(values, axis=0)
    return np.divide(total, count, out=np.zeros_like(total), where=count > 0).astype(np.float32)


@pytest.fixture
def inputs(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    arrays = []
    yy, xx = np.indices((12, 16))
    for frame in range(3):
        base = np.asarray((xx + yy + 8 + frame * 2) / 64, dtype=np.float32)
        data = np.stack([base, base * np.float32(.75), base * np.float32(.5)])
        data[:, 3, 5] = [1.25, 1.5, 1.125]
        data[:, -1, -1] = 0  # no contributing frame at this position
        if frame == 0:
            data[:, 2, 4] = 0  # some frames missing; not a zero-valued measurement
        arrays.append(data)
        _write(work / f"r_light_{frame + 1:05d}.fit", data)
    (work / "light_.seq").write_text(_seq(), encoding="utf-8")
    (work / "r_light_.seq").write_text(_seq("r_light_"), encoding="utf-8")
    return tmp_path, work, arrays


class MeanEngine:
    """Mean-only fake, with deliberate engine/output faults injected externally."""

    def __init__(self, source_work, arrays, action=None):
        self.source_work = source_work
        self.arrays = arrays
        self.action = action
        self.calls = []

    def __call__(self, executable, directory, name, commands, check):
        check()
        cd = next((line[3:].strip().strip('"') for line in commands if line.startswith("cd ")), None)
        attempt = Path(directory) if cd is None else Path(cd)
        if cd is not None and not attempt.is_absolute():
            attempt = Path(directory) / attempt
        attempt = attempt.resolve()
        assert attempt != self.source_work.resolve()
        stack_pos = next(i for i, line in enumerate(commands) if line.startswith("stack "))
        assert commands[stack_pos - 1] == "seqclean r_light -stat"
        command = commands[stack_pos]
        assert "-norm=addscale" in command
        assert "-filter-included" in command
        assert "-output_norm" not in command
        assert "rej w 3 3" in command
        path = attempt / "r_light_.seq"
        sequence = path.read_text(encoding="utf-8")
        selected = [int(line.split()[1]) for line in sequence.splitlines()
                    if line.startswith("I ") and int(line.split()[2]) == 1]
        arrays = []
        factors = []
        for number in selected:
            data, header = _read(attempt / f"r_light_{number:05d}.fit")
            assert header["BITPIX"] == -32
            assert header.get("ROWORDER", "BOTTOM-UP") == "BOTTOM-UP"
            original = self.arrays[number - 1]
            valid = original != 0
            assert not np.any(data[~valid] != 0)
            assert np.all(data[valid] > 0)
            assert np.all(data[valid] < 1)
            factor = float(original[valid][0] / data[valid][0])
            assert np.log2(factor).is_integer()
            np.testing.assert_array_equal(data, (original / factor).astype(np.float32))
            arrays.append(data)
            factors.append(factor)
        assert len(set(factors)) == 1
        out_name = next(part[5:] for part in command.split() if part.startswith("-out="))
        if not out_name.lower().endswith((".fit", ".fits")):
            out_name += ".fit"
        output = attempt / out_name
        mean = _oracle_mean(arrays)
        self.calls.append({"attempt": attempt, "factor": factors[0], "output": output,
                           "selected": selected, "sequence": sequence, "mean": mean.copy()})
        action = self.action(self, mean) if self.action else None
        if isinstance(action, str) and action == "no_output":
            return
        if isinstance(action, np.ndarray):
            mean = action
        _write(output, np.clip(mean, 0, 1).astype(np.float32))
        with fits.open(output, mode="update") as hdus:
            hdus[0].header["STACKCNT"] = len(selected)


def _snapshot(work):
    return {p.name: _sha(p) for p in work.glob("r_light_*.fit")}


def _assert_not_published(work):
    assert not (work / "sky_master.fit").exists()
    if (work / "stack_range.json").exists():
        record = json.loads((work / "stack_range.json").read_text(encoding="utf-8"))
        assert record.get("state") != "complete"


@pytest.fixture
def engine(tmp_path, monkeypatch):
    """Pin a fake executable without disabling production identity validation."""
    executable = tmp_path / "independent-fake-engine.exe"
    executable.write_bytes(b"MeanEngine numerical fixture; not executable")
    monkeypatch.setattr(sr, "ENGINE_SHA256", _sha(executable))
    return {"version": "1.4.4", "executable": str(executable), "available": True}


def _run(inputs, engine, runner, check=lambda: None):
    # This file is the contract 1.0.0 regression suite (all-positive inputs,
    # zero output refused); contract 1.1.0 is covered by test_stack_range_v2.py.
    directory, work, _ = inputs
    return sr.run_protected_stack(engine, directory, work, check, runner, contract=sr.contract(sr.VERSION_V1))


def _job(engine, work=None):
    saved = sr.contract(sr.VERSION_V1)
    result = {"schema_version": 2, "stack_range_mode": sr.MODE, "stack_range_contract": saved,
              "stack_range_sha256": hashlib.sha256(json.dumps(saved, sort_keys=True, allow_nan=False).encode()).hexdigest(),
              "engine": engine, "checkpoints": [], "checkpoint_files": {}}
    if work is not None:
        result["job_dir"] = str(work.parent)
        result["checkpoints"] = ["stack"]
        result["checkpoint_files"]["stack"] = [
            {"path": f"work/{name}", "sha256": _sha(work / name)}
            for name in ("sky_master.fit", sr.RECORD, sr.VALID)
        ]
    return result


def _complete(inputs, engine):
    _, work, arrays = inputs
    runner = MeanEngine(work, arrays)
    _run(inputs, engine, runner)
    sr.finalize_receipt(work)
    return runner, _job(engine, work)


def test_mean_oracle_restores_highlight_and_zero_support_without_touching_sources(inputs, engine):
    _, work, arrays = inputs
    original = _snapshot(work)
    seq_bytes = (work / "r_light_.seq").read_bytes()
    runner = MeanEngine(work, arrays)
    record = _run(inputs, engine, runner)
    actual, header = _read(work / "sky_master.fit")
    np.testing.assert_array_equal(actual, _oracle_mean(arrays))
    assert np.max(actual) > 1
    assert header["SSRGUNIT"] == "original"
    assert record["state"] == "decoded"
    assert len(runner.calls) == 1
    assert _snapshot(work) == original
    assert (work / "r_light_.seq").read_bytes() == seq_bytes
    valid = np.load(work / sr.VALID, allow_pickle=False)
    expected = np.logical_or.reduce([np.all(a > 0, axis=0) for a in arrays])
    np.testing.assert_array_equal(valid, expected)
    assert valid[2, 4] and not valid[-1, -1]
    assert all(actual[:, -1, -1] == 0)
    sr.finalize_receipt(work)
    assert sr.verify_receipt(work, _job(engine, work))["state"] == "complete"


@pytest.mark.parametrize("bad", ["negative", "nan", "infinity", "mixed_zero", "too_high", "underflow"])
def test_ineligible_samples_fail_without_engine_or_publication(inputs, engine, bad):
    _, work, arrays = inputs
    a = arrays[0].copy()
    value = {"negative": -0.01, "nan": np.nan, "infinity": np.inf,
             "mixed_zero": 0, "too_high": np.nextafter(np.float32(8), np.float32(np.inf)),
             "underflow": np.nextafter(np.float32(0), np.float32(1))}[bad]
    a[0, 4, 7] = value
    if bad == "underflow":
        a[:, 4, 7] = value
    _write(work / "r_light_00001.fit", a, overwrite=True)
    runner = MeanEngine(work, arrays)
    before = _snapshot(work)
    with pytest.raises(ValueError):
        _run(inputs, engine, runner)
    assert not runner.calls
    _assert_not_published(work)
    assert _snapshot(work) == before


@pytest.mark.parametrize("kind", ["missing", "extra", "duplicate", "no_reference", "excluded_reference", "wrong_count"])
def test_sequence_must_identify_exact_sources_selection_and_reference(inputs, engine, kind):
    _, work, arrays = inputs
    seq = work / "r_light_.seq"
    if kind == "missing":
        (work / "r_light_00003.fit").unlink()
    elif kind == "extra":
        _write(work / "r_light_00004.fit", arrays[0])
    elif kind == "duplicate":
        seq.write_text(seq.read_text().replace("I 3 1", "I 2 1"))
    elif kind == "no_reference":
        seq.write_text(_seq("r_light_", ref=-1))
    elif kind == "excluded_reference":
        seq.write_text(_seq("r_light_", selected=(True, False, True), ref=1))
    else:
        seq.write_text(seq.read_text().replace("1 3 3 5", "1 3 2 5"))
    runner = MeanEngine(work, arrays)
    with pytest.raises(ValueError):
        _run(inputs, engine, runner)
    assert not runner.calls
    _assert_not_published(work)


def test_excluded_frame_and_reference_stem_preserved_after_export(inputs, engine):
    _, work, arrays = inputs
    arrays.append(np.full_like(arrays[0], 4))
    _write(work / "r_light_00004.fit", arrays[-1])
    (work / "r_light_.seq").write_text(_seq("r_light_", selected=(True, False, True, True), ref=2))
    runner = MeanEngine(work, arrays)
    record = _run(inputs, engine, runner)
    assert record["reference"] == "r_light_00003"
    assert runner.calls[0]["selected"] == [1, 3, 4]
    np.testing.assert_array_equal(_read(work / "sky_master.fit")[0], _oracle_mean([arrays[i] for i in (0, 2, 3)]))


@pytest.mark.parametrize("kind", ["zero_valid", "negative_valid", "nonzero_missing", "nan", "no_output", "exception"])
def test_bad_engine_result_never_publishes_master(inputs, engine, kind):
    _, work, arrays = inputs
    def action(runner, mean):
        if kind == "exception":
            raise RuntimeError("independent engine fault")
        if kind == "no_output":
            return "no_output"
        out = mean.copy()
        if kind == "nonzero_missing":
            out[:, -1, -1] = .01
        else:
            out[0, 0, 0] = {"zero_valid": 0, "negative_valid": -.02, "nan": np.nan}[kind]
        return out
    runner = MeanEngine(work, arrays, action)
    original = _snapshot(work)
    with pytest.raises((ValueError, RuntimeError, FileNotFoundError)):
        _run(inputs, engine, runner)
    assert len(runner.calls) == 1
    _assert_not_published(work)
    assert _snapshot(work) == original


def test_upper_endpoint_retries_in_fresh_trial_with_larger_scale(inputs, engine):
    _, work, arrays = inputs
    def once(runner, mean):
        if len(runner.calls) == 1:
            mean[0, 0, 0] = 1
        return mean
    runner = MeanEngine(work, arrays, once)
    _run(inputs, engine, runner)
    assert len(runner.calls) == 2
    assert runner.calls[0]["attempt"] != runner.calls[1]["attempt"]
    assert runner.calls[1]["factor"] == 2 * runner.calls[0]["factor"]
    np.testing.assert_array_equal(_read(work / "sky_master.fit")[0], _oracle_mean(arrays))


def test_persistent_upper_endpoint_stops_after_three_trials(inputs, engine):
    _, work, arrays = inputs
    def always(runner, mean):
        mean[0, 0, 0] = 1
        return mean
    runner = MeanEngine(work, arrays, always)
    with pytest.raises(ValueError):
        _run(inputs, engine, runner)
    assert len(runner.calls) == 3
    assert len({c["attempt"] for c in runner.calls}) == 3
    _assert_not_published(work)


def test_cancelled_engine_has_original_exception_and_next_run_uses_fresh_attempt(inputs, engine):
    _, work, arrays = inputs
    error = Cancelled("independent cancellation")
    def stop(runner, mean):
        raise error
    first = MeanEngine(work, arrays, stop)
    with pytest.raises(Cancelled) as caught:
        _run(inputs, engine, first)
    assert caught.value is error
    _assert_not_published(work)
    second = MeanEngine(work, arrays)
    _run(inputs, engine, second)
    assert first.calls[0]["attempt"].parent != second.calls[0]["attempt"].parent
    np.testing.assert_array_equal(_read(work / "sky_master.fit")[0], _oracle_mean(arrays))


def test_cancellation_at_final_publication_boundary_keeps_original_unit_sources(inputs, engine):
    _, work, arrays = inputs
    original = _snapshot(work)
    def cancel_when_ready():
        if any(work.glob("range-master-*.tmp")):
            raise Cancelled("independent cancellation after decoded temporary write")
    with pytest.raises(Cancelled):
        _run(inputs, engine, MeanEngine(work, arrays), cancel_when_ready)
    _assert_not_published(work)
    assert _snapshot(work) == original


@pytest.mark.parametrize("target", ["source", "sequence"])
def test_source_change_during_engine_is_detected_before_publish(inputs, engine, target):
    _, work, arrays = inputs
    def tamper(runner, mean):
        path = work / ("r_light_00001.fit" if target == "source" else "r_light_.seq")
        with path.open("ab") as handle:
            handle.write(b"\n# independent mutation\n")
    runner = MeanEngine(work, arrays, tamper)
    with pytest.raises(ValueError):
        _run(inputs, engine, runner)
    _assert_not_published(work)


@pytest.mark.parametrize("target", ["sky_master.fit", "r_light_00001.fit", "r_light_.seq", "stack_range_valid.npy"])
def test_complete_receipt_rejects_changed_materials(inputs, engine, target):
    _, work, _ = inputs
    _, job = _complete(inputs, engine)
    with (work / target).open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError):
        sr.verify_receipt(work, job)


@pytest.mark.parametrize("field,value", [("state", "decoded"), ("divisor", 3), ("reference", "r_light_00001"),
                                          ("selected", ["r_light_00001"]), ("sources", [])])
def test_complete_receipt_rejects_metadata_change(inputs, engine, field, value):
    _, work, _ = inputs
    _, job = _complete(inputs, engine)
    path = work / sr.RECORD
    record = json.loads(path.read_text())
    record[field] = value
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError):
        sr.verify_receipt(work, job)


@pytest.mark.parametrize("change", ["version", "mode", "fingerprint", "schema"])
def test_positive_job_rejects_contract_downgrade_and_unknown_version(engine, change):
    job = _job(engine)
    if change == "version":
        job["stack_range_contract"]["version"] = "0.9.0"
    elif change == "mode":
        job["stack_range_mode"] = "legacy"
    elif change == "fingerprint":
        job["stack_range_sha256"] = "0" * 64
    else:
        job["schema_version"] = 1
    with pytest.raises(ValueError):
        sr.verify_job(job)


def test_legacy_job_stays_legacy_and_cannot_adopt_a_protected_receipt(inputs, engine):
    _, work, arrays = inputs
    legacy = {"schema_version": 1, "engine": engine}
    assert sr.verify_job(legacy) is False
    assert sr.verify_receipt(work, legacy) is None
    _run(inputs, engine, MeanEngine(work, arrays))
    sr.finalize_receipt(work)
    with pytest.raises(ValueError):
        sr.verify_receipt(work, legacy)


def test_metadata_copy_failure_does_not_publish_success_master(inputs, engine, monkeypatch):
    _, work, arrays = inputs
    original_copy = sr.shutil.copyfile
    def fail_on_valid(source, target, *args, **kwargs):
        if Path(target) == work / sr.VALID:
            raise OSError("independent disk-full sidecar failure")
        return original_copy(source, target, *args, **kwargs)
    monkeypatch.setattr(sr.shutil, "copyfile", fail_on_valid)
    with pytest.raises(OSError):
        _run(inputs, engine, MeanEngine(work, arrays))
    _assert_not_published(work)


def test_final_receipt_write_failure_does_not_publish_success_master(inputs, engine, monkeypatch):
    _, work, arrays = inputs
    original_atomic = sr.atomic_json
    def fail_on_final(path, value):
        if Path(path) == work / sr.RECORD:
            raise OSError("independent final-receipt write failure")
        return original_atomic(path, value)
    monkeypatch.setattr(sr, "atomic_json", fail_on_final)
    with pytest.raises(OSError):
        _run(inputs, engine, MeanEngine(work, arrays))
    _assert_not_published(work)


@pytest.mark.parametrize("kind", ["engine_version", "engine_content"])
def test_wrong_engine_fails_before_creating_attempt(inputs, engine, kind):
    directory, work, arrays = inputs
    if kind == "engine_version":
        engine["version"] = "1.4.5"
    else:
        Path(engine["executable"]).write_bytes(b"changed executable")
    with pytest.raises(ValueError):
        _run(inputs, engine, MeanEngine(work, arrays))
    assert not list(directory.glob("range-*"))
    _assert_not_published(work)


@pytest.mark.parametrize("kind", ["reference", "selection", "count"])
def test_engine_must_not_change_reference_selection_or_member_count(inputs, engine, kind):
    _, work, arrays = inputs
    runner = MeanEngine(work, arrays)
    def altered_engine(*args):
        runner(*args)
        attempt = runner.calls[-1]["attempt"]
        seq = attempt / "r_light_.seq"
        if kind == "reference":
            seq.write_text(_seq("r_light_", ref=0))
        elif kind == "selection":
            seq.write_text(_seq("r_light_", selected=(True, True, False), ref=1))
        else:
            with fits.open(attempt / "encoded_master.fit", mode="update") as hdus:
                hdus[0].header["STACKCNT"] = 2
    with pytest.raises(ValueError):
        _run(inputs, engine, altered_engine)
    _assert_not_published(work)


def test_core_strict_support_filters_before_encoding_and_keeps_original_units(inputs, engine, monkeypatch):
    from starstack import core, registered_support as rs
    directory, work, arrays = inputs
    _write(work / "light_00001.fit", arrays[0])
    expected = [a.copy() for a in arrays]
    events = []
    mean = MeanEngine(work, expected)
    def run_script(executable, where, name, commands, check):
        if name == "02_export":
            events.append("export")
            assert (work / rs.EXPORT_SEQUENCE).read_bytes() == (work / "light_.seq").read_bytes()
            return
        events.append("encoded-stack")
        assert events == ["export", "filter", "encoded-stack"]
        mean(executable, where, name, commands, check)
    def support_filter(*args, **kwargs):
        events.append("filter")
        for number, array in enumerate(expected, 1):
            array[:, 4, 7] = 0
            _write(work / f"r_light_{number:05d}.fit", array, overwrite=True)
        return {"state": "exports_filtered"}
    monkeypatch.setattr(core, "run_script", run_script)
    monkeypatch.setattr(core, "_validity_source", lambda *args: object())
    monkeypatch.setattr(core, "_grid_model", lambda *args: None)
    monkeypatch.setattr(core, "_update", lambda *args, **kwargs: None)
    monkeypatch.setattr(rs, "prepare_exports", support_filter)
    job = {"stack_range_mode": sr.MODE, "engine": engine}
    core._supported_stack(engine["executable"], directory, work, job, {}, True, {"dilation_px": 3}, lambda: None)
    assert events == ["export", "filter", "encoded-stack"]
    np.testing.assert_array_equal(_read(work / "sky_master.fit")[0], _oracle_mean(expected))
    for number, array in enumerate(expected, 1):
        np.testing.assert_array_equal(_read(work / f"r_light_{number:05d}.fit")[0], array)


def test_core_recompose_rejects_changed_registered_source_before_reading_candidates(inputs, engine, monkeypatch):
    from starstack import core, transient_stage as ts
    directory, work, _ = inputs
    _, job = _complete(inputs, engine)
    with (work / "r_light_00001.fit").open("ab") as handle:
        handle.write(b"changed after publication")
    monkeypatch.setattr(ts, "read_record", lambda *args, **kwargs: pytest.fail("must verify range before reading transient candidates"))
    with pytest.raises(ValueError):
        core._recompose_locked(directory, job, None, None, None, None, None)


def test_core_resume_rejects_changed_range_receipt_before_spawning(inputs, engine, monkeypatch):
    from starstack import core
    directory, work, _ = inputs
    _, job = _complete(inputs, engine)
    job.update(state="failed", inputs=[], input_kind="raw", recovery_blocked=False)
    (directory / "job.json").write_text(json.dumps(job), encoding="utf-8")
    path = work / sr.RECORD
    record = json.loads(path.read_text())
    record["divisor"] *= 2
    path.write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.setattr(core, "doctor", lambda *args: dict(engine))
    monkeypatch.setattr(core, "_spawn_worker", lambda *args: pytest.fail("must not start changed recipe"))
    with pytest.raises(ValueError):
        core._resume_locked(directory)


@pytest.fixture
def admission(tmp_path, engine, monkeypatch):
    """Real source inspection and job persistence; no worker or engine process."""
    from starstack import core
    source = tmp_path / "admission-input"
    source.mkdir()
    for number in range(3):
        header = fits.Header()
        header["ROWORDER"] = "BOTTOM-UP"
        header["EXPTIME"] = 20.0
        # No STACKCNT: these are individual synthetic RGB frames, not masters.
        fits.writeto(source / f"exposure_{number + 1:02d}.fit",
                     np.full((3, 8, 9), (number + 4) / 64, dtype=np.float32), header)
    launched = []
    monkeypatch.setattr(core, "doctor", lambda *args: dict(engine))
    monkeypatch.setattr(core.shutil, "disk_usage", lambda *args: SimpleNamespace(free=10**12))
    monkeypatch.setattr(core, "_spawn_worker", lambda directory: launched.append(Path(directory)))
    return SimpleNamespace(source=source, output=tmp_path / "admission-output", engine=engine,
                           launched=launched, originals={p.name: _sha(p) for p in source.iterdir()})


def _submit(admission, key, **kwargs):
    from starstack import core
    return core.submit_job(str(admission.source), str(admission.output), idempotency_key=key, **kwargs)


def test_admission_positive_signature_is_distinct_from_legacy_and_persists_pinned_contract(admission):
    legacy = _submit(admission, "legacy-job")
    positive = _submit(admission, "positive-job", stack_range_mode="positive_power2_v1")
    assert positive["signature"] != legacy["signature"]
    assert positive["inputs"] == legacy["inputs"]
    assert positive["support_mode"] == legacy["support_mode"] == "legacy"
    assert positive["schema_version"] == 2 and legacy["schema_version"] == 1
    assert positive["stack_range_mode"] == "positive_power2_v1"
    saved = positive["stack_range_contract"]
    assert saved["mode"] == "positive_power2_v1"
    assert saved["version"] == "1.1.0"  # new jobs get contract 1.1.0 (2026-09-29)
    assert saved["engine_sha256"] == _sha(admission.engine["executable"])
    assert saved["backend"] == "python_float32_power2_copy"
    assert saved["input_maximum"] == 8.0 and saved["target_maximum"] == .75
    assert saved["max_attempts"] == 3
    assert saved["stack_command"] == "stack r_light rej w 3 3 -norm=addscale -32b -filter-included -out=encoded_master.fit"
    expected_fingerprint = hashlib.sha256(json.dumps(saved, sort_keys=True, allow_nan=False).encode()).hexdigest()
    assert positive["stack_range_sha256"] == expected_fingerprint
    assert legacy["stack_range_contract"] is None and legacy["stack_range_sha256"] is None
    assert sr.verify_job(positive) is True and sr.verify_job(legacy) is False
    persisted = json.loads((Path(positive["job_dir"]) / "job.json").read_text(encoding="utf-8"))
    assert persisted["signature"] == positive["signature"]
    assert persisted["stack_range_contract"] == saved
    assert len(admission.launched) == 2
    assert {p.name: _sha(p) for p in admission.source.iterdir()} == admission.originals


@pytest.mark.parametrize("first,second", [("legacy", "positive_power2_v1"), ("positive_power2_v1", "legacy")])
def test_admission_same_key_different_mode_refused_without_new_worker_or_rewriting_job(admission, first, second):
    initial = _submit(admission, "same-client-request", stack_range_mode=first)
    record = Path(initial["job_dir"]) / "job.json"
    before = record.read_bytes()
    with pytest.raises(ValueError):
        _submit(admission, "same-client-request", stack_range_mode=second)
    assert record.read_bytes() == before
    assert len(admission.launched) == 1


def test_admission_identical_positive_retry_reuses_job_without_launching_again(admission):
    first = _submit(admission, "repeat-positive", stack_range_mode="positive_power2_v1")
    record = Path(first["job_dir"]) / "job.json"
    before = record.read_bytes()
    second = _submit(admission, "repeat-positive", stack_range_mode="positive_power2_v1")
    assert second["job_dir"] == first["job_dir"] and second["signature"] == first["signature"]
    assert record.read_bytes() == before
    assert len(admission.launched) == 1


@pytest.mark.parametrize("bad", [None, True, "automatic", "positive_power2_v0"])
def test_admission_invalid_mode_refused_before_engine_or_output_access(admission, monkeypatch, bad):
    from starstack import core
    monkeypatch.setattr(core, "doctor", lambda *args: pytest.fail("invalid mode must fail before engine inspection"))
    with pytest.raises(ValueError):
        _submit(admission, "bad-mode", stack_range_mode=bad)
    assert not admission.output.exists()
    assert not admission.launched


@pytest.mark.parametrize("bad", ["version", "binary"])
def test_admission_positive_mode_enforces_engine_build_before_job_creation(admission, bad):
    if bad == "version":
        admission.engine["version"] = "1.4.5"
    else:
        Path(admission.engine["executable"]).write_bytes(b"a different 1.4.4 executable")
    with pytest.raises(ValueError):
        _submit(admission, "wrong-engine", stack_range_mode="positive_power2_v1")
    assert not admission.output.exists()
    assert not admission.launched


def test_admission_legacy_signature_matches_pre_range_formula_and_reuses_old_record(admission):
    """Use the documented pre-range signature, independently of core hashing.

    EFF-D4 (efficiency P1c §4.2): a job admitted today adds admission_options.stack_settings to that preimage; the
    historical record (admitted before both the range modes and the stack-settings contract) is still found by its
    old signature and reused unchanged."""
    import d4_fakes
    from starstack import stack_settings
    with d4_fakes.admitted_before_contract():
        first = _submit(admission, "legacy-historical-request")
    previous_payload = {"files": first["inputs"], "mask": None, "frame_masks": None,
                        "engine": admission.engine, "transient_marks": None}
    if first["fits_input"]:
        previous_payload["fits_input"] = first["fits_input"]
    expected = hashlib.sha256(json.dumps(previous_payload, sort_keys=True).encode()).hexdigest()
    assert first["signature"] == expected
    record = Path(first["job_dir"]) / "job.json"
    historical = json.loads(record.read_text(encoding="utf-8"))
    for field in ("stack_range_mode", "stack_range_contract", "stack_range_sha256"):
        historical.pop(field)
    record.write_text(json.dumps(historical, ensure_ascii=False), encoding="utf-8")
    before = record.read_bytes()
    second = _submit(admission, "legacy-historical-request", stack_range_mode="legacy")
    assert second["signature"] == expected and second["schema_version"] == 1
    assert "stack_range_mode" not in second and "stack_settings" not in second
    assert sr.verify_job(second) is False
    assert record.read_bytes() == before
    assert len(admission.launched) == 1
    today = _submit(admission, "legacy-today")
    contract_payload = {**previous_payload, "admission_options": {"stack_settings": stack_settings.contract()}}
    assert today["signature"] == hashlib.sha256(json.dumps(contract_payload, sort_keys=True).encode()).hexdigest()
    assert today["admission_options"] == {"stack_settings": stack_settings.contract()}
