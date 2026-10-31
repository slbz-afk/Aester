"""Small independent checks for the real-image diagnostic orchestrator.

The mocked orchestration check is explicitly not real Siril evidence. Actual
engine smoke runs are separate artifacts produced by the runner CLI.
"""
import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/verify_image_holdout.py"
spec = importlib.util.spec_from_file_location("image_holdout_runner_tested", SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def _write(path, data, order="TOP-DOWN", extra=None):
    image = fits.PrimaryHDU(data=data if order == "TOP-DOWN" else data[:, ::-1])
    image.header["ROWORDER"] = order
    for key, value in (extra or {}).items():
        image.header[key] = value
    image.writeto(path)


def _plan(tmp_path, *, count=3):
    data = np.arange(3 * 16 * 24, dtype=np.float32).reshape(3, 16, 24) / 10000
    frames = []
    for index in range(count):
        path = tmp_path / f"input-{index}.fit"
        _write(path, data)
        frames.append({"id": f"input-{index}", "path": str(path.resolve()), "sha256": runner.digest(path)})
    return {"version": "1.0.0", "width": 24, "height": 16, "reference_index": 1,
            "inputs": frames, "test_rectangles_xyxy": [[0, 0, 8, 16], [17, 0, 24, 16]],
            "exclusion_guard_px": 2, "detection_guard_px": 1, "association_radius_px": 1.5}


def _nearest_free(reference, predicted, radius):
    # All contenders are counted on both sides; no nearest-winner fallback.
    near = np.sum((reference[:, None, :] - predicted[None, :, :])**2, axis=2) <= radius**2
    result = []
    for index, row in enumerate(near):
        choices = np.flatnonzero(row)
        unique = len(choices) == 1 and np.count_nonzero(near[:, choices[0]]) == 1
        source = int(choices[0]) if unique else None
        result.append({"reference_index": index, "source_index": source,
                       "status": "matched" if unique else "ambiguous" if len(choices) else "missing",
                       "candidate_count": len(choices),
                       "distance_px": float(np.linalg.norm(reference[index]-predicted[source])) if unique else None})
    return result


def test_masks_are_disjoint_and_exclusion_guard_is_not_test_data(tmp_path):
    plan = _plan(tmp_path)
    train, test = runner.role_masks(plan)
    expected_test = np.zeros((16, 24), bool)
    expected_test[:, :8] = expected_test[:, 17:] = True
    expected_train = np.zeros_like(expected_test)
    expected_train[:, 10:15] = True
    np.testing.assert_array_equal(test, expected_test)
    np.testing.assert_array_equal(train, expected_train)
    assert not (test & train).any()
    assert not train[:, 8:10].any() and not test[:, 8:10].any()


@pytest.mark.parametrize("order", ["TOP-DOWN", "BOTTOM-UP"])
def test_splitter_preserves_active_bits_row_order_and_original_source(tmp_path, order):
    data = np.arange(3 * 7 * 11, dtype=np.float32).reshape(3, 7, 11) / 999
    data[0, 3, 5] = np.float32(-0.)
    source, target = tmp_path / "original.fit", tmp_path / "view.fit"
    _write(source, data, order=order, extra={"GLOBALAV": .723})
    original_sha = runner.digest(source)
    allowed = np.zeros((7, 11), bool)
    allowed[1:6, 2:9] = True
    runner.materialize_view(source, target, allowed, rows=2)
    with fits.open(target) as hdul:
        actual = hdul[0].data.copy()
        if order == "BOTTOM-UP":
            actual = actual[:, ::-1]
        assert hdul[0].header["ROWORDER"] == order
        assert "GLOBALAV" not in hdul[0].header
    expected = data.copy()
    expected[:, ~allowed] = 0
    assert actual.astype(np.float32).tobytes() == expected.tobytes()
    assert runner.digest(source) == original_sha
    with pytest.raises(FileExistsError):
        runner.materialize_view(source, target, allowed)


def test_altering_all_hidden_pixels_and_global_headers_cannot_change_training_bytes(tmp_path):
    plan = _plan(tmp_path)
    train, test = runner.role_masks(plan)
    first = np.random.default_rng(94).normal(size=(3, 16, 24)).astype(np.float32)
    altered = first.copy()
    altered[:, ~train] += 19
    a, b = tmp_path / "a.fit", tmp_path / "b.fit"
    _write(a, first, extra={"GLOBALAV": float(first.mean())})
    _write(b, altered, extra={"GLOBALAV": float(altered.mean())})
    a_train, b_train = tmp_path / "a-train.fit", tmp_path / "b-train.fit"
    runner.materialize_view(a, a_train, train)
    runner.materialize_view(b, b_train, train)
    assert runner.digest(a_train) == runner.digest(b_train)
    a_test, b_test = tmp_path / "a-test.fit", tmp_path / "b-test.fit"
    runner.materialize_view(a, a_test, test)
    runner.materialize_view(b, b_test, test)
    assert runner.digest(a_test) != runner.digest(b_test)


@pytest.mark.parametrize("order", ["TOP-DOWN", "BOTTOM-UP"])
def test_unsigned16_fixed_scale_and_arbitrary_mask_match_scalar_truth(tmp_path, order):
    data = np.array([0, 1, 255, 32767, 32768, 65534, 65535], dtype=np.uint16)
    data = np.resize(data, (3, 7, 11))
    source, target = tmp_path / "u16.fit", tmp_path / "u16-view.fit"
    _write(source, data, order=order)
    saved = runner.digest(source)
    mask = np.random.default_rng(174).random((7, 11)) > .3
    runner.materialize_view(source, target, mask, rows=3)
    with fits.open(target) as hdul:
        actual = hdul[0].data.copy()
        if order == "BOTTOM-UP":
            actual = actual[:, ::-1]
        assert hdul[0].header["BITPIX"] == -32
        assert "BZERO" not in hdul[0].header
    expected = np.zeros(data.shape, np.float32)
    for c in range(3):
        for y in range(7):
            for x in range(11):
                if mask[y, x]:
                    expected[c, y, x] = np.float32(int(data[c, y, x])) / np.float32(65535.)
    assert actual.astype(np.float32).tobytes() == expected.tobytes()
    assert runner.digest(source) == saved
    changed = data.copy()
    changed[:, ~mask] ^= np.uint16(65535)
    alternate, alternate_view = tmp_path / "u16-alternate.fit", tmp_path / "u16-alternate-view.fit"
    _write(alternate, changed, order=order)
    runner.materialize_view(alternate, alternate_view, mask, rows=1)
    assert runner.digest(target) == runner.digest(alternate_view)


@pytest.mark.parametrize("key,value", [("width", True), ("height", 32767), ("reference_index", -1),
                                       ("reference_index", True), ("exclusion_guard_px", -1),
                                       ("detection_guard_px", .5), ("association_radius_px", 0.),
                                       ("association_radius_px", float("nan")), ("association_radius_px", True),
                                       ("test_rectangles_xyxy", [[0, 0, 25, 16]])])
def test_plan_invalid_boundaries_rejected(tmp_path, key, value):
    plan = _plan(tmp_path)
    plan[key] = value
    with pytest.raises(ValueError):
        runner.validate_plan(plan)


def test_plan_duplicate_paths_ids_unknown_fields_and_empty_training_rejected(tmp_path):
    plan = _plan(tmp_path)
    for key in ("path", "id"):
        bad = copy.deepcopy(plan)
        bad["inputs"][1][key] = bad["inputs"][0][key]
        with pytest.raises(ValueError):
            runner.validate_plan(bad)
    bad = dict(plan, tune_after_test=True)
    with pytest.raises(ValueError):
        runner.validate_plan(bad)
    plan["test_rectangles_xyxy"] = [[0, 0, 24, 16]]
    with pytest.raises(ValueError):
        runner.role_masks(plan)


def test_rebase_changes_coordinate_reference_without_modifying_any_fit():
    original = {"a": np.eye(3), "b": np.array([[1., 0., 5.], [0., 1., 3.], [0., 0., 1.]]),
                "c": np.array([[1., 0., 9.], [0., 1., 7.], [0., 0., 1.]])}
    saved = {k: v.copy() for k, v in original.items()}
    got = runner.rebase(original, "b")
    np.testing.assert_array_equal(got["b"], np.eye(3))
    np.testing.assert_array_equal(got["c"], [[1, 0, 4], [0, 1, 4], [0, 0, 1]])
    for key in original:
        np.testing.assert_array_equal(original[key], saved[key])
    assert runner.rebase(original, "absent") == {}


def test_missing_frame_prevents_global_isolation_and_no_candidate_gate_hides_large_error(tmp_path, monkeypatch):
    plan = _plan(tmp_path)
    monkeypatch.setattr(runner, "_associate", _nearest_free)
    xy = np.array([[3., 5.], [20., 7.]])
    detections = {"light_00001": xy.copy(), "light_00002": xy.copy(), "light_00003": xy[:1].copy()}
    test_h = {key: np.eye(3) for key in detections}
    tracks = runner.associate_test_tracks(plan, detections, test_h)
    assert tracks["reference_count"] == 2
    assert tracks["globally_hidden_reference_indices"] == [0]
    assert tracks["frames"]["light_00001"][1]["global_track_isolation"] == "unknown"
    candidate = {"models": {"homography": {"frames": {key: [[1, 0, 20], [0, 1, 0], [0, 0, 1]] for key in detections}, "distortion": None}}}
    result = runner.evaluate_candidate(plan, candidate, tracks)
    frame = result["models"]["homography"]["light_00001"]
    assert frame["all_conditionally_matched"]["measured"] == 2
    assert frame["all_conditionally_matched"]["max_px"] == 20
    assert frame["complete_hidden_tracks"]["measured"] == 1
    assert frame["complete_hidden_tracks"]["reference_denominator"] == 2
    assert result["judgement"]["status"] == "insufficient_data"
    assert "light_00002" not in result["models"]["homography"]


def test_guard_and_ambiguous_detections_stay_in_denominator(tmp_path, monkeypatch):
    plan = _plan(tmp_path)
    monkeypatch.setattr(runner, "_associate", _nearest_free)
    detections = {"light_00001": np.array([[3., 5.], [3.2, 5.], [.1, 8.]]),
                  "light_00002": np.array([[3., 5.], [.1, 8.]]),
                  "light_00003": np.array([[3., 5.], [.1, 8.]])}
    tracks = runner.associate_test_tracks(plan, detections, {key: np.eye(3) for key in detections})
    assert tracks["reference_count"] == 2
    assert tracks["frames"]["light_00001"][0]["status"] == "ambiguous"
    assert tracks["frames"]["light_00001"][1]["status"] == "reference_guard"
    assert not tracks["globally_hidden_reference_indices"]


def test_mocked_full_runner_reads_test_only_after_model_freeze(tmp_path, monkeypatch):
    plan = _plan(tmp_path)
    plan_path = tmp_path / "plan.json"
    runner.write_json(plan_path, plan)
    monkeypatch.setattr(runner, "source_manifest", lambda: {"fixture": "unchanged"})
    monkeypatch.setattr(runner.siril, "doctor", lambda executable: {"available": True, "executable": "FAKE", "version": "1.4.4"})
    from starstack import registered_support
    monkeypatch.setattr(registered_support, "require_engine", lambda engine: None)
    monkeypatch.setattr(runner.calibration, "calibrate", lambda *args, **kwargs: {"status": "unavailable", "reason": "explicit unit fixture"})
    monkeypatch.setattr(runner, "_associate", _nearest_free)
    calls = []

    def fake_probe(executable, phase):
        calls.append(phase.name)
        assert (phase.parent / "candidate.json").exists() == (phase.name == "test")
        if phase.name == "train":
            assert not (phase.parent / "test").exists()
        # Both mocked probes choose frame 1 automatically, different from the
        # plan's frame 2 comparison reference. No post-hoc seq header rewrite.
        lines = ["S 'light_' 1 3 3 5 0 6 0 0 0", "L 3"]
        lines += [f"I {i} 1" for i in range(1, 4)]
        lines += ["R1 3 3 1 1 0 300 H 1 0 0 0 1 0 0 0 1"] * 3
        (phase / "work/light_.seq").write_text("\n".join(lines) + "\n", encoding="utf-8")

    monkeypatch.setattr(runner, "run_probe", fake_probe)
    monkeypatch.setattr(runner, "load_detections", lambda work, stems, height: (
        {s: np.array([[3., 5.], [20., 7.]]) for s in stems}, {s: {"state": "unit_fixture"} for s in stems}))
    result = runner.run_plan(plan_path, tmp_path / "results")
    assert result["state"] == "completed", result
    assert calls == ["train", "test"]
    assert result["globally_hidden_tracks"] == 2
    assert result["precision_passed"] is False
    assert result["judgement"]["status"] == "insufficient_data"
    stages = [event["stage"] for event in result["events"]]
    assert stages.index("candidate_frozen") < stages.index("test_views_ready")


def test_explicit_freeze_covers_transitive_dependencies_and_excludes_unrelated_s4():
    manifest = runner.source_manifest()
    for name in ("__init__", "registration_diagnostics", "sip_header", "process_guard", "resample",
                 "siril_support_map", "support_anchor", "sky_validity", "records", "holdout_association",
                 "finite_stack", "stack_range", "range_tiles", "support_kernel"):
        assert str(Path("src/starstack") / (name + ".py")) in manifest
    assert str(Path("scripts/verify_image_holdout.py")) in manifest
    assert not any("transient" in name or Path(name).stem in ("core", "gui") for name in manifest)


def test_new_local_import_fails_freeze_instead_of_silently_omitting_dependency(monkeypatch):
    monkeypatch.setattr(runner, "_local_module_names", lambda text: {"transient_stage"})
    with pytest.raises(ValueError, match="unfrozen local imports"):
        runner.source_manifest()


def test_import_closure_parser_includes_lazy_relative_and_absolute_project_imports():
    source = ("from . import fits_crop, siril_sequence\n"
              "from .registration_models import apply_matrix\n"
              "def lazy():\n"
              "    from . import resample\n"
              "    from starstack.holdout_association import associate_holdout\n"
              "    import starstack.process_guard\n"
              "    from starstack import records\n")
    assert runner._local_module_names(source) == {"fits_crop", "siril_sequence", "registration_models", "resample",
                                                "holdout_association", "process_guard", "records"}
