"""Independent P0 review: fixed recipes, parallel equivalence and cache identity.

These tests use synthetic arrays/files; they do not establish photographic
quality or the calibration of a real camera.
"""
import hashlib
import json
import threading

import numpy as np
from PIL import Image
import pytest

from starstack import color_adjustments as ca, core, display_transform as dt, references, imaging, siril


def test_channel_gain_matrix_independent_derivation():
    # Independent basis, non-diagonal and non-orthogonal; multiplication order
    # bugs cannot hide behind identity matrices or equal WB gains.
    transform = np.array([[.7, .2, .1], [.11, .81, .08], [.03, .09, .88]], dtype=float)
    original = np.array([2.1, 1., 1.4])
    start = np.array([1.8, 1., 1.7])
    camera = np.random.default_rng(742).uniform(.001, .15, size=(13, 17, 3))
    master_matrix = transform @ np.diag(original)
    master = camera @ master_matrix.T
    metadata = {"gamut": "rec2020", "colour_basis": {"gains_rgb": original.tolist(),
                 "master_matrix": master_matrix.tolist()}}
    params = dt.Params(wb_gains_rgb=tuple(start))
    for warmth, tint, exposure in [(-100, 100, -4), (100, -100, 4), (23, -17, .7)]:
        chosen = ca.apply(params, metadata, {"warmth": warmth, "tint": tint, "exposure_ev": exposure})
        expected_gains = start * np.exp2([warmth/100+tint/200, 0, -warmth/100+tint/200])
        expected = (camera * expected_gains) @ transform.T * (2.**exposure)
        actual = master @ np.array(chosen.wb_matrix).T
        assert np.allclose(actual, expected, atol=1e-13, rtol=1e-13)
        assert chosen.anchor == params.anchor and chosen.white == params.white
    assert ca.apply(params, metadata, {}) is params


@pytest.mark.parametrize("workers", [2, 3, 4, 8])
@pytest.mark.parametrize("rows", [1, 7, 32, 128])
def test_parallel_is_exact_on_odd_noncontiguous_arrays(workers, rows):
    rng = np.random.default_rng(993)
    original = rng.uniform(-.002, .4, size=(259, 76, 3)).astype(np.float32)
    data = original[:, ::2, :]
    source = original.copy()
    saturation = rng.uniform(size=data.shape[:2]) > .9
    regions = np.indices(data.shape[:2])[0] % 3
    params = dt.Params(saturation=1.5, shadow_smooth_radius=16,
                       shadow_smooth_start=.01, shadow_smooth_end=.09,
                       shadow_chroma_end=.003, wb_matrix=((1.2,.02,0),(-.01,.9,.02),(.03,0,1.1)))
    serial_flags = np.zeros(data.shape[:2], np.uint8)
    parallel_flags = np.full(data.shape[:2], 255, np.uint8)
    serial, serial_stats = dt.render(data, params, camera_saturated=saturation, region=regions,
                                     flags=serial_flags, rows=rows)
    parallel, parallel_stats = dt.render(data, params, camera_saturated=saturation, region=regions,
                                        flags=parallel_flags, rows=rows, workers=workers)
    assert np.array_equal(serial, parallel)
    assert np.array_equal(serial_flags, parallel_flags)
    assert serial_stats == parallel_stats
    assert np.array_equal(original, source)


def test_callback_stays_serial_even_when_workers_requested():
    threads = set()
    def look(values):
        threads.add(threading.get_ident())
        return values
    dt.render(np.full((259, 8, 3), .03, np.float32), dt.Params(), look=look, workers=8)
    assert threads == {threading.get_ident()}


def _cache(tmp_path):
    values = np.full((24, 36, 3), .012, np.float32)
    saturation = np.zeros((24, 36), bool)
    np.save(tmp_path / "preview_linear.npy", values, allow_pickle=False)
    np.save(tmp_path / "preview_saturated.npy", saturation, allow_pickle=False)
    cache = {"version": 1, "linear": str(tmp_path / "preview_linear.npy"),
             "saturated": str(tmp_path / "preview_saturated.npy"),
             "linear_sha256": core._sha(tmp_path / "preview_linear.npy"),
             "saturated_sha256": core._sha(tmp_path / "preview_saturated.npy"),
             "master_metadata": {}, "base_params": dt.Params().to_json()}
    cache["signature"] = hashlib.sha256(json.dumps(cache, sort_keys=True, allow_nan=False).encode()).hexdigest()
    record = {"state": "succeeded", "reference_dir": str(tmp_path), "color_cache": cache}
    (tmp_path / "reference.json").write_text(json.dumps(record), encoding="utf-8")
    return values, saturation, cache


def test_cache_array_replacement_after_validation_cannot_change_render(tmp_path, monkeypatch):
    expected, saturated, cache = _cache(tmp_path)
    original = np.load
    changed = [False]
    def race(file, *args, **kwargs):
        if not changed[0]:
            np.save(cache["linear"], np.full_like(expected, .9), allow_pickle=False)
            changed[0] = True
        return original(file, *args, **kwargs)
    monkeypatch.setattr(np, "load", race)
    result = references.render_reference(tmp_path)
    assert changed[0]
    expected_rgb, _ = dt.render(expected, dt.Params(), camera_saturated=saturated)
    assert np.array_equal(np.asarray(Image.open(result["preview_path"])), dt.to_8bit(expected_rgb))


@pytest.mark.parametrize("kind", ["recipe", "array", "external"])
def test_cache_rejects_changed_identity_before_render(tmp_path, kind):
    expected, saturated, cache = _cache(tmp_path)
    if kind == "recipe":
        cache["base_params"]["target"] = .4
    elif kind == "array":
        np.save(cache["linear"], expected * 2, allow_pickle=False)
    else:
        elsewhere = tmp_path.parent / "elsewhere.npy"
        np.save(elsewhere, expected, allow_pickle=False)
        cache["linear"] = str(elsewhere)
        # Re-signed metadata is not a licence to leave the reference directory.
        payload = {k: v for k, v in cache.items() if k != "signature"}
        cache["signature"] = hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()
    record_path = tmp_path / "reference.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["color_cache"] = cache
    record_path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError):
        references.render_reference(tmp_path)


def test_parallel_invalid_array_is_value_error():
    with pytest.raises(ValueError):
        dt.render([[[.1,.1,.1]]], dt.Params(), workers=4)


def test_reference_colour_uses_staged_identity_and_refuses_source_change(tmp_path, monkeypatch):
    source_dir = tmp_path / "inputs"
    source_dir.mkdir()
    source = source_dir / "one.nef"
    source.write_bytes(b"original camera metadata and pixels")
    output = tmp_path / "output"
    monkeypatch.setattr(core, "doctor", lambda *_: {"available": True, "executable": "fixture", "version": "1.4.4"})
    monkeypatch.setattr(references, "_optional_raw_metadata", lambda *_: {"available": False})
    def convert(executable, directory, name, commands, check):
        imaging.write_linear(directory / "work" / "ref_00001.fit", np.full((24, 36, 3), .01, np.float32), "fixture")
        source.write_bytes(b"replacement of original during conversion")
    monkeypatch.setattr(siril, "run_script", convert)
    seen = []
    def colour(data, basis_source, directory, adjustments):
        seen.append(basis_source)
        assert basis_source != source
        assert basis_source.read_bytes() == b"original camera metadata and pixels"
        return np.zeros((24, 36, 3), np.uint8), None
    monkeypatch.setattr(references, "_prepare_color_preview", colour)
    with pytest.raises(RuntimeError, match="变化|改变"):
        references.prepare_reference(None, str(output), input_files=[str(source)], color_adjustments={})
    assert len(seen) == 1
    receipts = list(output.glob("reference-*/reference.json"))
    assert len(receipts) == 1
    assert json.loads(receipts[0].read_text(encoding="utf-8"))["state"] == "failed"


def test_job_colour_reference_replacement_during_basis_never_publishes_calibrated_result(tmp_path, monkeypatch):
    from starstack import color_export as ce, master_export as me
    source = tmp_path / "selected-first.nef"
    source.write_bytes(b"admitted raw identity")
    frozen_hash = core._sha(source)
    products = tmp_path / "products"
    products.mkdir()
    calls = []
    def basis(path):
        calls.append(path)
        source.write_bytes(b"replacement during metadata read")
        return {"would_have_been": "calibrated basis"}
    monkeypatch.setattr(ce, "job_colour_basis", basis)
    monkeypatch.setattr(ce, "export_job_products", lambda *a, **k: pytest.fail("changed source reached calibrated export"))
    job = {"input_kind": "raw", "inputs": [{"path": str(source), "sha256": frozen_hash}], "mask": None,
           "color_adjustments": {"warmth": 0., "tint": 0., "exposure_ev": 0.}}
    image = np.full((18, 24, 3), .02, np.float32)
    artifacts, record = core._colour_products(tmp_path, job, tmp_path, image, products,
                                              products / "base.fit", lambda: None)
    assert calls == [str(source)]
    assert record["state"] == "fallback_legacy"
    assert "颜色参考原片在读取期间发生变化" in record["error"]
    assert artifacts and all(item.get("contract") not in {"M", "D"} for item in artifacts)
    assert not (products / me.FILENAME).exists()
    assert "读取期间发生变化" in (tmp_path / "worker.log").read_text(encoding="utf-8")
