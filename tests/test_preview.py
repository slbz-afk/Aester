"""Display edits must preserve masters, geometry, and original job results."""
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image
import pytest
import tifffile

from starstack import core, imaging, preview


@pytest.fixture
def completed(tmp_path):
    job_dir = tmp_path / "job-completed"
    products = job_dir / "products" / "original"
    products.mkdir(parents=True)
    y, x = np.mgrid[:23, :31]
    data = np.stack((.03 + x * .001, .02 + y * .001, .01 + x * .0005), -1).astype(np.float32)
    data[0, 0] = (-.03, .3, 2.0)
    master = products / "master_linear.fit"
    imaging.write_linear(master, data, "Asymmetric test, including negative and HDR values")
    artifacts = [{"kind": "fits", "path": str(master), "label": "合成线性母版"}]
    artifacts += imaging.export_products(data, products, lambda: None)
    core.atomic_json(job_dir / "job.json", {"job_dir": str(job_dir), "state": "succeeded", "artifacts": artifacts})
    return job_dir, master, data


def test_variations_preserve_all_original_files_and_unity_matches_existing_export(completed, monkeypatch):
    directory, master, data = completed
    before = {p: core._sha(p) for p in directory.rglob("*") if p.is_file()}
    monkeypatch.setattr(core, "run_script", lambda *_: pytest.fail("Preview ran Siril"))
    result = preview.render_job_preview(str(directory))
    assert result["state"] == "succeeded"
    assert len(result["artifacts"]) == 4
    assert sum(a["kind"] == "jpeg" for a in result["artifacts"]) == 3
    for artifact in result["artifacts"]:
        path = Path(artifact["path"])
        original = master.parent / path.name
        # LittleCMS timestamps newly created ICC profiles. Equal rendering
        # means identical decoded pixels and colour metadata, not equal file
        # bytes across a wall-clock second boundary.
        if path.suffix == ".tif":
            with tifffile.TiffFile(path) as actual, tifffile.TiffFile(original) as expected:
                np.testing.assert_array_equal(actual.asarray(), expected.asarray())
                assert actual.pages[0].dtype == expected.pages[0].dtype == np.uint16
                for tag in ("ImageDescription", "PhotometricInterpretation", "BitsPerSample"):
                    assert actual.pages[0].tags[tag].value == expected.pages[0].tags[tag].value
                profile = actual.pages[0].tags["InterColorProfile"].value
                original_profile = expected.pages[0].tags["InterColorProfile"].value
        else:
            with Image.open(path) as actual, Image.open(original) as expected:
                assert actual.mode == expected.mode == "RGB"
                np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
                profile = actual.info["icc_profile"]
                original_profile = expected.info["icc_profile"]
        assert len(profile) == len(original_profile) and len(profile) >= 128
        assert profile[:24] == original_profile[:24]
        assert profile[36:] == original_profile[36:]
        assert artifact["sha256"] == core._sha(path)
    assert before == {p: core._sha(p) for p in before}
    np.testing.assert_array_equal(imaging.read_rgb(master), data)
    assert result["source_sha256"] == before[master]
    assert json.loads(Path(result["recipe_path"]).read_text(encoding="utf-8")) == result
    second = preview.render_job_preview(str(directory), red_gain=1.7, blue_gain=.7, exposure_ev=-.4)
    assert result["directory"] != second["directory"]
    assert before == {p: core._sha(p) for p in before}
    assert not (Path(second["directory"]) / "master_linear_32bit.tif").exists()


def test_exposure_and_channels_affect_only_expected_display_values():
    gradient = np.linspace(.01, .3, 120).reshape(10, 12)
    data = np.repeat(gradient[..., None], 3, axis=-1).astype(np.float32)
    original = data.copy()
    baseline = imaging.display_base(data)
    brighter = imaging.display_base(data, exposure_ev=1)
    red = imaging.display_base(data, red_gain=2)
    assert np.all(brighter >= baseline)
    assert np.count_nonzero(brighter > baseline) > 30
    np.testing.assert_array_equal(red[..., 1:], baseline[..., 1:])
    assert np.count_nonzero(red[..., 0] > baseline[..., 0]) > 10
    # The brightest source row is the displayed top; no second flip.
    assert baseline[0].mean() > baseline[-1].mean()
    np.testing.assert_array_equal(data, original)


@pytest.mark.parametrize("parameters", [
    {"red_gain": 0}, {"blue_gain": 4.1}, {"exposure_ev": -3.1},
    {"red_gain": float("nan")}, {"blue_gain": float("inf")},
    {"red_gain": True}, {"exposure_ev": "1"}, {"exposure_ev": None},
])
def test_invalid_parameters_rejected_before_output(completed, parameters):
    directory, _, _ = completed
    with pytest.raises(ValueError):
        preview.render_job_preview(str(directory), **parameters)
    assert not (directory / "preview-variations").exists()


def test_foreign_master_is_not_read_or_used(completed, tmp_path, monkeypatch):
    directory, master, _ = completed
    external = tmp_path / "master_linear.fit"
    external.write_bytes(master.read_bytes())
    job = core.load_job(directory)
    job["artifacts"][0]["path"] = str(external)
    core.atomic_json(directory / "job.json", job)
    monkeypatch.setattr(preview, "read_rgb", lambda _: pytest.fail("Read foreign pixels"))
    with pytest.raises(ValueError, match="任务目录"):
        preview.render_job_preview(str(directory))
    assert not (directory / "preview-variations").exists()


def test_failed_render_keeps_failed_recipe_without_published_artifacts(completed, monkeypatch):
    directory, master, _ = completed
    before = core._sha(master)
    def fail(*args, **kwargs):
        raise OSError("disk full")
    monkeypatch.setattr(preview, "export_display_products", fail)
    with pytest.raises(OSError, match="disk full"):
        preview.render_job_preview(str(directory))
    records = list((directory / "preview-variations").glob("*/recipe.json"))
    assert len(records) == 1
    record = json.loads(records[0].read_text(encoding="utf-8"))
    assert record["state"] == "failed" and record["artifacts"] == []
    assert core._sha(master) == before


def test_external_change_during_render_prevents_success(completed, monkeypatch):
    directory, _, _ = completed
    render = preview.export_display_products
    def changed(*args, **kwargs):
        result = render(*args, **kwargs)
        with (directory / "job.json").open("a", encoding="utf-8") as stream:
            stream.write("\n")
        return result
    monkeypatch.setattr(preview, "export_display_products", changed)
    with pytest.raises(ValueError, match="发生变化"):
        preview.render_job_preview(str(directory))
    record = json.loads(next((directory / "preview-variations").glob("*/recipe.json")).read_text(encoding="utf-8"))
    assert record["state"] == "failed" and record["artifacts"] == []


def test_relative_master_inside_job_is_supported_and_outputs_keep_size(completed):
    directory, master, data = completed
    job = core.load_job(directory)
    job["artifacts"][0]["path"] = str(master.relative_to(directory))
    core.atomic_json(directory / "job.json", job)
    result = preview.render_job_preview(str(directory), blue_gain=1.5)
    for artifact in result["artifacts"]:
        if artifact["kind"] == "jpeg":
            with Image.open(artifact["path"]) as image:
                assert image.size == (data.shape[1], data.shape[0])
        else:
            pixels = tifffile.imread(artifact["path"])
            assert pixels.shape == data.shape and pixels.dtype == np.uint16


@pytest.mark.parametrize("state", ["queued", "cancelled", "failed"])
def test_incomplete_job_cannot_render(completed, state):
    directory, _, _ = completed
    job = core.load_job(directory)
    job.update(state=state, updated_at=core.now())
    core.atomic_json(directory / "job.json", job)
    with pytest.raises(ValueError, match="已完成"):
        preview.render_job_preview(str(directory))


def test_extreme_finite_float_master_fails_explicitly_without_black_export(completed):
    directory, master, _ = completed
    data = np.linspace(-3.3e38, 3.3e38, 300, dtype=np.float64).astype(np.float32).reshape(10, 10, 3)
    assert np.isfinite(data).all()
    master.unlink()
    imaging.write_linear(master, data, "Finite but beyond current display arithmetic")
    before = core._sha(master)
    with pytest.raises(ValueError, match="数值"):
        preview.render_job_preview(str(directory))
    record = json.loads(next((directory / "preview-variations").glob("*/recipe.json")).read_text(encoding="utf-8"))
    assert record["state"] == "failed" and record["artifacts"] == []
    assert not list((directory / "preview-variations").glob("*/*.jpg"))
    assert core._sha(master) == before


def test_concurrent_variations_have_independent_recipes(completed):
    directory, master, _ = completed
    before = core._sha(master), core._sha(directory / "job.json")
    with ThreadPoolExecutor(max_workers=2) as pool:
        tasks = [pool.submit(preview.render_job_preview, str(directory), red_gain=g) for g in (1, 1.7)]
        results = [f.result(timeout=20) for f in tasks]
    assert len({r["directory"] for r in results}) == 2
    for result in results:
        assert result["state"] == "succeeded"
        assert json.loads(Path(result["recipe_path"]).read_text(encoding="utf-8")) == result
    assert before == (core._sha(master), core._sha(directory / "job.json"))


def test_final_recipe_io_failure_never_returns_success(completed, monkeypatch):
    directory, _, _ = completed
    write = core.atomic_json
    def fail_success(path, value):
        if value.get("state") == "succeeded":
            raise OSError("recipe commit failed")
        write(path, value)
    monkeypatch.setattr(core, "atomic_json", fail_success)
    with pytest.raises(OSError, match="recipe commit failed"):
        preview.render_job_preview(str(directory))
    record = json.loads(next((directory / "preview-variations").glob("*/recipe.json")).read_text(encoding="utf-8"))
    assert record["state"] == "failed" and record["artifacts"] == []
