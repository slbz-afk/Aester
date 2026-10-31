"""GUI/MCP preview re-render on M/D jobs goes through the single display transform (docs/PRE_RELEASE_CLOSURE_20260922.md G):
unity controls reproduce the job's own D byte for byte, gains/exposure map onto the recipe, legacy jobs are untouched."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile
from PIL import Image

from starstack import color_export as ce, core, imaging, master_export as me, preview
from starstack.photographic_job import _display_source

sys.path.insert(0, str(Path(__file__).resolve().parent))
import color_fixtures as cf  # noqa: E402


@pytest.fixture
def md_job(tmp_path):
    """A succeeded RAW job whose products were written by the M/D writer, the way core records it."""
    dng = cf.write_reference_dng(tmp_path / "frame00001.dng")
    cam, sky, *_ = cf.synthetic_camera_master()
    job_dir = tmp_path / "job-md"
    products = job_dir / "products" / "abc123"
    products.mkdir(parents=True)
    composite = products / "master_linear.fit"
    imaging.write_linear(composite, np.flipud(cam).copy(), "synthetic composite")
    mask = job_dir / "sky_mask.png"
    Image.fromarray((sky * 255).astype(np.uint8)).save(mask)
    artifacts, record = ce.export_job_products(np.flipud(cam).copy(), products, lambda: None, basis=ce.job_colour_basis(dng),
                                               source_fits=composite, sky_mask_path=mask)
    job = {"job_dir": str(job_dir), "state": "succeeded", "input_kind": "raw", "mask": {"path": str(mask), "sha256": core._sha(mask)},
           "artifacts": [{"path": str(composite), "kind": "fits", "label": "合成线性母版"}, *artifacts],
           "diagnostics": {"color_export": record}}
    core.atomic_json(job_dir / "job.json", job)
    return job_dir, products, record


def _tree(root: Path) -> dict:
    return {p: core._sha(p) for p in root.rglob("*") if p.is_file() and "preview-variations" not in p.parts}


def test_unity_preview_reproduces_the_job_d_and_leaves_the_job_untouched(md_job):
    job_dir, products, record = md_job
    before = _tree(job_dir)
    result = preview.render_job_preview(str(job_dir))
    assert result["state"] == "succeeded" and result["schema_version"] == 2 and result["source_contract"] == "M"
    kinds = [a["kind"] for a in result["artifacts"]]
    assert kinds == ["jpeg", "tiff"] and result["recipe_label"] == "photo-default-1"
    tiff = Path(result["artifacts"][1]["path"])
    assert Path(result["directory"]).parent == job_dir / "preview-variations" and tiff.parent == Path(result["directory"])
    np.testing.assert_array_equal(tifffile.imread(tiff), tifffile.imread(products / ce.D_TIFF_NAME))
    with Image.open(result["artifacts"][0]["path"]) as new, Image.open(products / ce.D_JPEG_NAME) as old:
        np.testing.assert_array_equal(np.asarray(new), np.asarray(old))
    shape, icc, meta = _display_source(tiff)                     # the preview TIFF satisfies the D contract
    assert meta["display_transform"] == record["photo"]["display_transform"] and meta["recipe"] == "photo-default-1"
    assert _tree(job_dir) == before and json.loads(Path(result["recipe_path"]).read_text(encoding="utf-8")) == result
    assert result["source_sha256"] == core._sha(Path(record["master"]["path"]))


def test_gains_and_exposure_map_onto_the_recipe(md_job):
    job_dir, products, record = md_job
    base = record["photo"]["display_transform"]
    result = preview.render_job_preview(str(job_dir), red_gain=1.3, blue_gain=.8, exposure_ev=1.0)
    transform = result["display_transform"]
    assert result["recipe_label"] == "photo-default-1+preview-gains" and transform["wb_mode"] == "as_shot"
    assert _display_source(Path(result["artifacts"][1]["path"]))[2]["recipe"] == "photo-default-1+preview-gains"
    expected = (base["wb_gains_rgb"][0] * 1.3, base["wb_gains_rgb"][1], base["wb_gains_rgb"][2] * .8)
    assert np.allclose(transform["wb_gains_rgb"], expected)
    assert transform["wb_matrix"] != base["wb_matrix"]
    # +1 EV halves the anchor measured under the new matrix; the measurement itself moves only through the matrix
    unity = preview.render_job_preview(str(job_dir), red_gain=1.3, blue_gain=.8, exposure_ev=0.0)["display_transform"]
    assert np.isclose(transform["anchor"], unity["anchor"] / 2.0) and unity["wb_matrix"] == transform["wb_matrix"]
    assert unity["anchor"] > 0 and unity["anchor"] != base["anchor"]              # measured under the re-balanced luminance
    assert not np.array_equal(tifffile.imread(result["artifacts"][1]["path"]), tifffile.imread(products / ce.D_TIFF_NAME))
    assert (Path(result["directory"]) / "photo_render.json").is_file() and not (Path(result["directory"]) / "render_flags.png").exists()


def test_legacy_job_without_m_keeps_the_three_tone_path(tmp_path):
    job_dir = tmp_path / "job-legacy"
    products = job_dir / "products" / "original"
    products.mkdir(parents=True)
    data = np.full((12, 16, 3), .02, np.float32)
    master = products / "master_linear.fit"
    imaging.write_linear(master, data, "legacy")
    artifacts = [{"kind": "fits", "path": str(master), "label": "合成线性母版"}] + imaging.export_products(data, products, lambda: None)
    core.atomic_json(job_dir / "job.json", {"job_dir": str(job_dir), "state": "succeeded", "artifacts": artifacts,
                                            "diagnostics": {"color_export": {"state": "legacy"}}})
    result = preview.render_job_preview(str(job_dir))
    assert result["schema_version"] == 1 and sum(a["kind"] == "jpeg" for a in result["artifacts"]) == 3


def test_master_recorded_outside_the_job_is_refused(md_job, tmp_path):
    job_dir, products, record = md_job
    job = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / me.FILENAME).write_bytes(Path(record["master"]["path"]).read_bytes())
    job["diagnostics"]["color_export"]["master"]["path"] = str(elsewhere / me.FILENAME)
    core.atomic_json(job_dir / "job.json", job)
    with pytest.raises(ValueError, match="任务目录内"):
        preview.render_job_preview(str(job_dir))


def test_mcp_service_returns_the_two_bounded_artifacts(md_job, tmp_path):
    from starstack.mcp_server import WorkspaceTools
    job_dir, products, record = md_job
    outputs = job_dir.parent
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    service = WorkspaceTools(input_root=inputs, output_root=outputs)
    result = service.render_job_preview(str(job_dir), red_gain=1.0, blue_gain=1.0, exposure_ev=0.0)
    assert [a["kind"] for a in result["artifacts"]] == ["jpeg", "tiff"]
    assert all(Path(a["path"]).is_relative_to(job_dir / "preview-variations") for a in result["artifacts"])
