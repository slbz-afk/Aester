"""Step 2 of the colour package (docs/COLOR_CONTRACT_PACKAGE_20260922.md §10): job products switch to M + D for RAW
inputs, legacy products stay for FITS inputs, and a failing colour writer falls back visibly (S2c–S2e synthetic)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile
from PIL import Image

from starstack import color_export as ce, core, imaging, master_export as me, photo_render as pr
from starstack.photographic_job import _display_source

sys.path.insert(0, str(Path(__file__).resolve().parent))
import color_fixtures as cf  # noqa: E402


@pytest.fixture
def composite(tmp_path):
    """A synthetic composite saved the way core saves it (bottom-up FITS) plus the reference DNG and the sky mask."""
    dng = cf.write_reference_dng(tmp_path / "frame00001.dng")
    cam, sky, saturated, nodata = cf.synthetic_camera_master()
    export_dir = tmp_path / "products" / "abc123"
    export_dir.mkdir(parents=True)
    bottom_up = np.flipud(cam).copy()
    fits_path = export_dir / "master_linear.fit"
    imaging.write_linear(fits_path, bottom_up, "synthetic composite")
    mask = tmp_path / "sky_mask.png"
    Image.fromarray((sky * 255).astype(np.uint8)).save(mask)
    return dng, cam, sky, bottom_up, export_dir, fits_path, mask


def test_export_job_products_writes_m_and_d_with_legacy_names_and_default_recipe(composite, tmp_path):
    dng, cam, sky, bottom_up, export_dir, fits_path, mask = composite
    basis = ce.job_colour_basis(dng)
    artifacts, record = ce.export_job_products(bottom_up, export_dir, lambda: None, basis=basis, source_fits=fits_path,
                                               sky_mask_path=mask)
    names = sorted(p.name for p in export_dir.iterdir())
    assert names == sorted(["master_linear.fit", me.FILENAME, me.MASK_FILENAME, me.RECIPE_FILENAME, ce.D_TIFF_NAME, ce.D_JPEG_NAME,
                            pr.RECORD_NAME, pr.FLAGS_NAME])
    assert [a["contract"] for a in artifacts] == ["M", "D", "D"] and [a["kind"] for a in artifacts] == ["tiff", "tiff", "jpeg"]
    assert record["state"] == "succeeded" and record["contract"] == "M+D photo-default-1" and record["photo"]["recipe"] == "photo-default-1"
    # M is the TOP-DOWN master of exactly this composite, reproducible by the standalone exporter (S2a)
    standalone = me.export_master(fits_path, basis, tmp_path / "M")
    assert standalone["master_sha256"] == record["master"]["sha256"]
    # D under the legacy name satisfies the display contract and equals a standalone default render (S2b)
    shape, icc_bytes, meta = _display_source(export_dir / ce.D_TIFF_NAME)
    assert meta["recipe"] == "photo-default-1" and meta["display_transform"]["wb_mode"] == "as_shot"
    data, mmeta, _ = me.read_master(standalone["master"])
    params = pr.default_params(mmeta, data, sky=sky)
    other = pr.render_photo(standalone["master"], tmp_path / "D", params, sky_mask=mask)
    assert np.array_equal(tifffile.imread(other["tiff"]), tifffile.imread(export_dir / ce.D_TIFF_NAME))
    assert other["metadata"]["display_transform"] == meta["display_transform"]
    # no legacy products (S2c)
    assert not any((export_dir / n).exists() for n in ce.LEGACY_NAMES)


def test_core_colour_products_uses_m_d_for_raw_jobs_and_legacy_otherwise(composite, tmp_path, monkeypatch):
    dng, cam, sky, bottom_up, export_dir, fits_path, mask = composite
    job_dir = tmp_path / "job"
    (job_dir / "work").mkdir(parents=True)
    Image.fromarray((sky * 255).astype(np.uint8)).save(job_dir / "sky_mask.png")
    monkeypatch.setattr(core, "sequence_reference", lambda seq, inputs: {"reference_input": str(dng)})
    raw_job = {"input_kind": "raw", "inputs": [{"path": str(dng)}], "mask": {"path": str(mask), "sha256": "0" * 64}}
    artifacts, record = core._colour_products(job_dir, raw_job, job_dir / "work", bottom_up, export_dir, fits_path, lambda: None)
    assert record["state"] == "succeeded" and record["reference_input"] == str(dng) and len(artifacts) == 3
    assert (export_dir / ce.D_JPEG_NAME).is_file() and not (export_dir / "cool.jpg").exists()
    # FITS inputs: legacy products, untouched behaviour (S2e)
    legacy_dir = tmp_path / "products" / "legacy"
    legacy_dir.mkdir()
    fits_job = {"input_kind": "fits", "inputs": [{"path": "x.fit"}], "mask": None}
    artifacts, record = core._colour_products(job_dir, fits_job, job_dir / "work", bottom_up, legacy_dir, fits_path, lambda: None)
    assert record["state"] == "legacy" and {Path(a["path"]).name for a in artifacts} == {"master_linear_32bit.tif", "natural_display_16bit.tif",
                                                                                          "natural.jpg", "cool.jpg", "warm.jpg"}


def test_core_falls_back_to_legacy_products_visibly_when_the_colour_writer_fails(composite, tmp_path, monkeypatch):
    dng, cam, sky, bottom_up, export_dir, fits_path, mask = composite
    job_dir = tmp_path / "job"
    (job_dir / "work").mkdir(parents=True)
    monkeypatch.setattr(core, "sequence_reference", lambda seq, inputs: {"reference_input": str(dng)})

    def boom(*args, **kwargs):
        (export_dir / me.FILENAME).write_bytes(b"partial")          # a half-written product must not survive the fallback
        raise RuntimeError("synthetic colour failure")
    monkeypatch.setattr(ce, "export_job_products", boom)
    raw_job = {"input_kind": "raw", "inputs": [{"path": str(dng)}], "mask": None}
    artifacts, record = core._colour_products(job_dir, raw_job, job_dir / "work", bottom_up, export_dir, fits_path, lambda: None)
    assert record["state"] == "fallback_legacy" and "synthetic colour failure" in record["error"]
    assert {Path(a["path"]).name for a in artifacts} == {"master_linear_32bit.tif", "natural_display_16bit.tif", "natural.jpg", "cool.jpg", "warm.jpg"}
    assert not (export_dir / me.FILENAME).exists()
    assert "synthetic colour failure" in (job_dir / "worker.log").read_text(encoding="utf-8")
    # cancellation is never swallowed
    monkeypatch.setattr(ce, "export_job_products", lambda *a, **k: (_ for _ in ()).throw(core.Cancelled()))
    with pytest.raises(core.Cancelled):
        core._colour_products(job_dir, raw_job, job_dir / "work", bottom_up, tmp_path / "products" / "c", fits_path, lambda: None)


def test_colour_basis_from_libraw_metadata_matches_the_dng_route(tmp_path):
    dng = cf.write_reference_dng(tmp_path / "ref.dng")
    via_dng = me.colour_basis_from_dng(dng)
    meta = cf.libraw_like_metadata()
    meta["fields"]["black_level_per_channel"] = {"value": [cf.Z8_BLACK] * 4}
    meta["fields"]["white_level"] = {"value": cf.Z8_WHITE}
    nef = tmp_path / "ref.nef"
    nef.write_bytes(b"not a real NEF; metadata supplied explicitly")
    via_raw = me.colour_basis_for_reference(nef, metadata=meta)
    assert np.allclose(via_raw["camera_to_linear_srgb"], via_dng["camera_to_linear_srgb"])
    assert via_raw["black_DN"] == via_dng["black_DN"] and via_raw["white_DN"] == via_dng["white_DN"]
    assert np.allclose(via_raw["as_shot_gains_rgb"], via_dng["as_shot_gains_rgb"], rtol=1e-6)
    assert via_raw["source"]["kind"] == "libraw"
    with pytest.raises(ValueError):
        me.colour_basis_from_raw(nef, metadata={"available": False, "status": "rawpy_unavailable"})


def test_mcp_service_exposes_export_master_and_render_photo(composite, tmp_path):
    dng, cam, sky, bottom_up, export_dir, fits_path, mask = composite
    from starstack.mcp_server import WorkspaceTools, AccessDenied
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    ref = inputs / "ref.dng"
    ref.write_bytes(dng.read_bytes())
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    job_products = outputs / "job" / "products" / "p1"
    job_products.mkdir(parents=True)
    (job_products / "master_linear.fit").write_bytes(fits_path.read_bytes())
    service = WorkspaceTools(input_root=inputs, output_root=outputs)
    master = service.export_master(str(job_products), reference_dng=str(ref))
    assert Path(master["directory"]).is_relative_to(outputs) and master["state"] == "succeeded"
    photo = service.render_photo(master["directory"], sky_mask=str(mask.rename(inputs / "sky.png")))
    assert photo["metadata"]["recipe"] == "photo-default-1" and Path(photo["directory"]).is_relative_to(outputs)
    with pytest.raises(AccessDenied):
        service.export_master(str(tmp_path / "products" / "abc123"), reference_dng=str(ref))     # outside the output root
    with pytest.raises(AccessDenied):
        service.render_photo(str(tmp_path / "elsewhere"))
