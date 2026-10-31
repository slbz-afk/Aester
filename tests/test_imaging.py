"""Numerical file-I/O contracts; these are not live Siril interoperability tests.

Fixtures describe a scene in ordinary top-to-bottom image coordinates. The
application's working arrays are bottom-up, with unsigned 16-bit ADU values
mapped to [0, 1] and floating point values preserved without normalization.
"""
from pathlib import Path
from types import SimpleNamespace
import shutil

import numpy as np
import pytest
from astropy.io import fits
from PIL import Image
import tifffile

from starstack import core, imaging


def scene(height=7, width=9):
    """Asymmetric spatial and channel markers reveal flips and axis swaps."""
    y, x = np.mgrid[:height, :width]
    return np.stack((.04 + .003 * y, .13 + .002 * x,
                     .24 + .001 * y + .0005 * x), axis=-1).astype(np.float32)


def external_fits(path, top_down_scene, roworder="BOTTOM-UP"):
    # Astropy writes FITS array indices without interpreting the ROWORDER keyword.
    rows = top_down_scene if roworder == "TOP-DOWN" else top_down_scene[::-1]
    header = fits.Header()
    if roworder is not None:
        header["ROWORDER"] = roworder
    fits.writeto(path, np.moveaxis(rows, -1, 0), header)
    return path


def artifact_path(artifacts, filename):
    return next(Path(a["path"]) for a in artifacts if Path(a["path"]).name == filename)


def test_unsigned_16bit_and_float_fits_describe_the_same_intensities(tmp_path):
    adu = np.array([
        [[0, 1, 32768], [65535, 65534, 16384]],
        [[1024, 8192, 4096], [49151, 32767, 65500]],
        [[7, 777, 7777], [10000, 20000, 30000]],
        [[65432, 43210, 21098], [54321, 32109, 10987]],
    ], dtype=np.uint16)
    normalized = adu.astype(np.float32) / 65535
    integer_path = external_fits(tmp_path / "unsigned.fit", adu)
    float_path = external_fits(tmp_path / "float.fit", normalized)
    # Confirm the fixture exercises FITS's signed storage plus unsigned offset.
    with fits.open(integer_path, do_not_scale_image_data=True) as hdus:
        assert hdus[0].header["BITPIX"] == 16
        assert hdus[0].header["BZERO"] == 32768
        assert hdus[0].header["BSCALE"] == 1
    actual = imaging.read_rgb(integer_path)
    np.testing.assert_array_equal(actual, imaging.read_rgb(float_path))
    np.testing.assert_allclose(actual, normalized[::-1], rtol=0, atol=1e-7)
    assert actual.dtype == np.float32
    assert actual.min() == 0
    assert actual.max() == 1


@pytest.mark.parametrize("roworder", ["BOTTOM-UP", "TOP-DOWN", None])
def test_reader_normalizes_explicit_row_direction_and_defaults_to_bottom_up(tmp_path, roworder):
    top = scene()
    path = external_fits(tmp_path / "oriented.fit", top, roworder)
    np.testing.assert_array_equal(imaging.read_rgb(path), top[::-1])


def test_float_values_are_not_rescaled_using_their_maximum(tmp_path):
    top = scene()
    top[0, 0] = (-.4, 1.5, 4.0)
    top[-1, -1] = (2.0, -.02, .0001)
    source = external_fits(tmp_path / "hdr.fit", top)
    np.testing.assert_array_equal(imaging.read_rgb(source), top[::-1])


def test_write_linear_declares_bottom_up_and_preserves_rgb_axes_and_float_values(tmp_path):
    top = scene()
    top[0, 0] = (-.125, .5, 1.75)
    path = tmp_path / "written.fit"
    imaging.write_linear(path, top[::-1], "Orientation and dynamic-range fixture")
    with fits.open(path, memmap=False) as hdus:
        assert hdus[0].header["BITPIX"] == -32
        assert hdus[0].header.get("ROWORDER") == "BOTTOM-UP"
        assert hdus[0].data.shape == (3, top.shape[0], top.shape[1])
        # FITS row zero is explicitly the bottom of this scene.
        np.testing.assert_array_equal(hdus[0].data[:, 0, 0], top[-1, 0])
        np.testing.assert_array_equal(hdus[0].data[:, -1, 0], top[0, 0])
    np.testing.assert_array_equal(imaging.read_rgb(path), top[::-1])


def test_linear_tiff_preserves_negative_hdr_values_and_ordinary_display_orientation(tmp_path):
    top = scene()
    top[0, 0] = (-.2, .5, 2.0)
    top[-1, -1] = (4.0, -.01, .07)
    working = top[::-1].copy()
    before = working.copy()
    artifacts = imaging.export_products(working, tmp_path / "exports", lambda: None)
    path = artifact_path(artifacts, "master_linear_32bit.tif")
    with tifffile.TiffFile(path) as tiff:
        actual = tiff.asarray()
        assert actual.dtype == np.float32
        assert tiff.pages[0].photometric.name == "RGB"
    np.testing.assert_array_equal(actual, top)
    np.testing.assert_array_equal(working, before)
    assert actual.min() < 0
    assert actual.max() > 1


@pytest.mark.parametrize("sky_order,ground_order", [
    ("BOTTOM-UP", "BOTTOM-UP"), ("TOP-DOWN", "TOP-DOWN"),
    ("BOTTOM-UP", "TOP-DOWN"), ("TOP-DOWN", "BOTTOM-UP"),
])
def test_png_mask_blends_the_intended_top_and_bottom_of_external_fits(
        tmp_path, sky_order, ground_order):
    sky_top = scene()
    ground_top = scene() + np.float32(.4)
    sky_file = external_fits(tmp_path / "sky.fit", sky_top, sky_order)
    ground_file = external_fits(tmp_path / "ground.fit", ground_top, ground_order)
    mask = np.zeros(sky_top.shape[:2], np.uint8)
    mask[:2] = 255  # The top two displayed rows belong to the sky.
    mask[2, 4] = 128  # A single known feathered pixel detects rounding mistakes.
    mask_file = tmp_path / "mask.png"
    Image.fromarray(mask).save(mask_file)
    result = imaging.blend_with_mask(imaging.read_rgb(sky_file), imaging.read_rgb(ground_file), mask_file)
    exports = imaging.export_products(result, tmp_path / "products", lambda: None)
    displayed = tifffile.imread(artifact_path(exports, "master_linear_32bit.tif"))
    np.testing.assert_array_equal(displayed[:2], sky_top[:2])
    np.testing.assert_array_equal(displayed[3:], ground_top[3:])
    np.testing.assert_array_equal(displayed[2, 0], ground_top[2, 0])
    expected_feather = sky_top[2, 4] * (128 / 255) + ground_top[2, 4] * (127 / 255)
    np.testing.assert_allclose(displayed[2, 4], expected_feather, rtol=0, atol=1e-7)


def test_median_fusion_uses_a_common_scale_for_integer_and_float_files(tmp_path):
    adu = np.round(scene() * 65535).astype(np.uint16)
    expected_top = adu.astype(np.float32) / 65535
    paths = [external_fits(tmp_path / "a.fit", adu),
             external_fits(tmp_path / "b.fit", adu),
             external_fits(tmp_path / "c.fit", expected_top)]
    result = imaging.median_foreground(paths, lambda: None)
    np.testing.assert_allclose(result, expected_top[::-1], rtol=0, atol=1e-7)


def test_median_fusion_aligns_mixed_row_order_across_chunk_boundaries(tmp_path):
    # Height deliberately spans multiple chunks and has a short final chunk.
    top = scene(height=131, width=5)
    paths = [external_fits(tmp_path / "bottom.fit", top, "BOTTOM-UP"),
             external_fits(tmp_path / "top_plus.fit", top + .03, "TOP-DOWN"),
             external_fits(tmp_path / "top_minus.fit", top - .03, "TOP-DOWN")]
    result = imaging.median_foreground(paths, lambda: None)
    np.testing.assert_allclose(result, top[::-1], rtol=0, atol=1e-7)
    # Explicit landmarks prevent a matching round-trip error hiding a flip.
    np.testing.assert_allclose(result[0, 0], top[-1, 0], rtol=0, atol=1e-7)
    np.testing.assert_allclose(result[-1, 0], top[0, 0], rtol=0, atol=1e-7)


def test_median_fusion_cancellation_releases_input_file_handles(tmp_path):
    paths = [external_fits(tmp_path / f"frame{i}.fit", scene(height=131)) for i in range(3)]

    def cancel():
        raise RuntimeError("stop numerical fusion")

    with pytest.raises(RuntimeError, match="stop numerical fusion"):
        imaging.median_foreground(paths, cancel)
    # Windows refuses deletion if an open FITS file/memmap retains its handle.
    for path in paths:
        path.unlink()
    assert not any(p.exists() for p in paths)


def test_mask_with_different_geometry_is_rejected(tmp_path):
    mask = tmp_path / "bad-mask.png"
    Image.new("L", (5, 4)).save(mask)
    with pytest.raises(ValueError):
        imaging.blend_with_mask(scene(), scene(), mask)


@pytest.mark.parametrize("input_order", ["TOP-DOWN", "BOTTOM-UP"])
def test_job_rejects_top_down_siril_frames_despite_reader_support(tmp_path, monkeypatch, input_order):
    # Reader-level support is intentionally broader than the current Siril job
    # contract. Do not let a future refactor silently claim end-to-end support.
    source = tmp_path / "input"
    source.mkdir()
    for i in range(3):
        external_fits(source / f"frame{i}.fit", scene() + i * .001, input_order)
    monkeypatch.setattr(core, "doctor", lambda *_: {
        "available": True, "version": "1.4.4", "executable": "never-executed.exe", "message": "Fake"})
    monkeypatch.setattr(core, "_spawn_worker", lambda *_: None)
    monkeypatch.setattr(core.shutil, "disk_usage", lambda _: SimpleNamespace(free=10**12))
    phases = []

    def fake_conversion(executable, directory, name, commands, check):
        phase = core.load_job(directory)["stage"]
        phases.append(phase)
        assert phase == "convert", "Unsupported coordinates reached the stacking engine"
        for i, path in enumerate(sorted((directory / "stage").glob("*.fit")), 1):
            target = directory / "work" / f"light_{i:05d}.fit"
            shutil.copyfile(path, target)
            fits.setval(target, "ROWORDER", value="TOP-DOWN")

    monkeypatch.setattr(core, "run_script", fake_conversion)
    if input_order == "TOP-DOWN":
        with pytest.raises(ValueError, match="BOTTOM-UP"):
            core.submit_job(str(source), str(tmp_path / "output"))
        assert not phases
        return
    job = core.submit_job(str(source), str(tmp_path / "output"))
    core.run_job(job["job_dir"])
    result = core.load_job(job["job_dir"])
    assert result["state"] == "failed"
    assert "BOTTOM-UP" in result["error"]
    assert phases == ["convert"]
    assert not result["artifacts"]
