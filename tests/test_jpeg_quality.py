"""2026-10-09 JPEG quality: validation, the size estimator and the stand-alone JPEG writers.

The workflow (frozen request, publication re-encode, trails versions) is tested in
tests/test_trail_default_and_extras.py; the GUI slider in tests/test_ui_mainwindow_tuoxing.py.
A JPEG's quality is read from its quantization tables (which depend only on the quality), its chroma
subsampling from Pillow's JpegImagePlugin.get_sampling (0 = 4:4:4, 2 = 4:2:0).
"""
from __future__ import annotations

import asyncio
import io
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile
from PIL import Image, JpegImagePlugin

import p1c_finishing_data as fd
from starstack import jpeg_estimate as je
from starstack import photographic_job as job
from starstack import star_reduction as sr


def tables(quality):
    """Quantization tables of a JPEG encoded at ``quality`` (content independent)."""
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), (40, 50, 60)).save(buffer, format="JPEG", quality=quality, subsampling=0)
    buffer.seek(0)
    with Image.open(buffer) as image:
        return image.quantization


def encoding(path):
    """(quantization tables, sampling, embedded ICC) of a JPEG file."""
    with Image.open(path) as image:
        return image.quantization, JpegImagePlugin.get_sampling(image), image.info.get("icc_profile")


def expected_jpeg(tiff, quality):
    """The bytes a set jpeg_quality delivers: round(code / 257), the TIFF's ICC, 4:4:4."""
    with tifffile.TiffFile(tiff) as tf:
        page = tf.pages[0]
        codes, icc = page.asarray(), page.tags[34675].value
    pixels = ((codes.astype(np.uint32) + 128) // 257).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="JPEG", quality=quality, subsampling=0, icc_profile=icc)
    return buffer.getvalue()


def scene(seed, height, width, noise=6.):
    """A night-sky-like 8-bit image: a smooth gradient, sensor noise, sparse stars and a dark foreground band."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    sky = 30 + 25 * yy / height + 10 * xx / width
    image = np.repeat(sky[..., None], 3, axis=2) + rng.normal(0, noise, (height, width, 3))
    count = height * width // 400
    image[rng.integers(0, height, count), rng.integers(0, width, count)] = rng.uniform(120, 255, (count, 3))
    image[int(height * .8):] *= .4
    return np.clip(np.rint(image), 0, 255).astype(np.uint8)


def full_size(array, quality):
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="JPEG", quality=quality, subsampling=0, icc_profile=je.srgb_icc())
    return buffer.tell()


# ------------------------------------------------------------------------------------------------ validation


@pytest.mark.parametrize("value", [60, 61, 95, 100])
def test_quality_accepts_whole_numbers_from_60_to_100(value):
    assert je.validate_quality(value) == value


@pytest.mark.parametrize("value", [59, 101, 0, -95, 95.0, 95.5, True, False, "95", None, [95], np.int64(95)])
def test_quality_refuses_everything_else(value):
    with pytest.raises(ValueError, match="jpeg_quality"):
        je.validate_quality(value)


def test_defaults_match_the_decision():
    assert (je.MINIMUM, je.MAXIMUM, je.DEFAULT, je.SUBSAMPLING) == (60, 100, 95, 0)


# ------------------------------------------------------------------------------------------------ estimator


def test_relative_size_table_is_exact_at_its_points_and_geometric_between_them():
    for quality, ratio in je.RELATIVE_SIZE:
        assert je.relative_size(quality) == ratio
    assert je.relative_size(96) == pytest.approx(math.sqrt(1.414))
    assert je.relative_size(60) == pytest.approx(.235 ** 2 / .341)          # the 70-80 ratio per step, extended
    sizes = [je.relative_size(q) for q in range(60, 101)]
    assert all(a < b for a, b in zip(sizes, sizes[1:]))


def test_low_quality_correction_steps_at_92():
    assert je.low_quality_correction(60) == je.low_quality_correction(92) == 1.07
    assert je.low_quality_correction(93) == je.low_quality_correction(95) == je.low_quality_correction(100) == 1.


def test_fallback_is_2_4_bits_per_pixel_at_q95_times_the_table():
    assert je.estimate_without_preview(8280, 5520, 95) == round(8280 * 5520 * 2.4 / 8)          # 13.7 MB
    assert je.estimate_without_preview(8280, 5520, 70) == round(8280 * 5520 * 2.4 / 8 * .235)
    assert je.estimate_without_preview(8280, 5520, 100) == round(8280 * 5520 * 2.4 / 8 * 3.41)
    for bad in ((0, 10), (10, -1), (10.5, 10), (True, 10)):
        with pytest.raises(ValueError):
            je.estimate_without_preview(*bad, 95)


def test_preview_of_the_full_frame_reproduces_the_encode_times_the_correction():
    image = scene(3, 120, 160)
    assert je.estimate_from_preview(image, 160, 120, 95) == full_size(image, 95)
    assert je.estimate_from_preview(image, 160, 120, 100) == full_size(image, 100)
    assert je.estimate_from_preview(image, 160, 120, 85) == round(full_size(image, 85) * 1.07)
    # A path, a PIL image and an array give the same estimate.
    assert je.estimate_from_preview(Image.fromarray(image), 160, 120, 90) == je.estimate_from_preview(image, 160, 120, 90)


def test_preview_estimate_scales_by_pixels_and_tracks_a_stationary_scene(tmp_path):
    """The first-frame preview is a separate, smaller image with the same kind of noise: its bytes per pixel predict
    the full frame's. Within 5 % at q95 and above on this synthetic scene; at q92 and below c(q) = 1.07 (the measured
    bias of real downscaled previews) makes the estimate up to about 12 % high here. A real scene is rougher."""
    full = scene(1, 1024, 1536)
    preview_path = tmp_path / "preview.png"
    Image.fromarray(scene(2, 512, 768)).save(preview_path)
    preview = je.load_preview(preview_path)
    assert preview.mode == "RGB" and preview.size == (768, 512)
    for quality in (60, 70, 80, 85, 90, 92, 95, 97, 100):
        estimate = je.estimate_from_preview(preview, 1536, 1024, quality)
        assert estimate == je.estimate_from_preview(preview_path, 1536, 1024, quality)
        low = .95 if quality >= 95 else .85
        assert low <= full_size(full, quality) / estimate <= 1.05, quality
    low, high = (je.estimate_from_preview(preview, 1536, 1024, q) for q in (70, 95))
    assert low < high
    assert je.estimate_from_preview(preview, 3072, 2048, 95) == pytest.approx(4 * high, rel=1e-6, abs=2)


def test_estimate_encodes_with_4_4_4_and_the_srgb_profile():
    image = scene(5, 64, 64)
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="JPEG", quality=95, subsampling=0, icc_profile=je.srgb_icc())
    assert je.encoded_size(image, 95) == buffer.tell()
    smaller = io.BytesIO()
    Image.fromarray(image).save(smaller, format="JPEG", quality=95)                 # Pillow's default 4:2:0
    assert smaller.tell() < buffer.tell()
    with pytest.raises(ValueError):
        je.encoded_size(np.zeros((8, 8), np.uint8), 95)


def test_a_full_frame_preview_is_area_reduced_to_the_standard_edge(tmp_path):
    """A FITS reference preview is the full decoded frame; the estimate's basis is at most 1024 px on its long side."""
    assert je.PREVIEW_EDGE == 1024
    big, small = tmp_path / "full.png", tmp_path / "small.png"
    Image.fromarray(scene(3, 1500, 2600)).save(big)
    Image.fromarray(scene(3, 300, 400)).save(small)
    reduced = je.load_preview(big, max_edge=je.PREVIEW_EDGE)
    assert reduced.mode == "RGB" and reduced.size[0] == 1024 and abs(reduced.size[1] - 1500 * 1024 / 2600) <= 1
    # area-averaged: each reduced pixel is the mean of its source block (no sharpening that would add bytes)
    whole = np.asarray(Image.open(big).convert("RGB"), np.float64)
    assert abs(np.asarray(reduced, np.float64).mean() - whole.mean()) < .5
    assert je.load_preview(big).size == (2600, 1500)
    assert je.load_preview(small, max_edge=je.PREVIEW_EDGE).size == (400, 300)


@pytest.mark.parametrize("size,text", [(13_767_480, "14"), (9_910_000, "9.9"), (4_300_000, "4.3"), (350_000, "0.35"),
                                       (12_000, "0.01"), (1_620, "0.01")])
def test_megabytes_are_short_decimal_figures(size, text):
    assert je.megabytes(size) == text


# ------------------------------------------------------------------------------------------------ stand-alone writers


def test_reduce_stars_keeps_95_4_2_0_unless_a_quality_is_given(tmp_path):
    stack = fd.layer_stack(8, (96, 64), paired=False)
    result = fd.make_separation(tmp_path / "sep", stack)
    old = sr.reduce_stars(result, tmp_path / "old", background_strength=.5, bright_radius=2)
    new = sr.reduce_stars(result, tmp_path / "new", background_strength=.5, bright_radius=2, jpeg_quality=72)
    old_jpeg, new_jpeg = (Path(r["artifacts"]["star_reduced.jpg"]["path"]) for r in (old, new))
    quantization, sampling, icc = encoding(old_jpeg)
    assert quantization == tables(95) and sampling == 2 and icc
    assert "jpeg_quality" not in old and new["jpeg_quality"] == 72
    quantization, sampling, _ = encoding(new_jpeg)
    assert quantization == tables(72) and sampling == 0
    tiff = Path(new["artifacts"]["star_reduced_srgb16.tif"]["path"])
    assert new_jpeg.read_bytes() == expected_jpeg(tiff, 72)
    # Only the JPEG differs: the 16-bit results are the same pixels.
    assert np.array_equal(tifffile.imread(tiff), tifffile.imread(old["artifacts"]["star_reduced_srgb16.tif"]["path"]))
    for bad in (59, 95.0, True):
        with pytest.raises(ValueError, match="jpeg_quality"):
            sr.reduce_stars(result, tmp_path / "bad", background_strength=.5, jpeg_quality=bad)
    assert not (tmp_path / "bad").exists()


def test_finish_milkyway_keeps_97_unless_a_quality_is_given(tmp_path):
    paths = fd.milkyway_files(tmp_path / "inputs")
    args = (paths["source"], None, paths["sky"], paths["protected"], paths["milkyway"])
    old = job.finish_milkyway(args[0], tmp_path / "old", *args[2:], milkyway_evidence=paths["evidence"])
    new = job.finish_milkyway(args[0], tmp_path / "new", *args[2:], milkyway_evidence=paths["evidence"],
                              jpeg_quality=66)
    assert old["state"] == new["state"] == "succeeded", (old.get("error"), new.get("error"))
    old_jpeg, new_jpeg = (Path(r["artifacts"]["milkyway_srgb.jpg"]["path"]) for r in (old, new))
    assert encoding(old_jpeg)[:2] == (tables(97), 0)
    assert encoding(new_jpeg)[:2] == (tables(66), 0)
    assert "jpeg_quality" not in old and new["jpeg_quality"] == 66
    assert new_jpeg.read_bytes() == expected_jpeg(Path(new["artifacts"]["milkyway_srgb16.tif"]["path"]), 66)
    assert old["artifacts"]["milkyway_srgb16.tif"]["sha256"] == new["artifacts"]["milkyway_srgb16.tif"]["sha256"]
    with pytest.raises(ValueError, match="jpeg_quality"):
        job.finish_milkyway(args[0], tmp_path / "bad", *args[2:], jpeg_quality=101)
    assert not (tmp_path / "bad").exists()


def test_finish_photo_keeps_97_unless_a_quality_is_given(tmp_path):
    paths = fd.milkyway_files(tmp_path / "inputs", shape=(160, 120))
    old = job.finish_photo(paths["source"], tmp_path / "old", paths["sky"], paths["protected"])
    new = job.finish_photo(paths["source"], tmp_path / "new", paths["sky"], paths["protected"], jpeg_quality=100)
    assert old["state"] == new["state"] == "succeeded", (old.get("error"), new.get("error"))
    old_jpeg, new_jpeg = (Path(r["artifacts"]["photographic_srgb.jpg"]["path"]) for r in (old, new))
    assert encoding(old_jpeg)[:2] == (tables(97), 0)
    assert encoding(new_jpeg)[:2] == (tables(100), 0)
    assert new_jpeg.read_bytes() == expected_jpeg(Path(new["artifacts"]["photographic_srgb16.tif"]["path"]), 100)


# ------------------------------------------------------------------------------------------------ CLI and MCP entries


@pytest.mark.parametrize("command,module,name", [
    (["reduce-stars", "--separation-result", "r.json"], sr, "reduce_stars"),
    (["finish-photo", "--source", "s.tif", "--sky-mask", "k.png", "--protected-mask", "p.png"], job, "finish_photo"),
    (["finish-milkyway", "--source", "s.tif", "--sky-mask", "k.png", "--protected-mask", "p.png",
      "--milkyway-mask", "m.png"], job, "finish_milkyway"),
])
def test_cli_forwards_jpeg_quality_only_when_given(monkeypatch, capsys, command, module, name):
    from starstack import cli
    calls = []
    monkeypatch.setattr(module, name, lambda *a, **k: calls.append(k) or {"state": "succeeded"})
    monkeypatch.setattr(sys, "argv", ["aester", *command, "--jpeg-quality", "77"])
    assert cli.main() == 0
    monkeypatch.setattr(sys, "argv", ["aester", *command])
    assert cli.main() == 0
    assert calls[0]["jpeg_quality"] == 77 and "jpeg_quality" not in calls[1]
    for bad in ("59", "101", "95.5", "high"):
        monkeypatch.setattr(sys, "argv", ["aester", *command, "--jpeg-quality", bad])
        with pytest.raises(SystemExit):
            cli.main()
    assert len(calls) == 2
    capsys.readouterr()


def test_mcp_tools_expose_strict_optional_jpeg_quality(tmp_path, monkeypatch):
    from mcp.shared.memory import create_connected_server_and_client_session
    from starstack.mcp_server import WorkspaceTools, create_server
    inputs, output = tmp_path / "inputs", tmp_path / "output"
    inputs.mkdir()
    output.mkdir()
    service = WorkspaceTools(inputs, output, None)
    calls = []
    for name in ("submit_workflow", "reduce_stars", "finish_photo", "finish_milkyway"):
        monkeypatch.setattr(service, name, lambda *a, _name=name, **k: calls.append((_name, k)) or {"state": "ok"})
    arguments = {"submit_workflow": {"input_files": ["a.NEF"]}, "reduce_stars": {"separation_result": "r.json"},
                 "finish_photo": {"source": "s.tif", "sky_mask": "k.png", "protected_mask": "p.png"},
                 "finish_milkyway": {"source": "s.tif", "sky_mask": "k.png", "protected_mask": "p.png",
                                     "milkyway_mask": "m.png"}}

    async def exercise():
        server = create_server(service)
        async with create_connected_server_and_client_session(server) as client:
            tools = {tool.name: tool for tool in (await client.list_tools()).tools}
            for name, base in arguments.items():
                schema = tools[name].inputSchema["properties"]["jpeg_quality"]
                text = json.dumps(schema)
                assert '"minimum": 60' in text and '"maximum": 100' in text and '"integer"' in text, name
                assert schema.get("default") is None, name
                assert "jpeg_quality" in tools[name].description, name
                for value in (59, 101, 95.5, True, "95"):
                    assert (await client.call_tool(name, {**base, "jpeg_quality": value})).isError, (name, value)
                assert not (await client.call_tool(name, {**base, "jpeg_quality": 81})).isError
                assert not (await client.call_tool(name, base)).isError

    asyncio.run(asyncio.wait_for(exercise(), timeout=30))
    by_tool = {}
    for name, kwargs in calls:
        by_tool.setdefault(name, []).append(kwargs)
    for name in arguments:
        sent, omitted = by_tool[name]
        assert sent["jpeg_quality"] == 81 and "jpeg_quality" not in omitted, name
