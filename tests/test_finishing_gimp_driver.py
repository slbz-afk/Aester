"""Pure decoder/compiler checks for the fixed driver; GI is not imported."""
import hashlib
from pathlib import Path
import struct
import tracemalloc
import zlib

import numpy as np
from PIL import Image
import pytest
import tifffile

from starstack import finishing_gimp_script as driver


def test_scalar_parameters_compile_to_xcf_storage_without_changing_recipe_values():
    class Config:
        def __init__(self):
            self.values = {}
        def set_property(self, key, value):
            self.values[key] = value
        def get_property(self, key):
            return self.values[key]

    requested = {"scale": 0.6, "black-level": 0.0}
    result = driver._configure_xcf_scalars(Config(), requested)
    # Actual failed NGC1893 XCF encoded 0x3f19999a, not double 0.6.
    assert result["effective"] == {"scale": 0.6000000238418579, "black-level": 0.0}
    assert requested == {"scale": 0.6, "black-level": 0.0}
    assert driver._configure_xcf_scalars(Config(), result["effective"])["effective"] == result["effective"]


def test_scalar_parameter_readback_mismatch_is_rejected():
    class ChangedConfig:
        def set_property(self, key, value):
            pass
        def get_property(self, key):
            return 0.5
    with pytest.raises(ValueError, match="readback"):
        driver._configure_xcf_scalars(ChangedConfig(), {"scale": 0.6})


@pytest.mark.parametrize("suffix,bits", [(".png", 8), (".png", 16), (".tif", 8), (".tif", 16)])
def test_mask_decoder_matches_independent_pixels(tmp_path, suffix, bits):
    rng = np.random.default_rng(924)
    values = rng.integers(0, 2 ** bits, (29, 47), dtype=np.uint16 if bits == 16 else np.uint8)
    path = tmp_path / ("weights" + suffix)
    if suffix == ".png":
        Image.fromarray(values).save(path)
    else:
        tifffile.imwrite(path, values, metadata=None, photometric="minisblack", byteorder=">")
    width, height, data, raw_sha = driver._mask_raster(path, bits)
    assert (width, height) == (47, 29)
    decoded = np.frombuffer(data, dtype=np.uint16).reshape(values.shape)
    assert np.array_equal(decoded, values.astype(np.uint16) * (257 if bits == 8 else 1))
    assert len(raw_sha) == 64


def test_curve_error_matches_independent_union_knot_calculation():
    points = [[0, 0], [0.133, 0.19], [0.335, 0.72], [1, 1]]
    grid = np.linspace(0, 1, 256)
    xs, ys = np.array(points).T
    knots = np.unique(np.r_[grid, xs])
    ideal = np.interp(knots, xs, ys)
    sampled = np.interp(knots, grid, np.interp(grid, xs, ys))
    assert driver._curve_approx_error(points) == pytest.approx(float(np.max(np.abs(ideal - sampled))), abs=1e-15)
    assert driver._curve_approx_error([[0, 0.2], [1, 0.8]]) < 1e-15
    assert driver._curve_approx_error([[0, 0], [0.001, 1], [1, 1]]) > 1 / 255


def _chunk(kind, data):
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


def test_png_expansion_is_bounded_by_declared_raster(tmp_path):
    path = tmp_path / "oversized-stream.png"
    compressed = zlib.compress(b"\0" * (4 * 1024 * 1024))
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0))
                     + _chunk(b"IDAT", compressed) + _chunk(b"IEND", b""))
    tracemalloc.start()
    try:
        with pytest.raises(ValueError, match="scanline"):
            driver._png_gray(path)
        _, peak = tracemalloc.get_traced_memory()
        assert peak < 1024 * 1024
    finally:
        tracemalloc.stop()
