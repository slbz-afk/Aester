"""Registration and the narrower first GIMP adapter have separate capabilities."""
import struct

import numpy as np
from PIL import ImageCms
import pytest
import tifffile

from starstack.finishing_assets import inspect_source
from starstack.finishing_backend import _format_supported


@pytest.mark.parametrize("options", [{"compression": "deflate"}, {"bigtiff": True}, {"tile": (16, 16)}])
def test_source_can_be_registered_but_unsupported_driver_format_is_explicit(tmp_path, options):
    path = tmp_path / "source.tif"
    pixels = np.arange(16 * 16 * 3, dtype=np.uint16).reshape(16, 16, 3)
    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    tifffile.imwrite(path, pixels, photometric="rgb", metadata=None,
                     extratags=[(34675, "B", len(icc), icc, False)], **options)
    assert inspect_source(path, encoding="srgb16")["width"] == 16
    with pytest.raises(ValueError, match="classic uncompressed"):
        _format_supported(path)


def test_lsb_first_mask_is_refused_before_raw_strip_interpretation(tmp_path):
    path = tmp_path / "lsb-mask.tif"
    tifffile.imwrite(path, np.full((8, 12), 128, dtype=np.uint8),
                     photometric="minisblack", metadata=None, description="reserved test tag")
    with tifffile.TiffFile(path) as tf:
        offset = tf.pages[0].tags[270].offset
        endian = tf.byteorder
    # Replace this fixture's unused description entry with FillOrder=2.
    with path.open("r+b") as stream:
        stream.seek(offset)
        stream.write(struct.pack(endian + "HHIHH", 266, 3, 1, 2, 0))
    with tifffile.TiffFile(path) as tf:
        assert int(tf.pages[0].fillorder) == 2
    with pytest.raises(ValueError, match="FillOrder=1"):
        _format_supported(path)


def test_classic_uncompressed_tiff_is_compatible(tmp_path):
    path = tmp_path / "mask.tif"
    tifffile.imwrite(path, np.zeros((8, 12), np.uint16), metadata=None)
    assert _format_supported(path) is None
