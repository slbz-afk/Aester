"""Reproducible tonal selections from actual source/candidate RGB16 pixels."""
from __future__ import annotations

import math
from pathlib import Path
import uuid

from .finishing_assets import inspect_mask, inspect_source, ordinary_path, sha256_file


def _knots(value):
    if (type(value) is not list or len(value) != 4
            or any(type(v) not in {int, float} or not math.isfinite(v) or not 0 <= v <= 1 for v in value)
            or any(a > b for a, b in zip(value, value[1:]))
            or value[0] == value[-1]):
        raise ValueError("Tonal knots must be four ordered finite numbers 0<=a<=b<=c<=d<=1, a<d")
    return [float(v) for v in value]


def tonal_coverage(directory, source, image_record, knots, *, invert=False, region=None):
    """Make a fixed mask, not a selection recomputed at every recipe operation.

    Brightness here is weighted encoded sRGB, not physical luminance. Inversion
    precedes region intersection, so a protected zero region remains zero.
    """
    import numpy as np
    from PIL import Image
    import tifffile

    knots = _knots(knots)
    if type(invert) is not bool:
        raise ValueError("invert must be a bool")
    image = inspect_source(image_record["path"], encoding="srgb16")
    if (image["sha256"] != image_record["sha256"]
            or (image["width"], image["height"]) != (source["width"], source["height"])):
        raise ValueError("Tonal selection image changed or has different source geometry")
    weights = None
    if region is not None:
        checked = inspect_mask(region["path"], source, source_sha256=source["sha256"])
        if any(region.get(key) != val for key, val in checked.items()):
            raise ValueError("Tonal region mask changed")
        if Path(region["path"]).suffix.lower() in {".tif", ".tiff"}:
            weights = tifffile.imread(region["path"])
        else:
            with Image.open(region["path"]) as im:
                weights = np.array(im)
    with tifffile.TiffFile(image["path"]) as tf:
        # Decode once; float intermediates below are limited to 128 rows.
        pixels = tf.pages[0].asarray()
    result = np.empty(pixels.shape[:2], dtype=np.uint16)
    a, b, c, d = knots
    for start in range(0, image["height"], 128):
        stop = start + 128
        rgb = pixels[start:stop].astype(np.int64)
        # One division of exact integer weights preserves equal-gray threshold
        # endpoints (e.g. [1,1,1] has y exactly float(1/65535)).
        y = (rgb @ np.array([2126, 7152, 722], dtype=np.int64)) / 655350000.0
        lo = (y >= b).astype(np.float64) if a == b else np.clip((y - a) / (b - a), 0, 1)
        hi = (y <= c).astype(np.float64) if c == d else np.clip((d - y) / (d - c), 0, 1)
        coverage = (lo * lo * (3 - 2 * lo)) * (hi * hi * (3 - 2 * hi))
        if invert:
            coverage = 1 - coverage
        if weights is not None:
            coverage *= weights[start:stop].astype(np.float64) / (2 ** region["bits"] - 1)
        result[start:stop] = np.rint(coverage * 65535).astype(np.uint16)
    if sha256_file(image["path"]) != image["sha256"]:
        raise ValueError("Tonal image changed during selection")
    if region is not None and sha256_file(region["path"]) != region["sha256"]:
        raise ValueError("Tonal region changed during selection")
    folder = Path(directory) / "previews"
    folder.mkdir(exist_ok=True)
    folder = ordinary_path(folder, directory=True)
    path = folder / ("tonal-" + uuid.uuid4().hex + ".tif")
    tifffile.imwrite(path, result, photometric="minisblack", metadata=None)
    return {"path": str(path), "geometry": {
        "kind": "encoded-srgb-tonal-range-v1", "image_sha256": image["sha256"],
        "source_sha256": source["sha256"], "knots": knots, "invert": invert,
        "brightness": "0.2126*encoded_R + 0.7152*encoded_G + 0.0722*encoded_B; RGB normalized to 0..1",
        "coverage": "smoothstep(a,b,y)*[1-smoothstep(c,d,y)]; equal endpoints use inclusive threshold",
        "intersection_mask_sha256": region["sha256"] if region else None,
        "order": "range, optional inversion, region multiplication, round-to-even uint16",
        "spatial_blur": False, "adaptive_at_render": False,
        "nonzero_pixels": int(np.count_nonzero(result)), "full_pixels": int(np.count_nonzero(result == 65535))}}
