"""The person's JPEG quality setting: validation and a rough size estimate (pure functions).

A caller that sets ``jpeg_quality`` (workflow ``submit_workflow``, ``reduce_stars``,
``finish_photo`` / ``finish_milkyway``; the GUI always does) gets every JPEG it
delivers encoded at that quality with 4:4:4 chroma and the image's sRGB ICC.
Leaving it unset keeps each writer's historical encoding.

Estimator (measured 2026-10-09 on ten delivered images, Pillow / libjpeg-turbo,
4:4:4 with the 588-byte sRGB ICC). Before a job the GUI has the 1024 px
first-frame preview and the full frame size: the preview's bytes at the chosen
quality, scaled by full pixels / preview pixels, gave actual / predicted
0.91-1.08 at q95 and 0.95-1.38 at q70 on the sampled scenes (single-frame
noise offsets the downscaling); ``low_quality_correction`` (1.07 for q <= 92)
offsets the low-quality bias. Without a preview: 2.4 bits per pixel at q95
times the measured mean size relative to q95. Both are rough estimates, never
a promise about a particular photograph.
"""
from __future__ import annotations

from functools import lru_cache
import io
import math
from pathlib import Path

MINIMUM = 60
MAXIMUM = 100
DEFAULT = 95
SUBSAMPLING = 0                       # 4:4:4: small coloured stars keep their colour
# Mean delivered size relative to q95 (4:4:4), ten delivered images, 2026-10-09.
RELATIVE_SIZE = ((70, .235), (80, .341), (85, .443), (90, .615), (92, .710), (95, 1.), (97, 1.414), (100, 3.41))
FALLBACK_BITS_PER_PIXEL = 2.4         # at q95, full-size camera stacks
LOW_QUALITY_LIMIT = 92
LOW_QUALITY_FACTOR = 1.07
PREVIEW_EDGE = 1024                   # the first-frame preview's long side (references.PREVIEW_MAX_SIDE)


def validate_quality(value):
    """``value`` when it is an int in 60..100 (a bool or a float is refused), else ValueError."""
    if type(value) is not int or not MINIMUM <= value <= MAXIMUM:
        raise ValueError("jpeg_quality 须为 60 至 100 的整数")
    return value


def relative_size(quality) -> float:
    """Mean JPEG size at ``quality`` relative to q95: the measured table, geometric between its points.

    Below q70 the 70-80 segment's ratio per quality step is extended.
    """
    quality = validate_quality(quality)
    table = dict(RELATIVE_SIZE)
    if quality in table:
        return table[quality]
    pairs = list(zip(RELATIVE_SIZE, RELATIVE_SIZE[1:]))
    (q0, r0), (q1, r1) = next((pair for pair in pairs if quality < pair[1][0]), pairs[-1])
    t = (quality - q0) / (q1 - q0)
    return math.exp(math.log(r0) + t * (math.log(r1) - math.log(r0)))


def low_quality_correction(quality) -> float:
    """c(q): the preview-based estimate's correction, 1.07 for q <= 92, else 1.0."""
    return LOW_QUALITY_FACTOR if validate_quality(quality) <= LOW_QUALITY_LIMIT else 1.


@lru_cache(maxsize=1)
def srgb_icc() -> bytes:
    """The sRGB profile the estimate embeds (same size as the delivered JPEGs' profile)."""
    from PIL import ImageCms
    return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


def load_preview(path, *, max_edge=None):
    """The preview image at ``path`` as a loaded RGB PIL image (the caller may cache it).

    ``max_edge``: a larger image (a FITS reference shows its full decoded frame) is area-averaged down to this
    long side, so that an estimate encodes no more than a standard preview; ``estimate_from_preview`` scales by
    full pixels / preview pixels, so the method is the same.
    """
    from PIL import Image
    with Image.open(Path(path)) as image:
        rgb = image.convert("RGB")
    if max_edge is not None and max(rgb.size) > max_edge:
        rgb.thumbnail((max_edge, max_edge), Image.Resampling.BOX)
    return rgb


def _rgb(image):
    from PIL import Image
    if isinstance(image, (str, Path)):
        return load_preview(image)
    if not isinstance(image, Image.Image):
        import numpy as np
        array = np.asarray(image)
        if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
            raise ValueError("需要 8 位 RGB 预览图")
        return Image.fromarray(array)
    return image if image.mode == "RGB" else image.convert("RGB")


def encoded_size(image, quality, *, icc=None) -> int:
    """Bytes of ``image`` (PIL image, HxWx3 uint8 array or path) as a 4:4:4 JPEG at ``quality`` with an sRGB ICC."""
    quality = validate_quality(quality)
    buffer = io.BytesIO()
    _rgb(image).save(buffer, format="JPEG", quality=quality, subsampling=SUBSAMPLING,
                     icc_profile=srgb_icc() if icc is None else icc)
    return buffer.tell()


def _pixels(width, height) -> int:
    if any(type(value) is not int or value <= 0 for value in (width, height)):
        raise ValueError("需要正整数的图像宽高")
    return width * height


def estimate_from_preview(preview, full_width, full_height, quality, *, icc=None) -> int:
    """Rough delivered JPEG bytes: the preview's bytes at ``quality`` x full pixels / preview pixels x c(q)."""
    image = _rgb(preview)
    scale = _pixels(full_width, full_height) / _pixels(*image.size)
    return round(encoded_size(image, quality, icc=icc) * scale * low_quality_correction(quality))


def estimate_without_preview(full_width, full_height, quality) -> int:
    """Rough delivered JPEG bytes without a preview: 2.4 bits per pixel at q95 x the relative-size table."""
    return round(_pixels(full_width, full_height) * FALLBACK_BITS_PER_PIXEL / 8 * relative_size(quality))


def megabytes(size) -> str:
    """``size`` bytes as a short decimal-megabyte figure: 14, 4.3, 0.35 (never below 0.01)."""
    value = size / 1e6
    if value >= 10:
        return f"{value:.0f}"
    if value >= 1:
        return f"{value:.1f}"
    return f"{max(value, .01):.2f}"
