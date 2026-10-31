"""Read-only input inspection for the optional NL-1 display-photo workflow."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct

MAX_PIXELS = 300_000_000


def ordinary_path(value, *, directory=False) -> Path:
    path = Path(value).expanduser().absolute()
    if ".." in path.parts:
        raise ValueError("Parent traversal is not accepted")
    for part in (path, *path.parents):
        if part.is_symlink() or getattr(part, "is_junction", lambda: False)():
            raise ValueError("Linked input or session paths are not accepted")
    path = path.resolve(strict=True)
    expected_kind = path.is_dir() if directory else path.is_file()
    if not expected_kind:
        raise ValueError("Expected an ordinary directory" if directory else "Expected an ordinary file")
    return path


def sha256_file(value) -> str:
    path = ordinary_path(value)
    try:
        from . import hashing
    except ImportError:
        # The installer bootstrap ships this module (for ordinary_path) without the rest of the package.
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    return hashing.sha256_file(path, source="finishing_assets.sha256_file")


def _identity(stat):
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def inspect_source(value, *, encoding: str) -> dict:
    """Inspect tags and hash bytes, without loading the full pixel array.

    The explicit encoding declaration is checked against colorimetric ICC tags;
    a profile name containing 'sRGB' alone is insufficient evidence.
    """
    if encoding != "srgb16":
        raise ValueError("NL-1 currently requires encoded sRGB16 TIFF")
    path = ordinary_path(value)
    if path.suffix.lower() not in {".tif", ".tiff"}:
        raise ValueError("NL-1 source must be a TIFF")
    import tifffile
    from .photographic_job import _validate_srgb

    before = _identity(path.stat())
    with tifffile.TiffFile(path) as tf:
        if len(tf.pages) != 1:
            raise ValueError("NL-1 source must contain one image page")
        page = tf.pages[0]
        if (page.dtype.kind != "u" or page.dtype.itemsize != 2
                or len(page.shape) != 3 or page.shape[-1] != 3
                or int(page.photometric) != 2 or int(page.planarconfig) != 1):
            raise ValueError("NL-1 requires contiguous RGB uint16 pixels")
        height, width, _ = page.shape
        if not (1 <= width <= 100_000 and 1 <= height <= 100_000
                and width * height <= MAX_PIXELS):
            raise ValueError("NL-1 source dimensions exceed the limit")
        _check_tiff_payload(page, before[2])
        orientation = page.tags.get("Orientation")
        if orientation is not None and orientation.value != 1:
            raise ValueError("Source orientation must already be TOP-DOWN")
        try:
            metadata = json.loads(page.description or "{}")
        except (ValueError, TypeError):
            metadata = {}
        if isinstance(metadata, dict):
            if metadata.get("scene_linear") is True or metadata.get("display_encoding_applied") is False:
                raise ValueError("Linear masters require a display conversion before NL-1")
            if any(metadata.get(key, "TOP-DOWN") != "TOP-DOWN" for key in ("roworder", "row_order")):
                raise ValueError("Source row order must be TOP-DOWN")
        tag = page.tags.get(34675)
        if tag is None:
            raise ValueError("Source must include a verified sRGB ICC profile")
        icc = bytes(tag.value)
        try:
            _validate_srgb(icc)
        except Exception as exc:
            raise ValueError("Source ICC is not the supported encoded sRGB profile") from exc
    digest = sha256_file(path)
    if _identity(path.stat()) != before:
        raise ValueError("Source changed while being inspected")
    return {"path": str(path), "sha256": digest, "bytes": before[2],
            "width": width, "height": height, "encoding": encoding,
            "icc_sha256": hashlib.sha256(icc).hexdigest()}


def recipe_source(asset: dict) -> dict:
    return {key: asset[key] for key in ("sha256", "width", "height", "encoding")}


def verify_source(asset: dict) -> None:
    current = inspect_source(asset["path"], encoding=asset["encoding"])
    if current != asset:
        raise ValueError("Registered source changed; create a new finishing session")


def _check_tiff_payload(page, size):
    # Tags alone can survive a truncated strip/tile payload. This preflight
    # catches out-of-file ranges; the renderer must still decode the pixels.
    offsets, counts = page.dataoffsets, page.databytecounts
    if not offsets or len(offsets) != len(counts):
        raise ValueError("Missing TIFF pixel segments")
    if any(offset <= 0 or count <= 0 or offset + count > size
           for offset, count in zip(offsets, counts)):
        raise ValueError("Truncated or invalid TIFF pixel segment")


def inspect_mask(value, source: dict, *, source_sha256: str) -> dict:
    """Admit literal grayscale coverage, without color conversion or feathering."""
    if source_sha256 != source["sha256"]:
        raise ValueError("Mask belongs to a different source")
    path = ordinary_path(value)
    before = _identity(path.stat())
    if path.suffix.lower() in {".tif", ".tiff"}:
        import tifffile
        with tifffile.TiffFile(path) as tf:
            if len(tf.pages) != 1:
                raise ValueError("Mask TIFF must have one page")
            page = tf.pages[0]
            if (len(page.shape) != 2 or page.dtype.kind != "u"
                    or page.dtype.itemsize not in {1, 2} or int(page.photometric) != 1):
                raise ValueError("Mask must be grayscale uint8/uint16, black means no effect")
            orientation = page.tags.get("Orientation")
            if orientation is not None and orientation.value != 1:
                raise ValueError("Mask orientation must be TOP-DOWN")
            if page.tags.get(34675) is not None:
                raise ValueError("Coverage masks must not carry a color profile")
            height, width = page.shape
            bits = page.dtype.itemsize * 8
            _check_tiff_payload(page, before[2])
            if (width, height) != (source["width"], source["height"]):
                raise ValueError("Mask dimensions must match the source")
            # Decode here: masks are an actual execution asset, not a thumbnail.
            page.asarray()
    elif path.suffix.lower() == ".png":
        from PIL import Image
        with path.open("rb") as handle:
            header = handle.read(33)
        if len(header) != 33 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
            raise ValueError("Invalid PNG mask")
        width, height, bits, color = struct.unpack(">IIBB", header[16:26])
        if color != 0 or bits not in {8, 16}:
            raise ValueError("Mask PNG must be grayscale uint8/uint16 without alpha")
        if (width, height) != (source["width"], source["height"]):
            raise ValueError("Mask dimensions must match the source")
        with Image.open(path) as im:
            if getattr(im, "n_frames", 1) != 1 or im.info.get("icc_profile"):
                raise ValueError("Coverage mask must be one image without a color profile")
            if "transparency" in im.info:
                raise ValueError("Coverage masks must not carry implicit alpha")
            if im.getexif().get(274, 1) != 1:
                raise ValueError("Mask orientation must be TOP-DOWN")
            im.load()
    else:
        raise ValueError("Mask must be a grayscale PNG or TIFF")
    digest = sha256_file(path)
    if _identity(path.stat()) != before:
        raise ValueError("Mask changed while being inspected")
    return {"path": str(path), "sha256": digest, "source_sha256": source_sha256,
            "width": width, "height": height, "bits": bits}


def inspect_reference(value) -> dict:
    """Reference pixels are view-only assets and never renderer input."""
    from PIL import Image
    path = ordinary_path(value)
    if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
        raise ValueError("Reference must be a JPEG, PNG or TIFF image")
    before = _identity(path.stat())
    if before[2] > 512 * 1024 * 1024:
        raise ValueError("Reference exceeds the size limit")
    with Image.open(path) as im:
        width, height = im.size
        if width * height > MAX_PIXELS or getattr(im, "n_frames", 1) != 1:
            raise ValueError("Reference must contain one image within the size limit")
        fmt = im.format
        im.verify()
    digest = sha256_file(path)
    if _identity(path.stat()) != before:
        raise ValueError("Reference changed while being inspected")
    return {"path": str(path), "sha256": digest, "bytes": before[2],
            "width": width, "height": height, "format": fmt}
