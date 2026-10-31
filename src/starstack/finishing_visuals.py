"""Source-coordinate masks and explicitly labeled view-only previews."""
from __future__ import annotations

import math
from pathlib import Path
import uuid

from .finishing_assets import ordinary_path


def _folder(directory):
    folder = Path(directory) / "previews"
    folder.mkdir(exist_ok=True)
    return ordinary_path(folder, directory=True)


def draw_coverage(directory, source, polygons, *, feather_px=0, invert=False):
    import numpy as np
    from PIL import Image, ImageFilter, ImageOps
    import tifffile
    if (type(polygons) is not list or not 1 <= len(polygons) <= 64
            or type(invert) is not bool or type(feather_px) not in {int, float}
            or not math.isfinite(feather_px) or not 0 <= feather_px <= 1024):
        raise ValueError("Invalid polygons, feather radius or invert flag")
    width, height = source["width"], source["height"]
    checked, count = [], 0
    for polygon in polygons:
        if type(polygon) is not list or not 3 <= len(polygon) <= 4096:
            raise ValueError("Each polygon needs 3 to 4096 source-coordinate points")
        points = []
        for point in polygon:
            if (type(point) is not list or len(point) != 2
                    or any(type(v) not in {int, float} or not math.isfinite(v) for v in point)
                    or not 0 <= point[0] <= width or not 0 <= point[1] <= height):
                raise ValueError("Mask points must stay within source dimensions")
            points.append(tuple(float(v) for v in point))
        checked.append(points)
        count += len(points)
    if count > 16384:
        raise ValueError("Mask point budget exceeded")
    raster = np.zeros((height, width), dtype=np.uint8)
    for points in checked:
        # Sample source pixel centres inside an even-odd polygon. Integer
        # coordinates describe pixel edges, so [2,6) covers exactly 4 pixels.
        starts, stops = {}, {}
        for edge, (p0, p1) in enumerate(zip(points, points[1:] + points[:1])):
            if p0[1] == p1[1]:
                continue
            if p0[1] > p1[1]:
                p0, p1 = p1, p0
            begin = max(0, math.ceil(p0[1] - 0.5))
            end = min(height, math.ceil(p1[1] - 0.5))
            if begin >= end:
                continue
            item = (p0[0], p0[1], (p1[0] - p0[0]) / (p1[1] - p0[1]))
            starts.setdefault(begin, []).append((edge, item))
            stops.setdefault(end, []).append(edge)
        if not starts:
            continue
        active = {}
        for row in range(min(starts), max(stops)):
            for edge in stops.get(row, ()):
                active.pop(edge, None)
            active.update(starts.get(row, ()))
            crossings = sorted(x + (row + 0.5 - y) * slope for x, y, slope in active.values())
            for left, right in zip(crossings[::2], crossings[1::2]):
                first, last = max(0, math.ceil(left - 0.5)), min(width, math.ceil(right - 0.5))
                raster[row, first:last] = 255
    image = Image.fromarray(raster)
    if feather_px:
        image = image.filter(ImageFilter.GaussianBlur(feather_px))
    if invert:
        image = ImageOps.invert(image)
    folder = _folder(directory)
    path = folder / ("drawn-" + uuid.uuid4().hex + ".tif")
    tifffile.imwrite(path, np.asarray(image), photometric="minisblack", metadata=None)
    return {"path": str(path), "geometry": {"polygons": polygons, "feather_px": float(feather_px),
            "invert": invert, "coordinates": "source-pixel edges",
            "raster": "pixel-centre even-odd polygon union (left/top included, right/bottom excluded), then Pillow L8 GaussianBlur"}}


def photo_preview(directory, source_record, *, crop=None, mask_record=None):
    """Return an sRGB JPEG view; numerical validation always uses the TIFF."""
    import numpy as np
    from PIL import Image
    import tifffile
    path = ordinary_path(source_record["path"])
    width, height = source_record["width"], source_record["height"]
    if crop is None:
        crop = [0, 0, width, height]
    if (type(crop) is not list or len(crop) != 4 or any(type(v) is not int for v in crop)
            or min(crop[:2]) < 0 or min(crop[2:]) <= 0
            or crop[0] + crop[2] > width or crop[1] + crop[3] > height):
        raise ValueError("Crop must be [x,y,width,height] within source pixels")
    x, y, cw, ch = crop
    with tifffile.TiffFile(path) as tf:
        page = tf.pages[0]
        pixels = page.asarray()[y:y + ch, x:x + cw]
        icc = bytes(page.tags[34675].value)
        view = ((pixels.astype(np.uint32) + 128) // 257).astype(np.uint8)
    image = Image.fromarray(view, "RGB")
    image.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
    if mask_record is not None:
        mask_path = ordinary_path(mask_record["path"])
        if mask_path.suffix.lower() in {".tif", ".tiff"}:
            weights = tifffile.imread(mask_path)
        else:
            with Image.open(mask_path) as mask:
                weights = np.array(mask)
        weights = weights[y:y + ch, x:x + cw].astype(np.float32) / (2 ** mask_record["bits"] - 1)
        alpha = Image.fromarray(weights, "F").resize(image.size, Image.Resampling.BILINEAR)
        coverage = np.asarray(alpha)[..., None] * 0.45
        overlay = np.asarray(image).astype(np.float32) * (1 - coverage) + np.array([255, 70, 40]) * coverage
        image = Image.fromarray(np.rint(overlay).astype(np.uint8), "RGB")
    output = _folder(directory) / ("preview-" + uuid.uuid4().hex + ".jpg")
    image.save(output, quality=95, icc_profile=icc)
    return {"path": str(output), "mime_type": "image/jpeg", "width": image.width, "height": image.height,
            "source_crop": crop, "source_width": width, "source_height": height,
            "native_scale": image.size == (cw, ch), "scale": image.width / cw,
            "scale_x": image.width / cw, "scale_y": image.height / ch,
            "coordinate_mapping": "source edge = crop origin + preview edge / scale_x or scale_y; scale is the horizontal alias",
            "source_sha256": source_record["sha256"],
            "mask_sha256": mask_record["sha256"] if mask_record else None,
            "kind": "mask_overlay" if mask_record else "photo_view",
            "transport": "local_path_only", "numerical_validation": False}


def reference_view(directory, record):
    """Make a color-managed, view-only reference; never an editing input."""
    import io
    from PIL import Image, ImageCms, ImageOps
    from .finishing_assets import inspect_reference, sha256_file
    actual = inspect_reference(record["path"])
    if any(record.get(key) != value for key, value in actual.items()):
        raise ValueError("Reference changed after registration")
    target_profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))
    with Image.open(actual["path"]) as original:
        original.load()
        profile = original.info.get("icc_profile")
        oriented = ImageOps.exif_transpose(original)
        alpha = oriented.convert("RGBA").getchannel("A") if "A" in oriented.getbands() or "transparency" in oriented.info else None
        image = oriented if oriented.mode in {"RGB", "CMYK", "L"} else oriented.convert("RGB")
        if profile:
            try:
                source_profile = ImageCms.ImageCmsProfile(io.BytesIO(profile))
                image = ImageCms.profileToProfile(image, source_profile, target_profile, outputMode="RGB")
            except (ValueError, OSError, ImageCms.PyCMSError) as exc:
                raise ValueError("Reference ICC cannot be converted to sRGB for display") from exc
            color_note = "embedded ICC converted to sRGB for display"
        else:
            if image.mode == "CMYK":
                raise ValueError("CMYK reference needs an embedded ICC for a reliable preview")
            image = image.convert("RGB")
            color_note = "untagged reference assumed sRGB for display; original color space unverified"
        if alpha is not None:
            background = Image.new("RGB", image.size, (32, 32, 32))
            background.paste(image, mask=alpha)
            image = background
            color_note += "; transparency displayed over dark gray"
        image.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
        if sha256_file(actual["path"]) != record["sha256"]:
            raise ValueError("Reference changed during preview")
        output = _folder(directory) / ("reference-" + uuid.uuid4().hex + ".png")
        image.save(output, icc_profile=target_profile.tobytes())
    return {"path": str(output), "mime_type": "image/png", "width": image.width,
            "height": image.height, "reference_id": record["id"], "kind": record["kind"],
            "preview_kind": "reference_view", "source_sha256": record["sha256"],
            "color_note": color_note, "reference_pixels_used": False, "numerical_validation": False}
