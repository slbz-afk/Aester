"""Fixed NL-1 GIMP render entry point. No work is done at import time.

The host owns process lifetime and supplies only a validated local request path.
"""

from __future__ import annotations

import array
import hashlib
import json
import os
import pathlib
import re
import struct
import sys
import zlib


def _require(test, message):
    if not test:
        raise ValueError(message)


def _sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _exact(obj, keys, name):
    _require(type(obj) is dict and set(obj) == set(keys), f"{name}: fields must be {keys}")
    return obj


def _absolute_file(value, name):
    _require(type(value) is str, f"{name}: expected path string")
    path = pathlib.Path(value)
    _require(path.is_absolute() and path.is_file(), f"{name}: expected existing absolute file")
    return path


def _hex(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _tiff(path):
    """Read classic TIFF metadata and raw, uncompressed interleaved strips."""
    blob = path.read_bytes()
    _require(len(blob) >= 8 and blob[:2] in (b"II", b"MM"), f"{path}: not TIFF")
    endian = "<" if blob[:2] == b"II" else ">"
    u16 = lambda off: struct.unpack_from(endian + "H", blob, off)[0]
    u32 = lambda off: struct.unpack_from(endian + "I", blob, off)[0]
    _require(u16(2) == 42, f"{path}: classic TIFF required")
    offset = u32(4)
    _require(8 <= offset < len(blob), f"{path}: bad IFD offset")
    count = u16(offset)
    _require(offset + 2 + 12 * count + 4 <= len(blob), f"{path}: truncated IFD")
    tags = {}
    for i in range(count):
        p = offset + 2 + i * 12
        tag, typ, n = struct.unpack_from(endian + "HHI", blob, p)
        unit = {1: 1, 2: 1, 3: 2, 4: 4, 7: 1}.get(typ)
        if unit is None or n > len(blob):
            continue
        size = n * unit
        start = p + 8 if size <= 4 else u32(p + 8)
        _require(start + size <= len(blob), f"{path}: tag {tag} outside file")
        part = blob[start:start + size]
        if typ == 3:
            values = struct.unpack(endian + "H" * n, part)
        elif typ == 4:
            values = struct.unpack(endian + "I" * n, part)
        elif typ in (1, 7):
            values = tuple(part)
        else:
            continue
        tags[tag] = values[0] if n == 1 else values
    next_ifd = u32(offset + 2 + count * 12)
    _require(next_ifd == 0, f"{path}: multipage TIFF rejected")
    one = lambda tag, default=None: tags.get(tag, default)
    width, height = one(256), one(257)
    _require(type(width) is int and type(height) is int and width > 0 and height > 0,
             f"{path}: invalid dimensions")
    _require(one(259, 1) == 1 and one(284, 1) == 1 and one(266, 1) == 1,
             f"{path}: only uncompressed interleaved FillOrder=1 TIFF supported")
    offsets, lengths = one(273), one(279)
    offsets = (offsets,) if type(offsets) is int else offsets
    lengths = (lengths,) if type(lengths) is int else lengths
    _require(type(offsets) is tuple and type(lengths) is tuple and len(offsets) == len(lengths),
             f"{path}: invalid strips")
    raw = bytearray()
    for off, length in zip(offsets, lengths):
        _require(off + length <= len(blob), f"{path}: strip outside file")
        raw += blob[off:off + length]
    return tags, width, height, endian, bytes(raw)


def _png_gray(path):
    blob = path.read_bytes()
    _require(blob[:8] == b"\x89PNG\r\n\x1a\n", f"{path}: not PNG")
    pos, data, header = 8, bytearray(), None
    while pos + 12 <= len(blob):
        size = struct.unpack_from(">I", blob, pos)[0]
        kind = blob[pos + 4:pos + 8]
        part = blob[pos + 8:pos + 8 + size]
        crc = struct.unpack_from(">I", blob, pos + 8 + size)[0]
        _require(zlib.crc32(kind + part) & 0xffffffff == crc, f"{path}: PNG CRC mismatch")
        pos += size + 12
        if kind == b"IHDR":
            header = struct.unpack(">IIBBBBB", part)
        elif kind == b"IDAT":
            data += part
        elif kind == b"IEND":
            break
    _require(header is not None, f"{path}: missing PNG header")
    width, height, bits, color, compression, filtering, interlace = header
    _require(bits in (8, 16) and color == 0 and compression == filtering == interlace == 0,
             f"{path}: only noninterlaced grayscale PNG8/16 supported")
    _require(0 < width <= 100000 and 0 < height <= 100000 and width * height <= 300000000,
             f"{path}: PNG dimensions exceed limits")
    stride, bpp = width * (bits // 8), bits // 8
    expected = height * (stride + 1)
    decoder = zlib.decompressobj()
    scan = decoder.decompress(data, expected + 1)
    _require(len(scan) == expected and decoder.eof and not decoder.unconsumed_tail
             and not decoder.unused_data, f"{path}: PNG scanline count mismatch")
    raw, prev, p = bytearray(), bytearray(stride), 0
    for _ in range(height):
        mode = scan[p]
        row = bytearray(scan[p + 1:p + 1 + stride])
        p += stride + 1
        _require(mode in (0, 1, 2, 3, 4), f"{path}: PNG filter unsupported")
        for i in range(stride):
            left = row[i - bpp] if i >= bpp else 0
            above = prev[i]
            upper_left = prev[i - bpp] if i >= bpp else 0
            if mode == 1:
                predictor = left
            elif mode == 2:
                predictor = above
            elif mode == 3:
                predictor = (left + above) // 2
            elif mode == 4:
                p0 = left + above - upper_left
                distances = (abs(p0 - left), abs(p0 - above), abs(p0 - upper_left))
                predictor = (left, above, upper_left)[distances.index(min(distances))]
            else:
                predictor = 0
            row[i] = (row[i] + predictor) & 255
        raw += row
        prev = row
    return width, height, bits, ">" if bits == 16 else None, bytes(raw)


def _mask_raster(path, expected_bits):
    if path.suffix.lower() == ".png":
        width, height, bits, endian, raw = _png_gray(path)
    elif path.suffix.lower() in (".tif", ".tiff"):
        tags, width, height, endian, raw = _tiff(path)
        bits = tags.get(258)
        _require(bits in (8, 16) and tags.get(277, 1) == 1 and tags.get(262) == 1
                 and tags.get(339, 1) == 1, f"{path}: unsigned min-black grayscale TIFF8/16 required")
    else:
        raise ValueError(f"{path}: grayscale PNG or TIFF required")
    _require(bits == expected_bits, f"{path}: mask bit depth mismatch")
    _require(len(raw) == width * height * bits // 8, f"{path}: mask raster byte count mismatch")
    if bits == 8:
        samples = array.array("H", (v * 257 for v in raw))
    else:
        samples = array.array("H")
        samples.frombytes(raw)
        if (endian == ">") != (sys.byteorder == "big"):
            samples.byteswap()
    return width, height, samples.tobytes(), hashlib.sha256(raw).hexdigest()


def _curve_value(points, x):
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return points[-1][1]


def _curve_approx_error(points, count=256):
    """Exact max of ideal-vs-sampled linear curves on [0,1].

    Both functions are affine between consecutive knots in the union of the
    control-point abscissas and the uniform sample grid. Their difference is
    affine there, so its absolute maximum occurs at a union knot.
    """
    samples = [_curve_value(points, i / (count - 1)) for i in range(count)]
    knots = {point[0] for point in points}
    knots.update(i / (count - 1) for i in range(count))
    maximum = 0.0
    for x in knots:
        position = x * (count - 1)
        left = min(int(position), count - 2)
        t = position - left
        approximate = samples[left] * (1 - t) + samples[left + 1] * t
        maximum = max(maximum, abs(_curve_value(points, x) - approximate))
    return maximum


def _configure_xcf_scalars(config, requested):
    """Compile scalar filter values to GIMP 3.2.6 XCF storage precision.

    XCF saves both G_TYPE_FLOAT and G_TYPE_DOUBLE arguments as gfloat.
    Configure that exact value before the first render, preserving requested
    values in the receipt. Curves use a different serialization path.
    """
    effective = {key: struct.unpack("<f", struct.pack("<f", value))[0]
                 for key, value in requested.items()}
    for key, value in effective.items():
        config.set_property(key, value)
    actual = {key: config.get_property(key) for key in effective}
    _require(actual == effective, "scalar filter parameter readback differs")
    return {"compiler": "xcf-float32-v1", "requested": requested,
            "effective": effective, "initial_readback": actual}


def _export(image, path, kind, Gimp, Gio):
    proc = Gimp.get_pdb().lookup_procedure(kind)
    _require(proc is not None, f"missing exporter {kind}")
    config = proc.create_config()
    names = {p.name for p in config.list_properties()}
    values = {"run-mode": Gimp.RunMode.NONINTERACTIVE,
              "image": image, "file": Gio.File.new_for_path(str(path)),
              "include-color-profile": True, "include-thumbnail": False}
    if kind == "file-tiff-export":
        values["compression"] = "none"
    else:
        values["quality"] = 0.95
    for key, value in values.items():
        _require(key in names, f"exporter {kind}: missing {key}")
        config.set_property(key, value)
    result = proc.run(config)
    _require(result.index(0) == Gimp.PDBStatusType.SUCCESS and path.is_file(), f"{kind} failed")


def run_request(path):
    """Validate a host request, render once, then write receipt.json last."""
    import gi
    gi.require_version("Gimp", "3.0")
    gi.require_version("Gegl", "0.4")
    from gi.repository import Gimp, Gegl, Gio
    _require(Gimp.version() == "3.2.6", "GIMP version requires new operator probe")

    script_dir = str(pathlib.Path(__file__).resolve().parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    from finishing_recipe import normalize_recipe, recipe_digest

    request_path = _absolute_file(str(path), "request")
    request = json.loads(request_path.read_text(encoding="utf8"))
    request_v2 = request.get("schema") == "aester-finishing-render-2"
    _exact(request, ("schema", "source", "recipe", "masks", "output_dir", "host_stage") if request_v2
           else ("schema", "source", "recipe", "masks", "output_dir"), "request")
    _require(request["schema"] in {"aester-finishing-render-1", "aester-finishing-render-2"}, "request schema mismatch")
    source = _exact(request["source"], ("path", "sha256", "bytes", "width", "height", "encoding", "icc_sha256"), "source")
    recipe = normalize_recipe(request["recipe"])
    curve_errors = {}
    for op in recipe["operations"]:
        if op["enabled"] and op["type"] == "curves":
            error = _curve_approx_error(op["params"]["points"])
            _require(error <= 1 / 255, f"curve {op['id']}: 256-sample error {error:.9g} exceeds 1/255")
            curve_errors[op["id"]] = error
    source_path = _absolute_file(source["path"], "source.path")
    _require(_hex(source["sha256"]) and _hex(source["icc_sha256"]), "source hash format")
    _require(type(source["bytes"]) is int and source["bytes"] > 0, "source.bytes invalid")
    _require(source_path.stat().st_size == source["bytes"] and _sha(source_path) == source["sha256"], "source bytes/hash mismatch")
    _require(recipe["source"] == {key: source[key] for key in ("sha256", "width", "height", "encoding")}, "recipe source mismatch")
    _require(source["encoding"] == "srgb16", "source encoding mismatch")
    tags, width, height, _, raw = _tiff(source_path)
    _require((width, height) == (source["width"], source["height"]), "source dimensions mismatch")
    sample_format = tags.get(339, (1, 1, 1))
    _require(tags.get(258) == (16, 16, 16) and tags.get(277) == 3 and tags.get(262) == 2
             and tags.get(274, 1) == 1 and sample_format in (1, (1, 1, 1))
             and len(raw) == width * height * 6,
             "source must be ordinary interleaved unsigned RGB16 TIFF")
    # Release the temporary whole-image strip copy before GIMP creates layers.
    del raw, tags
    _require(type(request["masks"]) is dict, "masks must be object")
    mask_data = {}
    for mask_id, item in request["masks"].items():
        _require(re.fullmatch(r"[a-z][a-z0-9_-]{0,47}", mask_id) is not None, "invalid mask id")
        _exact(item, ("path", "sha256", "source_sha256", "width", "height", "bits"), f"mask {mask_id}")
        mask_path = _absolute_file(item["path"], f"mask {mask_id}.path")
        _require(_hex(item["sha256"]) and _sha(mask_path) == item["sha256"], f"mask {mask_id}: SHA mismatch")
        _require(item["source_sha256"] == source["sha256"], f"mask {mask_id}: source binding mismatch")
        _require(type(item["width"]) is int and type(item["height"]) is int
                 and (item["width"], item["height"]) == (width, height), f"mask {mask_id}: dimensions mismatch")
        _require(type(item["bits"]) is int and item["bits"] in (8, 16), f"mask {mask_id}: bits invalid")
        mw, mh, pixels, raw_hash = _mask_raster(mask_path, item["bits"])
        _require((mw, mh) == (width, height), f"mask {mask_id}: raster dimensions mismatch")
        mask_data[mask_id] = (pixels, raw_hash, item["bits"])
    for op in recipe["operations"]:
        _require(op["mask_id"] is None or op["mask_id"] in mask_data, f"operation {op['id']}: missing mask")
    out = pathlib.Path(request["output_dir"])
    _require(out.is_absolute() and out.is_dir() and not any(out.iterdir()), "output_dir must be absolute, existing, empty")

    layer_ops = [op for op in recipe["operations"] if op["type"] == "astrophotographic_layers"]
    _require(bool(layer_ops) == request_v2, "host-stage request version mismatch")
    host_stage = request.get("host_stage")
    stage_summary = None
    stage_path = None
    if layer_ops and layer_ops[0]["enabled"]:
        _exact(host_stage, ("source_sha256", "recipe_sha256", "bundle_sha256", "implementation",
                            "image", "pixel_sha256", "details"), "host_stage")
        _require(host_stage["source_sha256"] == source["sha256"]
                 and host_stage["recipe_sha256"] == recipe_digest(recipe)
                 and host_stage["bundle_sha256"] == layer_ops[0]["params"]["bundle_sha256"],
                 "host-stage upstream binding differs")
        _require(host_stage["implementation"] == "aester-layer-stage-teacher-1"
                 and type(host_stage["details"]) is dict, "host-stage implementation missing")
        stage_image = _exact(host_stage["image"], tuple(source), "host_stage.image")
        stage_path = _absolute_file(stage_image["path"], "host_stage.image.path")
        _require(stage_path == request_path.parent / "stages" / "layers.tif", "host-stage path is not candidate-owned")
        _require(all(stage_image[key] == source[key] for key in ("width", "height", "encoding", "icc_sha256")),
                 "host-stage image geometry or domain differs")
        _require(type(stage_image["bytes"]) is int and stage_path.stat().st_size == stage_image["bytes"]
                 and _sha(stage_path) == stage_image["sha256"], "host-stage file identity differs")
        stage_tags, sw, sh, endian, raw = _tiff(stage_path)
        _require(34675 in stage_tags and hashlib.sha256(bytes(stage_tags[34675])).hexdigest() == source["icc_sha256"],
                 "host-stage actual ICC differs")
        _require((sw, sh) == (width, height) and stage_tags.get(258) == (16, 16, 16)
                 and stage_tags.get(277) == 3 and stage_tags.get(262) == 2
                 and stage_tags.get(274, 1) == 1 and stage_tags.get(339, (1, 1, 1)) in (1, (1, 1, 1))
                 and len(raw) == width * height * 6, "host-stage RGB16 raster differs")
        # Canonical pixel identity is little-endian RGB16, independent of TIFF byte order.
        if endian == ">":
            converted = array.array("H")
            converted.frombytes(raw)
            converted.byteswap()
            raw = converted.tobytes()
            del converted
        _require(_hex(host_stage["pixel_sha256"]) and hashlib.sha256(raw).hexdigest() == host_stage["pixel_sha256"],
                 "host-stage decoded pixel identity differs")
        del raw, stage_tags
        stage_summary = {key: host_stage[key] for key in
                         ("source_sha256", "recipe_sha256", "bundle_sha256", "implementation", "pixel_sha256")}
    else:
        _require(host_stage is None, "unexpected enabled host stage")

    image = Gimp.file_load(Gimp.RunMode.NONINTERACTIVE, Gio.File.new_for_path(str(source_path)))
    _require(image is not None, "source GIMP load failed")
    flat = None
    reopened = None
    replay = None
    try:
        _require((image.get_width(), image.get_height()) == (width, height), "GIMP source size mismatch")
        _require(str(image.get_precision()) == "250" and len(image.get_layers()) == 1,
                 "GIMP source must load as one U16 nonlinear layer")
        profile = image.get_color_profile()
        _require(profile is not None and hashlib.sha256(bytes(profile.get_icc_profile())).hexdigest() == source["icc_sha256"],
                 "GIMP source ICC mismatch")
        applied, details = [], []
        for op in recipe["operations"]:
            if not op["enabled"]:
                continue
            if op["type"] == "astrophotographic_layers":
                layer = Gimp.file_load_layer(Gimp.RunMode.NONINTERACTIVE, image, Gio.File.new_for_path(str(stage_path)))
                _require(layer is not None and image.insert_layer(layer, None, 0), "host-stage layer import failed")
                layer.set_name(op["id"])
                layer.set_offsets(0, 0)
                layer.set_opacity(100.0)
                layer.set_mode(Gimp.LayerMode.NORMAL)
                layer.set_visible(True)
                _require((layer.get_width(), layer.get_height()) == (width, height), "host-stage layer size differs")
                # Read the actual imported layer, not merely its source TIFF hash.
                actual = bytes(layer.get_buffer().get(Gegl.Rectangle.new(0, 0, width, height),
                    1.0, "R'G'B' u16", Gegl.AbyssPolicy.NONE))
                _require(hashlib.sha256(actual).hexdigest() == host_stage["pixel_sha256"],
                         "GIMP host-stage import changed encoded pixels")
                del actual
                applied.append(op["id"])
                details.append({"id": op["id"], "mask_id": None, "operator": "host-stage-snapshot",
                    "implementation": host_stage["implementation"], "pixel_sha256": host_stage["pixel_sha256"],
                    "editability": "Cumulative RGB16 snapshot; rerun recipe to change these parameters"})
                continue
            layer = Gimp.Layer.new_from_visible(image, image, op["id"])
            _require(layer is not None and image.insert_layer(layer, None, 0), f"{op['id']}: cannot insert visible layer")
            layer.set_mode(Gimp.LayerMode.NORMAL)
            layer.set_blend_space(Gimp.LayerColorSpace.RGB_NON_LINEAR)
            layer.set_composite_space(Gimp.LayerColorSpace.RGB_NON_LINEAR)
            layer.set_composite_mode(Gimp.LayerCompositeMode.UNION)
            _require(layer.get_mode() == Gimp.LayerMode.NORMAL
                     and layer.get_blend_space() == Gimp.LayerColorSpace.AUTO
                     and layer.get_composite_space() == Gimp.LayerColorSpace.RGB_NON_LINEAR
                     and layer.get_composite_mode() == Gimp.LayerCompositeMode.UNION,
                     f"{op['id']}: compositing mode mismatch")
            mask_id = op["mask_id"]
            if mask_id is not None:
                mask = layer.create_mask(Gimp.AddMaskType.WHITE)
                _require(layer.add_mask(mask), f"{op['id']}: cannot add mask")
                buffer = mask.get_buffer()
                rect = Gegl.Rectangle.new(0, 0, width, height)
                pixels, raw_hash, bits = mask_data[mask_id]
                buffer.set(rect, "Y u16", pixels)
                buffer.flush()
                _require(bytes(buffer.get(rect, 1.0, "Y u16", Gegl.AbyssPolicy.NONE)) == pixels,
                         f"{op['id']}: mask readback differs")
            else:
                raw_hash, bits = None, None
            if op["type"] == "curves":
                points = op["params"]["points"]
                curve = Gimp.Curve.new()
                curve.set_curve_type(Gimp.CurveType.FREE)
                curve.set_n_samples(256)
                _require(curve.get_n_samples() == 256, "curve sample count unsupported")
                for i in range(256):
                    x = i / 255
                    curve.set_sample(x, _curve_value(points, x))
                filter_obj = Gimp.DrawableFilter.new(layer, "gimp:curves", op["id"])
                config = filter_obj.get_config()
                config.set_property("curve", curve)
                config.set_property("channel", getattr(Gimp.HistogramChannel, op["params"]["channel"].upper()))
                config.set_property("trc", Gimp.TRCType.NON_LINEAR)
                mapping = {"operator": "gimp:curves", "trc": "nonlinear", "samples": 256,
                           "curve_contract": "256-sample approximation of piecewise-linear mapping",
                           "curve_backend": "gimp-free-256-v1",
                           "max_ideal_sample_curve_error": curve_errors[op["id"]]}
            elif op["type"] == "exposure":
                filter_obj = Gimp.DrawableFilter.new(layer, "gegl:exposure", op["id"])
                config = filter_obj.get_config()
                scalars = _configure_xcf_scalars(config, {"exposure": op["params"]["ev"], "black-level": 0.0})
                mapping = {"operator": "gegl:exposure", "exposure_ev": scalars["effective"]["exposure"],
                           "black_level": 0.0, "scalar_parameters": scalars}
            elif op["type"] == "local_contrast_unsharp":
                filter_obj = Gimp.DrawableFilter.new(layer, "gegl:unsharp-mask", op["id"])
                _require(filter_obj is not None, "gegl:unsharp-mask is unavailable")
                config = filter_obj.get_config()
                scalars = _configure_xcf_scalars(config, {"std-dev": op["params"]["sigma_px"],
                    "scale": op["params"]["amount"], "threshold": 0.0})
                mapping = {"operator": "gegl:unsharp-mask", "implementation": "gimp-3.2.6-unsharp-v2-xcf-float32",
                           "sigma_source_pixels": scalars["effective"]["std-dev"], "amount": scalars["effective"]["scale"],
                           "scalar_parameters": scalars,
                           "threshold": 0.0, "internal_space": "GEGL native; not declared scene-linear or Lightroom-equivalent"}
            else:
                filter_obj = Gimp.DrawableFilter.new(layer, "gegl:saturation", op["id"])
                config = filter_obj.get_config()
                scalars = _configure_xcf_scalars(config, {"scale": op["params"]["scale"]})
                config.set_property("colorspace", "Native")
                mapping = {"operator": "gegl:saturation", "scale": scalars["effective"]["scale"],
                           "colorspace": "Native", "scalar_parameters": scalars}
            _require(filter_obj is not None, f"{op['id']}: filter creation failed")
            layer.append_filter(filter_obj)
            applied.append(op["id"])
            details.append({"id": op["id"], "mask_id": mask_id, "mask_raw_sha256": raw_hash,
                            "mask_bits": bits,
                            "mask_u16_sha256": hashlib.sha256(mask_data[mask_id][0]).hexdigest() if mask_id else None,
                            "layer_mode": "NORMAL", "blend_space": "AUTO (GIMP coerces NORMAL mode)",
                            "composite_space": "RGB_NON_LINEAR", "composite_mode": "UNION",
                            **mapping})
        xcf = out / "result.xcf"
        _require(Gimp.file_save(Gimp.RunMode.NONINTERACTIVE, image, Gio.File.new_for_path(str(xcf)), None), "XCF save failed")
        flat = image.duplicate()
        flat.flatten()
        _export(flat, out / "result.tif", "file-tiff-export", Gimp, Gio)
        _export(flat, out / "result.jpg", "file-jpeg-export", Gimp, Gio)
        reopened = Gimp.file_load(Gimp.RunMode.NONINTERACTIVE, Gio.File.new_for_path(str(xcf)))
        _require(reopened is not None and len(reopened.get_layers()) == len(image.get_layers())
                 and str(reopened.get_precision()) == "250", "XCF reopen mismatch")
        replay_profile = reopened.get_color_profile()
        _require(replay_profile is not None and hashlib.sha256(bytes(replay_profile.get_icc_profile())).hexdigest()
                 == source["icc_sha256"], "XCF reopen ICC mismatch")
        reopened_layers = {layer.get_name(): layer for layer in reopened.get_layers()}
        for detail in details:
            layer = reopened_layers.get(detail["id"])
            _require(layer is not None, "XCF operation layer missing")
            if "scalar_parameters" in detail:
                filters = layer.get_filters()
                _require(len(filters) == 1 and filters[0].get_operation_name() == detail["operator"],
                         "XCF scalar filter missing or different")
                saved_config = filters[0].get_config()
                expected = detail["scalar_parameters"]["effective"]
                saved = {key: saved_config.get_property(key) for key in expected}
                _require(saved == expected, "XCF scalar filter parameters changed")
                detail["scalar_parameters"]["reopened_readback"] = saved
            if detail["mask_id"] is None:
                continue
            _require(layer is not None and layer.get_mask() is not None, "XCF mask missing")
            mask_pixels = bytes(layer.get_mask().get_buffer().get(
                Gegl.Rectangle.new(0, 0, width, height), 1.0, "Y u16", Gegl.AbyssPolicy.NONE))
            _require(hashlib.sha256(mask_pixels).hexdigest() == detail["mask_u16_sha256"],
                     "XCF mask pixel round trip mismatch")
        replay = reopened.duplicate()
        replay.flatten()
        _export(replay, out / "replay.tif", "file-tiff-export", Gimp, Gio)
        receipt = {"schema": "aester-finishing-gimp-receipt-1", "ok": True,
                   "source_sha256": source["sha256"], "recipe_sha256": recipe_digest(recipe),
                   "width": width, "height": height, "gimp_version": Gimp.version(),
                   "applied_operations": applied, "precision": "U16 nonlinear", "icc_sha256": source["icc_sha256"],
                   "mask_pixel_format": "Y u16 linear coverage", "layers": len(image.get_layers()),
                   "operations_detail": details}
        if request_v2:
            receipt["host_stage"] = stage_summary
        (out / "receipt.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf8")
        return receipt
    finally:
        if replay is not None:
            replay.delete()
        if reopened is not None:
            reopened.delete()
        if flat is not None:
            flat.delete()
        image.delete()


if __name__ == "__main__":
    request = os.environ.get("AESTER_FINISHING_REQUEST")
    _require(request, "AESTER_FINISHING_REQUEST absent")
    print(json.dumps(run_request(request), ensure_ascii=False), flush=True)
