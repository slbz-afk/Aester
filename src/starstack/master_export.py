"""M · scene-referred linear master export (docs/COLOR_CONTRACT_PACKAGE_20260922.md §2.1, §3.1).

Input: a job's camera-RGB master (``products/<uuid>/master_linear.fit`` or ``work/sky_master.fit``; Siril
camera RGB DN/65535, black not subtracted, no WB, no matrix).  Colour basis: the reference frame's DNG tags
(``BlackLevel``/``WhiteLevel``/``ColorMatrix``/``AsShotNeutral``) or an existing ``raw_color`` recipe.  Output: a
float32 TOP-DOWN RGB TIFF in a linear working gamut with the product's own ICC profile, a camera-saturation mask,
and a replayable recipe.  Nothing is clipped; exact-zero (no-data) pixels stay zero and are counted.

The camera → sRGB matrix comes from ``raw_color.cam_xyz_to_rgb_cam`` (the LibRaw convention already used by
the product); white balance is either the daylight-type neutral implied by that matrix (LibRaw ``pre_mul``,
constant per camera model) or the reference frame's as-shot neutral.  Both are recorded and interconvertible.
The Z8 DCP ForwardMatrix route is accepted through ``--color-basis dcp`` (matrix only, never HueSatMap).
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import io
import json
import os
import platform

import numpy as np
from PIL import Image
import tifffile

from . import __version__, hashing, icc_profiles as icc, raw_color
from .raw_audit import read_dng_tags

VERSION = "master-export-1"
SCHEMA = "starstack-master-1"
FILENAME = "master_scene_linear.tif"
MASK_FILENAME = "camera_saturation_mask.png"
RECIPE_FILENAME = "master_recipe.json"
SATURATION_FRACTION = 0.97
WB_MODES = ("daylight", "as_shot")
ROWS = 256


def sha256_file(path, *, source: str = "master_export.sha256_file", fresh: bool = False) -> str:
    """SHA-256 of the file's bytes through :mod:`starstack.hashing` (inside a run scope an unchanged file may be
    answered from the run ledger).  ``Path(path)`` is opened, as before, so odd values fail as they always did.
    ``fresh``: the caller has just written and closed ``path`` (see :func:`starstack.hashing.sha256_file`)."""
    return hashing.sha256_file(Path(path), source=source, fresh=fresh)


def save_image_with_digest(image, path, *, source: str, **params) -> str | None:
    """``image.save(path, **params)``; inside a hash run scope (not "off") encoded in memory first, then written
    with its digest recorded as written by this process, so a later :func:`sha256_file` of the unchanged file is a
    ledger hit instead of a read-back.  The bytes are the same: Pillow's encoders do not depend on the sink, and the
    format comes from the file extension exactly as ``Image.save`` chooses it; a failed write deletes a file this
    call created, as ``Image.save`` does.  Returns the digest, or None when the plain ``image.save`` ran (outside a
    run scope, or with ``AESTER_HASH_LEDGER=off``)."""
    if not hashing.digest_writes():
        image.save(path, **params)
        return None
    extension = os.path.splitext(os.fspath(path))[1].lower()
    image_format = Image.registered_extensions().get(extension)
    if image_format is None:
        raise ValueError(f"unknown file extension: {extension}")
    buffer = io.BytesIO()
    image.save(buffer, format=image_format, **params)
    view = buffer.getbuffer()
    try:
        return hashing.write_bytes_with_digest(path, view, source=source, discard=hashing.DISCARD_CREATED)
    finally:
        view.release()


def _json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _numeric(record):
    if record is None:
        return None
    return record.get("numeric", record.get("value"))


def colour_basis_from_dng(path, *, wb_mode: str = "daylight") -> dict:
    """Colour basis from one reference DNG: black/white levels, ColorMatrix (D65 preferred), as-shot and daylight WB."""
    path = Path(path).resolve(strict=True)
    if wb_mode not in WB_MODES:
        raise ValueError(f"wb_mode 只能是 {WB_MODES}")
    tags_record = read_dng_tags(path)
    if tags_record.get("status") != "ok":
        raise ValueError("需要可读取的 DNG 参考帧")
    tags = tags_record["tags"]

    def need(name):
        if name not in tags:
            raise ValueError(f"参考帧缺少 {name} 标签")
        return _numeric(tags[name])

    cfa = [int(v) for v in need("CFAPattern")]
    dims = [int(v) for v in need("CFARepeatPatternDim")]
    if dims != [2, 2] or len(cfa) != 4 or sorted(cfa) != [0, 1, 1, 2]:
        raise ValueError("只支持 2×2 RGGB 系 Bayer 参考帧")
    pattern = np.array(cfa, dtype=int).reshape(2, 2)
    black_values = np.asarray(need("BlackLevel"), dtype=np.float64).reshape(-1)
    black_dims = [int(v) for v in _numeric(tags["BlackLevelRepeatDim"])] if "BlackLevelRepeatDim" in tags else [1, 1]
    if black_values.size == 1:
        black_grid = np.full((2, 2), float(black_values[0]))
    elif black_dims == [2, 2] and black_values.size == 4:
        black_grid = black_values.reshape(2, 2)
    else:
        raise ValueError("不支持的 BlackLevel 布局")
    black_rgb = [float(black_grid[pattern == c].mean()) for c in (0, 1, 2)]
    white = float(np.asarray(need("WhiteLevel"), dtype=np.float64).reshape(-1)[0])
    if not white > max(black_rgb):
        raise ValueError("白位必须高于黑位")
    illuminants = {n: (int(np.asarray(_numeric(tags[n])).reshape(-1)[0]) if n in tags else None)
                   for n in ("CalibrationIlluminant1", "CalibrationIlluminant2")}
    chosen = None
    for cm_name, il_name in (("ColorMatrix2", "CalibrationIlluminant2"), ("ColorMatrix1", "CalibrationIlluminant1")):
        if cm_name in tags and illuminants[il_name] == 21:
            chosen = (cm_name, 21)
            break
    if chosen is None:
        for cm_name, il_name in (("ColorMatrix2", "CalibrationIlluminant2"), ("ColorMatrix1", "CalibrationIlluminant1")):
            if cm_name in tags:
                chosen = (cm_name, illuminants[il_name])
                break
    if chosen is None:
        raise ValueError("参考帧没有 ColorMatrix")
    cam_xyz = np.asarray(_numeric(tags[chosen[0]]), dtype=np.float64).reshape(3, 3)
    neutral = np.asarray(need("AsShotNeutral"), dtype=np.float64).reshape(-1)
    if neutral.size != 3 or np.any(neutral <= 0):
        raise ValueError("AsShotNeutral 无效")
    analog = np.asarray(_numeric(tags["AnalogBalance"]), dtype=np.float64).reshape(-1) if "AnalogBalance" in tags else np.ones(3)
    if not np.allclose(analog, 1.0):
        raise ValueError("AnalogBalance 不为 1 的参考帧暂不支持")
    for name in ("CameraCalibration1", "CameraCalibration2"):
        if name in tags and not np.allclose(np.asarray(_numeric(tags[name]), dtype=np.float64).reshape(3, 3), np.eye(3)):
            raise ValueError(f"{name} 不是单位矩阵的参考帧暂不支持")
    baseline = float(np.asarray(_numeric(tags["BaselineExposure"])).reshape(-1)[0]) if "BaselineExposure" in tags else 0.0
    camera = None
    with tifffile.TiffFile(path) as tf:
        page = tf.pages[0]
        for name in ("UniqueCameraModel", "Model", "Make"):
            tag = page.tags.get(name)
            if tag is not None and isinstance(tag.value, str):
                camera = tag.value if camera is None else camera
        make = page.tags.get("Make")
        model = page.tags.get("Model")
        identity = {"unique_camera_model": camera,
                    "make": make.value if make is not None else None, "model": model.value if model is not None else None}
    metadata = {"available": True, "white_balance_applied": False, "color_correction_applied": False,
                "versions": {"source": "dng_tags", "tifffile": tifffile.__version__},
                "fields": {"num_colors": {"value": 3}, "color_desc": {"value": "RGBG"},
                           "raw_pattern": {"value": pattern.tolist()},
                           "rgb_xyz_matrix": {"value": np.vstack([cam_xyz, np.zeros((1, 3))]).tolist()},
                           "camera_whitebalance": {"value": [1 / neutral[0], 1 / neutral[1], 1 / neutral[2], 1 / neutral[1]]}}}
    recipe = raw_color.build_recipe(metadata, sha256_file(path, source="master_export.colour_basis"), white,
                                    black_rgb[1], camera_identity=identity)
    return _finish_basis(recipe, wb_mode, black_rgb=black_rgb, source={"kind": "dng", "path": str(path),
                         "sha256": recipe["reference"]["source_sha256"], "color_matrix_tag": chosen[0],
                         "calibration_illuminant": chosen[1], "as_shot_neutral": neutral.tolist(),
                         "baseline_exposure_ev": baseline, "baseline_exposure_applied": False, "camera": identity})


def colour_basis_from_raw(path, *, wb_mode: str = "daylight", metadata: dict | None = None) -> dict:
    """Colour basis for a non-DNG RAW from LibRaw metadata (``raw_metadata.read_raw_metadata``; needs rawpy — Windows).

    ``metadata`` may be supplied (tests, or a metadata record saved by an earlier inspection); its structure is the
    LibRaw-shaped dict ``raw_color.build_recipe`` accepts.  Black per channel = LibRaw ``black_level_per_channel`` mapped
    [R, mean(G1, G2), B]; white = ``white_level``; the D65 matrix and both white balances follow the DNG route.
    """
    path = Path(path).resolve(strict=True)
    if wb_mode not in WB_MODES:
        raise ValueError(f"wb_mode 只能是 {WB_MODES}")
    if metadata is None:
        from .raw_metadata import read_raw_metadata
        metadata = read_raw_metadata(path)
    if not isinstance(metadata, dict) or metadata.get("available") is not True:
        raise ValueError(f"无法读取 RAW 元数据：{metadata.get('status') if isinstance(metadata, dict) else metadata}")
    fields = metadata.get("fields") or {}

    def value(name):
        record = fields.get(name)
        if not isinstance(record, dict) or record.get("value") is None:
            raise ValueError(f"RAW 元数据缺少 {name}")
        return record["value"]

    black = np.asarray(value("black_level_per_channel"), dtype=np.float64).reshape(-1)
    white = float(np.asarray(value("white_level"), dtype=np.float64).reshape(-1)[0])
    if black.size != 4 or not white > black.max():
        raise ValueError("RAW 黑位/白位无效")
    black_rgb = [float(black[0]), float((black[1] + black[3]) / 2), float(black[2])]
    recipe = raw_color.build_recipe(metadata, sha256_file(path, source="master_export.colour_basis"), white, black_rgb[1],
                                    camera_identity={"source": "libraw_metadata", "versions": metadata.get("versions")})
    # The body from the container tags (EXIF Make/Model/Software; None when the container is not TIFF-readable), so a
    # render can pick the look registered for it (camera_looks); the colour basis itself does not depend on it.
    from .devices import identify_camera
    ident = identify_camera(path)
    camera = ({"make": ident["make"], "model": ident["model"], "software": ident["software"], "source": "container_tags"}
              if ident["identified"] else None)
    return _finish_basis(recipe, wb_mode, black_rgb=black_rgb,
                         source={"kind": "libraw", "path": str(path), "sha256": recipe["reference"]["source_sha256"],
                                 "libraw_versions": metadata.get("versions"), "baseline_exposure_applied": False,
                                 "camera": camera})


def colour_basis_for_reference(path, *, wb_mode: str = "daylight", metadata: dict | None = None) -> dict:
    """DNG → tag route; any other RAW → LibRaw route."""
    path = Path(path)
    if path.suffix.lower() == ".dng":
        return colour_basis_from_dng(path, wb_mode=wb_mode)
    return colour_basis_from_raw(path, wb_mode=wb_mode, metadata=metadata)


def colour_basis_from_recipe(recipe: dict, *, wb_mode: str = "daylight", source_path=None) -> dict:
    """Colour basis from an existing ``raw_color`` recipe (e.g. one recorded by an earlier development run)."""
    if wb_mode not in WB_MODES:
        raise ValueError(f"wb_mode 只能是 {WB_MODES}")
    recipe = raw_color.validate_recipe(recipe)
    black = recipe["reference"]["common_black_for_scale_DN"]
    if black is None or recipe["reference"]["white_DN"] is None:
        raise ValueError("配方缺少参考白位/黑位，无法确定母版单位")
    return _finish_basis(recipe, wb_mode, black_rgb=[float(black)] * 3,
                         source={"kind": "recipe", "path": None if source_path is None else str(source_path),
                                 "sha256": recipe["reference"]["source_sha256"]})


def _finish_basis(recipe: dict, wb_mode: str, *, black_rgb, source) -> dict:
    diag = np.asarray(recipe["matrix"]["daylight_normalization_diagnostic"], dtype=np.float64)
    daylight = (diag / diag[1]).tolist()
    as_shot = list(recipe["white_balance"]["gains_rgb"])
    gains = daylight if wb_mode == "daylight" else as_shot
    white = float(recipe["reference"]["white_DN"])
    return {
        "schema": "starstack-colour-basis-1", "wb_mode": wb_mode, "gains_rgb": gains,
        "daylight_gains_rgb": daylight, "as_shot_gains_rgb": as_shot,
        "black_DN": [float(v) for v in black_rgb], "common_black_DN": float(recipe["reference"]["common_black_for_scale_DN"]),
        "white_DN": white, "scale_DN": float(recipe["scale"]["common_denominator_DN"]),
        "camera_to_linear_srgb": recipe["matrix"]["values"], "matrix_construction": recipe["matrix"]["construction"],
        "cam_xyz": recipe["camera"]["model"]["cam_xyz"], "source": source, "raw_color_recipe": recipe,
        "recipe_sha256": hashlib.sha256(json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
    }


def switch_white_balance_matrix(basis: dict, new_gains, gamut: str) -> np.ndarray:
    """3×3 acting on finished master pixels (working gamut) that re-balances them to ``new_gains``: since the master is
    ``S·R·diag(g)·x``, the change is ``S·R·diag(g'/g)·(S·R)⁻¹``.  Exact in float; used for the as-shot candidate."""
    sr = icc.SRGB.matrix_to(icc.Gamut(gamut)) @ np.asarray(basis["camera_to_linear_srgb"], dtype=np.float64)
    ratio = np.asarray(new_gains, dtype=np.float64) / np.asarray(basis["gains_rgb"], dtype=np.float64)
    return sr @ np.diag(ratio) @ np.linalg.inv(sr)


def master_matrix(basis: dict, gamut: str, forward_matrix=None) -> np.ndarray:
    """Camera DN (black-subtracted, /scale) → working-gamut linear.

    LibRaw route: ``S·R·diag(g)`` with R the recipe's camera→linear-sRGB matrix.  DCP route (``forward_matrix`` = the
    profile's PCS-normalised ForwardMatrix, camera-after-WB → XYZ D50): ``inv(colorants_D50)·FM·diag(g)``, i.e. exactly
    the inverse of what the master's ICC profile declares."""
    g = icc.Gamut(gamut)
    gains = np.diag(np.asarray(basis["gains_rgb"], dtype=np.float64))
    if forward_matrix is not None:
        fm = np.asarray(forward_matrix, dtype=np.float64)
        if fm.shape != (3, 3) or not np.isfinite(fm).all():
            raise ValueError("ForwardMatrix 无效")
        return g.matrix_from_xyz_d50() @ fm @ gains
    r = np.asarray(basis["camera_to_linear_srgb"], dtype=np.float64)
    return icc.SRGB.matrix_to(g) @ r @ gains


Z8_CAM_XYZ = np.array([[1.1423, -.4564, -.1123], [-.4816, 1.2895, .2119], [-.021, .1061, .7282]], np.float32)


def require_z8_family(cam_xyz) -> None:
    """The pinned public profile belongs to the audited Z8 matrix family only (same check as experimental_dcp)."""
    if cam_xyz is None or not np.array_equal(np.asarray(cam_xyz, np.float32)[:3], Z8_CAM_XYZ):
        raise ValueError("DCP 路线只支持已审计的 Z8 矩阵家族")


def z8_forward_matrix(basis: dict, profile_path) -> tuple[np.ndarray, str]:
    """The pinned public Z8 profile's normalised ForwardMatrix (experimental_dcp contract), for the audited Z8 matrix family."""
    from . import experimental_dcp as dcp
    require_z8_family(basis["cam_xyz"])
    profile = dcp.load_z8_profile(profile_path)
    return np.asarray(profile["forward_matrix"], dtype=np.float64), profile["sha256"]


def _apply_3x3(block: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Deterministic elementwise 3×3 (no BLAS kernel choice), float64 in and out."""
    r = block[..., 0] * m[0, 0] + block[..., 1] * m[0, 1] + block[..., 2] * m[0, 2]
    g = block[..., 0] * m[1, 0] + block[..., 1] * m[1, 1] + block[..., 2] * m[1, 2]
    b = block[..., 0] * m[2, 0] + block[..., 1] * m[2, 1] + block[..., 2] * m[2, 2]
    return np.stack([r, g, b], axis=-1)


def convert_camera_master(camera_top_down: np.ndarray, basis: dict, gamut: str, *, forward_matrix=None, rows: int = ROWS):
    """Camera RGB (DN/65535, TOP-DOWN, float32) → (float32 working-gamut linear, saturation mask, nodata mask).

    ``forward_matrix``: optional PCS-normalised DCP ForwardMatrix; when given it replaces the LibRaw matrix route."""
    if camera_top_down.ndim != 3 or camera_top_down.shape[2] != 3 or camera_top_down.dtype.kind != "f":
        raise ValueError("需要浮点 HWC 相机 RGB")
    h, w = camera_top_down.shape[:2]
    black = np.asarray(basis["black_DN"], dtype=np.float64)
    white = float(basis["white_DN"])
    scale = float(basis["scale_DN"])
    m = master_matrix(basis, gamut, forward_matrix)
    out = np.empty((h, w, 3), dtype=np.float32)
    saturated = np.zeros((h, w), dtype=bool)
    nodata = np.zeros((h, w), dtype=bool)

    def convert(bounds):
        # rows-sized blocks from a block boundary: identical arithmetic whether one range or several run in parallel
        first, last = bounds
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            for y0 in range(first, last, rows):
                y1 = min(y0 + rows, last)
                block = np.asarray(camera_top_down[y0:y1], dtype=np.float64)
                if not np.isfinite(block).all():
                    raise ValueError("相机母版包含非有限值")
                dn = block * 65535.0
                nd = (block == 0).all(axis=2)
                saturated[y0:y1] = (dn >= SATURATION_FRACTION * white).any(axis=2)
                nodata[y0:y1] = nd
                lin = _apply_3x3((dn - black) / scale, m)
                lin[nd] = 0.0
                out[y0:y1] = lin.astype(np.float32)

    from .performance import render_workers, run_row_ranges
    run_row_ranges(convert, h, rows, render_workers(h, w, rows=rows), name="aester-master")
    if not np.isfinite(out).all():
        raise ValueError("母版转换产生非有限值")
    return out, saturated, nodata


def read_camera_master(path) -> np.ndarray:
    """Job FITS master → TOP-DOWN float32 camera RGB in DN/65535 units."""
    from .imaging import read_rgb
    return np.flipud(read_rgb(path)).copy()


def export_master_array(camera_top_down: np.ndarray, basis: dict, output_dir, *, source_path, source_sha256: str,
                        gamut: str = "rec2020", dcp_profile=None, exist_ok: bool = False,
                        stack_note: str = "Siril addscale per-channel normalisation; post-stack matrix approximation") -> dict:
    """Write M from an in-memory TOP-DOWN camera RGB array (DN/65535) into ``output_dir``.  Returns the record."""
    output_dir = Path(output_dir)
    icc.Gamut(gamut)
    forward_matrix, dcp_profile_sha256 = (None, None) if dcp_profile is None else z8_forward_matrix(basis, dcp_profile)
    output_dir.mkdir(parents=True, exist_ok=exist_ok)
    for name in (FILENAME, MASK_FILENAME, RECIPE_FILENAME):
        if (output_dir / name).exists():
            raise FileExistsError(f"{output_dir / name} 已存在")
    linear, saturated, nodata = convert_camera_master(camera_top_down, basis, gamut, forward_matrix=forward_matrix)
    profile = icc.write_linear_profile(gamut)
    mask_path = output_dir / MASK_FILENAME
    save_image_with_digest(Image.fromarray((saturated * 255).astype(np.uint8)), mask_path,
                           source="master_export.mask_png", optimize=True)
    mask_sha = sha256_file(mask_path, source="master_export.mask_png")
    metadata = {
        "schema": SCHEMA, "product": "scene_linear_master", "version": VERSION, "starstack": __version__,
        "scene_linear": True, "display_encoding_applied": False, "roworder": "TOP-DOWN", "row_order": "TOP-DOWN",
        "color_space": f"linear_{gamut}_" + ("D50" if icc.GAMUTS[gamut]["white"] == icc.D50_XY else "D65"),
        "gamut": gamut, "white_point": "D50" if icc.GAMUTS[gamut]["white"] == icc.D50_XY else "D65",
        "units": "reference_frame_sensor_white_after_WB_on_G_equals_1", "clipping_applied": False, "negative_preserved": True,
        "icc": {"description": icc.GAMUTS[gamut]["description"], "sha256": hashlib.sha256(profile).hexdigest(), "trc": "linear"},
        "nodata": {"value": 0, "count": int(nodata.sum())},
        "camera_saturation": {"fraction_of_white": SATURATION_FRACTION, "count": int(saturated.sum()),
                              "mask_file": MASK_FILENAME, "mask_sha256": mask_sha},
        "colour_basis": {"source_recipe_sha256": basis["recipe_sha256"], "wb_mode": basis["wb_mode"], "gains_rgb": basis["gains_rgb"],
                         "daylight_gains_rgb": basis["daylight_gains_rgb"], "as_shot_gains_rgb": basis["as_shot_gains_rgb"],
                         "black_DN": basis["black_DN"], "white_DN": basis["white_DN"], "scale_DN": basis["scale_DN"],
                         "camera_to_linear_srgb": basis["camera_to_linear_srgb"], "cam_xyz": basis["cam_xyz"],
                         "route": "dcp_forward_matrix" if forward_matrix is not None else "libraw_matrix",
                         "dcp_profile_sha256": dcp_profile_sha256,
                         "forward_matrix_pcs_normalised": None if forward_matrix is None else forward_matrix.tolist(),
                         "master_matrix": master_matrix(basis, gamut, forward_matrix).tolist(), "reference": basis["source"]},
        "source": {"path": str(source_path), "sha256": source_sha256, "units": "camera_RGB_DN_div_65535"},
        "stack_note": stack_note,
        "statistics": {"min": float(linear.min()), "max": float(linear.max()),
                       "negative_channels": int((linear < 0).sum()), "above_one_channels": int((linear > 1).sum())},
    }
    path = output_dir / FILENAME
    tifffile.imwrite(path, linear, photometric="rgb", iccprofile=profile, metadata=metadata)
    del linear
    record = {"version": VERSION, "state": "succeeded", "master": str(path),
              "master_sha256": sha256_file(path, source="master_export.master_tiff", fresh=True),
              "mask": str(mask_path), "mask_sha256": mask_sha, "metadata": metadata, "basis": _json_ready(basis),
              "gamut": gamut, "environment": {"python": platform.python_version(), "numpy": np.__version__,
                                              "tifffile": tifffile.__version__, "platform": platform.platform()}}
    (output_dir / RECIPE_FILENAME).write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return record


def export_master(source, basis: dict, output_dir, *, gamut: str = "rec2020", dcp_profile=None,
                  stack_note: str = "Siril addscale per-channel normalisation; post-stack matrix approximation") -> dict:
    """Write M for ``source`` (a job FITS master) into ``output_dir`` (created, must not exist).  Returns the record."""
    source = Path(source).resolve(strict=True)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"{output_dir} 已存在")
    source_sha = sha256_file(source, source="master_export.source")
    camera = read_camera_master(source)
    return export_master_array(camera, basis, output_dir, source_path=source, source_sha256=source_sha, gamut=gamut,
                               dcp_profile=dcp_profile, stack_note=stack_note)


def read_master(path):
    """Open an M TIFF: returns (memmap or array HxWx3 float32, metadata dict, icc bytes).  Validates the contract."""
    path = Path(path).resolve(strict=True)
    with tifffile.TiffFile(path) as tf:
        page = tf.pages[0]
        if page.dtype != np.float32 or len(page.shape) != 3 or page.shape[2] != 3:
            raise ValueError("需要 float32 RGB 母版")
        if int(page.photometric) != 2 or int(page.planarconfig) != 1:
            raise ValueError("需要交织 RGB 母版")
        profile = page.tags[34675].value if 34675 in page.tags else None
        if not isinstance(profile, bytes):
            raise ValueError("母版缺少 ICC 配置")
        try:
            metadata = json.loads(page.description)
        except (TypeError, ValueError):
            raise ValueError("母版缺少元数据") from None
        if not isinstance(metadata, dict) or metadata.get("schema") != SCHEMA or metadata.get("scene_linear") is not True \
                or metadata.get("display_encoding_applied") is not False or metadata.get("roworder") != "TOP-DOWN":
            raise ValueError("不是本产品的场景参照母版")
        gamut = metadata.get("gamut")
        if icc.identify_linear_profile(profile) != gamut:
            raise ValueError("母版 ICC 与记录的色域不一致")
        if hashlib.sha256(profile).hexdigest() != metadata.get("icc", {}).get("sha256"):
            raise ValueError("母版 ICC 与记录的哈希不一致")
        # The mask is a portable sibling asset, never a capability to read a
        # different directory supplied through embedded TIFF metadata.
        saturation = metadata.get("camera_saturation", {})
        if saturation.get("storage") == "tiff-page-1":
            if len(tf.pages) != 2 or saturation.get("page") != 1 or "mask_file" in saturation:
                raise ValueError("母版内嵌蒙版页声明无效")
            mask = tf.pages[1]
            if mask.shape != page.shape[:2] or mask.dtype != np.uint8 or int(mask.photometric) != 1:
                raise ValueError("母版内嵌蒙版尺寸／类型不符")
        else:
            if saturation.get("storage") is not None or len(tf.pages) != 1:
                raise ValueError("母版需为单页 TIFF 或明确声明的内嵌蒙版 TIFF")
            mask_name = saturation.get("mask_file")
            if not isinstance(mask_name, str) or not mask_name or mask_name in (".", "..") \
                    or any(c in mask_name for c in '/\\:') or Path(mask_name).name != mask_name:
                raise ValueError("母版饱和蒙版必须是同目录文件名")
            mask_path = path.parent / mask_name
            if mask_path.is_symlink() or getattr(mask_path, 'is_junction', lambda: False)() \
                    or mask_path.resolve().parent != path.parent:
                raise ValueError("母版饱和蒙版不能指向其他目录")
    try:
        data = tifffile.memmap(path, page=0, mode="r")
    except (ValueError, OSError):
        data = tifffile.imread(path, key=0)
    return data, metadata, profile


def read_saturation_mask(path, metadata, shape):
    """Read only the explicitly declared, integrity-checked saturation asset."""
    path = Path(path).resolve(strict=True)
    item = metadata["camera_saturation"]
    if item.get("storage") == "tiff-page-1":
        mask = tifffile.imread(path, key=1)
        if mask.dtype != np.uint8 or mask.shape != tuple(shape) or np.any((mask != 0) & (mask != 255)):
            raise ValueError("母版内嵌饱和蒙版无效")
        if hashing.sha256_bytes(mask.tobytes(), source="master_export.mask_pixels") != item.get("pixels_sha256"):
            raise ValueError("母版内嵌饱和蒙版哈希不符")
        saturated = mask != 0
        if int(saturated.sum()) != item.get("count"):
            raise ValueError("母版内嵌饱和蒙版计数不符")
        return saturated
    # read_master validates the sibling-only name and redirection boundary.
    read_master(path)
    mask_path = path.parent / item["mask_file"]
    if not mask_path.is_file():
        if item["count"]:
            raise ValueError("母版记录了相机饱和像素但蒙版文件缺失")
        return None
    if sha256_file(mask_path, source="master_export.mask_png") != item["mask_sha256"]:
        raise ValueError("相机饱和蒙版与母版记录的哈希不一致")
    from .photo_render import load_mask
    return load_mask(mask_path, shape)


def package_portable_master(source, destination, *, display_transform=None, check=lambda: None):
    """Self-contained TIFF, with identical primary float32 bits and ICC.

    Page zero is the ordinary RGB image. Page one is a declared uint8 mask;
    external editors may ignore it, while our reader validates it explicitly.
    Historical paths remain provenance only, never dependencies.
    """
    source, destination = Path(source).resolve(strict=True), Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    before = sha256_file(source, source="master_export.package_source")
    data, metadata, profile = read_master(source)
    mask = read_saturation_mask(source, metadata, data.shape[:2])
    mask = np.zeros(data.shape[:2], np.uint8) if mask is None else mask.astype(np.uint8) * 255
    metadata = json.loads(json.dumps(metadata))
    original = metadata["camera_saturation"]
    metadata["camera_saturation"] = {k: v for k, v in original.items()
                                      if k not in ("mask_file", "mask_sha256", "storage", "page", "pixels_sha256")}
    metadata["camera_saturation"].update(storage="tiff-page-1", page=1,
        count=int(np.count_nonzero(mask)),
        pixels_sha256=hashing.sha256_bytes(mask.tobytes(), source="master_export.mask_pixels"))
    metadata["portable_master"] = {"version": "aester-portable-master-1", "source_sha256": before,
                                    "primary_pixels": "float32-bit-identical", "display_transform": display_transform}
    check()
    with tifffile.TiffWriter(destination, bigtiff=data.nbytes + mask.nbytes > 2**32 - 2**25) as writer:
        writer.write(data, photometric="rgb", iccprofile=profile, metadata=metadata)
        check()
        # Deflate is lossless and has tifffile's stdlib zlib fallback; the
        # ordinary float RGB page remains uncompressed and directly mappable.
        writer.write(mask, photometric="minisblack", compression="deflate",
                     metadata={"schema": "aester-camera-saturation-1"})
    check()
    copied, copied_meta, copied_icc = read_master(destination)
    if copied_icc != profile or copied.shape != data.shape:
        raise ValueError("母版打包改变了色彩描述或尺寸")
    for start in range(0, data.shape[0], ROWS):
        check()
        if not np.array_equal(data[start:start+ROWS].view(np.uint32), copied[start:start+ROWS].view(np.uint32)):
            raise ValueError("母版打包改变了主图浮点位模式")
    read_saturation_mask(destination, copied_meta, copied.shape[:2])
    if sha256_file(source, source="master_export.package_source") != before:
        raise ValueError("母版打包期间源文件改变")
    return {"source_sha256": before, "primary_pixels_bit_identical": True, "icc_identical": True,
            "embedded_mask_sha256": copied_meta["camera_saturation"]["pixels_sha256"]}
