"""D · finished photo from an M master (docs/COLOR_CONTRACT_PACKAGE_20260922.md §2.2).

One display transform (``display_transform.render``) produces the 16-bit sRGB TIFF; the JPEG is the same array rounded
to 8 bits.  The TIFF carries the lcms sRGB profile and the metadata that ``finish-photo`` and the other display-domain
commands require.  ``render_candidates`` writes a reduced-size grid of explicit parameter choices for the user to pick
from; nothing here fits anything to the scene except the recorded anchor luminance.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import platform
import struct

import numpy as np
from PIL import Image, ImageCms
import tifffile

from . import __version__, icc_profiles as icc, display_transform as dt, master_export as me
from .whole_color import resize_signed

VERSION = "photo-render-2"
DEFAULT_WB = "as_shot"                 # user decision U2 (2026-09-22): finished photos default to the as-shot neutral
SHADOW_SMOOTH_START_SIGMAS = 10.0      # fixed multiples of the measured noise (package doc §9)
SHADOW_SMOOTH_END_SIGMAS = 40.0
SHADOW_SMOOTH_RADIUS = 3
# The frozen default finishing recipe (user decision U3, 2026-09-22: candidate as-shot k12 s1.00 with shadow smoothing).
# Every value is explicit; a render whose parameters equal these is labelled with the recipe name in its metadata.
DEFAULT_RECIPE = {"name": "photo-default-1", "decided": "2026-09-22 U3 (candidate as-shot k12 s1.00)",
                  "wb_mode": DEFAULT_WB, "asinh_strength": 12.0, "saturation": 1.0, "target": 0.22, "black": 0.0,
                  "highlight_desat_start": 0.8, "highlight_desat_end": 0.95, "look": "none", "shadow_smoothing": True,
                  "shadow_smooth_radius": SHADOW_SMOOTH_RADIUS, "shadow_smooth_start_sigmas": SHADOW_SMOOTH_START_SIGMAS,
                  "shadow_smooth_end_sigmas": SHADOW_SMOOTH_END_SIGMAS, "shadow_chroma_end": 0.0}


def matches_default_recipe(params: dt.Params, master_meta: dict | None = None) -> bool:
    """True when ``params`` realise DEFAULT_RECIPE exactly (the measured anchor/noise/WB matrix are data, not recipe).
    With ``master_meta`` the look must be the one registered for the master's body (``default_look_fields``)."""
    r = DEFAULT_RECIPE
    smoothing_expected = r["shadow_smoothing"] and (params.noise_sigma or 0) > 0
    look = default_look_fields(master_meta) if master_meta is not None else {"look": r["look"], "look_id": None, "look_strength": 1.0}
    return (params.wb_mode == r["wb_mode"] and params.asinh_strength == r["asinh_strength"] and params.saturation == r["saturation"]
            and params.target == r["target"] and params.black == r["black"] and params.highlight_desat_start == r["highlight_desat_start"]
            and params.highlight_desat_end == r["highlight_desat_end"] and params.look == look["look"]
            and params.look_id == look["look_id"] and params.look_strength == look["look_strength"]
            and params.shadow_chroma_end == r["shadow_chroma_end"]
            and (not smoothing_expected or (params.shadow_smooth_radius == r["shadow_smooth_radius"]
                                            and params.shadow_smooth_start == r["shadow_smooth_start_sigmas"] * params.noise_sigma
                                            and params.shadow_smooth_end == r["shadow_smooth_end_sigmas"] * params.noise_sigma))
            and (smoothing_expected or not params.shadow_smoothing))
TIFF_NAME = "photo_srgb16.tif"
JPEG_NAME = "photo.jpg"
FLAGS_NAME = "render_flags.png"
RECORD_NAME = "photo_render.json"
JPEG_QUALITY = 95
CANDIDATE_CONTRAST = (6.0, 12.0, 24.0)
CANDIDATE_SATURATION = (0.85, 1.0)
CANDIDATE_TARGET = 0.22
LOOK_BYTES_PER_PIXEL = 1536      # conservative live temporaries per row pixel when a HueSatMap look runs (vs 768 without)


def srgb_icc() -> bytes:
    profile = bytearray(ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes())
    # lcms stamps newly created profiles with the wall clock. This date belongs
    # to our profile revision, not the photograph, so equivalent renders must
    # not acquire different file identities just by crossing a second boundary.
    # Only the creation date and dependent ICC ID change; all colour tags remain.
    struct.pack_into(">6H", profile, 24, 2026, 9, 24, 0, 0, 0)
    hashed = bytearray(profile)
    hashed[44:48] = b"\0" * 4
    hashed[64:68] = b"\0" * 4
    hashed[84:100] = b"\0" * 16
    profile[84:100] = hashlib.md5(bytes(hashed)).digest()
    return bytes(profile)


def load_mask(path, shape) -> np.ndarray:
    """TOP-DOWN single-channel mask (PNG/TIFF) → bool HxW (>0)."""
    path = Path(path).resolve(strict=True)
    if path.suffix.lower() in (".tif", ".tiff"):
        a = tifffile.imread(path)
    else:
        with Image.open(path) as im:
            a = np.asarray(im.convert("L") if im.mode not in ("L", "I;16", "I", "F", "1") else im)
    if a.ndim != 2 or a.shape != tuple(shape):
        raise ValueError("蒙版必须与母版同尺寸、单通道")
    return np.asarray(a) > 0


def make_dcp_hsm_look(profile_path, gamut: str, *, cam_xyz=None):
    """Z8 HueSatMap look on working-gamut linear pixels (the user's photographic default for the Z8, decision v11).

    Same mathematics as ``experimental_dcp.render_camera``: ProPhoto HSV, negative ProPhoto pixels bypass, S/V not
    clipped.  Returns (callable, record).  Only the pinned public profile is accepted (its loader enforces the hash),
    and only for a master whose colour basis is the audited Z8 matrix family (``cam_xyz``; None skips the check).
    """
    from . import experimental_dcp as dcp
    if cam_xyz is not None:
        me.require_z8_family(cam_xyz)
    profile = dcp.load_z8_profile(profile_path)
    look, counters = hsm_look(profile["huesat"], gamut)
    record = {"look": "dcp_hsm", "profile_sha256": profile["sha256"], "profile_name": profile["name"],
              "space": "ProPhoto HSV via XYZ D50 (experimental_dcp constants)", "counters": counters,
              "range_policy": "negative_ProPhoto_bypasses_HSM; no_S_or_V_clipping"}
    return look, record


def hsm_look(table, gamut: str):
    """(callable, counters) applying a 2-D HueSatMap ``table`` in ProPhoto HSV; shared by ``dcp_hsm`` and the camera looks.

    Arithmetic equals ``experimental_dcp.rgb_to_hsv``/``hsv_to_rgb`` and ``experimental_huesat.lookup_huesat`` bit for bit
    (``huesat_fast``).  Each call is independent of the others and the counters are updated under a lock, so the
    callable is marked ``thread_safe`` and the display transform may run it on parallel row ranges."""
    import threading
    from . import experimental_dcp as dcp, huesat_fast as hf
    g = icc.Gamut(gamut)
    to_srgb = g.matrix_to(icc.SRGB)
    from_srgb = icc.SRGB.matrix_to(g)
    srgb_to_xyz_d50 = np.linalg.inv(dcp.XYZ_TO_SRGB)
    to_pro = dcp.XYZ_TO_PROPHOTO @ srgb_to_xyz_d50 @ to_srgb          # gamut linear → ProPhoto linear (D50)
    back = from_srgb @ dcp.XYZ_TO_SRGB @ dcp.PROPHOTO_TO_XYZ          # ProPhoto → gamut linear
    prepared = hf.prepare(table)
    counters = {"processed": 0, "bypassed_negative": 0, "saturation_above_one": 0}
    lock = threading.Lock()

    def look(block):
        pro = block @ to_pro.T
        good = (pro >= 0).all(axis=-1)
        bypassed = int((~good).sum())
        processed = int(good.size - bypassed)
        above = 0
        all_good = bypassed == 0
        selected = pro.reshape(-1, 3) if all_good else pro[good]      # same values in the same order
        if selected.size:
            h, s, v = hf.rgb_to_hsv(selected)
            mods = hf.lookup(prepared, h, np.clip(s, 0.0, 1.0))
            s = s * mods[:, 1]
            v = v * mods[:, 2]
            h = h + mods[:, 0]
            above = int((s > 1).sum())
            rgb = hf.hsv_to_rgb(h, s, v)
            if all_good:
                pro = rgb.reshape(pro.shape)
            else:
                pro[good] = rgb
        out = pro @ back.T
        if not all_good:
            out[~good] = block[~good]
        with lock:
            counters["bypassed_negative"] += bypassed
            counters["processed"] += processed
            counters["saturation_above_one"] += above
        return out

    look.thread_safe = True
    return look, counters


def display_workers(params: dt.Params, height: int, width: int, *, preview: bool = False) -> int:
    """Row-range workers for rendering ``params`` (the shared policy in ``performance.render_workers``): looks count
    with their larger temporaries; previews and candidates parallelise from ``PREVIEW_MIN_PIXELS``.  The output is the
    same for any worker count."""
    from .performance import render_workers, PREVIEW_MIN_PIXELS
    kwargs = {"bytes_per_pixel": LOOK_BYTES_PER_PIXEL if params.look != "none" else 768}
    if preview:
        kwargs["min_pixels"] = PREVIEW_MIN_PIXELS
    return render_workers(height, width, **kwargs)


def default_look_fields(master_meta: dict) -> dict:
    """The look part of the default recipe for this master: the camera look registered for its body (user decision per
    body, ``devices_registry`` ``colour_look``), otherwise ``DEFAULT_RECIPE["look"]``."""
    from . import camera_looks
    chosen = camera_looks.default_for(master_meta)
    if chosen is None:
        return {"look": DEFAULT_RECIPE["look"], "look_id": None, "look_strength": 1.0}
    return {"look": camera_looks.LOOK_NAME, "look_id": chosen["look_id"], "look_strength": chosen["strength"]}


def wb_choice(master_meta: dict, wb_mode: str):
    """(wb_mode, matrix or None, gains) realising ``wb_mode`` on a master exported with another white balance."""
    basis = master_meta["colour_basis"]
    if wb_mode == "master" or wb_mode == basis["wb_mode"]:
        return "master" if wb_mode == "master" else wb_mode, None, tuple(basis["gains_rgb"])
    if wb_mode not in ("daylight", "as_shot"):
        raise ValueError("wb_mode 只能是 master、daylight 或 as_shot")
    gains = basis["daylight_gains_rgb"] if wb_mode == "daylight" else basis["as_shot_gains_rgb"]
    m = me.switch_white_balance_matrix({"camera_to_linear_srgb": basis["camera_to_linear_srgb"], "gains_rgb": basis["gains_rgb"]},
                                       gains, master_meta["gamut"])
    return wb_mode, tuple(tuple(float(v) for v in row) for row in m), tuple(float(v) for v in gains)


def render_photo(master_path, output_dir, params: dt.Params, *, sky_mask=None, look_profile=None,
                 write_flags: bool = True, label: str | None = None, tiff_name: str = TIFF_NAME, jpeg_name: str = JPEG_NAME,
                 exist_ok: bool = False, recipe_label: str | None = None) -> dict:
    """Render D from an M file into ``output_dir`` (created unless ``exist_ok``; the product files must not exist yet).
    Returns the record (also written).  ``recipe_label`` overrides the recorded recipe name for renders whose white
    balance was deliberately shifted from the recipe (the GUI/MCP preview controls)."""
    master_path = Path(master_path).resolve(strict=True)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=exist_ok)
    for name in (tiff_name, jpeg_name, RECORD_NAME):
        if (output_dir / name).exists():
            raise FileExistsError(f"{output_dir / name} 已存在")
    data, meta, _ = me.read_master(master_path)
    h, w = data.shape[:2]
    if params.gamut != meta["gamut"]:
        raise ValueError("显示参数的色域与母版不一致")
    params = params.validate()
    saturated = me.read_saturation_mask(master_path, meta, (h, w))
    region = None
    sky = None
    if sky_mask is not None:
        sky = load_mask(sky_mask, (h, w))
        region = sky.astype(np.uint8)
    look, look_record = None, {"look": "none"}
    if params.look == "dcp_hsm":
        if look_profile is None:
            raise ValueError("dcp_hsm 观感需要 --dcp-profile")
        look, look_record = make_dcp_hsm_look(look_profile, params.gamut, cam_xyz=meta["colour_basis"].get("cam_xyz"))
    elif params.look == "camera_hsm":
        from . import camera_looks
        look, look_record = camera_looks.make_look(params.look_id, params.look_strength, params.gamut)
    flags = np.zeros((h, w), dtype=np.uint8) if write_flags else None
    from .performance import render_workers
    if look is None:
        workers = render_workers(h, w)
    elif getattr(look, "thread_safe", False) is True:
        workers = render_workers(h, w, bytes_per_pixel=LOOK_BYTES_PER_PIXEL)
    else:
        workers = 1
    from . import resources
    resources.shadow_pool("py.render_rows", actual=workers, items=(h + 127) // 128, width=w)   # P1b-2: shadow only
    rgb16, stats = dt.render(data, params, camera_saturated=saturated, look=look, region=region, flags=flags, workers=workers)
    icc_bytes = srgb_icc()
    tiff_meta = {"schema": "starstack-photo-1", "product": "finished_photo", "version": VERSION, "starstack": __version__,
                 "roworder": "TOP-DOWN", "row_order": "TOP-DOWN", "color_space": "encoded_sRGB", "scene_linear": False,
                 "display_encoding_applied": True, "display_transform": params.to_json(), "look": look_record,
                 "recipe": recipe_label or (DEFAULT_RECIPE["name"] if matches_default_recipe(params, meta) else "custom"),
                 "source_master": {"path": str(master_path),
                                   "sha256": me.sha256_file(master_path, source="photo_render.source_master"),
                                   "gamut": meta["gamut"], "colour_basis": meta["colour_basis"]},
                 "sky_mask": None if sky_mask is None else str(Path(sky_mask).resolve()),
                 "label": label, "statistics": stats}
    tiff_path = output_dir / tiff_name
    tifffile.imwrite(tiff_path, rgb16, photometric="rgb", iccprofile=icc_bytes, metadata=tiff_meta)
    rgb8 = dt.to_8bit(rgb16)
    jpeg_path = output_dir / jpeg_name
    # Encoded in memory and written with its digest (same bytes); the read-back below is then a ledger hit in a run.
    me.save_image_with_digest(Image.fromarray(rgb8), jpeg_path, source="photo_render.jpeg",
                              quality=JPEG_QUALITY, subsampling=0, icc_profile=icc_bytes)
    record = {"version": VERSION, "state": "succeeded", "tiff": str(tiff_path),
              "tiff_sha256": me.sha256_file(tiff_path, source="photo_render.tiff", fresh=True),
              "jpeg": str(jpeg_path), "jpeg_sha256": me.sha256_file(jpeg_path, source="photo_render.jpeg"),
              "jpeg_quality": JPEG_QUALITY,
              "jpeg_subsampling": "4:4:4", "jpeg_source": "round(uint16/257) of the TIFF array",
              "metadata": tiff_meta, "render_workers": workers, "environment": {"python": platform.python_version(), "numpy": np.__version__,
                                                     "tifffile": tifffile.__version__, "Pillow": Image.__version__,
                                                     "platform": platform.platform()}}
    if flags is not None:
        flags_path = output_dir / FLAGS_NAME
        me.save_image_with_digest(Image.fromarray(flags), flags_path, source="photo_render.flags", optimize=True)
        record["flags"] = str(flags_path)
        record["flags_sha256"] = me.sha256_file(flags_path, source="photo_render.flags")
        record["flag_bits"] = {"camera_saturated": dt.FLAG_CAMERA_SATURATED, "highlight_desaturated": dt.FLAG_HIGHLIGHT_DESATURATED,
                               "gamut_negative": dt.FLAG_GAMUT_NEGATIVE, "gamut_over_white": dt.FLAG_GAMUT_OVER_WHITE,
                               "at_or_below_black": dt.FLAG_AT_OR_BELOW_BLACK, "rounded_to_zero": dt.FLAG_ROUNDED_TO_ZERO}
    (output_dir / RECORD_NAME).write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return record


def default_params(master_meta: dict, data: np.ndarray, *, sky=None, wb_mode: str = DEFAULT_WB,
                   shadow_smoothing: bool = DEFAULT_RECIPE["shadow_smoothing"], **overrides) -> dt.Params:
    """Params with the white balance realised as a matrix, the anchor and the noise measured on the (re-balanced)
    master (anchor sky-masked when given), the shadow-chroma thresholds as fixed multiples of the noise; everything
    else explicit defaults, overridable.  ``look`` defaults to ``"auto"``: the look registered for the master's body
    (``default_look_fields``); an explicit look ("none", "dcp_hsm", "camera_hsm" with ``look_id``) replaces it."""
    gamut = master_meta["gamut"]
    mode, matrix, gains = wb_choice(master_meta, wb_mode)
    anchor = dt.measure_anchor(data, gamut, mask=sky, wb_matrix=matrix)
    sigma = dt.measure_noise_sigma(data, gamut, wb_matrix=matrix)
    fields = {"gamut": gamut, "anchor": anchor, "wb_mode": mode, "wb_matrix": matrix, "wb_gains_rgb": gains, "noise_sigma": sigma,
              "asinh_strength": DEFAULT_RECIPE["asinh_strength"], "saturation": DEFAULT_RECIPE["saturation"],
              "target": DEFAULT_RECIPE["target"], "black": DEFAULT_RECIPE["black"],
              "highlight_desat_start": DEFAULT_RECIPE["highlight_desat_start"], "highlight_desat_end": DEFAULT_RECIPE["highlight_desat_end"],
              "shadow_chroma_end": DEFAULT_RECIPE["shadow_chroma_end"]}
    look = overrides.pop("look", "auto")
    fields.update(default_look_fields(master_meta) if look == "auto" else {"look": look, "look_id": None, "look_strength": 1.0})
    if shadow_smoothing and sigma > 0:
        fields.update(shadow_smooth_radius=SHADOW_SMOOTH_RADIUS, shadow_smooth_start=SHADOW_SMOOTH_START_SIGMAS * sigma,
                      shadow_smooth_end=SHADOW_SMOOTH_END_SIGMAS * sigma)
    fields.update(overrides)
    return dt.Params(**fields).validate()


def render_candidates(master_path, output_dir, *, sky_mask=None, look_profile=None, max_side: int = 1600,
                      target: float = CANDIDATE_TARGET, wb_mode: str = DEFAULT_WB, base_overrides: dict | None = None) -> dict:
    """Reduced-size JPEG grid: contrast × saturation at the default white balance, the other white balance, shadow
    smoothing off, the optional chroma fade, and (Z8) the DCP HueSatMap look."""
    master_path = Path(master_path).resolve(strict=True)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    data, meta, _ = me.read_master(master_path)
    h, w = data.shape[:2]
    sky = None if sky_mask is None else load_mask(sky_mask, (h, w))
    base = default_params(meta, data, sky=sky, wb_mode=wb_mode, target=target, **(base_overrides or {}))
    small = resize_signed(np.asarray(data), max_side)
    scale_down = small.shape[0] / h
    sat_small = None
    full = me.read_saturation_mask(master_path, meta, (h, w))
    if full is not None:
        sat_small = np.asarray(Image.fromarray(full.astype(np.uint8) * 255).resize((small.shape[1], small.shape[0]),
                                                                                  Image.Resampling.BOX)) > 0
    icc_bytes = srgb_icc()
    basis = meta["colour_basis"]
    items = []
    # the box smoothing radius is in pixels: scale it with the preview so the candidates show the full-size behaviour
    radius_small = max(1, int(round(base.shadow_smooth_radius * scale_down))) if base.shadow_smoothing else 0

    def emit(name, params, *, look=None, note=""):
        rgb16, stats = dt.render(small, params, camera_saturated=sat_small, look=look,
                                 workers=display_workers(params, small.shape[0], small.shape[1], preview=True))
        path = output_dir / f"{name}.jpg"
        Image.fromarray(dt.to_8bit(rgb16)).save(path, quality=JPEG_QUALITY, subsampling=0, icc_profile=icc_bytes)
        items.append({"name": name, "file": path.name, "params": params.to_json(), "note": note,
                      "statistics": {k: stats[k] for k in ("camera_saturated", "highlight_desaturated", "gamut_negative_mapped",
                                                          "gamut_over_white_mapped", "shadow_chroma_smoothed")}})

    def variant(**kw):
        fields = {k: getattr(base, k) for k in dt.Params.__dataclass_fields__}
        if base.shadow_smoothing:
            fields["shadow_smooth_radius"] = radius_small
        fields.update(kw)
        return dt.Params(**fields).validate()

    for k in CANDIDATE_CONTRAST:
        for sat in CANDIDATE_SATURATION:
            emit(f"candidate_k{k:g}_s{sat:.2f}", variant(asinh_strength=k, saturation=sat),
                 note=f"contrast k={k:g}, saturation {sat:.2f}, {base.wb_mode} WB, shadow chroma smoothing on")
    mid = CANDIDATE_CONTRAST[1]
    # the other white balance
    other_mode = "daylight" if base.wb_mode == "as_shot" else "as_shot"
    o_mode, o_matrix, o_gains = wb_choice(meta, other_mode)
    o_anchor = dt.measure_anchor(data, meta["gamut"], mask=sky, wb_matrix=o_matrix)
    o_sigma = dt.measure_noise_sigma(data, meta["gamut"], wb_matrix=o_matrix)
    for sat in CANDIDATE_SATURATION:
        emit(f"candidate_{o_mode.replace('_', '')}_k{mid:g}_s{sat:.2f}",
             variant(asinh_strength=mid, saturation=sat, wb_mode=o_mode, wb_matrix=o_matrix, wb_gains_rgb=o_gains, anchor=o_anchor,
                     noise_sigma=o_sigma, shadow_smooth_start=SHADOW_SMOOTH_START_SIGMAS * o_sigma if base.shadow_smoothing else 0.0,
                     shadow_smooth_end=SHADOW_SMOOTH_END_SIGMAS * o_sigma if base.shadow_smoothing else 0.0),
             note=f"{o_mode} WB gains {[round(v, 4) for v in o_gains]}, contrast k={mid:g}, saturation {sat:.2f}")
    # shadow chroma comparisons at the middle contrast, saturation 1.0
    emit(f"candidate_k{mid:g}_s1.00_noshadowsmooth", variant(asinh_strength=mid, saturation=1.0, shadow_smooth_start=0.0,
                                                           shadow_smooth_end=0.0, shadow_smooth_radius=0),
         note=f"shadow chroma smoothing OFF (stage-1 behaviour), contrast k={mid:g}, saturation 1.00")
    if base.noise_sigma:
        emit(f"candidate_k{mid:g}_s1.00_chromafade", variant(asinh_strength=mid, saturation=1.0,
                                                           shadow_chroma_end=SHADOW_SMOOTH_END_SIGMAS * base.noise_sigma),
             note=f"optional chroma fade to neutral below {SHADOW_SMOOTH_END_SIGMAS:g}σ (in addition to smoothing), contrast k={mid:g}")
    look_record = None
    if look_profile is not None:
        look, look_record = make_dcp_hsm_look(look_profile, meta["gamut"], cam_xyz=basis.get("cam_xyz"))
        for sat in CANDIDATE_SATURATION:
            emit(f"candidate_dcphsm_k{mid:g}_s{sat:.2f}", variant(asinh_strength=mid, saturation=sat, look="dcp_hsm", look_id=None,
                                                                   look_strength=1.0), look=look,
                 note=f"Z8 DCP HueSatMap look (user decision v11), {base.wb_mode} WB, contrast k={mid:g}, saturation {sat:.2f}")
    anchor = base.anchor
    record = {"version": VERSION, "state": "succeeded", "master": str(master_path),
              "master_sha256": me.sha256_file(master_path, source="photo_render.source_master"),
              "max_side": max_side, "anchor_full_master": anchor, "target": target, "sky_mask": None if sky_mask is None else str(sky_mask),
              "wb_mode": base.wb_mode, "noise_sigma": base.noise_sigma, "look": look_record, "items": items}
    (output_dir / "CANDIDATES.json").write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# 候选影调（缩小图，供用户挑选；未选中的不是默认）", "",
             f"母版：`{master_path}`；白平衡 {base.wb_mode}；锚点亮度 {anchor:.5f} → 编码 {target}；噪声 σ {base.noise_sigma:.3g}；最大边 {max_side} px。", ""]
    for it in items:
        lines.append(f"- `{it['file']}`：{it['note']}")
    (output_dir / "CANDIDATES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return record
