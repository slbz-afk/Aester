"""Colour-contract acceptance on a real job (docs/COLOR_CONTRACT_PACKAGE_20260922.md §4: M1, M4, M5, M6, M7, M8).

Usage (project root, Windows venv):
    python scripts/c1_accept_color_contract.py --job <job dir> --output outputs/fable-color-20260922
        [--reference <RAW/DNG>] [--sky-mask <png>] [--dcp-profile <Z8 dcp>] [--wb daylight|as_shot] [--gamut rec2020]
        [--look auto|none]

The reference frame (any RAW; DNG → tags, other RAW → LibRaw metadata, same routing as the job path) defaults to the
frame the job's colour export used (``diagnostics.color_export.reference_input``); jobs without a colour-export record
fall back to ``diagnostics.reference_input``.  ``--reference-dng`` is kept as an alias of ``--reference``.  The sky mask
defaults to the job's mask.  Everything is written into one new directory; the job and its products are only read.  Criteria and
thresholds are the pre-registered ones; the script never adapts them to what it measures.  Verdicts follow the
criteria as revised with the user's decisions of 2026-09-22 (修订 2: M6 on 8×8 block means; 修订 3: M5 excludes zeros
produced by rounding); the pre-registered pixel-wise / unrevised numbers are still reported.  Stage 1b adds M9 (shadow
chroma, package doc §9): the old v21 TIFF is measured with the same code, never copied by hand.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import datetime as _dt
import json
import platform
import shutil
import sys
import time
import traceback
import uuid

import numpy as np
from PIL import Image
import tifffile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from starstack import __version__, color_metrics as cm, color_products as cp, display_transform as dt, icc_profiles as icc  # noqa: E402
from starstack import master_export as me, photo_render as pr  # noqa: E402
from starstack.imaging import read_rgb  # noqa: E402
from starstack.photographic_job import _display_source  # noqa: E402

VERSION = "c1-accept-color-contract-2"
CRITERIA_REVISION = "修订 2、3 采纳（用户 2026-09-22）；阶段 1b M9（§9）"
OLD_TIFF_DEFAULT = ROOT / "outputs/star-controls-20260921/real/tengger/recommended/star-reduction-f3f262006945/star_reduced_srgb16.tif"
PATCH = {"x0": 2880, "y0": 5000, "size": 256}          # deep-shadow patch of the Tengger frame (package doc §9)
M9_LOCAL_MEAN_RADIUS = 4                                # 9×9 local mean for the chroma-noise residual
CRITERIA_DOC = "docs/COLOR_CONTRACT_PACKAGE_20260922.md"
THRESHOLDS = {"M4_spread_codes": 2, "M4_near_white_luminance": 0.95, "M4_edge_min_fraction": 0.97,
              "M6_pixel_median": 0.5, "M6_pixel_p99": 2.0, "M6_block_median": 0.5, "M6_block_p99": 2.0, "M6_row_stride": 2,
              "M9c_mean_ab_delta": 0.5, "M9e_luminance_codes": 2}


def _json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _job_defaults(job_dir: Path):
    record = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
    if record.get("state") != "succeeded":
        raise ValueError("作业未成功完成")
    diagnostics = record.get("diagnostics") or {}
    mask = record.get("mask")
    if isinstance(mask, dict):            # jobs record the mask as {path, sha256}
        mask = mask.get("path")
    if "color_export" in diagnostics:
        # docs/SONY_ADAPTATION_PREREG_20260925.md C4: the frame the job's colour export actually used; never fall back to
        # the geometric reference, which differs from it for workflow jobs (first selected input).
        reference = (diagnostics.get("color_export") or {}).get("reference_input")
        if not reference:
            raise ValueError("作业有颜色导出记录但没有参考帧；请用 --reference 指定")
    else:
        reference = diagnostics.get("reference_input")
        if not reference:
            raise ValueError("作业没有记录参考帧；请用 --reference 指定")
    return record, reference, mask


def reference_source(record: dict) -> str:
    """Where the default reference comes from: the colour-export record, or the legacy geometric reference."""
    return "color_export" if "color_export" in (record.get("diagnostics") or {}) else "diagnostics_legacy"


def _nodata_count_from_fits(path: Path) -> int:
    data = read_rgb(path)
    return int((data == 0).all(axis=2).sum())


def check_m1(master_rec: dict, master_rec_replay: dict, source_fits: Path) -> dict:
    data, meta, profile = me.read_master(master_rec["master"])
    import io
    from PIL import ImageCms
    lcms = ImageCms.ImageCmsProfile(io.BytesIO(profile)).profile
    finite = bool(np.isfinite(np.asarray(data)).all())
    keys = ["schema", "scene_linear", "display_encoding_applied", "roworder", "color_space", "white_point", "units",
            "clipping_applied", "nodata", "camera_saturation", "colour_basis", "source", "stack_note", "icc"]
    missing = [k for k in keys if k not in meta]
    nodata_source = _nodata_count_from_fits(source_fits)
    stats = meta["statistics"]
    result = {"dtype": str(np.asarray(data).dtype), "shape": list(np.asarray(data).shape), "finite": finite,
              "icc_identified_as": icc.identify_linear_profile(profile), "lcms_description": lcms.profile_description,
              "metadata_missing": missing, "nodata_master": meta["nodata"]["count"], "nodata_source": nodata_source,
              "max": stats["max"], "min": stats["min"], "negative_channels": stats["negative_channels"],
              "above_one_channels": stats["above_one_channels"],
              "replay_sha256_equal": master_rec["master_sha256"] == master_rec_replay["master_sha256"],
              "master_sha256": master_rec["master_sha256"], "replay_sha256": master_rec_replay["master_sha256"]}
    result["pass"] = (finite and not missing and result["icc_identified_as"] == meta["gamut"]
                      and meta["nodata"]["count"] == nodata_source and result["replay_sha256_equal"])
    return result


def _iter_rows(array, rows=256):
    for y0 in range(0, array.shape[0], rows):
        yield y0, np.asarray(array[y0:y0 + rows])


def check_m4_m5(photo_rec: dict, master_rec: dict, sky: np.ndarray | None) -> tuple[dict, dict]:
    tif = tifffile.memmap(photo_rec["tiff"], mode="r")
    flags = np.asarray(Image.open(photo_rec["flags"]))
    saturated = np.asarray(Image.open(master_rec["mask"])) > 0
    h, w = tif.shape[:2]
    t = THRESHOLDS
    m4 = {"a_camera_saturated_pixels": int(saturated.sum()), "a_violations": 0, "b_near_white_pixels": 0, "b_violations": 0,
          "c_gamut_edge_pixels": 0, "c_gamut_edge_sky": 0, "c_gamut_edge_ground": 0, "max_spread_camera_saturated": 0,
          "max_spread_near_white": 0}
    m5 = {"i_gamut_negative_mapped": 0, "i_sky": 0, "i_ground": 0, "ii_single_zero_not_from_gamut": 0,
          "ii_of_which_rounded_to_zero": 0, "ii_not_from_gamut_nor_rounding": 0,
          "iii_all_zero": 0, "iii_nodata_plus_at_or_below_black": 0, "iii_all_zero_not_black_nor_rounding": 0,
          "single_or_double_zero_total": 0,
          "R_zero": 0, "G_zero": 0, "B_zero": 0, "R_65535": 0, "G_65535": 0, "B_65535": 0}
    nodata_master = master_rec["metadata"]["nodata"]["count"]
    for y0, block in _iter_rows(tif):
        b = block.astype(np.int32)
        f = flags[y0:y0 + b.shape[0]]
        s = saturated[y0:y0 + b.shape[0]]
        region = None if sky is None else sky[y0:y0 + b.shape[0]]
        spread = b.max(axis=2) - b.min(axis=2)
        lum = icc.srgb_decode(b / 65535.0) @ icc.SRGB.luma
        near_white = lum >= t["M4_near_white_luminance"]
        m4["a_violations"] += int((spread[s] > t["M4_spread_codes"]).sum())
        if s.any():
            m4["max_spread_camera_saturated"] = max(m4["max_spread_camera_saturated"], int(spread[s].max()))
        m4["b_near_white_pixels"] += int(near_white.sum())
        m4["b_violations"] += int((spread[near_white] > t["M4_spread_codes"]).sum())
        if near_white.any():
            m4["max_spread_near_white"] = max(m4["max_spread_near_white"], int(spread[near_white].max()))
        edge = (b.max(axis=2) == 65535) & (b.min(axis=2) < t["M4_edge_min_fraction"] * 65535)
        m4["c_gamut_edge_pixels"] += int(edge.sum())
        if region is not None:
            m4["c_gamut_edge_sky"] += int((edge & region).sum())
            m4["c_gamut_edge_ground"] += int((edge & ~region).sum())
        neg = (f & dt.FLAG_GAMUT_NEGATIVE) > 0
        m5["i_gamut_negative_mapped"] += int(neg.sum())
        if region is not None:
            m5["i_sky"] += int((neg & region).sum())
            m5["i_ground"] += int((neg & ~region).sum())
        zero_r, zero_g, zero_b = (b[..., c] == 0 for c in range(3))
        single = (zero_r | zero_b) & (b[..., 1] > 0)
        rounded = (f & dt.FLAG_ROUNDED_TO_ZERO) > 0
        m5["ii_single_zero_not_from_gamut"] += int((single & ~neg).sum())
        m5["ii_of_which_rounded_to_zero"] += int((single & ~neg & rounded).sum())
        m5["ii_not_from_gamut_nor_rounding"] += int((single & ~neg & ~rounded).sum())
        all_zero = zero_r & zero_g & zero_b
        below = (f & dt.FLAG_AT_OR_BELOW_BLACK) > 0
        m5["iii_all_zero"] += int(all_zero.sum())
        m5["iii_nodata_plus_at_or_below_black"] += int(below.sum())
        m5["iii_all_zero_not_black_nor_rounding"] += int((all_zero & ~below & ~rounded).sum())
        m5["single_or_double_zero_total"] += int(((b.min(axis=2) == 0) & (b.max(axis=2) > 0)).sum())
        for name, z in (("R_zero", zero_r), ("G_zero", zero_g), ("B_zero", zero_b)):
            m5[name] += int(z.sum())
        for c, name in enumerate(("R_65535", "G_65535", "B_65535")):
            m5[name] += int((b[..., c] == 65535).sum())
    m5["iii_nodata_recorded_in_master"] = nodata_master
    m5["pixels"] = int(h * w)
    m4["pass"] = m4["a_violations"] == 0 and m4["b_violations"] == 0
    m5["iii_difference"] = m5["iii_all_zero"] - m5["iii_nodata_plus_at_or_below_black"]
    m5["pass_preregistered"] = m5["ii_single_zero_not_from_gamut"] == 0 and m5["iii_difference"] <= 0
    m5["pass_revision3"] = m5["ii_not_from_gamut_nor_rounding"] == 0 and m5["iii_all_zero_not_black_nor_rounding"] == 0
    m5["pass"] = m5["pass_revision3"]
    return m4, m5


def check_m6(photo_rec: dict) -> dict:
    tif = tifffile.memmap(photo_rec["tiff"], mode="r")
    with Image.open(photo_rec["jpeg"]) as im:
        jpg = np.asarray(im.convert("RGB"))
    stride = THRESHOLDS["M6_row_stride"]
    pixel = []
    for y0, block in _iter_rows(tif, rows=512):
        t8 = dt.to_8bit(block)[::stride]
        j8 = jpg[y0:y0 + block.shape[0]][::stride]
        pixel.append(cm.ciede2000(cm.lab_from_encoded_srgb8(j8), cm.lab_from_encoded_srgb8(t8)).astype(np.float32).ravel())
    pixel = np.concatenate(pixel)
    t8_full = dt.to_8bit(np.asarray(tif))
    block = cm.ciede2000(cm.lab_from_linear_srgb(cm.block_means_linear(jpg)), cm.lab_from_linear_srgb(cm.block_means_linear(t8_full)))
    t = THRESHOLDS
    result = {"pixel_median": float(np.median(pixel)), "pixel_p99": float(np.quantile(pixel, 0.99)),
              "pixel_p999": float(np.quantile(pixel, 0.999)), "pixel_max": float(pixel.max()), "pixel_samples": int(pixel.size),
              "block8_median": float(np.median(block)), "block8_p99": float(np.quantile(block, 0.99)), "block8_max": float(block.max()),
              "jpeg_quality": photo_rec["jpeg_quality"], "jpeg_subsampling": photo_rec["jpeg_subsampling"]}
    result["pass_preregistered_pixelwise"] = result["pixel_median"] <= t["M6_pixel_median"] and result["pixel_p99"] <= t["M6_pixel_p99"]
    result["pass_revision2_block_means"] = result["block8_median"] <= t["M6_block_median"] and result["block8_p99"] <= t["M6_block_p99"]
    result["pass"] = result["pass_revision2_block_means"]
    return result


def _patch_slices(shape):
    h, w = shape[:2]
    y0, x0, n = PATCH["y0"], PATCH["x0"], PATCH["size"]
    if y0 + n > h or x0 + n > w:                       # smaller images (tests): the bottom-right corner patch
        n = min(n, h, w)
        y0, x0 = h - n, w - n
    return slice(y0, y0 + n), slice(x0, x0 + n)


def chroma_noise_std(rgb8_patch: np.ndarray) -> tuple[float, float, float, float, float]:
    """(L* noise, a* noise, b* noise, mean a*, mean b*): std of the residual after a (2r+1)² local mean, plus means."""
    lab = cm.lab_from_encoded_srgb8(rgb8_patch)
    res = lab - dt.box_mean(lab, M9_LOCAL_MEAN_RADIUS)
    std = res.reshape(-1, 3).std(axis=0)
    return float(std[0]), float(std[1]), float(std[2]), float(lab[..., 1].mean()), float(lab[..., 2].mean())


def check_m9(photo_rec: dict, photo_off_rec: dict, old_tiff, sky: np.ndarray | None) -> dict:
    """Stage 1b (package doc §9): shadow chroma noise vs the old v21 output, colour and luminance preservation."""
    tif = tifffile.memmap(photo_rec["tiff"], mode="r")
    off = tifffile.memmap(photo_off_rec["tiff"], mode="r")
    ys, xs = _patch_slices(tif.shape)
    new8 = dt.to_8bit(np.asarray(tif[ys, xs]))
    off8 = dt.to_8bit(np.asarray(off[ys, xs]))
    nL, na, nb, ma, mb = chroma_noise_std(new8)
    oL, oa, ob, oma, omb = chroma_noise_std(off8)
    result = {"patch": {"y0": ys.start, "x0": xs.start, "size": ys.stop - ys.start},
              "new": {"L_noise": nL, "a_noise": na, "b_noise": nb, "mean_a": ma, "mean_b": mb},
              "smoothing_off": {"L_noise": oL, "a_noise": oa, "b_noise": ob, "mean_a": oma, "mean_b": omb},
              "c_mean_a_delta": abs(ma - oma), "c_mean_b_delta": abs(mb - omb)}
    # M9e: luminance unchanged by the smoothing, whole frame, in 16-bit codes of the encoded luminance
    max_delta = 0
    for y0, block in _iter_rows(tif):
        a = icc.srgb_decode(block.astype(np.float64) / 65535.0) @ icc.SRGB.luma
        b = icc.srgb_decode(np.asarray(off[y0:y0 + block.shape[0]]).astype(np.float64) / 65535.0) @ icc.SRGB.luma
        d = np.abs(np.rint(icc.srgb_encode(a) * 65535.0) - np.rint(icc.srgb_encode(b) * 65535.0))
        max_delta = max(max_delta, int(d.max()))
    result["e_luminance_max_delta_codes"] = max_delta
    # ground B=0 fractions, new (smoothing on / off)
    def ground_zero_fraction(array, channel=2):
        if sky is None:
            return None
        count = total = 0
        for y0, block in _iter_rows(array):
            g = ~sky[y0:y0 + block.shape[0]]
            count += int((block[..., channel][g] == 0).sum()); total += int(g.sum())
        return count / total if total else None
    result["b_ground_B_zero_fraction_new"] = ground_zero_fraction(tif)
    result["b_ground_B_zero_fraction_smoothing_off"] = ground_zero_fraction(off)
    old_path = Path(old_tiff) if old_tiff else None
    if old_path is not None and old_path.is_file():
        old = tifffile.memmap(old_path, mode="r")
        result["old_tiff"] = {"path": str(old_path), "sha256": me.sha256_file(old_path), "shape": list(old.shape)}
        if tuple(old.shape) == tuple(tif.shape) and old.dtype == np.uint16:
            oL2, oa2, ob2, _, _ = chroma_noise_std(dt.to_8bit(np.asarray(old[ys, xs])))
            result["old"] = {"L_noise": oL2, "a_noise": oa2, "b_noise": ob2}
            result["b_ground_B_zero_fraction_old"] = ground_zero_fraction(old)
            result["a_pass"] = na <= oa2 and nb <= ob2
            result["b_pass"] = (result["b_ground_B_zero_fraction_new"] is not None and result["b_ground_B_zero_fraction_old"] is not None
                                and result["b_ground_B_zero_fraction_new"] <= result["b_ground_B_zero_fraction_old"])
        else:
            result["old_tiff"]["error"] = "geometry or dtype differs; M9a/M9b not evaluated"
            result["a_pass"] = result["b_pass"] = False
    else:
        result["old_tiff"] = None
        result["a_pass"] = result["b_pass"] = False
        result["note"] = "old TIFF missing; M9a/M9b not evaluated"
    result["c_pass"] = result["c_mean_a_delta"] <= THRESHOLDS["M9c_mean_ab_delta"] and result["c_mean_b_delta"] <= THRESHOLDS["M9c_mean_ab_delta"]
    result["e_pass"] = max_delta <= THRESHOLDS["M9e_luminance_codes"]
    result["pass"] = bool(result["a_pass"] and result["b_pass"] and result["c_pass"] and result["e_pass"])
    return result


def check_s2(job_dir: Path, source_fits: Path, master_rec: dict, photo_rec: dict) -> dict | None:
    """Step 2 (package doc §10): when the job itself published M + D next to its composite, they must equal the standalone
    export/render byte for byte (M) and pixel for pixel with identical parameters (D), and the legacy products must be gone."""
    from starstack import color_export as ce
    products = source_fits.parent
    built_master = products / me.FILENAME
    built_photo = products / ce.D_TIFF_NAME
    if not built_master.is_file():
        return None
    job = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
    record = ((job.get("diagnostics") or {}).get("color_export") or {})
    result = {"products_dir": str(products), "job_color_export": {k: record.get(k) for k in ("state", "contract", "error", "reference_input")}}
    result["a_master_sha256_built"] = me.sha256_file(built_master)
    result["a_master_sha256_standalone"] = master_rec["master_sha256"]
    result["a_pass"] = result["a_master_sha256_built"] == result["a_master_sha256_standalone"]
    if built_photo.is_file():
        built = tifffile.memmap(built_photo, mode="r")
        fresh = tifffile.memmap(photo_rec["tiff"], mode="r")
        equal = built.shape == fresh.shape
        if equal:
            for y0, block in _iter_rows(fresh):
                if not np.array_equal(block, np.asarray(built[y0:y0 + block.shape[0]])):
                    equal = False
                    break
        with tifffile.TiffFile(built_photo) as tf:
            built_meta = json.loads(tf.pages[0].description)
        result["b_pixels_equal"] = bool(equal)
        result["b_params_equal"] = built_meta.get("display_transform") == photo_rec["metadata"]["display_transform"]
        result["b_recipe"] = built_meta.get("recipe")
        result["b_pass"] = bool(equal and result["b_params_equal"] and built_meta.get("recipe") == pr.DEFAULT_RECIPE["name"])
        built_jpeg = products / ce.D_JPEG_NAME
        result["c_natural_jpg_is_d"] = built_jpeg.is_file() and (products / pr.RECORD_NAME).is_file()
    else:
        result["b_pass"] = False
        result["c_natural_jpg_is_d"] = False
    result["c_legacy_present"] = [n for n in ce.LEGACY_NAMES if (products / n).exists()]
    result["c_pass"] = not result["c_legacy_present"] and bool(result["c_natural_jpg_is_d"])
    result["pass"] = bool(result["a_pass"] and result["b_pass"] and result["c_pass"])
    return result


def legacy_style_statistics(photo_rec: dict, sky: np.ndarray | None) -> dict:
    """The F2-style numbers (per-channel 65535/0 fractions, sampled every 4 rows) for direct comparison with the old output."""
    tif = tifffile.memmap(photo_rec["tiff"], mode="r")
    sample = np.asarray(tif[::4])
    region = None if sky is None else sky[::4]
    out = {"rows_sampled": int(sample.shape[0]), "p50": np.median(sample, axis=(0, 1)).tolist(),
           "p999": np.quantile(sample.reshape(-1, 3), 0.999, axis=0).tolist(), "max": sample.max(axis=(0, 1)).tolist()}
    for c, name in enumerate("RGB"):
        out[f"{name}_65535_fraction"] = float((sample[..., c] == 65535).mean())
        out[f"{name}_zero_fraction"] = float((sample[..., c] == 0).mean())
        if region is not None:
            out[f"{name}_zero_fraction_sky"] = float((sample[..., c][region] == 0).mean()) if region.any() else None
            out[f"{name}_zero_fraction_ground"] = float((sample[..., c][~region] == 0).mean()) if (~region).any() else None
    return out


def run(args) -> int:
    started = time.perf_counter()
    job_dir = Path(args.job).expanduser().resolve(strict=True)
    if args.reference:
        job_record = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
        if job_record.get("state") != "succeeded":      # same gate as _job_defaults, before any output is created
            raise ValueError("作业未成功完成")
        mask_default = job_record.get("mask")
        mask_default = mask_default.get("path") if isinstance(mask_default, dict) else mask_default
        reference_default, source = args.reference, "explicit"
    else:
        job_record, reference_default, mask_default = _job_defaults(job_dir)
        source = reference_source(job_record)
    reference_dng = Path(reference_default).expanduser()
    if not reference_dng.is_file():
        raise ValueError("找不到参考帧；请用 --reference 指定")
    sky_mask = args.sky_mask or mask_default
    out_root = Path(args.output).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = out_root / f"accept-color-{stamp}-{uuid.uuid4().hex[:6]}"
    out.mkdir()
    record = {"version": VERSION, "criteria": CRITERIA_DOC, "criteria_revision": CRITERIA_REVISION, "thresholds": THRESHOLDS,
              "state": "running", "started_utc": stamp, "photo_wb": args.wb_photo, "old_tiff": str(args.old_tiff) if args.old_tiff else None,
              "job": str(job_dir), "job_state": job_record.get("state"), "reference_dng": str(reference_dng),
              "reference": str(reference_dng), "reference_source": source, "sky_mask": sky_mask,
              "wb_mode": args.wb, "gamut": args.gamut, "dcp_profile": args.dcp_profile, "look": args.look, "starstack": __version__,
              "environment": {"python": platform.python_version(), "numpy": np.__version__, "tifffile": tifffile.__version__,
                              "Pillow": Image.__version__, "platform": platform.platform()}, "steps": {}}
    _json(out / "COLOR_ACCEPTANCE.json", record)
    try:
        source_fits = cp.resolve_job_master(job_dir)
        record["source_master"] = {"path": str(source_fits), "sha256": me.sha256_file(source_fits)}
        t0 = time.perf_counter()
        master_rec = cp.export_master(source_fits, out, reference_dng=reference_dng, wb_mode=args.wb, gamut=args.gamut)
        record["steps"]["export_master"] = {"directory": master_rec["directory"], "seconds": time.perf_counter() - t0,
                                            "sha256": master_rec["master_sha256"], "basis": master_rec["metadata"]["colour_basis"]}
        t0 = time.perf_counter()
        replay_rec = cp.export_master(source_fits, out / "replay", reference_dng=reference_dng, wb_mode=args.wb, gamut=args.gamut)
        record["steps"]["export_master_replay"] = {"seconds": time.perf_counter() - t0, "sha256": replay_rec["master_sha256"]}
        m1 = check_m1(master_rec, replay_rec, source_fits)
        shutil.rmtree(out / "replay")           # the replay copy exists only for the hash comparison (548 MB)
        record["steps"]["export_master_replay"]["copy_removed_after_hash"] = True
        sky = None
        if sky_mask:
            sky = pr.load_mask(sky_mask, tuple(m1["shape"][:2]))
        t0 = time.perf_counter()
        photo_rec = cp.render_photo(master_rec["directory"], out, sky_mask=sky_mask, dcp_profile=args.dcp_profile,
                                    candidates=True, look=args.look, wb_mode=args.wb_photo)
        record["steps"]["render_photo"] = {"directory": photo_rec["directory"], "seconds": time.perf_counter() - t0,
                                           "params": photo_rec["metadata"]["display_transform"],
                                           "statistics": photo_rec["metadata"]["statistics"], "tiff_sha256": photo_rec["tiff_sha256"],
                                           "jpeg_sha256": photo_rec["jpeg_sha256"],
                                           "candidates": [i["name"] for i in photo_rec.get("candidates", {}).get("items", [])]}
        t0 = time.perf_counter()
        photo_off_rec = cp.render_photo(master_rec["directory"], out, sky_mask=sky_mask, candidates=False, look=args.look,
                                        wb_mode=args.wb_photo, shadow_smoothing=False, write_flags=False)
        record["steps"]["render_photo_smoothing_off"] = {"directory": photo_off_rec["directory"], "seconds": time.perf_counter() - t0,
                                                         "statistics": photo_off_rec["metadata"]["statistics"], "purpose": "M9c/M9e comparison"}
        if args.dcp_profile:
            t0 = time.perf_counter()
            hsm_rec = cp.render_photo(master_rec["directory"], out, sky_mask=sky_mask, dcp_profile=args.dcp_profile,
                                      look="dcp_hsm", candidates=False, wb_mode=args.wb_photo)
            record["steps"]["render_photo_dcp_hsm"] = {"directory": hsm_rec["directory"], "seconds": time.perf_counter() - t0,
                                                       "look": hsm_rec["metadata"]["look"], "statistics": hsm_rec["metadata"]["statistics"],
                                                       "report_only": True}
        t0 = time.perf_counter()
        m4, m5 = check_m4_m5(photo_rec, master_rec, sky)
        m6 = check_m6(photo_rec)
        m9 = check_m9(photo_rec, photo_off_rec, args.old_tiff, sky)
        s2 = check_s2(job_dir, source_fits, master_rec, photo_rec)
        try:
            _display_source(photo_rec["tiff"])
            m7 = {"finish_photo_display_source_accepts": True, "pass": True}
        except Exception as exc:  # noqa: BLE001
            m7 = {"finish_photo_display_source_accepts": False, "error": str(exc), "pass": False}
        record["steps"]["checks"] = {"seconds": time.perf_counter() - t0}
        record["legacy_style_statistics"] = legacy_style_statistics(photo_rec, sky)
        record["criteria_results"] = {"M1": m1, "M4": m4, "M5": m5, "M6": m6, "M7": m7, "M9": m9}
        if s2 is not None:
            record["criteria_results"]["S2"] = s2
        record["seconds"] = time.perf_counter() - started
        record["criteria_results"]["M8"] = {"exit_code": 0, "seconds_total": record["seconds"],
                                            "pass": all(record["criteria_results"][k]["pass"] for k in ("M1", "M4", "M5", "M6", "M7", "M9"))}
        record["state"] = "succeeded"
        record["verdict"] = {k: ("PASS" if v["pass"] else "FAIL") for k, v in record["criteria_results"].items()}
        record["verdict"]["M5_preregistered_unrevised"] = "PASS" if m5["pass_preregistered"] else "FAIL"
        record["verdict"]["M6_preregistered_pixelwise"] = "PASS" if m6["pass_preregistered_pixelwise"] else "FAIL"
    except BaseException:
        record["state"] = "failed"
        record["error"] = traceback.format_exc()
        record["seconds"] = time.perf_counter() - started
        _json(out / "COLOR_ACCEPTANCE.json", record)
        raise
    _json(out / "COLOR_ACCEPTANCE.json", record)
    (out / "COLOR_ACCEPTANCE.md").write_text(_markdown(record), encoding="utf-8")
    print(json.dumps({"state": record["state"], "directory": str(out), "verdict": record["verdict"], "seconds": record["seconds"]},
                     ensure_ascii=False, indent=2))
    return 0


def _frac(value) -> str:
    return "n/a" if value is None else f"{value:.3g}"


def _markdown(r: dict) -> str:
    c = r["criteria_results"]
    lines = [f"# 色彩合同真实验收（{r['started_utc']}）", "",
             f"判据：`{r['criteria']}`（预登记；结论按 {r['criteria_revision']}，未修订的数字照报）。作业 `{r['job']}`，参考帧 `{r['reference_dng']}`，"
             f"母版白平衡 `{r['wb_mode']}`、成片白平衡 `{r['photo_wb']}`，色域 `{r['gamut']}`。",
             f"总耗时 {r['seconds']:.1f} s；导出 M {r['steps']['export_master']['seconds']:.1f} s，渲染 D {r['steps']['render_photo']['seconds']:.1f} s。", "",
             "| 判据 | 结果 | 数字 |", "|---|---|---|"]
    m1, m4, m5, m6, m7, m8, m9 = (c[k] for k in ("M1", "M4", "M5", "M6", "M7", "M8", "M9"))
    lines.append(f"| M1 母版合同 | {r['verdict']['M1']} | dtype {m1['dtype']}，ICC={m1['icc_identified_as']}，元数据缺失 {m1['metadata_missing']}，有限 {m1['finite']}，"
                 f"无数据 {m1['nodata_master']}/{m1['nodata_source']}，min {m1['min']:.4g} max {m1['max']:.4g}，负通道 {m1['negative_channels']}，重导出哈希一致 {m1['replay_sha256_equal']} |")
    lines.append(f"| M4 高光去饱和 | {r['verdict']['M4']} | (a) 相机饱和 {m4['a_camera_saturated_pixels']} 像素，违例 {m4['a_violations']}（最大通道差 {m4['max_spread_camera_saturated']}）；"
                 f"(b) 近白 {m4['b_near_white_pixels']} 像素，违例 {m4['b_violations']}（最大通道差 {m4['max_spread_near_white']}）；(c) 色域边界 {m4['c_gamut_edge_pixels']}（天空 {m4['c_gamut_edge_sky']} / 地景 {m4['c_gamut_edge_ground']}，只报告） |")
    lines.append(f"| M5 暗部零通道（修订 3） | {r['verdict']['M5']} | (i) 色域负通道收零 {m5['i_gamut_negative_mapped']}（天空 {m5['i_sky']} / 地景 {m5['i_ground']}，只报告）；"
                 f"(ii) 非色域来源单通道零 {m5['ii_single_zero_not_from_gamut']}，其中舍入 {m5['ii_of_which_rounded_to_zero']}、既非色域也非舍入 {m5['ii_not_from_gamut_nor_rounding']}；"
                 f"(iii) 全零 {m5['iii_all_zero']}，其中非无数据/非黑位/非舍入 {m5['iii_all_zero_not_black_nor_rounding']}（未修订口径差 {m5['iii_difference']}，未修订结论 {r['verdict']['M5_preregistered_unrevised']}）；"
                 f"R=0 {m5['R_zero']} G=0 {m5['G_zero']} B=0 {m5['B_zero']}；R=65535 {m5['R_65535']} G=65535 {m5['G_65535']} B=65535 {m5['B_65535']} |")
    lines.append(f"| M6 JPEG=TIFF（修订 2：8×8 块均值） | {r['verdict']['M6']} | 块均值 ΔE00 中位 {m6['block8_median']:.3f}，p99 {m6['block8_p99']:.3f}，max {m6['block8_max']:.3f}；"
                 f"逐像素（未修订口径，结论 {r['verdict']['M6_preregistered_pixelwise']}）中位 {m6['pixel_median']:.3f}，p99 {m6['pixel_p99']:.3f}，p99.9 {m6['pixel_p999']:.3f}，max {m6['pixel_max']:.2f} |")
    lines.append(f"| M7 下游合同 | {r['verdict']['M7']} | `_display_source` 接受 = {m7['finish_photo_display_source_accepts']} |")
    old = m9.get("old")
    old_txt = (f"旧 v21 同法 a*/b* {old['a_noise']:.3f}/{old['b_noise']:.3f}（L* {old['L_noise']:.3f}）" if old else f"旧 TIFF 未评估（{m9.get('note') or m9.get('old_tiff', {}).get('error')}）")
    lines.append(f"| M9 阴影色度（阶段 1b） | {r['verdict']['M9']} | 暗斑 ({m9['patch']['x0']},{m9['patch']['y0']}) {m9['patch']['size']}²：新 a*/b* 噪声 {m9['new']['a_noise']:.3f}/{m9['new']['b_noise']:.3f}（L* {m9['new']['L_noise']:.3f}），"
                 f"平滑关 {m9['smoothing_off']['a_noise']:.3f}/{m9['smoothing_off']['b_noise']:.3f}，{old_txt} → M9a {'PASS' if m9['a_pass'] else 'FAIL'}；"
                 f"地景 B=0 新 {_frac(m9['b_ground_B_zero_fraction_new'])}、平滑关 {_frac(m9['b_ground_B_zero_fraction_smoothing_off'])}、旧 {_frac(m9.get('b_ground_B_zero_fraction_old'))} → M9b {'PASS' if m9['b_pass'] else 'FAIL'}；"
                 f"M9c Δmean a*/b* {m9['c_mean_a_delta']:.3f}/{m9['c_mean_b_delta']:.3f} → {'PASS' if m9['c_pass'] else 'FAIL'}；"
                 f"M9e 亮度最大差 {m9['e_luminance_max_delta_codes']} 码 → {'PASS' if m9['e_pass'] else 'FAIL'}；M9d = M4 重跑 {r['verdict']['M4']} |")
    lines.append(f"| M8 真实运行 | {r['verdict']['M8']} | 退出码 0，总耗时 {m8['seconds_total']:.1f} s，候选 {len(r['steps']['render_photo']['candidates'])} 张 |")
    s2 = c.get("S2")
    if s2 is not None:
        lines.append(f"| S2 作业内建 M+D（第二步） | {r['verdict']['S2']} | 作业 color_export {s2['job_color_export']}；S2a 内建 M sha {s2['a_master_sha256_built'][:12]} vs 独立 {s2['a_master_sha256_standalone'][:12]} → {'PASS' if s2['a_pass'] else 'FAIL'}；"
                     f"S2b 内建 D 像素相等 {s2.get('b_pixels_equal')}、参数相等 {s2.get('b_params_equal')}、recipe {s2.get('b_recipe')} → {'PASS' if s2['b_pass'] else 'FAIL'}；"
                     f"S2c 旧产物残留 {s2['c_legacy_present']}、natural.jpg 为 D {s2['c_natural_jpg_is_d']} → {'PASS' if s2['c_pass'] else 'FAIL'} |")
    else:
        lines.append("| S2 作业内建 M+D（第二步） | 不适用 | 该作业 products 里没有内建 M（旧作业或 FITS 作业） |")
    ls = r["legacy_style_statistics"]
    lines += ["", "## 与 F2 同口径的统计（每 4 行抽样）", "",
              f"p50 R/G/B = {ls['p50']}；p99.9 = {[round(v) for v in ls['p999']]}；max = {ls['max']}。",
              f"R=65535 比例 {ls['R_65535_fraction']:.3g}、G {ls['G_65535_fraction']:.3g}、B {ls['B_65535_fraction']:.3g}；"
              f"R=0 比例 {ls['R_zero_fraction']:.3g}、G {ls['G_zero_fraction']:.3g}、B {ls['B_zero_fraction']:.3g}。"]
    if "R_zero_fraction_sky" in ls:
        lines.append(f"天空 R=0 {_frac(ls['R_zero_fraction_sky'])}、B=0 {_frac(ls['B_zero_fraction_sky'])}；地景 R=0 {_frac(ls['R_zero_fraction_ground'])}、B=0 {_frac(ls['B_zero_fraction_ground'])}"
                     f"（F2 旧值：天空 R=0 1.8e−4；地景 R=0 1.42%、B=0 0.70%）。")
    st = r["steps"]["render_photo"]["statistics"]
    lines += ["", "## 渲染统计", "", f"参数：`{json.dumps(r['steps']['render_photo']['params'], ensure_ascii=False)}`",
              f"相机饱和 {st['camera_saturated']}，高光去饱和 {st['highlight_desaturated']}，亮度≤黑位 {st['luminance_at_or_below_black']}，"
              f"色域负通道 {st['gamut_negative_mapped']}，色域超白 {st['gamut_over_white_mapped']}，亮度超白 {st['luminance_over_white']}，"
              f"阴影色度平滑 {st.get('shadow_chroma_smoothed')} 像素。"]
    if "render_photo_dcp_hsm" in r["steps"]:
        hs = r["steps"]["render_photo_dcp_hsm"]
        lines.append(f"DCP HueSatMap 对照（只报告）：目录 `{hs['directory']}`，处理 {hs['look']['counters']['processed']}，绕过负值 {hs['look']['counters']['bypassed_negative']}，"
                     f"饱和度>1 {hs['look']['counters']['saturation_above_one']}。")
    lines += ["", "候选集：`" + r["steps"]["render_photo"]["directory"] + "/candidates/CANDIDATES.md`（审美决定归用户）。", ""]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--job", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--reference", "--reference-dng", dest="reference", help="参考帧（任意 RAW；DNG 走标签，其他走 LibRaw）")
    p.add_argument("--sky-mask")
    p.add_argument("--dcp-profile")
    p.add_argument("--wb", choices=("daylight", "as_shot"), default="daylight", help="母版白平衡（M 记录）")
    p.add_argument("--wb-photo", choices=("as_shot", "daylight", "master"), default=pr.DEFAULT_WB, help="成片白平衡（U2 默认 as_shot）")
    p.add_argument("--old-tiff", default=str(OLD_TIFF_DEFAULT), help="M9 对照用的旧 v21 16 位 TIFF；同法现算")
    p.add_argument("--gamut", choices=("rec2020", "prophoto", "srgb"), default="rec2020")
    p.add_argument("--look", choices=("auto", "none"), default="auto",
                   help="成片观感：auto = 该机型登记的默认观感，与作业内建 D 一致（docs/SONY_A7RV_LOOK_PREREG_20260926.md）；"
                        "none = 无观感。平滑开/关对照（M9）用同一观感")
    return p


def main(argv=None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
