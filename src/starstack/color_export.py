"""Job product writer for the M + D contract (docs/COLOR_CONTRACT_PACKAGE_20260922.md §10, step 2).

Called by ``core`` after a composite is written: when the job's reference frame yields a colour basis (RAW inputs),
the products directory gets the scene-referred master (M) and the finished photo (D, default recipe
``photo-default-1``) instead of the legacy display products.  FITS inputs have no camera basis and keep the legacy
writer.  Any failure here is reported to the caller, which falls back to the legacy products and records the error.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import json

import numpy as np

from . import display_transform as dt, master_export as me, photo_render as pr

VERSION = "color-export-1"
D_TIFF_NAME = "natural_display_16bit.tif"          # legacy file name kept for GUI/MCP consumers; content follows the D contract
D_JPEG_NAME = "natural.jpg"
LEGACY_NAMES = ("master_linear_32bit.tif", "cool.jpg", "warm.jpg")


def sha256_file(path) -> str:
    return me.sha256_file(path, source="color_export.sha256_file")


def job_colour_basis(reference_path, *, wb_mode: str = "daylight", metadata: dict | None = None) -> dict:
    """The colour basis of a job's recorded reference frame (DNG tags or LibRaw metadata)."""
    if reference_path is None:
        raise ValueError("作业没有记录参考帧，无法确定颜色底子")
    return me.colour_basis_for_reference(reference_path, wb_mode=wb_mode, metadata=metadata)


def basis_summary(basis: dict) -> dict:
    """Compact, JSON-ready record of a basis for job diagnostics."""
    return {"source": basis["source"], "recipe_sha256": basis["recipe_sha256"], "wb_mode": basis["wb_mode"],
            "gains_rgb": basis["gains_rgb"], "daylight_gains_rgb": basis["daylight_gains_rgb"],
            "as_shot_gains_rgb": basis["as_shot_gains_rgb"], "black_DN": basis["black_DN"], "white_DN": basis["white_DN"],
            "camera_to_linear_srgb": basis["camera_to_linear_srgb"]}


def export_job_products(output_bottom_up: np.ndarray, export_dir, cancel_check, *, basis: dict, source_fits,
                        sky_mask_path=None, gamut: str = "rec2020", color_adjustments=None) -> tuple[list[dict], dict]:
    """Write M and D for a composite (FITS bottom-up float32, DN/65535) already saved as ``source_fits`` in ``export_dir``.

    Returns (artifact entries, record).  The anchor and noise are measured on the master (anchor within the sky mask
    when the job has one) with the frozen default recipe; nothing else is estimated.
    """
    export_dir = Path(export_dir)
    source_fits = Path(source_fits)
    cancel_check()
    camera = np.flipud(np.asarray(output_bottom_up, dtype=np.float32)).copy()      # TOP-DOWN for the M contract
    master = me.export_master_array(camera, basis, export_dir, source_path=source_fits, source_sha256=sha256_file(source_fits),
                                    gamut=gamut, exist_ok=True)
    del camera
    cancel_check()
    data, meta, _ = me.read_master(master["master"])
    sky = None
    if sky_mask_path is not None:
        sky = pr.load_mask(sky_mask_path, data.shape[:2])
    params = pr.default_params(meta, data, sky=sky)
    from . import color_adjustments as ca
    adjustments = ca.normalize(color_adjustments)
    params = ca.apply(params, meta, adjustments)
    del data
    cancel_check()
    photo = pr.render_photo(master["master"], export_dir, params, sky_mask=sky_mask_path, tiff_name=D_TIFF_NAME,
                            jpeg_name=D_JPEG_NAME, exist_ok=True,
                            recipe_label="photo-user-adjusted-1" if any(adjustments.values()) else None)
    cancel_check()
    artifacts = [
        {"path": master["master"], "kind": "tiff", "label": f"后期母版（场景参照线性 float32，{meta['color_space']}）", "contract": "M"},
        {"path": photo["tiff"], "kind": "tiff", "label": "成片（16 位 sRGB）", "contract": "D"},
        {"path": photo["jpeg"], "kind": "jpeg", "label": "成片 JPEG（与 16 位成片同源）", "contract": "D"},
    ]
    record = {"version": VERSION, "state": "succeeded", "contract": "M+D " + photo["metadata"]["recipe"],
              "color_adjustments": adjustments, "adjustments_version": ca.VERSION,
              "master": {"path": master["master"], "sha256": master["master_sha256"], "mask": master["mask"],
                         "recipe": str(export_dir / me.RECIPE_FILENAME)},
              "photo": {"tiff": photo["tiff"], "tiff_sha256": photo["tiff_sha256"], "jpeg": photo["jpeg"],
                        "jpeg_sha256": photo["jpeg_sha256"], "record": str(export_dir / pr.RECORD_NAME),
                        "recipe": photo["metadata"]["recipe"], "display_transform": photo["metadata"]["display_transform"]},
              "basis": basis_summary(basis), "sky_mask": None if sky_mask_path is None else str(sky_mask_path)}
    return artifacts, record
