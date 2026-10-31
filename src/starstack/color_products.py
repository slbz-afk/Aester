"""Command-level entry points for the M (master) and D (photo) products: unique new directories, explicit inputs,
JSON records.  Thin orchestration over ``master_export`` and ``photo_render``; nothing here changes a job."""
from __future__ import annotations

from pathlib import Path
import json
import time
import uuid

import numpy as np

from . import display_transform as dt, master_export as me, photo_render as pr

MASTER_SOURCES = ("master_linear.fit", "sky_master.fit")


def _new_dir(root, prefix: str) -> Path:
    root = Path(root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{prefix}-{uuid.uuid4().hex[:12]}"


def resolve_job_master(source) -> Path:
    """``source`` may be a FITS master, a job directory, or a job's ``products/<uuid>`` directory."""
    path = Path(source).expanduser().resolve(strict=True)
    if path.is_file():
        return path
    if (path / "master_linear.fit").is_file():
        return path / "master_linear.fit"
    job = path / "job.json"
    if job.is_file():
        record = json.loads(job.read_text(encoding="utf-8"))
        if record.get("state") != "succeeded":
            raise ValueError("只接受已成功完成的作业")
        masters = [Path(a["path"]) for a in record.get("artifacts", [])
                   if Path(a["path"]).name == "master_linear.fit" and a.get("group") != "transients"]
        if len(masters) == 1 and masters[0].is_file():
            return masters[0]
        raise ValueError("作业没有唯一的已发布合成母版")
    raise ValueError("需要 FITS 母版、作业目录或 products/<uuid> 目录")


def export_master(source, output_root, *, reference_dng=None, recipe=None, wb_mode: str = "daylight",
                  gamut: str = "rec2020", dcp_profile=None) -> dict:
    """CLI ``export-master``: M into ``<output_root>/master-<uuid>``."""
    started = time.perf_counter()
    if (reference_dng is None) == (recipe is None):
        raise ValueError("需要且只能给一个颜色底子来源：--reference-dng 或 --recipe")
    if reference_dng is not None:
        # Same routing as the job path (color_export.job_colour_basis): DNG → tags, any other RAW → LibRaw metadata.
        basis = me.colour_basis_for_reference(reference_dng, wb_mode=wb_mode)
    else:
        recipe_path = Path(recipe).resolve(strict=True)
        basis = me.colour_basis_from_recipe(json.loads(recipe_path.read_text(encoding="utf-8")), wb_mode=wb_mode,
                                            source_path=recipe_path)
    fits = resolve_job_master(source)
    out = _new_dir(output_root, "master")
    record = me.export_master(fits, basis, out, gamut=gamut, dcp_profile=dcp_profile)
    record["directory"] = str(out)
    record["seconds"] = time.perf_counter() - started
    (out / me.RECIPE_FILENAME).write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return record


_R = pr.DEFAULT_RECIPE


def render_photo(master, output_root, *, sky_mask=None, look: str = "auto", dcp_profile=None, contrast: float = _R["asinh_strength"],
                 saturation: float = _R["saturation"], target: float = _R["target"], black: float = _R["black"], anchor=None,
                 highlight_desat_start: float = _R["highlight_desat_start"], highlight_desat_end: float = _R["highlight_desat_end"],
                 wb_mode: str = _R["wb_mode"], shadow_smoothing: bool = _R["shadow_smoothing"],
                 shadow_chroma_end: float = _R["shadow_chroma_end"], candidates: bool = False, write_flags: bool = True) -> dict:
    """CLI ``render-photo``: D into ``<output_root>/photo-<uuid>`` (plus ``candidates/`` when requested).  ``look="auto"``
    is the look registered for the master's body (same as the job's D); "none" or "dcp_hsm" replace it."""
    started = time.perf_counter()
    master_path = Path(master).expanduser().resolve(strict=True)
    if master_path.is_dir():
        master_path = master_path / me.FILENAME
    data, meta, _ = me.read_master(master_path)
    sky = None if sky_mask is None else pr.load_mask(sky_mask, data.shape[:2])
    overrides = {"black": black, "target": target, "asinh_strength": contrast, "saturation": saturation,
                 "highlight_desat_start": highlight_desat_start, "highlight_desat_end": highlight_desat_end, "look": look,
                 "shadow_chroma_end": shadow_chroma_end}
    if anchor is not None:
        overrides["anchor"] = float(anchor)
    params = pr.default_params(meta, data, sky=sky, wb_mode=wb_mode, shadow_smoothing=shadow_smoothing, **overrides)
    del data
    out = _new_dir(output_root, "photo")
    record = pr.render_photo(master_path, out, params, sky_mask=sky_mask, look_profile=dcp_profile, write_flags=write_flags)
    record["directory"] = str(out)
    record["anchor_source"] = "explicit" if anchor is not None else ("measured (median luminance%s)" % (" within sky mask" if sky is not None else ""))
    record["noise_sigma_source"] = "measured (darkest decile, horizontal difference MAD)"
    if candidates:
        record["candidates"] = pr.render_candidates(master_path, out / "candidates", sky_mask=sky_mask, look_profile=dcp_profile,
                                                    target=target, wb_mode=wb_mode)
    record["seconds"] = time.perf_counter() - started
    (out / pr.RECORD_NAME).write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return record
