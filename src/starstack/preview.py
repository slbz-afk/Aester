"""Reproducible display variations from a completed, unchanged job.

Jobs whose products carry the M/D contract (``diagnostics.color_export.state == succeeded``, docs/COLOR_CONTRACT_PACKAGE_20260922.md
§10) are re-rendered through the single display transform from their M master (docs/PRE_RELEASE_CLOSURE_20260922.md G): the
red/blue gains multiply the default (as-shot) white-balance gains and are realised as a matrix, the exposure moves the
midtone anchor (anchor × 2^-EV), everything else is the frozen recipe ``photo-default-1``.  Older jobs and FITS jobs keep
the legacy three-tone display byte for byte.
"""
from pathlib import Path
import math
import uuid

from . import core
from .imaging import read_rgb, export_display_products


def _real_path(path: Path) -> Path:
    for part in (path, *path.parents):
        if part.is_symlink() or getattr(part, "is_junction", lambda: False)():
            raise ValueError("预览路径不能包含符号链接或目录联接")
    return path.resolve(strict=True)


def _require(path: Path, directory: Path) -> None:
    """A source removed by a cache cleanup is reported as such, not as a broken job.

    The in-app cleanup keeps every file read here; the uninstaller's cleanup removes the masters.
    """
    if not path.exists():
        refusal = core.cache_cleanup_refusal(directory, "重新生成预览")
        if refusal:
            raise ValueError(refusal)


def _parameter(value, name: str, low: float, high: float) -> float:
    if isinstance(value, (bool, str)):
        raise ValueError(f"{name} 必须为 {low} 到 {high} 之间的有限数值")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} 必须为有限数值") from exc
    if not math.isfinite(number) or not low <= number <= high:
        raise ValueError(f"{name} 必须为 {low} 到 {high} 之间的有限数值")
    return number


def render_job_preview(job_dir: str, red_gain: float = 1.0, blue_gain: float = 1.0,
                       exposure_ev: float = 0.0) -> dict:
    """Add a unique variation without running Siril or rewriting job.json.

    Gains are display channel balance, applied after the existing common
    percentile normalization and before the display stretch. They are not
    sensor white-balance coefficients or a camera-to-sRGB transformation.
    """
    parameters = {"red_gain": _parameter(red_gain, "红通道", .25, 4),
                  "blue_gain": _parameter(blue_gain, "蓝通道", .25, 4),
                  "exposure_ev": _parameter(exposure_ev, "曝光", -3, 3)}
    directory = _real_path(Path(job_dir).expanduser().absolute())
    job_path = _real_path(directory / "job.json")
    job = core.load_job(directory)
    if job.get("state") != "succeeded":
        raise ValueError("只有已完成的任务可以重新生成预览")
    candidates = [Path(a["path"]) for a in job.get("artifacts", [])
                  if a.get("kind") == "fits" and Path(a["path"]).name == "master_linear.fit"]
    if len(candidates) != 1:
        raise ValueError("任务缺少唯一的合成线性母版，请重新处理素材")
    source = candidates[0]
    if not source.is_absolute():
        source = directory / source
    if ".." in source.parts or not source.is_relative_to(directory):
        raise ValueError("线性母版必须位于该任务目录内")
    _require(source, directory)
    source = _real_path(source)
    if not source.is_relative_to(directory) or not source.is_file():
        raise ValueError("线性母版必须位于该任务目录内")
    core._validate_master(source)
    master = _scene_master(directory, job)
    if master is not None:
        return _render_display_transform(directory, job, job_path, master, parameters)
    source_hash = core._sha(source)
    job_hash = core._sha(job_path)
    data = read_rgb(source)
    variations = directory / "preview-variations"
    if variations.exists():
        _real_path(variations)
    variations.mkdir(exist_ok=True)
    output = variations / uuid.uuid4().hex
    output.mkdir()
    recipe_path = output / "recipe.json"
    recipe = {"schema_version": 1, "renderer": "starstack.display.v1",
              "state": "rendering", "created_at": core.now(), "directory": str(output),
              "job_dir": str(directory), "job_sha256": job_hash,
              "source_path": str(source), "source_sha256": source_hash,
              "parameters": parameters, "width": data.shape[1], "height": data.shape[0],
              "color_calibrated": False, "linear_master_modified": False,
              "method": "common percentiles 0.003/0.998; display gains and exposure; asinh(6); sRGB transfer",
              "limitations": ["预览通道平衡，非完整相机校色", "预览为非线性图像，亮部可能裁切"],
              "artifacts": [], "recipe_path": str(recipe_path)}
    core.atomic_json(recipe_path, recipe)
    try:
        artifacts = export_display_products(data, output, lambda: None, **parameters)
        if core._sha(source) != source_hash or core._sha(job_path) != job_hash:
            raise ValueError("渲染期间原任务或线性母版发生变化，未发布此预览")
        for artifact in artifacts:
            artifact["sha256"] = core._sha(Path(artifact["path"]))
        recipe.update(state="succeeded", artifacts=artifacts, completed_at=core.now())
        core.atomic_json(recipe_path, recipe)
    except Exception as exc:
        recipe.update(state="failed", artifacts=[], error=str(exc))
        core.atomic_json(recipe_path, recipe)
        raise
    return recipe


def _scene_master(directory: Path, job: dict) -> Path | None:
    """The job's M master when its products follow the M/D contract, bounded to the job directory; None otherwise."""
    colour = (job.get("diagnostics") or {}).get("color_export") or {}
    if colour.get("state") != "succeeded":
        return None
    recorded = (colour.get("master") or {}).get("path")
    if not recorded:
        return None
    master = Path(recorded)
    if not master.is_absolute():
        master = directory / master
    if ".." in master.parts or not master.is_relative_to(directory):
        raise ValueError("后期母版必须位于该任务目录内")
    _require(master, directory)
    master = _real_path(master)
    if not master.is_relative_to(directory) or not master.is_file():
        raise ValueError("后期母版必须位于该任务目录内")
    return master


def _render_display_transform(directory: Path, job: dict, job_path: Path, master: Path, parameters: dict) -> dict:
    """D re-rendered from M with the preview controls mapped onto the single display transform (recipe photo-default-1)."""
    from . import color_export as ce, display_transform as dt, master_export as me, photo_render as pr
    master_hash = core._sha(master)
    job_hash = core._sha(job_path)
    sky_path = directory / "sky_mask.png" if job.get("mask") and (directory / "sky_mask.png").is_file() else None
    variations = directory / "preview-variations"
    if variations.exists():
        _real_path(variations)
    variations.mkdir(exist_ok=True)
    output = variations / uuid.uuid4().hex
    output.mkdir()
    recipe_path = output / "recipe.json"
    recipe = {"schema_version": 2, "renderer": "starstack.display-transform-2", "state": "rendering", "created_at": core.now(),
              "directory": str(output), "job_dir": str(directory), "job_sha256": job_hash,
              "source_path": str(master), "source_sha256": master_hash, "source_contract": "M",
              "parameters": parameters, "base_recipe": pr.DEFAULT_RECIPE["name"],
              "color_calibrated": True, "linear_master_modified": False,
              "method": "photo-default-1 display transform from the M master; red/blue gains multiply the as-shot white-balance "
                        "gains (realised as a matrix); exposure moves the midtone anchor (anchor × 2^-EV), not a linear multiply",
              "limitations": ["通道增益是成片白平衡的偏移，不是重新标定", "曝光作用于中间调锚点，高光压缩随之变化"],
              "artifacts": [], "recipe_path": str(recipe_path)}
    core.atomic_json(recipe_path, recipe)
    try:
        data, meta, _ = me.read_master(master)
        recipe.update(width=int(data.shape[1]), height=int(data.shape[0]))
        sky = pr.load_mask(sky_path, data.shape[:2]) if sky_path is not None else None
        _, _, base_gains = pr.wb_choice(meta, pr.DEFAULT_WB)
        gains = (float(base_gains[0]) * parameters["red_gain"], float(base_gains[1]), float(base_gains[2]) * parameters["blue_gain"])
        basis = meta["colour_basis"]
        matrix = me.switch_white_balance_matrix({"camera_to_linear_srgb": basis["camera_to_linear_srgb"], "gains_rgb": basis["gains_rgb"]},
                                                gains, meta["gamut"])
        matrix = tuple(tuple(float(v) for v in row) for row in matrix)
        anchor = dt.measure_anchor(data, meta["gamut"], mask=sky, wb_matrix=matrix) / (2.0 ** parameters["exposure_ev"])
        params = pr.default_params(meta, data, sky=sky, wb_mode=pr.DEFAULT_WB, wb_matrix=matrix, wb_gains_rgb=gains, anchor=anchor)
        del data
        unity = parameters["red_gain"] == 1.0 and parameters["blue_gain"] == 1.0 and parameters["exposure_ev"] == 0.0
        photo = pr.render_photo(master, output, params, sky_mask=sky_path, tiff_name=ce.D_TIFF_NAME, jpeg_name=ce.D_JPEG_NAME,
                                exist_ok=True, label="preview variation", write_flags=False,
                                recipe_label=None if unity else pr.DEFAULT_RECIPE["name"] + "+preview-gains")
        if core._sha(master) != master_hash or core._sha(job_path) != job_hash:
            raise ValueError("渲染期间原任务或后期母版发生变化，未发布此预览")
        artifacts = [{"path": photo["jpeg"], "kind": "jpeg", "label": "成片预览", "sha256": photo["jpeg_sha256"]},
                     {"path": photo["tiff"], "kind": "tiff", "label": "16 位成片 TIFF", "sha256": photo["tiff_sha256"]}]
        recipe.update(state="succeeded", artifacts=artifacts, display_transform=params.to_json(),
                      recipe_label=photo["metadata"]["recipe"], completed_at=core.now())
        core.atomic_json(recipe_path, recipe)
    except Exception as exc:
        recipe.update(state="failed", artifacts=[], error=str(exc))
        core.atomic_json(recipe_path, recipe)
        raise
    return recipe

