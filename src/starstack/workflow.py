"""Durable shared RAW -> master/photo -> optional star-layer workflow.

The stack job and its states remain independent. A workflow never upgrades a
failed stack or a legacy colour fallback to a successful M+D delivery. Model
weights are supplied by the caller; adapter semantics are explicit and frozen.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import functools
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import traceback
import uuid

from . import core, frame_pool, hashing, orientation, profiling
from .locking import Busy, file_lock, is_locked, pid_alive
from .records import atomic_json, read_json

VERSION = "workflow-1"
TERMINAL = {"succeeded", "failed", "cancelled"}
BACKEND_DEFAULTS = {"tile": 512, "stride": 256, "batch": 1,
                    "layout": "nhwc", "channel_order": "rgb", "value_range": "0..1",
                    "stitch": "blend", "intra_op_threads": None}
TRAIL_DEFAULTS = (None, "meteor", "review_only")
# Optional star-layer images copied from the verified layer pack
# (delivered name, pack file, label). Starless - subtract + add reproduces the
# photograph before star tidying; the layers do not depend on the strength.
STAR_EXTRAS = {"save_starless": (("starless_srgb16.tif", "00_background.tif", "无星图（16 位）"),),
               "save_star_layer": (("stars_add_srgb16.tif", "02_stars_add.tif", "星点加层（16 位，相加）"),
                                   ("stars_subtract_srgb16.tif", "01_stars_subtract.tif", "星点减层（16 位，相减）"))}
EXTRA_ROLES = {"starless_srgb16.tif": "starless", "stars_add_srgb16.tif": "stars_add",
               "stars_subtract_srgb16.tif": "stars_subtract", "trails_layer_rgba16.tif": "trails_layer"}
# Hash handover of one stack job launch (P1): the workflow worker puts a fresh
# token here while it submits or resumes the child job, so the job worker it
# spawns inherits it; the job worker exports its run ledger for the job
# directory to hashing.handover_path(token) before the job becomes terminal,
# and the workflow worker imports it once the terminal job is validated.
HANDOVER_TOKEN_ENV = "AESTER_HASH_HANDOVER_TOKEN"


class WorkflowCancelled(KeyboardInterrupt):
    """Also lets separate_stars record cancellation, rather than failure."""


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode("utf-8")).hexdigest()


def _scoped(label):
    """Run the decorated call inside a hashing run scope (shared with an enclosing one)."""
    def decorate(function):
        @functools.wraps(function)
        def scoped(*args, **kwargs):
            with hashing.run_scope(label):
                return function(*args, **kwargs)
        return scoped
    return decorate


def _sha(path, source, *, trust=hashing.TRUST_LEDGER):
    """``core._sha`` (``Path(path).open("rb")``) through the shared hashing module."""
    return hashing.sha256_file(Path(path), source=source, trust=trust)


def _file(path, *, source="workflow._file", trust=hashing.TRUST_LEDGER):
    path = Path(path).expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError("需要普通文件：" + str(path))
    return {"path": str(path), "sha256": hashing.sha256_file(path, source=source, trust=trust),
            "bytes": path.stat().st_size}


def _regular(path, prepare=None):
    """What ``_file(prepare(path))`` checks before hashing: the resolved path of a regular file."""
    if prepare is not None:
        path = prepare(path)
    path = Path(path).expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError("需要普通文件：" + str(path))
    return path


def _files(paths, *, prepare=None, source="workflow._file", trust=hashing.TRUST_LEDGER, fresh=False):
    """``[_file(prepare(p)) for p in paths]``: same records and first failure, files hashed in parallel.

    A path that cannot be prepared or is not a regular file fails only after
    every earlier file has been hashed, exactly where the serial loop failed.
    A file whose hash fails (it went away while earlier files were hashed) is
    checked again as ``_file`` checked it before hashing, so it fails with the
    same error; each size is read right after that file's digest, as ``_file``
    does.  ``fresh=True``: every file was just written and closed by the
    caller (``hashing.sha256_many``'s ``fresh``; records and failures unchanged).
    """
    originals, resolved, failure = [], [], None
    for path in paths:
        try:
            resolved.append(_regular(path, prepare))
        except Exception as exc:
            failure = exc
            break
        originals.append(path)
    sizes = [None] * len(resolved)

    def size(index):
        sizes[index] = resolved[index].stat().st_size

    def recheck(index, exc):
        if isinstance(exc, OSError):
            _regular(originals[index], prepare)   # raises what _file raised before hashing, if it now fails
        return None

    digests = hashing.sha256_many(resolved, source=source, trust=trust, on_error=recheck, after=size, fresh=fresh)
    records = [{"path": str(path), "sha256": digest, "bytes": size}
               for path, digest, size in zip(resolved, digests, sizes)]
    if failure is not None:
        raise failure
    return records


def _changed(entry):
    return ValueError("输入或已完成阶段的文件已改变／缺失：" + str(Path(entry["path"])))


def _verify_files(entries, *, trust=hashing.TRUST_LEDGER):
    """Each entry's file is a regular file of the recorded size and digest; the first failing entry raises.

    Identical paths in one call are hashed once; the files are hashed in parallel.
    """
    hashing.verify_entries(entries, source="workflow._verify_files", error=_changed, trust=trust)


def _number(value, name, low=0., high=1.):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} 须为 [{low}, {high}] 内有限数值")
    return float(value)


def _options(options, *, stars_enabled):
    """Admitted backend options, as frozen in the request and its signature.

    EFF-NOX (P1c K1): a request that runs nox (``stars_enabled``) and names no
    thread count (absent or null) gets the explicit
    ``starless_runtime.NOX_INTRA_OP_THREADS``; an explicit integer is kept.
    Other requests keep null, so their digests, idempotency keys and child job
    directories are unchanged; a frozen null still runs ORT-auto on resume.
    """
    from . import starless_runtime as runtime
    if options is not None and not isinstance(options, dict):
        raise ValueError("backend_options 须为对象")
    if set(options or {}) - BACKEND_DEFAULTS.keys():
        raise ValueError("未知 backend_options 参数")
    out = {**BACKEND_DEFAULTS, **(options or {})}
    if stars_enabled and out["intra_op_threads"] is None:
        out["intra_op_threads"] = runtime.NOX_INTRA_OP_THREADS
    runtime.resource_plan(**{k: out[k] for k in ("tile", "stride", "stitch", "batch")})
    runtime._validate_adapter_options(out["layout"], out["channel_order"], out["value_range"], out["intra_op_threads"])
    return out


def _read(directory):
    directory = Path(directory).expanduser().resolve(strict=True)
    record = read_json(directory / "workflow.json")  # polled while the worker replaces it
    if record.get("version") != VERSION or record.get("workflow_dir") != str(directory):
        raise ValueError("不是受支持的原位工作流记录")
    if record.get("stack_job_dir") and not Path(record["stack_job_dir"]).resolve().is_relative_to(directory / "stack"):
        raise ValueError("堆栈作业目录不在此工作流范围内")
    return record


def _update(directory, **fields):
    directory = Path(directory)
    with file_lock(directory / "metadata.lock", timeout=30):
        record = _read(directory)
        record.update(fields, updated_at=core.now())
        atomic_json(directory / "workflow.json", record)
    return record


def _verify(record, *, admission=False):
    """Frozen request, inputs, completed stages and publications are unchanged.

    ``admission``: the worker's first check of an attempt reads the request
    files for real (never from a ledger or a handover).
    """
    if core.cache_cleanup_marked(record["workflow_dir"]):
        raise ValueError("此作业的大文件缓存已清理；作品仍可使用，继续处理请新建作业")
    if _digest(record["request"]) != record["signature"]:
        raise ValueError("工作流参数或输入绑定已改变")
    if record.get("config") != record["request"].get("config"):
        raise ValueError("工作流配置与冻结请求不一致")
    if record["request"].get("delivery_root") != record["config"].get("delivery_root"):
        raise ValueError("作品目录与冻结配置不一致")
    if record["request"].get("delivery_format") != record["config"].get("delivery_format"):
        raise ValueError("作品格式与冻结配置不一致")
    if record["request"].get("jpeg_quality") != record["config"].get("jpeg_quality"):
        raise ValueError("JPEG 质量与冻结配置不一致")
    if record["request"].get("delivery_format") not in (None, "aester-delivery-2") \
            or (record["request"].get("delivery_format") and not record["request"].get("delivery_root")):
        raise ValueError("未知作品交付格式")
    _verify_files(record["request"]["files"], trust=hashing.TRUST_REAL if admission else hashing.TRUST_LEDGER)
    request = record["request"]
    rotation = orientation.normalize_rotation(request.get("image_rotation", 0))
    if rotation != orientation.normalize_rotation(record["config"].get("image_rotation", 0)):
        raise ValueError("工作流方向与冻结配置不一致")
    selected = ({"input_files": request["input_files"]}
                if request.get("selection_mode") == "files" else {})
    sources = core.inspect_sources(request["input_dir"], **selected)
    if sources["files"] != record["request"]["input_files"]:
        raise ValueError("素材文件夹的曝光列表已改变")
    directory = Path(record["workflow_dir"])
    for name in record.get("checkpoints", {}):
        _cached(directory, name)
    if record.get("state") == "succeeded":
        _verify_published(record)


def _verify_published(record):
    """Current artifacts and every recorded v2 publication still match their frozen hashes."""
    request = record["request"]
    directory = Path(record["workflow_dir"])
    for item in record.get("artifacts", []):
        if request.get("delivery_root"):
            from .storage import delivery_file
            delivery_file(record, item["path"])
        else:
            _local(item["path"], directory)
    _verify_files(record.get("artifacts", []))
    if request.get("delivery_format") == "aester-delivery-2":
        from .storage import delivery_receipt, verify_delivery
        required = {record["delivery_dir"], record.get("base_delivery_dir", record["delivery_dir"])}
        required.update(item["delivery_dir"] for item in record.get("review_deliveries", []))
        for published in required:
            verify_delivery(published, receipt=delivery_receipt(record, published))


def _stack_job_dir(directory, signature) -> Path:
    """The child stack job's folder: core.submit_job's job-<sha256(idempotency key)[:24]> under <workflow>/stack,
    for the key _wait_for_stack passes."""
    key = "workflow-stack-" + signature
    return Path(directory) / "stack" / ("job-" + hashlib.sha256(key.encode()).hexdigest()[:24])


def validate_stack_binding(record, job):
    """Verify the child job against its admitted parent before acting on it.

    Directory ownership alone is insufficient: a locally edited child record
    must not select different inputs, masks or executable on resume/cancel.
    The hash checks are integrity checks, not a signature against an attacker
    who can also rewrite every parent identity and its digest.
    """
    request = record["request"]
    if _digest(request) != record.get("signature") or record.get("config") != request.get("config"):
        raise ValueError("工作流请求绑定已改变")
    directory = Path(record["workflow_dir"]).resolve(strict=True)
    expected_dir = _stack_job_dir(directory, record["signature"])
    actual_dir = Path(job.get("job_dir", "")).resolve()
    if actual_dir != expected_dir or (record.get("stack_job_dir") and Path(record["stack_job_dir"]).resolve() != expected_dir):
        raise ValueError("堆栈目录与父工作流的固定提交编号不符")
    inventory = {entry["path"]: entry for entry in request["files"]}
    expected_inputs = [{"path": path, "size": inventory[path]["bytes"], "sha256": inventory[path]["sha256"]}
                       for path in request["input_files"]]
    if job.get("inputs") != expected_inputs or job.get("input_kind") != request["input_kind"]:
        raise ValueError("堆栈源列表／原片身份与父工作流不符")
    assets = request["assets"]
    expected_mask = ({"path": assets["sky_mask_path"]["path"], "sha256": assets["sky_mask_path"]["sha256"]}
                     if assets["sky_mask_path"] else None)
    # After publication, advanced trail review may append a new derivative to
    # this child job. It does not change the parent's frozen base or products.
    transient_edit = job.get("transient_marks") and record.get("state") != "succeeded"
    if job.get("mask") != expected_mask or job.get("frame_masks") or transient_edit:
        raise ValueError("堆栈蒙版／亮迹参数与父工作流不符")
    if (job.get("support_mode", "legacy") != request["support_mode"]
            or job.get("stack_range_mode", "legacy") != request["stack_range_mode"]):
        raise ValueError("堆栈支持／数值范围模式与父工作流不符")
    if (orientation.normalize_rotation(job.get("image_rotation", 0))
            != orientation.normalize_rotation(request.get("image_rotation", 0))):
        raise ValueError("堆栈方向与父工作流不符")
    if (orientation.normalize_rotation(record["config"].get("image_rotation", 0))
            != orientation.normalize_rotation(request.get("image_rotation", 0))):
        raise ValueError("工作流方向与冻结配置不一致")
    # New requests freeze effect intent; old records retain their original
    # implicit scan/model contract when these keys are absent.
    if "trails_enabled" in request:
        expected_transients = "review" if request["trails_enabled"] else "off"
        if job.get("transient_mode") != expected_transients:
            raise ValueError("堆栈亮迹开关与父工作流不符")
    if "color_adjustments" in request and job.get("color_adjustments") != request["color_adjustments"]:
        raise ValueError("堆栈颜色调节与父工作流不符")
    if job.get("compose_feather_mode") != request.get("compose_feather_mode"):
        raise ValueError("天空地景过渡方式与父工作流不符")
    if request.get("selection_mode") == "files":
        expected_selection = {"mode": "files", "files": request["input_files"],
                              "input_roots": list(dict.fromkeys(str(Path(p).parent) for p in request["input_files"]))}
        if job.get("input_selection") != expected_selection:
            raise ValueError("堆栈未使用冻结的所选照片清单")
    engine = job.get("engine") or {}
    expected_engine = assets["siril_path"]
    if Path(engine.get("executable", "")).resolve() != Path(expected_engine["path"]):
        raise ValueError("堆栈几何引擎路径与父工作流不符")
    _verify_files([expected_engine])
    finite = assets["stack_engine_path"]
    child_finite = job.get("stack_engine")
    if finite:
        if (not isinstance(child_finite, dict)
                or Path(child_finite.get("executable", "")).resolve() != Path(finite["path"])
                or child_finite.get("sha256") != finite["sha256"]):
            raise ValueError("独立堆栈引擎与父工作流不符")
        _verify_files([finite])
    elif child_finite:
        raise ValueError("父工作流未授权独立堆栈引擎")


@_scoped("workflow.submit")
def submit_workflow(input_dir, output_root, *, siril_path=None, sky_mask_path=None,
                    stack_engine_path=None, star_model_path=None, background_strength=None,
                    milkyway_offset=0.0, backend_options=None, idempotency_key=None,
                    stack_range_mode="legacy", input_files=None,
                    trails_enabled=False, stars_enabled=False, color_adjustments=None,
                    regions_path=None, milkyway_enabled=False, milkyway_strength=.6,
                    delivery_root=None, delivery_format=None, image_rotation=None,
                    trail_default=None, save_starless=False, save_star_layer=False, jpeg_quality=None):
    """Start a detached, resumable workflow; strengths use the interval 0..1.

    Effects are off until explicitly requested, even with an installed model.
    A model run requires a sky mask: absence of a mask does not certify
    that a landscape photograph contains only sky. A saved canvas project
    supplies managed masks and explicit human-reviewed Milky Way evidence.

    trail_default='meteor' (with trails_enabled) adds back, after the base
    photograph is published, every candidate whose identity is meteor and whose
    measurement is trustworthy, with algorithm provenance; the base stays.
    None/'review_only' keeps the legacy review-only behaviour. save_starless /
    save_star_layer (with stars_enabled) also deliver the starless image and
    the signed star layer pair. jpeg_quality (an int in 60..100) is frozen in
    the request: the delivered final.jpg (and the before-reduction JPEG where
    the layout delivers one) are encoded from the delivered 16-bit TIFF at that
    quality with 4:4:4 chroma, and every later trails version of the job uses
    it; None keeps each JPEG's historical encoding. Unset options do not change
    the request digest, except that a nox request (stars_enabled) without an
    ORT thread count freezes the explicit EFF-NOX default (see ``_options``).
    """
    from .color_adjustments import normalize
    if jpeg_quality is not None:
        from .jpeg_estimate import validate_quality
        jpeg_quality = validate_quality(jpeg_quality)
    if any(type(v) is not bool for v in (trails_enabled, stars_enabled, milkyway_enabled)):
        raise ValueError("trails_enabled、stars_enabled 与 milkyway_enabled 必须是 bool")
    if trail_default is not None and (type(trail_default) is not str or trail_default not in TRAIL_DEFAULTS):
        raise ValueError("trail_default 须为 meteor、review_only 或省略")
    if type(save_starless) is not bool or type(save_star_layer) is not bool:
        raise ValueError("save_starless 与 save_star_layer 必须是 bool")
    if (save_starless or save_star_layer) and not stars_enabled:
        raise ValueError("保存无星图或星点图层需要开启整理星点")
    if trail_default == "meteor" and not trails_enabled:
        raise ValueError("默认加回流星需要开启亮迹识别")
    adjustments = normalize(color_adjustments)
    rotation = None if image_rotation is None else orientation.normalize_rotation(image_rotation)
    options = _options(backend_options, stars_enabled=stars_enabled)
    offset = _number(milkyway_offset, "milkyway_offset", -1., 1.)
    if offset != 0 and not regions_path:
        raise ValueError("非零银河 offset 需要源图与银河蒙版绑定的存在依据；请使用进阶缩星入口")
    mw_strength = _number(milkyway_strength, "milkyway_strength")
    strength = None if background_strength is None else _number(background_strength, "background_strength")
    if idempotency_key is not None and (not isinstance(idempotency_key, str) or not idempotency_key.strip()):
        raise ValueError("idempotency_key 须为非空字符串")
    selected = {"input_files": input_files} if input_files is not None else {}
    sources = core.inspect_sources(input_dir, **selected)
    root = Path(output_root).expanduser().resolve()
    source_roots = {Path(p).parent for p in sources["files"]}
    if any(root == parent or root.is_relative_to(parent) for parent in source_roots):
        raise ValueError("结果文件夹必须位于原片文件夹之外")
    if delivery_root is not None:
        from .storage import admit_delivery_root
        delivery_root = admit_delivery_root(delivery_root, root, sources["files"])
        delivery_format = delivery_format or "aester-delivery-2"
        if delivery_format not in ("aester-delivery-1", "aester-delivery-2"):
            raise ValueError("未知作品交付格式")
    elif delivery_format is not None:
        raise ValueError("作品交付格式仅用于分离式发布")
    region_assets = None
    if regions_path:
        if sky_mask_path:
            raise ValueError("画布分区与外部天空蒙版不能同时指定")
        from . import regions
        project = regions.load_project(regions_path, input_files=sources["files"])
        bound_rotation = orientation.normalize_rotation(project["source_binding"]["reference"].get("image_rotation", 0))
        if rotation is not None and rotation != bound_rotation:
            raise ValueError("素材方向与绘制分区不一致；旋转后请重新绘制分区")
        rotation = bound_rotation
        region_assets = regions.build_masks(project, root / "region-assets", input_files=sources["files"])
        sky_mask_path = region_assets["sky_mask_path"]
        if (milkyway_enabled or offset != 0) and not region_assets.get("milkyway_mask_path"):
            raise ValueError("银河增强／差别缩星需要已绘制的银河画布区域")
    elif milkyway_enabled:
        raise ValueError("银河增强需要已绘制的银河画布区域")
    if rotation is None:
        # An old standalone PNG has no orientation binding. Preserve its
        # historical native frame unless the caller explicitly supplies one.
        rotation = orientation.source_rotation(sources["files"][0]) if input_files is not None and not sky_mask_path else 0
    if sources["kind"] == "fits" and any(adjustments.values()):
        raise ValueError("FITS 的基础显示暂不支持白平衡与曝光调节；请将颜色调节复位")
    if stack_engine_path and not sky_mask_path:
        raise ValueError("显式有限值堆栈需要天空蒙版")
    if stack_range_mode not in ("legacy", "positive_power2_v1", "finite_validity_v1"):
        raise ValueError("未知堆栈范围模式")
    if stack_range_mode == "finite_validity_v1" and not stack_engine_path:
        raise ValueError("finite_validity_v1 需要专用堆栈引擎")
    if stars_enabled and not star_model_path:
        raise ValueError("尚未配置星点整理后端；请先选择模型或关闭整理星点")
    if stars_enabled and not sky_mask_path:
        raise ValueError("自动去星需要显式天空蒙版；纯天空可提供全白 PNG，不能将未知地景当作天空")
    info = core.doctor(siril_path)
    if not info.get("available"):
        raise ValueError(info.get("message", "Siril 不可用"))
    files = _files(sources["files"])
    assets = {}
    for name, path in (("siril_path", info["executable"]), ("sky_mask_path", sky_mask_path),
                       ("stack_engine_path", stack_engine_path),
                       ("star_model_path", star_model_path if stars_enabled else None)):
        assets[name] = None if path is None else _file(path)
        if assets[name]:
            files.append(assets[name])
    if sky_mask_path and Path(assets["sky_mask_path"]["path"]).suffix.lower() != ".png":
        raise ValueError("天空蒙版须为 PNG")
    if region_assets:
        for path in (regions_path, region_assets["project_path"], region_assets.get("milkyway_mask_path"),
                     region_assets["source_binding"]["reference"]["linear_path"]):
            if path is not None:
                files.append(_file(path))
    mode = "finite_validity_v1" if stack_engine_path else stack_range_mode
    config = {"input_dir": sources["input_dir"], "output_root": str(root),
              **{name: value["path"] if value else None for name, value in assets.items()},
              "background_strength": strength, "milkyway_offset": offset,
              "backend_options": options, "stack_range_mode": mode,
              "selection_mode": "files" if input_files is not None else "directory",
              "trails_enabled": trails_enabled, "stars_enabled": stars_enabled,
              "color_adjustments": adjustments, "regions_path": str(Path(regions_path).resolve()) if regions_path else None,
              "milkyway_enabled": milkyway_enabled, "milkyway_strength": mw_strength,
              "image_rotation": rotation,
              "compose_feather_mode": "canvas-inward-feather-1" if region_assets else None}
    if input_files is not None:
        config["input_files"] = sources["files"]
    # New intents enter the frozen request only when requested, so older
    # callers keep identical digests and idempotency keys.
    if trail_default == "meteor":
        config["trail_default"] = "meteor"
    if save_starless:
        config["save_starless"] = True
    if save_star_layer:
        config["save_star_layer"] = True
    if jpeg_quality is not None:
        config["jpeg_quality"] = jpeg_quality
    if delivery_root is not None:
        config["delivery_root"] = delivery_root
        if delivery_format == "aester-delivery-2":
            config["delivery_format"] = delivery_format
    request = {"input_dir": sources["input_dir"], "input_files": sources["files"],
               "input_kind": sources["kind"], "assets": assets, "files": files,
               "background_strength": strength, "milkyway_offset": offset,
               "backend_options": options,
               "support_mode": "strict_siril_1_4_4" if stack_engine_path else "legacy",
               "stack_range_mode": mode, "config": config}
    request.update({key: config[key] for key in
                    ("selection_mode", "trails_enabled", "stars_enabled", "color_adjustments",
                     "regions_path", "milkyway_enabled", "milkyway_strength", "compose_feather_mode", "image_rotation")})
    request.update({key: config[key] for key in ("trail_default", "save_starless", "save_star_layer", "jpeg_quality")
                    if key in config})
    request["region_assets"] = region_assets
    if delivery_root is not None:
        request["delivery_root"] = delivery_root
        if delivery_format == "aester-delivery-2":
            request["delivery_format"] = delivery_format
    signature = _digest(request)
    ident = hashlib.sha256(idempotency_key.encode()).hexdigest()[:24] if idempotency_key else uuid.uuid4().hex[:16]
    root.mkdir(parents=True, exist_ok=True)
    directory = root / ("workflow-" + ident)
    if any(Path(path).is_relative_to(directory) for path in sources["files"]):
        raise ValueError("工作流目录不能包含原片")
    with file_lock(root / ".workflow-admission.lock", timeout=30):
        if directory.exists():
            prior = _read(directory)
            if prior["signature"] != signature:
                raise ValueError("相同提交编号对应不同素材／模型／参数")
            _verify(prior)
            return workflow_status(directory)
        # S-8: the child stack job's engine must be able to write its outputs (L workflows are never checked).
        refusal = core.stack_engine_path_refusal(root, _stack_job_dir(directory, signature), mode)
        if refusal:
            raise ValueError(refusal)
        directory.mkdir()
        limits = [("量测可靠的流星候选在基础成片完成后自动加回（算法建议，不认证身份）；其余候选待复核；堆栈仍可能残留亮迹。"
                   if trail_default == "meteor" else "基础成片不自动加入未复核亮迹；堆栈仍可能残留亮迹。"),
                  "模型分层为实验能力，可重合成不代表无星摄影质量通过。"]
        if not (region_assets or {}).get("milkyway_mask_path"):
            limits.append("没有人工确认的银河范围，未启用银河增强或差别缩星。")
        if mode == "legacy":
            limits.append("未提供有限值堆栈引擎，沿用 legacy；不保证配准／堆栈保留全部负值和超界值。")
        elif mode == "positive_power2_v1":
            limits.append("高光范围保护保留高光余量；单通道零值／负值照常堆栈，暗端与 legacy 相同在0处截断并计数，不是完整有符号前端。")
        if not sky_mask_path:
            limits.append("未提供天空蒙版，未进行天空／地景分离合成。")
        record = {"version": VERSION, "workflow_dir": str(directory), "state": "queued",
                  "stage": "queued", "progress": 0, "message": "等待处理", "error": None,
                  "stack_job_dir": None, "worker_pid": None, "request": request, "config": config,
                  "signature": signature, "created_at": core.now(), "updated_at": core.now(),
                  "checkpoints": {}, "artifacts": [], "attempt": 1, "limitations": limits}
        atomic_json(directory / "workflow.json", record)
        _spawn(directory)
    return workflow_status(directory)


def _spawn(directory):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1]) + os.pathsep + env.get("PYTHONPATH", "")
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    try:
        with (directory / "worker.log").open("ab") as log:
            process = subprocess.Popen([sys.executable, "-m", "starstack.workflow", str(directory)],
                env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                creationflags=flags, start_new_session=(os.name != "nt"))
        _update(directory, worker_pid=process.pid)
    except OSError as exc:
        _update(directory, state="failed", stage="launch", error=str(exc), message="无法启动处理进程")
        raise


def workflow_status(workflow_dir):
    """The workflow record, with derived fields that are never stored.

    A stopped worker is reported as failed. A cleaned cache (marker written by the
    in-app cleanup or by the uninstaller) adds ``cache_cleaned: true`` and a short
    ``cache_cleanup`` summary; uncleaned records are returned unchanged.
    """
    directory = Path(workflow_dir).expanduser().resolve(strict=True)
    record = _read(directory)
    if record["state"] not in TERMINAL and not is_locked(directory / "worker.lock"):
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(record["updated_at"])).total_seconds()
        pid = record.get("worker_pid")
        if age > 60:
            record.update(state="failed", error="Workflow worker stopped before publication",
                          message="工作流未持有运行锁；可核验完整阶段后恢复", worker_stopped=True,
                          recorded_pid_alive=bool(pid and pid_alive(pid)))
    if core.cache_cleanup_marked(directory):
        record.update(cache_cleaned=True, cache_cleanup=_cleanup_summary(directory))
    return record


def _cleanup_summary(directory):
    """Who cleaned the cache and how far it got, read from the markers (never trusted for any decision).

    The in-app cleanup's marker is summarised when present, else the uninstaller's; ``cleaned_by_all``
    names every cleanup that left a marker (the two remove different files).
    """
    summaries = []
    for name, default, version in ((core.APP_CACHE_CLEANUP_MARKER, "aester-app", "aester-app-cache-cleaned-1"),
                                   (core.CACHE_CLEANUP_MARKER, "uninstaller", "aester-cache-cleaned-1")):
        marker = directory / name
        if not marker.exists():
            continue
        try:
            data = read_json(marker)
        except (OSError, ValueError):
            data = None
        if not isinstance(data, dict):
            summaries.append({"state": "unknown", "cleaned_by": "unknown", "at": None})
            continue
        by = data.get("cleaned_by") or (default if data.get("version") == version else "unknown")
        summaries.append({"state": data.get("state") if isinstance(data.get("state"), str) else "unknown",
                          "cleaned_by": by if isinstance(by, str) else "unknown",
                          "at": data.get("finished_at") or data.get("started_at")})
    first = summaries[0] if summaries else {"state": "unknown", "cleaned_by": "unknown", "at": None}
    return {**first, "cleaned_by_all": [summary["cleaned_by"] for summary in summaries]}


def cancel_workflow(workflow_dir):
    directory = Path(workflow_dir).expanduser().resolve(strict=True)
    initial = _read(directory)
    if initial.get("stack_job_dir"):
        validate_stack_binding(initial, core.load_job(initial["stack_job_dir"]))
    with file_lock(directory / "metadata.lock", timeout=30):
        record = _read(directory)
        if record["state"] not in TERMINAL:
            (directory / "cancel.request").touch()
            record.update(state="cancel_requested", message="正在停止；已完成母版保留", updated_at=core.now())
            atomic_json(directory / "workflow.json", record)
    if record.get("stack_job_dir") and record["state"] == "cancel_requested":
        core.cancel_job(record["stack_job_dir"])
    return workflow_status(directory)


@_scoped("workflow.resume")
def resume_workflow(workflow_dir):
    directory = Path(workflow_dir).expanduser().resolve(strict=True)
    with file_lock(directory.parent / ".workflow-admission.lock", timeout=30):
        if is_locked(directory / "worker.lock"):
            raise ValueError("工作流进程仍在运行，请等待停止")
        record = workflow_status(directory)
        if record["state"] not in {"failed", "cancelled"}:
            raise ValueError("只有失败或已取消的工作流可恢复")
        _verify(record)
        if record.get("stack_job_dir"):
            validate_stack_binding(record, core.load_job(record["stack_job_dir"]))
        (directory / "cancel.request").unlink(missing_ok=True)
        _update(directory, state="queued", error=None, worker_pid=None, stage="resume",
                attempt=record.get("attempt", 1) + 1, message="核验通过，准备恢复")
        _spawn(directory)
    return workflow_status(directory)


def _cancel_check(directory):
    if (directory / "cancel.request").exists():
        raise WorkflowCancelled("用户请求停止")


def _checkpoint(directory, name, data, paths, *, fresh=False):
    """Record checkpoint ``name``; ``fresh=True``: the caller has just written and closed every one of ``paths``."""
    _cancel_check(directory)
    records = _files(sorted(set(map(str, paths))), prepare=lambda path: _local(path, directory),
                     source="workflow._checkpoint", fresh=fresh)
    with file_lock(directory / "metadata.lock", timeout=30):
        record = _read(directory)
        record["checkpoints"][name] = {"data": data, "data_sha256": _digest(data), "files": records}
        record["updated_at"] = core.now()
        atomic_json(directory / "workflow.json", record)
    return data


def _cached(directory, name):
    checkpoint = _read(directory)["checkpoints"].get(name)
    if checkpoint is None:
        return None
    if _digest(checkpoint["data"]) != checkpoint["data_sha256"]:
        raise ValueError("阶段记录发生改变")
    for item in checkpoint["files"]:
        _local(item["path"], directory)
    _verify_files(checkpoint["files"])
    data = checkpoint["data"]
    bound = {str(Path(item["path"]).resolve()): item for item in checkpoint["files"]}

    def bound_file(path):
        p = _local(path, directory)
        if str(p) not in bound:
            raise ValueError("阶段引用未绑定到冻结文件：" + str(p))
        return p

    if name == "stack":
        snapshots = [Path(path) for path in bound if Path(path).name.startswith("stack_snapshot-")]
        # Pre-integration runs froze the child index itself. Accept that old
        # immutable checkpoint only while its original bytes still match.
        if not snapshots:
            snapshots = [Path(path) for path in bound if Path(path).name == "job.json"]
        if len(snapshots) != 1 or json.loads(snapshots[0].read_text(encoding="utf-8")) != data["job"]:
            raise ValueError("堆栈缓存与不可变快照不符")
        if data.get("job_snapshot") and bound_file(data["job_snapshot"]) != snapshots[0]:
            raise ValueError("堆栈快照引用不一致")
        validate_stack_binding(_read(directory), data["job"])
        if _read(directory)["request"].get("trails_enabled"):
            revision = data["job"].get("transient_revision")
            if revision:
                bound_file(revision["record"])
                if _sha(revision["record"], "workflow._cached") != revision["sha256"]:
                    raise ValueError("亮迹复核记录与冻结堆栈不符")
        colour = data["colour"]
        for section, keys in (("master", ("path", "recipe", "mask")), ("photo", ("tiff", "jpeg", "record"))):
            for key in keys:
                if colour[section].get(key):
                    bound_file(colour[section][key])
        original = (data["job"].get("diagnostics") or {}).get("color_export") or {}
        if original.get("state") == "succeeded":
            if colour != original:
                raise ValueError("缓存颜色产物与已完成堆栈快照不符")
        elif data["job"].get("input_kind") == "fits" and original.get("state") == "legacy":
            names = {Path(a["path"]).name: str(Path(a["path"]).resolve())
                     for a in data["job"]["artifacts"] if a.get("group") != "transients"}
            required = [(colour["master"]["path"], "master_linear_32bit.tif"),
                        (colour["photo"]["tiff"], "natural_display_16bit.tif"),
                        (colour["photo"]["jpeg"], "natural.jpg")]
            if colour.get("contract") != "legacy_input_RGB" or any(str(Path(p).resolve()) != names.get(n) for p, n in required):
                raise ValueError("FITS 缓存产物与已完成堆栈快照不符")
        else:
            raise ValueError("缓存没有已成功的颜色产物依据")
    elif name == "masks":
        bound_file(data["sky_mask"])
        bound_file(data["protected_mask"])
    elif name == "legacy_display":
        bound_file(data["path"])
        _local(data["source"], directory)
    elif name == "separation":
        result_path = bound_file(data["result_path"])
        if json.loads(result_path.read_text(encoding="utf-8")) != data["record"]:
            raise ValueError("分层缓存与冻结结果不符")
    elif name in ("advice", "reduction", "layers", "milkyway"):
        result_path = bound_file(Path(data["processing_directory"]) / "result.json")
        if json.loads(result_path.read_text(encoding="utf-8")) != data:
            raise ValueError("后期缓存与冻结结果不符")
    return data


@profiling.staged("milkyway_region")
def _mapped_milkyway(directory, request, stack):
    """Move a first-frame sky annotation into the actual stacked sky frame.

    Foreground masks remain camera-footprint masks. A sky object follows the
    fitted registration, including the native/undistorted/native roundtrip.
    An unavailable first-frame transform is an error, never an identity guess.
    """
    cached = _cached(directory, "milkyway_region")
    if cached:
        return cached
    assets = request.get("region_assets") or {}
    if not assets.get("milkyway_mask_path"):
        return None
    import numpy as np
    import tifffile
    from .resample import warp_image
    from .registration_distortion import RadialDistortion
    from .registration_models import apply_matrix
    from .photographic_job import _mask, _display_source
    source = _display_for_finishing(directory, stack)
    shape = _display_source(Path(source))[0][:2]
    mask = _mask(assets["milkyway_mask_path"], shape)
    geometry = (stack["job"].get("diagnostics") or {}).get("first_frame_geometry") or {}
    if (geometry.get("status") != "succeeded" or geometry.get("shape") != list(shape)
            or geometry.get("input") != stack["job"]["inputs"][0]):
        raise ValueError("首帧缺少绑定到堆栈的几何快照，请新建作业以可靠映射银河范围")
    h = np.asarray(geometry["matrix"], dtype=np.float64)
    model = RadialDistortion(**geometry["distortion"]) if geometry.get("distortion") is not None else None
    if h is None or np.asarray(h).shape != (3, 3) or not np.isfinite(h).all():
        raise ValueError("首帧未成功配准，不能将银河画布区域可靠映射到堆栈；请另选参考素材")
    inverse = np.linalg.inv(h)
    def mapping(native):
        _cancel_check(directory)
        points = model.apply(native) if model is not None else native
        points = apply_matrix(inverse, points)
        points = model.inverse(points) if model is not None else points
        if not np.isfinite(points).all():
            raise ValueError("银河范围几何映射产生非有限坐标")
        return points
    result = warp_image(mask, mapping, kernel="bilinear", fill=0.)
    # Restrict the edit using actual composite support, not the hand drawing.
    masks = _masks(directory, stack)
    sky = _mask(masks["sky_mask"], shape)
    result[sky != 1] = 0
    target = directory / ("milkyway-region-" + uuid.uuid4().hex[:10] + ".tif")
    tifffile.imwrite(target, result, metadata={"roworder": "TOP-DOWN"}, photometric="minisblack")
    data = {"path": str(target), "project_path": assets["project_path"],
            "region_revision": assets["revision"], "first_frame_to_reference": np.asarray(h).tolist(),
            "distortion": geometry.get("distortion"),
            "shape": list(shape), "coordinate_contract": "first-native-to-stacked-reference-native"}
    return _checkpoint(directory, "milkyway_region", data, [target], fresh=True)


@profiling.staged("evidence")
def _milkyway_inputs(directory, request, stack, source):
    region = _mapped_milkyway(directory, request, stack)
    if region is None:
        return {}
    from . import regions
    from .milkyway_gate import VERSION as EVIDENCE_VERSION
    project = regions.load_project(request["regions_path"], input_files=request["input_files"])
    declaration = project["milkyway"]
    if not regions.milkyway_active(project):
        raise ValueError("当前画布没有已绘制的银河区域")
    method = ('explicit-region-agent' if declaration.get('authority') == 'agent' else
              'explicit-region-human' if declaration.get('selection') == regions.DRAWING else 'human-review')
    evidence = {"version": EVIDENCE_VERSION, "status": "present", "method": method,
                "reviewer": declaration["reviewer"], "rationale": declaration["rationale"],
                "source_sha256": _sha(source, "workflow._milkyway_inputs"),
                "milkyway_mask_sha256": _sha(region["path"], "workflow._milkyway_inputs")}
    path = directory / ("milkyway-evidence-" + _digest(evidence)[:24] + ".json")
    if not path.exists():
        atomic_json(path, evidence)
    elif json.loads(path.read_text(encoding="utf-8")) != evidence:
        raise ValueError("银河依据缓存已改变")
    return {"milkyway_mask": region["path"], "milkyway_evidence": str(path)}


@profiling.staged("milkyway")
def _milkyway_finish(directory, request, stack):
    if not request.get("milkyway_enabled"):
        return None
    cached = _cached(directory, "milkyway")
    if cached:
        return cached
    from .photographic_job import finish_milkyway
    source = _display_for_finishing(directory, stack)
    inputs = _milkyway_inputs(directory, request, stack, source)
    if not inputs:
        raise ValueError("银河增强没有已绑定的人工确认区域")
    masks = _masks(directory, stack)
    _update(directory, stage="milkyway", progress=76, message="调整圈选银河内的结构与色彩")
    # P1c T-5a: the render's row blocks see the stop request (a cancel removes the staging folder)
    result = finish_milkyway(source, directory / "finishing" / "milkyway",
                            masks["sky_mask"], masks["protected_mask"],
                            strength=request.get("milkyway_strength", .6), **inputs,
                            check=lambda: _cancel_check(directory))
    if result.get("state") != "succeeded":
        raise ValueError("银河增强失败：" + str(result.get("error")))
    _checkpoint(directory, "milkyway", result, [*_result_paths(result), *inputs.values()])
    return result


def _local(path, directory):
    path = Path(path).resolve(strict=True)
    if not path.is_file() or not path.is_relative_to(directory.resolve()):
        raise ValueError("产物不在当前工作流中：" + str(path))
    return path


def _result_paths(result):
    paths = [item["path"] for item in result.get("artifacts", {}).values()]
    if result.get("processing_directory"):
        paths.append(str(Path(result["processing_directory"]) / "result.json"))
    elif paths:
        paths.append(str(Path(paths[0]).parent / "result.json"))
    return paths


def _stack(directory, request):
    with profiling.stage("stack_checkpoint"):
        prior = _cached(directory, "stack")
        if prior:
            validate_stack_binding(_read(directory), core.load_job(prior["job"]["job_dir"]))
            return prior
    # A fresh token per launch: a handover left by an earlier attempt (or by a
    # job this attempt did not start) is never accepted.
    token = uuid.uuid4().hex if hashing.active_mode() != "off" else None
    if token is not None:
        # Housekeeping: handovers of launches whose worker was stopped while its job ran are never read.
        hashing.prune_handovers()
    try:
        with profiling.stage("stack_wait"):
            job = _wait_for_stack(directory, request, handover_token=token)
        with profiling.stage("stack_checkpoint"):
            return _checkpoint_stack(directory, request, job, handover_token=token)
    finally:
        if token is not None:
            _discard_handover(token)


@contextmanager
def _handover_token(token):
    """Expose ``token`` to a stack job worker spawned inside the block (its environment is inherited)."""
    if token is None:
        yield
        return
    previous = os.environ.get(HANDOVER_TOKEN_ENV)
    os.environ[HANDOVER_TOKEN_ENV] = token
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(HANDOVER_TOKEN_ENV, None)
        else:
            os.environ[HANDOVER_TOKEN_ENV] = previous


def _discard_handover(token):
    """Remove an unconsumed handover of this launch (housekeeping; never fails the worker)."""
    try:
        hashing.handover_path(token).unlink(missing_ok=True)
    except (OSError, ValueError):
        pass


def _wait_for_stack(directory, request, *, handover_token=None):
    """Submit (or resume) the child stack job and wait until it is terminal."""
    job_dir = _read(directory).get("stack_job_dir")
    assets = request["assets"]
    if job_dir is None:
        _cancel_check(directory)
        kwargs = {"siril_path": assets["siril_path"]["path"],
                  "sky_mask_path": assets["sky_mask_path"]["path"] if assets["sky_mask_path"] else None,
                  "support_mode": request["support_mode"], "stack_range_mode": request["stack_range_mode"],
                  "idempotency_key": "workflow-stack-" + _read(directory)["signature"]}
        if assets["stack_engine_path"]:
            kwargs["stack_engine_path"] = assets["stack_engine_path"]["path"]
        if request.get("selection_mode") == "files":
            kwargs["input_files"] = request["input_files"]
        if "trails_enabled" in request:
            kwargs["transient_mode"] = "review" if request["trails_enabled"] else "off"
        if "color_adjustments" in request:
            kwargs["color_adjustments"] = request["color_adjustments"]
        if request.get("compose_feather_mode") is not None:
            kwargs["compose_feather_mode"] = request["compose_feather_mode"]
        kwargs["image_rotation"] = orientation.normalize_rotation(request.get("image_rotation", 0))
        with _handover_token(handover_token):
            job = core.submit_job(request["input_dir"], str(directory / "stack"), **kwargs)
        validate_stack_binding(_read(directory), job)
        job_dir = job["job_dir"]
        if not Path(job_dir).resolve().is_relative_to(directory):
            raise ValueError("堆栈作业位于工作流之外")
        _update(directory, stack_job_dir=job_dir)
    else:
        job = core.load_job(job_dir)
        validate_stack_binding(_read(directory), job)
    if job["state"] in {"failed", "cancelled"} and _read(directory).get("attempt", 1) > 1:
        with _handover_token(handover_token):
            job = core.resume_job(job_dir)
        validate_stack_binding(_read(directory), job)
    while True:
        if (directory / "cancel.request").exists():
            validate_stack_binding(_read(directory), core.load_job(job_dir))
            core.cancel_job(job_dir)
            # Do not claim stopped while the independent stack is still running.
            job = core.load_job(job_dir)
            if job["state"] in core.TERMINAL:
                raise WorkflowCancelled("堆栈已停止")
        else:
            job = core.load_job(job_dir)
            if job["state"] in core.TERMINAL:
                break
        _update(directory, stage="stack", progress=min(75, round(job.get("progress", 0) * .75, 1)),
                message=job.get("message", "堆栈处理中"), stack_state=job["state"])
        time.sleep(.5)
    return job


def _checkpoint_stack(directory, request, job, *, handover_token=None):
    """Verify the terminal child job and freeze it as the workflow's stack checkpoint."""
    _cancel_check(directory)
    validate_stack_binding(_read(directory), job)
    if handover_token is not None:
        # Digests the job worker computed for its own files, with their stat;
        # a lookup still needs the file's current stat to match exactly.
        hashing.import_handover(None, job["job_dir"], token=handover_token)
    if job["state"] != "succeeded":
        raise RuntimeError("堆栈作业未成功：" + str(job.get("error") or job["state"]))
    colour = (job.get("diagnostics") or {}).get("color_export") or {}
    if job.get("input_kind") == "fits" and colour.get("state") == "legacy":
        by_name = {Path(a["path"]).name: a["path"] for a in job["artifacts"] if a.get("group") != "transients"}
        # The legacy FITS writer is intentional, not a failed RAW colour basis.
        colour = {"state": "succeeded", "contract": "legacy_input_RGB",
                  "master": {"path": by_name["master_linear_32bit.tif"],
                             "sha256": _sha(by_name["master_linear_32bit.tif"], "workflow._checkpoint_stack")},
                  "photo": {"tiff": by_name["natural_display_16bit.tif"], "jpeg": by_name["natural.jpg"],
                            "tiff_sha256": _sha(by_name["natural_display_16bit.tif"], "workflow._checkpoint_stack"),
                            "jpeg_sha256": _sha(by_name["natural.jpg"], "workflow._checkpoint_stack")},
                  "limitation": "输入 RGB 未经相机颜色标定；线性堆栈不是相机 M 合同母版。"}
    if colour.get("state") != "succeeded":
        raise RuntimeError("堆栈完成，但 M+D 导出未成功：" + str(colour.get("error") or colour.get("state")))
    snapshot = directory / ("stack_snapshot-" + uuid.uuid4().hex[:12] + ".json")
    atomic_json(snapshot, job)
    paths = [snapshot, colour["master"]["path"],
             colour["photo"]["tiff"], colour["photo"]["jpeg"]]
    for section, field in (("master", "recipe"), ("photo", "record")):
        if colour[section].get(field):
            paths.append(colour[section][field])
    if colour["master"].get("mask"):
        paths.append(colour["master"]["mask"])
    desc = ((job.get("diagnostics") or {}).get("sky_validity") or {}).get("compose_weight")
    if desc:
        paths.extend([desc["path"], desc["composite"]])
    revision = job.get("transient_revision")
    if request.get("trails_enabled") and revision:
        record_path = _local(revision["record"], directory)
        if _sha(record_path, "workflow._checkpoint_stack") != revision["sha256"]:
            raise ValueError("亮迹复核记录哈希不符")
        paths.append(record_path)
    for path in paths:
        _local(path, directory)
    products = ((colour["master"], "sha256", "path"), (colour["photo"], "tiff_sha256", "tiff"),
                (colour["photo"], "jpeg_sha256", "jpeg"))
    if hashing.first_mismatch([Path(item[path_name]) for item, _, path_name in products],
                              [item.get(hash_name) for item, hash_name, _ in products],
                              source="workflow._checkpoint_stack") is not None:
        raise ValueError("堆栈颜色产物哈希不符")
    return _checkpoint(directory, "stack", {"job": job, "colour": colour,
                                             "job_snapshot": str(snapshot)}, paths)


@profiling.staged("masks")
def _masks(directory, stack):
    cached = _cached(directory, "masks")
    if cached:
        return cached
    import numpy as np
    import tifffile
    from PIL import Image
    from .compose_evidence import read_weight
    from .photographic_job import _display_source
    job = stack["job"]
    source = Path(stack["colour"]["photo"]["tiff"])
    shape = _display_source(Path(_display_for_finishing(directory, stack)))[0][:2]
    descriptor = ((job.get("diagnostics") or {}).get("sky_validity") or {}).get("compose_weight")
    if job.get("mask") or descriptor:
        if not isinstance(descriptor, dict):
            raise ValueError("有地景的成片缺少实际 compose_weight，不允许推断去星范围")
        composites = [Path(a["path"]).resolve() for a in job.get("artifacts", [])
                      if Path(a["path"]).name == "master_linear.fit" and a.get("group") != "transients"]
        if len(composites) != 1 or composites[0].parent != source.parent:
            raise ValueError("基础成片与实际合成权重未绑定同一导出目录")
        weight = read_weight(Path(job["job_dir"]), descriptor, composites[0])
        if weight.shape != shape:
            raise ValueError("实际合成权重与成片尺寸不符")
        sky_scope = "actual-compose-weight-pure-sky"
    else:
        raise ValueError("未能证明成片无地景，不能自动使用全白去星范围")
    masks = directory / ("masks-" + uuid.uuid4().hex[:10])
    masks.mkdir()
    sky = masks / "sky_weight.tiff"
    tifffile.imwrite(sky, weight, photometric="minisblack", metadata={"roworder": "TOP-DOWN"})
    protect = masks / "protected.png"
    # This workflow takes only the base colour product. It never silently adds
    # candidate trails, nor invents a protection mask from unreviewed detections.
    Image.fromarray(np.zeros(shape, np.uint8)).save(protect)
    data = {"sky_mask": str(sky), "protected_mask": str(protect), "scope": sky_scope,
            "protection": "base-photo-only-no-additional-reviewed-trail-layer"}
    return _checkpoint(directory, "masks", data, [sky, protect], fresh=True)


@profiling.staged("legacy_display")
def _display_for_finishing(directory, stack):
    """Label an authenticated legacy FITS display copy, preserving its codes."""
    source = str(_local(stack["colour"]["photo"]["tiff"], directory))
    if stack["colour"].get("contract") != "legacy_input_RGB":
        return source
    cached = _cached(directory, "legacy_display")
    if cached:
        return cached["path"]
    import numpy as np
    import tifffile
    from .photographic_job import _validate_srgb, _display_source
    with tifffile.TiffFile(source) as tf:
        if len(tf.pages) != 1 or tf.pages[0].dtype != np.uint16:
            raise ValueError("FITS 成片不是单页 uint16 TIFF")
        page = tf.pages[0]
        icc = page.tags[34675].value
        _validate_srgb(icc)
        metadata = json.loads(page.description)
        if metadata.get("linear") is not False or metadata.get("purpose") != "uncalibrated display rendering":
            raise ValueError("不是已知 legacy FITS 显示写出格式")
        pixels = page.asarray()
    target = directory / ("legacy-display-" + uuid.uuid4().hex[:10] + ".tif")
    tifffile.imwrite(target, pixels, photometric="rgb", iccprofile=icc,
        metadata={"roworder": "TOP-DOWN", "scene_linear": False, "display_encoding_applied": True,
                  "color_space": "encoded_sRGB", "source_sha256": _sha(source, "workflow._display_for_finishing"),
                  "color_contract": "legacy_input_RGB", "pixel_transform": "identity"})
    _display_source(target)
    if not np.array_equal(tifffile.imread(target), pixels):
        raise ValueError("显示域包装改变了像素")
    data = _checkpoint(directory, "legacy_display", {"path": str(target), "source": source}, [target])
    return data["path"]


@profiling.staged("separation")
def _separate(directory, request, stack, source=None):
    cached = _cached(directory, "separation")
    if cached:
        return cached
    from . import starless_runtime as runtime
    from .star_separation import separate_stars
    masks = _masks(directory, stack)
    _cancel_check(directory)
    _update(directory, stage="separation", progress=76, message="生成可编辑星空图层")
    options = request["backend_options"]
    from . import resources
    resources.shadow_nox(actual_intra_op_threads=options.get("intra_op_threads"))   # P1b-2: record only (P3/P4)
    # The frozen value as admitted: EFF-NOX's 8, a caller's count, or None (a request frozen before P1c: ORT auto).
    model = runtime.OnnxModel(request["assets"]["star_model_path"]["path"],
        **{k: options[k] for k in ("layout", "channel_order", "value_range", "intra_op_threads")})
    if model.identity.get("model_sha256") != request["assets"]["star_model_path"]["sha256"]:
        raise ValueError("加载的去星模型与提交时不同")

    class CancellableModel:
        identity = model.identity
        expects_tile = getattr(model, "expects_tile", None)
        expects_batch = getattr(model, "expects_batch", None)

        def __call__(self, array):
            _cancel_check(directory)
            result = model(array)
            _cancel_check(directory)
            return result

    def progress(done, total):
        _cancel_check(directory)
        _update(directory, stage="separation", progress=76 + 14 * done / max(1, total),
                message=f"生成星空图层 {done}/{total}")

    def provider(source, estimate, _stage):
        # P1c §2.2: nox separation is an exclusive section; a frame pool started meanwhile runs serially
        with frame_pool.exclusive("nox"):
            return runtime.run_tiled(source, estimate, CancellableModel(), progress=progress,
                **{k: options[k] for k in ("tile", "stride", "batch", "stitch")})

    result = separate_stars(source or _display_for_finishing(directory, stack), directory / "finishing" / "separation",
        masks["sky_mask"], masks["protected_mask"], provider=provider,
        provider_identity={**model.identity, **options, "uses_learned_model": True})
    paths = _result_paths(result)
    result_path = str(Path(result["artifacts"]["diffuse_srgb16.tif"]["path"]).parent / "result.json")
    return _checkpoint(directory, "separation", {"record": result, "result_path": result_path}, paths)


def _finish(directory, request, stack):
    milkyway = _milkyway_finish(directory, request, stack)
    if not milkyway and (request.get("stars_enabled") is False or request["assets"]["star_model_path"] is None):
        return {"state": "skipped", "reason": "not-requested" if request.get("stars_enabled") is False else "no-star-model",
                "background_strength_applied": None}
    source = (milkyway["artifacts"]["milkyway_srgb16.tif"]["path"] if milkyway
              else _display_for_finishing(directory, stack))
    mw_outputs = ({"milkyway": milkyway,
                   "final_photo": {"tiff": source,
                                   "jpeg": milkyway["artifacts"]["milkyway_srgb.jpg"]["path"]}}
                  if milkyway else {})
    if request.get("stars_enabled") is False:
        return {"state": "succeeded" if milkyway else "skipped", "reason": "star-reduction-not-requested" if milkyway else "not-requested",
                "background_strength_applied": None, **mw_outputs}
    if request["assets"]["star_model_path"] is None:
        return {"state": "succeeded" if milkyway else "skipped", "reason": "no-star-model",
                "background_strength_applied": None, **mw_outputs}
    from .star_reduction import suggest_star_reduction, reduce_stars
    from .star_layer_export import export_star_layers
    separation = _separate(directory, request, stack, source)
    mw_inputs = _milkyway_inputs(directory, request, stack, source)
    result_path = separation["result_path"]
    _cancel_check(directory)
    with profiling.stage("advice"):
        advice = _cached(directory, "advice")
        if advice is None:
            _update(directory, stage="density_advice", progress=91, message="评估星点密度")
            advice = suggest_star_reduction(result_path, directory / "finishing" / "advice", allowed_root=directory, **mw_inputs)
            _checkpoint(directory, "advice", advice, [*_result_paths(advice), *mw_inputs.values()])
    strength = request["background_strength"]
    unavailable = strength is None and not advice["recommendation"]["available"]
    if strength is None and not unavailable:
        strength = advice["recommendation"]["background_strength"]
    with profiling.stage("reduction"):
        reduction = _cached(directory, "reduction")
        if reduction is None and not unavailable:
            _cancel_check(directory)
            _update(directory, stage="star_reduction", progress=93, message="整理背景星点")
            reduction = reduce_stars(result_path, directory / "finishing" / "reduction", allowed_root=directory,
                                     background_strength=strength, milkyway_offset=request.get("milkyway_offset", 0.),
                                     **mw_inputs, check=lambda: _cancel_check(directory))   # P1c T-5b: a cancel point
            _checkpoint(directory, "reduction", reduction, [*_result_paths(reduction), *mw_inputs.values()])
    _cancel_check(directory)
    with profiling.stage("layers"):
        layers = _cached(directory, "layers")
        if layers is None:
            _update(directory, stage="layer_export", progress=96, message="导出可编辑图层")
            layers = export_star_layers(result_path, directory / "finishing" / "layers", allowed_root=directory)
            _checkpoint(directory, "layers", layers, _result_paths(layers))
    return {"state": "succeeded" if reduction else "reduction_skipped", "advice": advice, **mw_outputs,
            "background_strength_applied": strength, "separation_result": result_path,
            "reduction": reduction, "layers": layers,
            "reason": "density-advice-unavailable-kept-base-photo" if unavailable else None}


def _encode_jpeg(source, target, quality):
    """``target``: the 16-bit RGB TIFF ``source`` as a JPEG at ``quality`` (request ``jpeg_quality``).

    8-bit codes are round(code / 257) (``display_transform.to_8bit``, applied per 128-row block); the TIFF's own
    ICC profile is embedded and chroma is 4:4:4. The 128-row conversion runs on the frame pool (P1c T-5x (c))
    under its own note key.
    """
    import numpy as np
    import tifffile
    from PIL import Image
    from . import performance, resources
    from .display_transform import to_8bit
    from .jpeg_estimate import SUBSAMPLING
    from .master_export import save_image_with_digest
    from .photographic_job import _close_map, _jpeg_rows, _read_display
    with tifffile.TiffFile(source) as tf:
        page = tf.pages[0]
        if len(tf.pages) != 1 or page.dtype != np.uint16 or len(page.shape) != 3 or page.shape[2] != 3:
            raise ValueError("JPEG 来源须为单页 16 位 RGB TIFF")
        shape = page.shape
        icc = page.tags[34675].value if 34675 in page.tags else None
    picture = _read_display(source)
    try:
        pool_stats = {}
        pixels = _jpeg_rows(picture, to_8bit, pool_stats=pool_stats)
        performance.note_pool("jpeg_rows.publish", pool_stats)
        resources.shadow_pool("py.jpeg_rows", actual=pool_stats.get("workers"), items=-(-shape[0] // 128),
                              width=shape[1])
    finally:
        _close_map(picture)
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    profile = {"icc_profile": icc} if isinstance(icc, bytes) else {}
    save_image_with_digest(Image.fromarray(pixels), target, source="workflow._encode_jpeg", quality=quality,
                           subsampling=SUBSAMPLING, **profile)
    return target


ENCODED_JPEGS = "jpeg-"        # <workflow>/jpeg-<token>/: one publication attempt's encoded JPEGs (removed after it)
_PHOTO_JPEG_SOURCE = "round(uint16/257) of the TIFF array"   # photo_render's statement about its JPEG


def _sweep_encoded_jpegs(directory):
    """Remove the jpeg-<token> folders an earlier, cancelled or failed publication attempt left; never raises.

    Only the worker (under worker.lock) publishes, so no folder of this pattern is in use when an attempt starts.
    """
    try:
        stale = [path for path in directory.glob(ENCODED_JPEGS + "*")
                 if path.is_dir() and len(path.name) == len(ENCODED_JPEGS) + 12
                 and all(c in "0123456789abcdef" for c in path.name[len(ENCODED_JPEGS):])]
    except OSError:
        return
    for path in stale:
        shutil.rmtree(path, ignore_errors=True)


def _deliver_jpeg_quality(directory, names, photo, quality, folder, *, images_only, delivered_dir):
    """``names`` (delivered name -> source path) with its JPEGs at the request's ``jpeg_quality``, in place.

    Each JPEG the delivery carries is encoded afresh from the TIFF it accompanies into ``folder`` (a v2,
    images-only delivery carries final.jpg alone, so nothing else is encoded). Frozen stages keep their own
    JPEGs. A JPEG that is the stack photograph's photo_render JPEG, already round(uint16/257) of the same TIFF
    with its ICC at 4:4:4 and this quality, is delivered as it is (the same bytes an encode would give). When
    the stack photograph's JPEG is replaced and its photo_render.json is delivered (local and aester-delivery-1
    layouts), a copy of the record that describes the delivered JPEG is delivered instead; the frozen JPEG's
    fields stay under ``frozen_jpeg``. ``delivered_dir``: the local layout's delivery folder (None: external,
    where the record names the JPEG relative to the delivery folder).
    """
    record = None
    if photo.get("record"):
        record = json.loads(_local(photo["record"], directory).read_text(encoding="utf-8"))
    frozen_matches = (record is not None and record.get("jpeg_quality") == quality
                      and record.get("jpeg_subsampling") == "4:4:4" and record.get("jpeg_source") == _PHOTO_JPEG_SOURCE
                      and record.get("jpeg_sha256") == photo.get("jpeg_sha256")
                      and record.get("tiff_sha256") == photo.get("tiff_sha256"))
    replaced = None
    for jpeg, tiff in (("final.jpg", "final_srgb16.tif"),
                       ("stack_before_reduction.jpg", "stack_before_reduction_srgb16.tif")):
        if jpeg not in names or (images_only and jpeg != "final.jpg"):
            continue
        from_photo = names[jpeg] == photo["jpeg"] and names[tiff] == photo["tiff"]
        if from_photo and frozen_matches:
            continue
        _cancel_check(directory)
        names[jpeg] = str(_encode_jpeg(_local(names[tiff], directory), folder / jpeg, quality))
        if from_photo:
            replaced = jpeg
    if replaced and record is not None and "photo_render.json" in names and not images_only:
        encoded = Path(names[replaced])
        record["frozen_jpeg"] = {key: record.get(key) for key in ("jpeg", "jpeg_sha256", "jpeg_quality", "jpeg_subsampling")}
        record.update(jpeg=replaced if delivered_dir is None else str(delivered_dir / replaced),
                      jpeg_sha256=_sha(encoded, "workflow._publish"), jpeg_quality=quality, jpeg_subsampling="4:4:4",
                      jpeg_source=_PHOTO_JPEG_SOURCE, jpeg_encoded_at="publication (request jpeg_quality)")
        path = folder / "photo_render.json"
        atomic_json(path, record)
        names["photo_render.json"] = str(path)


@profiling.staged("publish")
def _publish(directory, stack, finishing):
    profiling.switch(".stage")
    _cancel_check(directory)
    _update(directory, stage="publish", progress=98, message="保存母版、成片与处理记录")
    colour = stack["colour"]
    publication_record = _read(directory)
    external = publication_record["request"].get("delivery_root") is not None
    master_name = "master_input_linear.tif" if colour.get("contract") == "legacy_input_RGB" else "master_scene_linear.tif"
    names = {master_name: colour["master"]["path"],
             "stack_before_reduction_srgb16.tif": colour["photo"]["tiff"],
             "stack_before_reduction.jpg": colour["photo"]["jpeg"]}
    if colour["master"].get("recipe"):
        names["master_recipe.json"] = colour["master"]["recipe"]
    if colour["photo"].get("record"):
        names["photo_render.json"] = colour["photo"]["record"]
    if colour["master"].get("mask"):
        names["camera_saturation_mask.png"] = colour["master"]["mask"]
    extra_labels = {}
    for option, files in STAR_EXTRAS.items():
        if not publication_record["request"].get(option):
            continue
        layers = finishing.get("layers") or {}
        if layers.get("state") != "succeeded":
            raise ValueError("已请求保存无星图／星点图层，但本次没有生成可用的星点图层包")
        for name, layer_name, label in files:
            item = (layers.get("artifacts") or {}).get(layer_name)
            if not item:
                raise ValueError("星点图层包缺少文件：" + layer_name)
            path = _local(item["path"], directory)
            if _sha(path, "workflow._publish") != item["sha256"]:
                raise ValueError("星点图层文件与冻结记录不符")
            names[name] = str(path)
            extra_labels[name] = label
    reduced = finishing.get("reduction")
    final_photo = finishing.get("final_photo") or colour["photo"]
    # New basic deliveries need one final TIFF/JPEG, not a duplicate copy of
    # the same bytes labelled "before reduction". The internal immutable stack
    # remains available for later editing. Old workflows keep their old layout.
    if not reduced and "stars_enabled" in publication_record["request"] and not (external and final_photo != colour["photo"]):
        names.pop("stack_before_reduction_srgb16.tif")
        names.pop("stack_before_reduction.jpg")
    names["final_srgb16.tif"] = (str(Path(reduced["processing_directory"]) / "star_reduced_srgb16.tif")
                                if reduced else final_photo["tiff"])
    names["final.jpg"] = (str(Path(reduced["processing_directory"]) / "star_reduced.jpg")
                          if reduced else final_photo["jpeg"])
    token = uuid.uuid4().hex[:12]
    staging = directory / (".partial-delivery-" + token)
    target = directory / ("delivery-" + token)
    quality = publication_record["request"].get("jpeg_quality")
    encoded = directory / (ENCODED_JPEGS + token)
    if quality is not None:
        # The person's JPEG quality: the delivered JPEGs are encoded afresh from the TIFFs they accompany (frozen
        # stages and their JPEGs stay as they are); requests without it publish the frozen JPEGs unchanged.
        from .jpeg_estimate import validate_quality
        validate_quality(quality)
        _sweep_encoded_jpegs(directory)
        _deliver_jpeg_quality(directory, names, colour["photo"], quality, encoded,
                              images_only=external and publication_record["request"].get("delivery_format")
                              == "aester-delivery-2", delivered_dir=None if external else target)
    artifacts = []
    entries = {}
    for name, path in names.items():
        kind = {".tif": "tiff", ".jpg": "jpeg", ".png": "png", ".json": "json"}[Path(name).suffix]
        label = {master_name: "后期母版", "final.jpg": "最终 JPEG", "final_srgb16.tif": "最终 16 位成片",
                 "stack_before_reduction_srgb16.tif": "缩星前 16 位成片", "stack_before_reduction.jpg": "缩星前 JPEG",
                 **extra_labels}.get(name, name)
        entries[name] = {"path": str(_local(path, directory)), "kind": kind, "label": label}
    if external:
        from .storage import stage_delivery
        package = stage_delivery(publication_record, entries, target.name, check=lambda: _cancel_check(directory))
        staging, target, artifacts = package["stage"], package["target"], package["artifacts"]
    else:
        staging.mkdir()
        for name, item in entries.items():
            _cancel_check(directory)
            source = Path(item["path"])
            frozen = _file(source)
            with hashing.writing(staging / name):
                shutil.copyfile(source, staging / name)
            if _sha(staging / name, "workflow._publish") != frozen["sha256"]:
                raise ValueError("导出复制校验失败")
            artifacts.append({**frozen, "path": str(target / name), "name": name,
                              "kind": item["kind"], "label": item["label"]})
    profiling.switch(".verify")
    _verify(_read(directory))
    profiling.switch(".commit")
    with file_lock(directory / "metadata.lock", timeout=30):
        _cancel_check(directory)
        record = _read(directory)
        if external:
            from .storage import publication_parent, verify_staged
            if publication_parent(record) != target.parent:
                raise ValueError("作品发布位置改变")
            verify_staged(package)
        staging.rename(target)
        if external and package.get("receipt"):
            record.setdefault("publication_receipts", {})[str(target)] = package["receipt"]
        trail_review = _trail_review(record["request"], stack["job"])
        # The meteor default is added right after this base publication by the
        # same worker; the record stays non-terminal until that finishes.
        automatic = record["request"].get("trail_default") == "meteor" and trail_review["state"] == "needs_review"
        if automatic and colour.get("contract") == "legacy_input_RGB":
            # Trails versions need the calibrated camera colour replay.
            automatic = False
            trail_review["message"] = "未标定 FITS 暂不支持自动加回亮迹；候选仍可查看与复核。"
        extras = [{"role": EXTRA_ROLES[item["name"]], "path": item["path"], "sha256": item["sha256"]}
                  for item in artifacts if item["name"] in EXTRA_ROLES]
        record.update(state="running" if automatic else "succeeded", stage="trail_addback" if automatic else "complete",
                      progress=99 if automatic else 100, error=None,
                      message=("基础成片已保存；正在自动加回量测可靠的流星候选" if automatic
                               else "基础成片已保存；亮迹等待复核" if trail_review["state"] == "needs_review"
                               else "基础成片已保存；亮迹扫描未完成，尚未加回亮迹" if trail_review["state"] == "unavailable"
                               else "母版与成片已保存"), delivery_dir=str(target), artifacts=artifacts,
                      base_delivery_dir=str(target), base_artifacts=artifacts, extras=extras,
                      finishing=finishing, separation_result=finishing.get("separation_result"),
                      trail_review=trail_review, needs_review=trail_review["state"] == "needs_review",
                      color_contract=colour.get("contract"), updated_at=core.now())
        if automatic:
            record["trail_addback_pending"] = True
        else:
            record.pop("trail_addback_pending", None)
        atomic_json(directory / "workflow.json", record)
    if quality is not None:
        # Copied and verified into the delivery: nothing reads the encoded files again (best effort). A cancelled
        # or failed attempt's folder is swept when the next attempt starts (_sweep_encoded_jpegs).
        shutil.rmtree(encoded, ignore_errors=True)
    return record


def _trail_review(request, job):
    """A base delivery never claims that candidate trails were reviewed/added."""
    requested = request.get("trails_enabled") is True
    state = "not_requested" if "trails_enabled" in request else "legacy_unreviewed"
    diagnostics = job.get("diagnostics") or {}
    scan = diagnostics.get("transients") or {}
    if requested:
        state = "needs_review" if scan.get("status") == "needs_review" else "unavailable"
    meteor = request.get("trail_default") == "meteor"
    candidates = (scan.get("summary") or {}).get("candidates")
    return {"state": state, "requested": requested, "automatic_addback": False,
            "default_policy": "meteor-default-keep-1" if meteor else "explicit-keep-only-1",
            "kept_count": 0, "kept_ids": [], "kept_by": {"algorithm": 0, "human": 0, "agent": 0},
            "pending_count": int(candidates or 0) if state == "needs_review" and isinstance(candidates, int) else 0,
            "job_dir": job["job_dir"], "scan": scan,
            "message": (("正在自动加回量测可靠的流星候选；其余候选稍后可复核。" if meteor else
                         "请复核候选身份及保留选择，再生成含亮迹版本。") if state == "needs_review"
                        else "亮迹扫描未完成；基础成片没有自动加入候选。" if state == "unavailable"
                        else "旧版工作流基础成片未自动加入亮迹；旧候选和派生继续保留。" if state == "legacy_unreviewed"
                        else "未请求亮迹识别与加回。")}


def _automatic_trails(directory):
    """Add back the meteor default after the base publication; never lose the base.

    Any failure or a stop request leaves the workflow succeeded with the base
    photograph current and the candidates available for review.
    """
    from . import workflow_edit
    try:
        return workflow_edit.apply_default_trails(directory)["workflow"]
    except (WorkflowCancelled, KeyboardInterrupt):
        return _finish_without_trails(directory, "已停止自动加回流星；基础成片已保存，可稍后在亮迹复核中继续。", None)
    except hashing.HashLedgerMismatch:   # shadow mode: a wrong ledger answer must fail the run, not be skipped
        raise
    except Exception as exc:
        with (directory / "worker.log").open("a", encoding="utf-8") as log:
            traceback.print_exc(file=log)
        return _finish_without_trails(directory, "基础成片已保存；自动加回流星未完成，可在亮迹复核中重试。",
                                      f"{type(exc).__name__}: {exc}")


def _finish_without_trails(directory, message, error):
    profiling.switch(".fallback")
    with file_lock(directory / "metadata.lock", timeout=30):
        record = _read(directory)
        (directory / "cancel.request").unlink(missing_ok=True)
        review = dict(record.get("trail_review") or {})
        review.update(automatic_addback=False, message=message)
        if error:
            review["automatic_addback_error"] = error
        record.pop("trail_addback_pending", None)
        record.update(state="succeeded", stage="complete", progress=100, error=None, message=message,
                      trail_review=review, updated_at=core.now())
        atomic_json(directory / "workflow.json", record)
    return record


def run_workflow(workflow_dir):
    """Worker entry; public callers should use submit/resume instead."""
    directory = Path(workflow_dir).expanduser().resolve(strict=True)
    try:
        with file_lock(directory / "worker.lock"):
            record = _read(directory)
            if record["state"] == "succeeded":
                return record
            timer = profiling.StageTimer(host=True, initial="unstaged", peak_sampler_hz=10)   # P1b-2 local peaks
            core._note_siril_pin()  # diagnostics only; the stack job's own receipt records it too
            from . import resources
            # P1b-2 governor, shadow only: notes under this timer's timings; never raises, never changes a setting
            resources.shadow_start("workflow", timer=timer, work_dir=directory, request=record.get("request"))
            prior = _prior_timings(record)
            result = None
            try:
                # One hashing run scope for the whole attempt: stack, finishing,
                # publication and the automatic trail add-back.
                with profiling.activate(timer), hashing.run_scope("workflow.worker"):
                    try:
                        result = _run_attempt(directory, record)
                    finally:
                        # What the worker's run ledger did (reads, hits, shadow checks and
                        # mismatches), in this attempt's timings: diagnostics only.
                        hashing.note_scope_report()
            finally:
                result = _record_timings(directory, timer, prior, result)
            return result
    except Busy:
        return workflow_status(directory)


def _run_attempt(directory, record):
    if record.get("trail_addback_pending") is True:
        # The base photograph is already published; only the automatic
        # meteor add-back remains (e.g. after the worker was stopped).
        with profiling.stage("admission"):
            _update(directory, state="running", worker_pid=os.getpid(), error=None, stage="trail_addback",
                    progress=99, message="基础成片已保存；正在自动加回量测可靠的流星候选")
        with profiling.stage("trail_addback"):
            try:
                return _automatic_trails(directory)
            except hashing.HashLedgerMismatch as exc:   # shadow mode only (see _automatic_trails)
                with (directory / "worker.log").open("a", encoding="utf-8") as log:
                    traceback.print_exc(file=log)
                return _update(directory, state="failed", message="处理失败，可核验后恢复",
                               error=f"{type(exc).__name__}: {exc}")
    try:
        with profiling.stage("admission"):
            # P1c §3.5: finishing runs frame pools, so a malformed AESTER_FRAME_WORKERS / AESTER_OVERLAP fails the
            # workflow here, before any work (the trail add-back branch above validates in workflow_edit._apply_body)
            frame_pool.validate_environment()
            _verify(record, admission=True)
            _cancel_check(directory)
            _update(directory, state="running", worker_pid=os.getpid(), error=None,
                    stage="preflight", message="核验素材与处理参数")
        stack = _stack(directory, record["request"])
        _cancel_check(directory)
        finishing = _finish(directory, record["request"], stack)
        _cancel_check(directory)
        published = _publish(directory, stack, finishing)
        if published.get("trail_addback_pending") is True:
            with profiling.stage("trail_addback"):
                return _automatic_trails(directory)
        return published
    except (WorkflowCancelled, KeyboardInterrupt) as exc:
        return _update(directory, state="cancelled", message="已停止，可从完整阶段恢复",
                       error=str(exc) or "用户中断", artifacts=[])
    except Exception as exc:
        with (directory / "worker.log").open("a", encoding="utf-8") as log:
            traceback.print_exc(file=log)
        return _update(directory, state="failed", message="处理失败，可核验后恢复",
                       error=f"{type(exc).__name__}: {exc}", artifacts=[])


def _prior_timings(record):
    """Timing receipts of earlier worker attempts (diagnostics; absent in old records)."""
    attempts = record.get("timing_attempts")
    attempts = list(attempts) if isinstance(attempts, list) else []
    if isinstance(record.get("timings"), dict):
        attempts.append(record["timings"])
    return attempts


def _record_timings(directory, timer, prior, result):
    """Store this attempt's stage timings in workflow.json; never fails the worker.

    ``timings``/``timing_attempts`` are diagnostics outside the request
    signature, every checkpoint's data digest and every delivery receipt.
    ``updated_at`` is left as the worker's last state change.
    """
    try:
        timings = timer.finish()
        # Diagnostics only: never keep worker.lock waiting long for them.
        with file_lock(directory / "metadata.lock", timeout=5):
            latest = _read(directory)
            timings.update(state=latest.get("state"), recorded_at=core.now())
            latest.update(timings=timings, timing_attempts=prior)
            atomic_json(directory / "workflow.json", latest)
        return latest if result is not None else result
    except Exception:
        try:
            with (directory / "worker.log").open("a", encoding="utf-8") as log:
                traceback.print_exc(file=log)
        except OSError:
            pass
        return result


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        raise SystemExit("usage: python -m starstack.workflow WORKFLOW_DIRECTORY")
    result = run_workflow(args[0])
    return 0 if result["state"] == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
