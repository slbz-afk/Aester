"""Source-bound, optimistic batch trail review shared by desktop and MCP.

Classification and photographic disposition are independent. This module only
updates the child trail layer; workflow_edit owns final colour/effect delivery.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import uuid
from pathlib import Path

from . import hashing
from .locking import file_lock, is_locked
from .records import atomic_json

VERSION = "trail-review-session-1"
# Legacy sessions ("explicit-keep-only-1"): a classification is not
# authorization to add pixels to the photograph; only an explicit keep adds.
REVIEW_POLICY = {label: "omit" for label in ("meteor", "satellite", "aircraft", "artifact", "unknown")}
EXPLICIT_POLICY_VERSION = "explicit-keep-only-1"
# 2026-09-27 user decision: meteors are the only class kept by default. A
# measured, trustworthy meteor identity (suggested or labelled) is added back
# unless explicitly omitted; every other class needs an explicit keep. Records
# mark such keeps as algorithm provenance, distinct from human/agent choices.
METEOR_POLICY_VERSION = "meteor-default-keep-1"
DEFAULT_OPTIONS = {"export_trail_layer": False}


def _file_digest(path):
    return hashing.sha256_file(Path(path), source="trail_review_session._file_digest")


def _checkpoint_digest(directory, job, path):
    path = _within(path, directory)
    for entries in (job.get("checkpoint_files") or {}).values():
        for entry in entries:
            candidate = _within(directory / entry["path"], directory)
            if candidate == path:
                return entry.get("sha256")
    return None


def _verify_file(path, expected, description):
    if not expected or not Path(path).is_file() or _file_digest(path) != expected:
        raise ValueError(f"{description}缺失、已改变或缺少内容校验")


def _verify_inputs(inputs):
    """``for source in inputs: _verify_file(source["path"], source["sha256"], "原始曝光")``, hashed in parallel.

    Same checks in the same order per input, the same message, and the first
    failing input (in order) is the one reported.
    """
    hashing.verify_entries(inputs, source="trail_review_session._file_digest",
                           error=lambda source: ValueError("原始曝光缺失、已改变或缺少内容校验"),
                           path=lambda source: source["path"], expected=lambda source: source["sha256"],
                           precheck=lambda source, path: bool(source["sha256"]) and Path(path).is_file())


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _within(path, root):
    path = Path(path).resolve()
    try:
        path.relative_to(Path(root).resolve())
    except ValueError as exc:
        raise ValueError("复核文件超出本作业目录") from exc
    return path


def _scope(directory):
    """Old mutable-index workflow freezes cannot safely acquire derivatives."""
    for parent in (directory, *directory.parents):
        path = parent / "workflow.json"
        if not path.is_file():
            continue
        workflow = json.loads(path.read_text(encoding="utf8"))
        # The workflow worker itself applies the meteor default right after
        # publishing the base photograph, while still holding its run lock.
        worker_addback = (workflow.get("trail_addback_pending") is True and workflow.get("state") == "running"
                          and is_locked(parent / "worker.lock"))
        if workflow.get("state") != "succeeded" and not worker_addback:
            raise ValueError("完整出图尚未成功结束，不能应用亮迹复核")
        checkpoint = (workflow.get("checkpoints") or {}).get("stack") or {}
        snapshots = [f for f in checkpoint.get("files", [])
                     if Path(str(f.get("path", ""))).name.startswith("stack_snapshot-")]
        if not snapshots:
            raise ValueError("旧任务未冻结独立堆栈快照，只能读取；请使用新的完整出图任务")
        for item in snapshots:
            snapshot = _within(item["path"], parent)
            digest = hashing.sha256_file(snapshot, source="trail_review_session._scope")
            if not item.get("sha256") or digest != item["sha256"]:
                raise ValueError("完整任务的堆栈快照缺失或已改变")
        break


def _workflow_record(directory):
    """The enclosing workflow record, if this job lives inside one; never trusted for pixels."""
    for parent in Path(directory).resolve().parents:
        path = parent / "workflow.json"
        if not path.is_file():
            continue
        try:
            record = json.loads(path.read_text(encoding="utf8"))
        except (OSError, UnicodeError, ValueError):
            return parent, {}
        return parent, record if isinstance(record, dict) else {}
    return None


def _policy_version(directory):
    """New sessions follow the workflow's submitted trail default.

    A workflow submitted without trail_default=meteor (older records and older
    callers) keeps the explicit-keep-only contract; a standalone job uses the
    current meteor default.
    """
    found = _workflow_record(directory)
    if found is None:
        return METEOR_POLICY_VERSION
    request = found[1].get("request") or {}
    return METEOR_POLICY_VERSION if request.get("trail_default") == "meteor" else EXPLICIT_POLICY_VERSION


def _options(options, current=None):
    if options is None:
        return {**DEFAULT_OPTIONS, **(current or {})}
    if type(options) is not dict or set(options) - set(DEFAULT_OPTIONS) \
            or any(type(value) is not bool for value in options.values()):
        raise ValueError("复核选项须为对象，且只包含布尔值 export_trail_layer")
    return {**DEFAULT_OPTIONS, **(current or {}), **options}


def _verified_jpeg(items, name="final.jpg"):
    """A delivery JPEG bound by the workflow record's hash, or None (never an unverified file)."""
    for item in items or []:
        if not isinstance(item, dict) or item.get("name") != name:
            continue
        try:
            path = Path(item["path"]).resolve(strict=True)
            if path.is_file() and item.get("sha256") and _file_digest(path) == item["sha256"]:
                return str(path)
        except (OSError, KeyError, TypeError, ValueError):
            return None
    return None


def _current(directory):
    from .core import cache_cleanup_refusal, gc_refusal, load_job
    from .transient_stage import read_record
    job = load_job(str(directory))
    if job.get("state") != "succeeded":
        raise ValueError("请等出图结束后再复核亮迹")
    # Once the frames are gone a review can no longer be applied (nor its neighbour crops shown): refuse
    # opening, saving and applying up front, not after the user finished reviewing.
    refusal = gc_refusal(job, "复核亮迹") or cache_cleanup_refusal(directory, "复核亮迹")
    if refusal:
        raise ValueError(refusal)
    revision = job.get("transient_revision")
    if not isinstance(revision, dict) or not revision.get("sha256"):
        raise ValueError("此任务没有可复核的不可变亮迹记录；需开启亮迹检测后重新出图")
    record = read_record(directory / "work", job=job)
    if record is None:
        raise ValueError("此任务没有亮迹候选记录")
    return job, record


def _identity(job, record):
    inputs = job.get("inputs") or []
    identities = []
    ids = set()
    for candidate in record.get("candidates", []):
        cid = candidate.get("id")
        if not isinstance(cid, str) or not cid or cid in ids:
            raise ValueError("候选编号缺失或重复")
        ids.add(cid)
        provenance = candidate.get("provenance") or {}
        index = provenance.get("frame_index")
        if isinstance(index, bool) or not isinstance(index, int) or not 1 <= index <= len(inputs):
            raise ValueError("候选缺少明确的原始曝光身份")
        source = provenance.get("input") or {}
        expected = inputs[index - 1]
        if not source.get("sha256") or any(source.get(key) != expected.get(key) for key in ("path", "sha256")):
            raise ValueError("候选原片身份与作业输入不符")
        if candidate.get("stem") != f"light_{index:05d}":
            raise ValueError("候选帧序号与原片身份不符")
        identities.append({"id": cid, "source": source, "frame_index": index,
                           "geometry": candidate.get("geometry"), "measurement": candidate.get("measurement")})
    return {"job_signature": job.get("signature"), "inputs": inputs,
            "candidate_identity_sha256": _digest(identities), "candidate_ids": sorted(ids)}


def _session_path(directory, revision):
    return _within(directory / "review" / ("draft-" + _digest(revision)[:24] + ".json"), directory)


def _load(path):
    path = Path(path).resolve()
    value = json.loads(path.read_text(encoding="utf8"))
    if value.get("version") != VERSION:
        raise ValueError("不支持的亮迹复核草稿版本")
    directory = Path(value["job_dir"]).resolve()
    if path != _session_path(directory, value["base_revision"]).resolve():
        raise ValueError("复核草稿路径与作业修订身份不符")
    if value.get("session_path") != str(path):
        raise ValueError("复核草稿位置已改变")
    return value, directory


def _validate(value, job, record):
    if value["base_revision"] != job.get("transient_revision"):
        raise ValueError("亮迹记录已被另一操作更新，请重新打开复核；草稿不会套用到新编号")
    if value.get("binding") != _identity(job, record):
        raise ValueError("复核草稿的原片或候选位置绑定已改变")


def _geometry(directory, job):
    from .fits_crop import read_header
    path = directory / "work" / "sky_master.fit"
    _verify_file(path, _checkpoint_digest(directory, job, path), "整图坐标参考")
    header, _ = read_header(path)
    return {"width": int(header["NAXIS1"]), "height": int(header["NAXIS2"]),
            "row_order": "TOP-DOWN", "coordinates": "zero-based pixel centres"}


def _disposition(candidate, meteor):
    """The record's disposition before this draft, in the session's policy."""
    from .transient_stage import keep_view
    if candidate.get("override") == "keep":
        return "明确保留"
    if candidate.get("override") == "omit":
        return "明确省略"
    if meteor:
        recorded = keep_view(candidate, meteor_default=True)
        if recorded["default_keep"]:
            return "默认保留（流星）"
        if recorded["identity"] == "meteor":
            return "流星量测不可靠，默认不加回"
        return "默认省略"
    return "建议保留（未采用）" if candidate.get("decision") == "keep" else "默认省略"


def _view(value, job, record):
    from .transient_review import candidate_view
    from .transient_stage import keep_view
    result = copy.deepcopy(value)
    directory = Path(value["job_dir"])
    version = value.get("application_policy_version", EXPLICIT_POLICY_VERSION)
    meteor = version == METEOR_POLICY_VERSION
    decisions = value.get("decisions") or {}
    result["application_policy_version"] = version
    result["options"] = _options(None, value.get("options"))
    result["geometry"] = _geometry(directory, job)
    result["candidates"] = []
    for candidate in record["candidates"]:
        view = candidate_view(candidate)
        view["endpoints_native_reference"] = (candidate.get("geometry") or {}).get("native_reference")
        view["frame_index"] = candidate["provenance"]["frame_index"]
        view["input_sha256"] = candidate["provenance"]["input"]["sha256"]
        view["explicitly_kept"] = candidate.get("override") == "keep"
        view["review_disposition"] = _disposition(candidate, meteor)
        # identity/default_keep/effective_keep/keep_source include this draft's saved decisions.
        view.update(keep_view(candidate, meteor_default=meteor, decision=decisions.get(candidate["id"]),
                              reviewer=value.get("reviewer")))
        if view.get("review_png"):
            path = _within(view["review_png"], directory)
            expected = candidate.get("review_png_sha256")
            if expected:
                _verify_file(path, expected, "候选缩略图")
                view["review_png"] = str(path)
            else:
                view["review_png"] = None
                view["preview_note"] = "旧候选缩略图没有内容校验；请使用下方经过校验的实际曝光对照"
        result["candidates"].append(view)
    result["preview_path"] = None
    photo = ((job.get("diagnostics") or {}).get("color_export") or {}).get("photo") or {}
    if photo.get("jpeg") and photo.get("jpeg_sha256"):
        path = _within(photo["jpeg"], directory)
        _verify_file(path, photo["jpeg_sha256"], "整图预览")
        result["preview_path"] = str(path)
    result["base_preview_path"] = result["preview_path"]
    found = _workflow_record(directory)
    workflow = found[1] if found else {}
    if found and workflow.get("stack_job_dir") and Path(workflow["stack_job_dir"]).resolve() == directory.resolve():
        base_dir = workflow.get("base_delivery_dir") or workflow.get("delivery_dir")
        with_trails = bool(workflow.get("delivery_dir") and base_dir and workflow["delivery_dir"] != base_dir)
        base_items = workflow.get("base_artifacts") or (None if with_trails else workflow.get("artifacts"))
        result["base_preview_path"] = _verified_jpeg(base_items) or result["base_preview_path"]
        if with_trails:
            result["preview_path"] = _verified_jpeg(workflow.get("artifacts")) or result["preview_path"]
    result["notice"] = ("量测可靠的流星默认加回（算法建议，不认证身份）；其余类别只在明确保留时加回。未知不等于卫星。保留只加回已测量范围。"
                        if meteor else "自动建议不是身份认证；未知不等于卫星。保留只加回已测量范围。")
    return result


def open_review(job_dir: str) -> dict:
    """Open/reopen the draft for exactly the current immutable candidate revision."""
    directory = Path(job_dir).resolve()
    # Checking state before mkdir avoids making new directories for bad jobs.
    _current(directory)
    with file_lock(directory / "trail_review.lock"):
        job, record = _current(directory)
        path = _session_path(directory, job["transient_revision"])
        path.parent.mkdir(exist_ok=True)
        if path.is_file():
            value, _ = _load(path)
            _validate(value, job, record)
        else:
            value = {"version": VERSION, "job_dir": str(directory), "session_path": str(path),
                     "base_revision": copy.deepcopy(job["transient_revision"]),
                     "binding": _identity(job, record), "draft_revision": uuid.uuid4().hex,
                     "state": "editing", "reviewer": None, "decisions": {},
                     "application_policy_version": _policy_version(directory),
                     "options": dict(DEFAULT_OPTIONS)}
            atomic_json(path, value)
        return _view(value, job, record)


def read_review(session_path: str) -> dict:
    """Return a saved draft after checking the current source/revision binding."""
    value, directory = _load(session_path)
    job, record = _current(directory)
    _validate(value, job, record)
    return _view(value, job, record)


def save_review(session_path: str, decisions: dict, *, reviewer: str = "human",
                expected_draft_revision: str, options: dict | None = None) -> dict:
    """Replace the batch intent, never the source candidates; missing fields stay unchanged.

    ``options`` ({"export_trail_layer": bool}) is stored with the draft and
    becomes part of the published trails delivery's identity; None keeps it.
    """
    from .review_decisions import normalize_decisions
    if reviewer not in ("human", "agent"):
        raise ValueError("复核来源须明确为 human 或 agent")
    _, directory = _load(session_path)
    with file_lock(directory / "trail_review.lock"):
        value, _ = _load(session_path)
        if value.get("state") != "editing":
            raise ValueError("该草稿已应用或正在应用；请重新打开当前亮迹记录")
        if not expected_draft_revision or value.get("draft_revision") != expected_draft_revision:
            raise ValueError("草稿已在另一窗口更改，请重新打开后再编辑")
        job, record = _current(directory)
        _validate(value, job, record)
        clean = normalize_decisions(decisions, value["binding"]["candidate_ids"])
        clean_options = _options(options, value.get("options"))
        value.update(decisions=clean, reviewer=reviewer, options=clean_options, draft_revision=uuid.uuid4().hex)
        atomic_json(Path(session_path), value)
        return _view(value, job, record)


def apply_review(session_path: str, *, expected_revision: dict | None = None,
                 expected_draft_revision: str | None = None) -> dict:
    """Apply the saved batch once. Caller must publish through workflow_edit afterwards.

    Under meteor-default-keep-1 an empty draft still applies the default once
    (meteor keeps with algorithm provenance); later empty drafts change nothing.
    Legacy explicit-keep-only-1 sessions keep their original contract.
    """
    from .core import load_job, recompose_transients
    from .review_decisions import normalize_decisions
    from .transient_stage import METEOR_DEFAULT_POLICY, METEOR_DEFAULT_RULE, UNTRUSTWORTHY_FLAGS
    _, directory = _load(session_path)
    with file_lock(directory / "trail_review.lock"):
        value, _ = _load(session_path)
        if expected_draft_revision is not None and expected_draft_revision != value.get("draft_revision"):
            raise ValueError("应用前草稿已在另一窗口更改，请重新打开并确认选择")
        if expected_revision is not None and expected_revision != value["base_revision"]:
            raise ValueError("请求的亮迹修订与草稿不符")
        if value.get("state") == "applied":
            job = load_job(str(directory))
            if job.get("transient_revision") != value.get("applied_revision"):
                raise ValueError("已应用草稿后又产生新修订，请重新打开")
            return {"job": job, "session": copy.deepcopy(value), "changed": False}
        if value.get("state") != "editing":
            raise ValueError("上次应用尚未确认完成，请重新读取作业状态，避免重复应用")
        _scope(directory)
        job, record = _current(directory)
        _validate(value, job, record)
        decisions = normalize_decisions(value["decisions"], value["binding"]["candidate_ids"])
        meteor = value.get("application_policy_version", EXPLICIT_POLICY_VERSION) == METEOR_POLICY_VERSION
        already_default = (record.get("default_keep_rule") or {}).get("version") == METEOR_DEFAULT_RULE
        if not decisions and (not meteor or already_default):
            return {"job": job, "session": copy.deepcopy(value), "changed": False}
        if decisions and value.get("reviewer") not in ("human", "agent"):
            raise ValueError("草稿缺少明确的复核来源")
        _verify_inputs(job["inputs"])
        kwargs = {plural: {cid: item[field] for cid, item in decisions.items() if field in item}
                  for field, plural in (("label", "labels"), ("override", "overrides"), ("note", "notes"))}
        reviewer = value.get("reviewer") if decisions else None
        kwargs["reviewer"] = reviewer
        value.update(state="applying", application_id=uuid.uuid4().hex)
        if meteor:
            kwargs["policy"] = dict(METEOR_DEFAULT_POLICY)
            kwargs["default_keep_rule"] = METEOR_DEFAULT_RULE
            value["application_policy"] = {"version": METEOR_POLICY_VERSION, "policy": dict(METEOR_DEFAULT_POLICY),
                                           "default_keep_rule": METEOR_DEFAULT_RULE,
                                           "untrustworthy_quality_flags": list(UNTRUSTWORTHY_FLAGS),
                                           "reset_disposition": "policy", "classification_does_not_authorize_addback": False,
                                           "meteor_identity_authorizes_default_keep": True, "identity_certified": False,
                                           "applied_by": reviewer or "algorithm"}
        else:
            kwargs["policy"] = dict(REVIEW_POLICY)
            value["application_policy"] = {"version": EXPLICIT_POLICY_VERSION, "policy": dict(REVIEW_POLICY),
                                           "reset_disposition": "omit", "classification_does_not_authorize_addback": True,
                                           "applied_by": reviewer}
        atomic_json(Path(session_path), value)
        try:
            job = recompose_transients(str(directory), **kwargs, expected_revision=value["base_revision"])
        except Exception:
            # A post-publication error still cannot replay: expected_revision will reject.
            value["state"] = "editing"
            atomic_json(Path(session_path), value)
            raise
        value.update(state="applied", applied_revision=copy.deepcopy(job["transient_revision"]))
        atomic_json(Path(session_path), value)
        return {"job": job, "session": copy.deepcopy(value), "changed": True}


def candidate_context(session_path: str, candidate_id: str) -> dict:
    """Real adjacent-exposure cutouts with checkpoint-verified pixels, on demand.

    Neighbours mean preceding/following *input exposures*, including explicit
    unavailable slots when registration rejected them. They are not candidate
    list neighbours. Prefer registered frames when checkpoint digests exist;
    otherwise use verified converted exposures in fixed native coordinates,
    explicitly reporting that sky motion has not been aligned. All available
    panels use one shared grayscale stretch.
    """
    import numpy as np
    from PIL import Image
    from .transient_trails import Planes
    value, directory = _load(session_path)
    job, record = _current(directory)
    _validate(value, job, record)
    candidate = next((c for c in record["candidates"] if c["id"] == candidate_id), None)
    if candidate is None:
        raise ValueError("候选不在当前修订中")
    index = candidate["provenance"]["frame_index"]
    available_indices = [n for n in (index-1, index, index+1) if 1 <= n <= len(job["inputs"])]
    registered = all(_checkpoint_digest(directory, job, directory / "work" / f"r_light_{n:05d}.fit") for n in available_indices)
    points = (candidate.get("geometry") or {}).get("grid" if registered else "native_source")
    if points is None:
        return {"candidate_id": candidate_id, "panels": [], "notice": "候选未测量，缺少可靠位置，无法生成邻帧对照"}
    p = np.asarray(points, dtype=float)
    if p.shape != (2, 2) or not np.isfinite(p).all():
        raise ValueError("候选位置无效")
    geometry = _geometry(directory, job)
    radius = min(2048, max(64, float(np.max(np.abs(p[1] - p[0]))) * .6 + 24))
    cx, cy = np.mean(p, axis=0)
    x0, x1 = max(0, int(cx-radius)), min(geometry["width"], int(cx+radius)+1)
    y0, y1 = max(0, int(cy-radius)), min(geometry["height"], int(cy+radius)+1)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("候选位置在图像之外")
    panels, arrays = [], []
    for number, role in ((index-1, "previous"), (index, "source"), (index+1, "next")):
        panel = {"role": role, "frame_index": number, "available": False}
        panels.append(panel)
        if not 1 <= number <= len(job["inputs"]):
            panel["reason"] = "序列边界，没有相邻曝光"
            continue
        panel.update(input=job["inputs"][number-1]["path"], input_sha256=job["inputs"][number-1]["sha256"])
        stem = f"light_{number:05d}"
        frame = directory / "work" / (f"r_{stem}.fit" if registered else f"{stem}.fit")
        expected = _checkpoint_digest(directory, job, frame)
        if not frame.is_file():
            panel["reason"] = "此曝光未保留已校验的图像，不能冒充已有邻帧证据"
            continue
        if not expected:
            panel["reason"] = "此曝光缺少检查点内容校验，不能当作可靠复核证据"
            continue
        try:
            _verify_file(frame, expected, "曝光检查点")
        except ValueError as exc:
            panel["reason"] = str(exc)
            continue
        plane = Planes(frame)
        array = plane.crop(1, x0, x1, y0, y1)
        step = max(1, math.ceil(max(array.shape) / 420))
        arrays.append((panel, array[::step, ::step].copy()))
        # Verify after reading too; cache never becomes evidence for a moving file.
        _verify_file(frame, expected, "曝光检查点")
        panel.update(available=True, grid="registered TOP-DOWN" if registered else "native TOP-DOWN, unaligned",
                     checkpoint_sha256=expected, box=[x0, y0, x1, y1], downsample=step)
    finite = [array[np.isfinite(array) & (array != 0)] for _, array in arrays]
    samples = np.concatenate([a for a in finite if a.size]) if any(a.size for a in finite) else np.array([0., 1.])
    low, high = np.percentile(samples, [.5, 99.7])
    high = max(high, low + 1e-12)
    cache = _within(directory / "review" / "previews" / (_digest([value["base_revision"], candidate_id])[:24]), directory)
    cache.mkdir(parents=True, exist_ok=True)
    for panel, array in arrays:
        scaled = np.arcsinh(np.maximum(0, (np.nan_to_num(array)-low)/(high-low))*10) / np.arcsinh(10)
        pixels = np.uint8(np.clip(scaled, 0, 1)*255 + .5)
        path = cache / f"{panel['role']}.png"
        temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
        try:
            Image.fromarray(pixels).save(temporary, format="PNG")
            with hashing.writing(path):
                temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        panel["path"] = str(path)
    alignment = "实际配准曝光" if registered else "实际转换曝光的同一原图区域（未对齐，星空可随时间移动）"
    return {"candidate_id": candidate_id, "panels": panels, "notice": alignment + "，G 通道、同一显示拉伸；用于检查连续轨迹，不代表最终色彩"}


def list_trail_candidates(job_dir) -> dict:
    """Every recorded trail candidate of a finished stack job, in the compact review view, plus its trail artifacts.

    Read-only. Used by the desktop review panel; the imports stay function-local so tests can patch
    ``core.load_job`` and ``transient_stage.read_record``.
    """
    from .core import load_job
    from .transient_stage import read_record
    from .transient_review import candidate_view
    directory = Path(job_dir).resolve()
    job = load_job(str(directory))
    record = read_record(directory / "work", job=job)
    artifacts = [a for a in job.get("artifacts", []) if a.get("group") == "transients" or str(a.get("label", "")).startswith("含亮迹")]
    return {"state": "succeeded", "job_dir": str(directory), "candidates": [candidate_view(c) for c in (record or {}).get("candidates", [])],
            "artifacts": artifacts}
