"""Publish reviewed trails as a new delivery of an existing Aester workflow.

The frozen stack and finishing checkpoints remain the base for every edit.
Trail RGB is rendered with the base's frozen display transform; it is never
subtracted from a separately auto-stretched photograph. Signed display-code
differences are added after finishing with a common, reported headroom guard.

A trails version is published as ``delivery-trails-<n>kept-<token>`` beside
the untouched base. When nothing is kept (or the kept trails change no pixel)
no new image folder is written and the base stays the current photograph.
"""
from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
import json
import shutil
import uuid
import zlib

import numpy as np
import tifffile
from PIL import Image

from . import core, frame_pool, hashing, performance, profiling, workflow as wf
from .locking import file_lock, is_locked
from .records import atomic_json

VERSION = "workflow-reviewed-trails-1"
LAYER_NAME = "trails_layer_rgba16.tif"
LAYER_VERSION = "reviewed-trails-layer-1"
# P1c T-5c (docs/EFFICIENCY_P1C_PREREG_20261005.md §3.5): add_signed_trails' row blocks on the frame pool, at most 8
# workers; one 128-row block's memory measured at W=8280 (three int32 RGB inputs, float64 headroom ratios), scaled
# by rows x W (pair_unit_bytes).
PAIR_WORKER_CAP = 8
PAIR_UNIT_BYTES = 210_000_000          # measured 175.6 MB on F48-size images (serial unit peak)
PAIR_UNIT_WIDTH = 8280


def pair_unit_bytes(rows, width):
    """unit_bytes of one add_signed_trails row block (P1c T-5c, R17): measured at 128 rows x W=8280, scaled by
    rows x W, never below the measured value."""
    scaled = -(-PAIR_UNIT_BYTES * int(rows) * int(width) // (128 * PAIR_UNIT_WIDTH))
    return max(PAIR_UNIT_BYTES, scaled)


def add_signed_trails(source, background, finished, output, paired_background, *, rows=128, check=None):
    """F-Q=S-B, with Q equal to the finished image wherever T=S-B is zero.

    Only pixels with a nonzero trail contribution may use the common headroom
    guard. This avoids rolling back unrelated finished highlights or shadows.
    P1c T-5c: the row blocks run on the frame pool (each writes only its own
    rows; the two counts are folded in block order); ``check`` is passed only
    when the workflow worker owns the run (the automatic add-back).
    """
    arrays = (source, background, finished, output, paired_background)
    if any(a.dtype != np.uint16 or a.shape != source.shape or a.ndim != 3 or a.shape[-1] != 3 for a in arrays):
        raise ValueError("亮迹配对需要同尺寸 uint16 RGB")
    if any(not a.flags.writeable for a in (output, paired_background)):
        raise ValueError("配对输出不可写")
    if any(np.shares_memory(a,b) for i,a in enumerate(arrays) for b in arrays[i+1:]):
        raise ValueError("配对输入与输出不可共享内存")
    stats = {"residual_pixels": 0, "headroom_limited_pixels": 0, "residual_mismatched_channels": 0,
             "outside_residual_changed_channels": 0, "physical_flux_preserved": False}

    def block(y, unit):
        probe = unit.check if frame_pool.in_pool() else None   # R18: parallel path only
        sl = slice(y,y+rows)
        s = source[sl].astype(np.int32)
        if probe: probe()
        b = background[sl].astype(np.int32)
        if probe: probe()
        proposed = finished[sl].astype(np.int32)
        if probe: probe()
        t = s-b
        active = np.any(t != 0, axis=-1)
        change = proposed-b
        if probe: probe()
        head = np.divide(65534-np.maximum(s,b), change,
                         out=np.full(s.shape,np.inf), where=change>0)
        if probe: probe()
        foot = np.divide(np.minimum(s,b)-1, -change,
                         out=np.full(s.shape,np.inf), where=change<0)
        if probe: probe()
        alpha = np.clip(np.minimum(head,foot).min(-1),0,1)
        if probe: probe()
        alpha[np.any((s==0)|(s==65535)|(b==0)|(b==65535),axis=-1)] = 0
        if probe: probe()
        q = b+np.trunc(change*alpha[...,None]).astype(np.int32)
        if probe: probe()
        q[~active] = proposed[~active]
        f = q+t
        if probe: probe()
        if f.min()<0 or f.max()>65535 or q.min()<0 or q.max()>65535:
            raise ValueError("配对输出越界")
        if np.any((f-q)!=t):
            raise ValueError("亮迹有符号残差改变")
        if probe: probe()
        output[sl], paired_background[sl] = f.astype(np.uint16), q.astype(np.uint16)
        return int(active.sum()), int(np.count_nonzero(active & np.any(q!=proposed,axis=-1)))

    def fold(i, value):          # R2: the serial loop's two integer sums, in block order on the coordinator
        stats["residual_pixels"] += value[0]
        stats["headroom_limited_pixels"] += value[1]

    starts = range(0, source.shape[0], rows)
    unit_bytes = pair_unit_bytes(rows, source.shape[1])
    workers = performance.frame_workers(len(starts), unit_bytes, PAIR_WORKER_CAP, blas=False)
    pool_stats = {}
    with performance.operation("trail_pair"):
        frame_pool.run_ordered(block, starts, workers=workers, name="trail_pair", check=check, on_result=fold,
                               unit_bytes=unit_bytes, blas=False, stats=pool_stats)
    performance.note_pool("trail_pair", pool_stats)
    from . import resources
    resources.shadow_pool("py.trail_pair", actual=pool_stats.get("workers"), items=len(starts),
                          width=-(-source.shape[1]*rows//128))
    return stats


def _frozen_final(stack, finishing, *, directory):
    """Resolve through authenticated checkpoints, rejecting a mutable index swap."""
    reduction=wf._cached(directory,"reduction")
    milkyway=wf._cached(directory,"milkyway")
    if finishing.get("reduction")!=reduction or finishing.get("milkyway")!=milkyway:
        raise ValueError("后期索引与冻结阶段不一致")
    if reduction:
        root = Path(reduction["processing_directory"])
        return {"tiff":str(root/"star_reduced_srgb16.tif"), "jpeg":str(root/"star_reduced.jpg")}
    if milkyway:
        expected={"tiff":milkyway["artifacts"]["milkyway_srgb16.tif"]["path"],
                  "jpeg":milkyway["artifacts"]["milkyway_srgb.jpg"]["path"]}
        if finishing.get("final_photo")!=expected:
            raise ValueError("最终显色索引与冻结银河阶段不一致")
        return expected
    if finishing.get("final_photo") is not None:
        raise ValueError("最终显色索引没有对应冻结阶段")
    return stack["colour"]["photo"]


def _copy_verified(source,target):
    """Copy a frozen file; the bytes copied are hashed and the copy is read back for real.

    In a hash run scope (not "off") the copy digest proves exactly the bytes
    written came from the frozen source, so the source needs no separate
    re-verification afterwards; otherwise the former copy, read-back and
    source re-verification.
    """
    frozen=wf._file(source)
    if not hashing.digest_writes():
        with hashing.writing(target):
            shutil.copyfile(source,target)
        if core._sha(Path(target))!=frozen["sha256"] or Path(target).stat().st_size!=frozen["bytes"]:
            raise ValueError("复核成片复制校验失败")
        wf._verify_files([frozen])
        return
    copied=hashing.copy_with_digest(source,target,source="workflow_edit._copy_verified",expected=frozen["sha256"],
                                    exclusive=False,fsync=False)
    if copied.dst_sha256!=frozen["sha256"] or Path(target).stat().st_size!=frozen["bytes"]:
        raise ValueError("复核成片复制校验失败")


def _base_render_params(stack):
    from .display_transform import Params
    record=json.loads(Path(stack["colour"]["photo"]["record"]).read_text(encoding="utf-8"))
    params=Params.from_json(record["metadata"]["display_transform"])
    if params.look not in ("none","camera_hsm"):     # camera_hsm is carried entirely by the recorded parameters
        raise ValueError("此旧配方没有冻结外部 DCP 路径，不能可靠重放；请使用当前完整工作流")
    return params,record


def _render_fixed(directory, stack, job):
    """Render the current reviewed M with the immutable base D parameters."""
    from . import display_transform as dt, photo_render as pr, master_export as me
    colour = ((job.get("diagnostics") or {}).get("transients") or {}).get("color_export") or {}
    if not ((job.get("diagnostics") or {}).get("transients") or {}).get("composite"):
        return None
    if colour.get("state") != "succeeded":
        raise ValueError("含亮迹版本未成功完成相机显色，不能混用未标定预览")
    expected = colour["master"]["sha256"]
    master = wf._local(colour["master"]["path"], directory.parent)
    if wf._sha(master, "workflow_edit._render_fixed") != expected:
        raise ValueError("含亮迹母版发生改变")
    base = stack["colour"]
    from .photographic_job import _close_map
    base_array, base_meta, _ = me.read_master(base["master"]["path"])
    new_array, new_meta, _ = me.read_master(master)
    same_shape = base_array.shape == new_array.shape
    _close_map(base_array); _close_map(new_array)
    if not same_shape:
        raise ValueError("含亮迹母版与基础母版几何不一致")
    if base_meta["gamut"] != new_meta["gamut"] or base_meta["colour_basis"] != new_meta["colour_basis"]:
        raise ValueError("含亮迹母版与基础母版的相机颜色基底不同")
    params,photo_record = _base_render_params(stack)
    path = (Path(job["job_dir"])/"sky_mask.png") if job.get("mask") else None
    if path is not None and wf._sha(path, "workflow_edit._render_fixed")!=job["mask"]["sha256"]:
        raise ValueError("作业内天空蒙版与冻结请求不同")
    result = pr.render_photo(master, directory/"fixed-colour", params, sky_mask=path,
                             recipe_label=photo_record["metadata"]["recipe"])
    if wf._sha(master, "workflow_edit._render_fixed") != expected:
        raise ValueError("含亮迹母版在渲染期间改变")
    return result


def _write_trail_layer(path, source, background, finished, final, icc, meta, *, rows=64):
    """RGBA16 layer: the with-trails pixels where T=S-B is nonzero, alpha 65535 there and 0 elsewhere.

    A normal blend over the base final_srgb16.tif reproduces the with-trails
    final_srgb16.tif bit-exactly; that is verified strip by strip after
    writing. Encoded-sRGB display codes, not scene-linear light.
    """
    height, width = final.shape[:2]
    counted = {"pixels": 0}
    layer_meta = {**meta, "operation": LAYER_VERSION, "alpha": "binary-65535-where-signed-trail-difference-nonzero",
                  "blend": "normal-100-percent-over-base-final_srgb16", "reproduces": "trails final_srgb16.tif bit-exactly"}

    def strips():
        for y in range(0, height, rows):
            active = np.any(np.asarray(source[y:y+rows]) != np.asarray(background[y:y+rows]), axis=-1)
            block = np.zeros((active.shape[0], width, 4), np.uint16)
            block[..., :3][active] = np.asarray(final[y:y+rows])[active]
            block[..., 3][active] = 65535
            counted["pixels"] += int(active.sum())
            yield zlib.compress(block.astype("<u2", copy=False).tobytes(), 6)

    tifffile.imwrite(path, strips(), shape=(height, width, 4), dtype=np.uint16, photometric="rgb",
                     extrasamples=["unassalpha"], compression="zlib", predictor=False, rowsperstrip=rows,
                     iccprofile=icc, metadata=layer_meta)
    with tifffile.TiffFile(path) as tf:
        page = tf.pages[0]
        if len(tf.pages) != 1 or page.shape != (height, width, 4) or page.dtype != np.uint16:
            raise ValueError("亮迹图层写出格式不符")
        covered = 0
        for segment, index, _ in page.segments(sort=True):
            y, data = index[2], segment[0]
            alpha = data[..., 3:4].astype(np.uint64)
            if np.any((alpha != 0) & (alpha != 65535)):
                raise ValueError("亮迹图层透明度不是二值")
            base = np.asarray(finished[y:y+data.shape[0]]).astype(np.uint64)
            blended = (data[..., :3].astype(np.uint64) * alpha + base * (65535 - alpha)) // 65535
            if not np.array_equal(blended, np.asarray(final[y:y+data.shape[0]])):
                raise ValueError("亮迹图层叠加不能精确重现含亮迹成片")
            covered += data.shape[0]
        if covered != height:
            raise ValueError("亮迹图层行数不完整")
    return {"file": LAYER_NAME, "trail_pixels": counted["pixels"], "blend_verified": True,
            "encoding": "encoded sRGB display codes (same ICC as final_srgb16.tif); not scene-linear"}


TRAILS_JPEG_QUALITY = 97        # a trails version's final.jpg when its workflow froze no jpeg_quality (4:4:4)


def _write_pair(directory, stack, reviewed, finished, *, layer=False, check=None, quality=None):
    """The trails version's final_srgb16.tif, final.jpg and optional layer in ``directory``.

    ``quality``: the workflow's frozen ``jpeg_quality``; None (a workflow without it) keeps 97.
    """
    from .photographic_job import _display_source, _read_display, _close_map, _jpeg_rows
    shape, icc, _ = _display_source(Path(finished["tiff"]))
    inputs = [reviewed["tiff"], stack["colour"]["photo"]["tiff"], finished["tiff"]]
    identities = wf._files(inputs, source="workflow_edit._write_pair")
    if any(_display_source(Path(path))[0] != shape for path in inputs):
        raise ValueError("含亮迹版本与冻结成片尺寸不符")
    # The memory-mapped outputs are in-place writes: no ledger may answer for
    # them from before or during the write (closed after the maps below).
    with ExitStack() as writers:
        arrays=[]
        try:
            arrays = [_read_display(path) for path in inputs]
            meta={"roworder":"TOP-DOWN", "scene_linear":False,"display_encoding_applied":True,
                  "color_space":"encoded_sRGB","operation":VERSION,"physical_flux_preserved":False,
                  "signed_trail_source_sha256":identities[0]["sha256"],
                  "signed_trail_background_sha256":identities[1]["sha256"],
                  "finished_base_sha256":identities[2]["sha256"]}
            final=directory/"final_srgb16.tif"
            background=directory/"finished_background_srgb16.tif"
            for path in (final,background):
                writers.enter_context(hashing.writing(path))
                arrays.append(tifffile.memmap(path,shape=shape,dtype=np.uint16,photometric="rgb",iccprofile=icc,metadata=meta))
            stats=add_signed_trails(*arrays,check=check)
            for array in arrays[3:]:array.flush()
            # P1c T-5x (c): the same expression per 128-row block into one uint8 array (no whole-image floats)
            pool_stats={}
            pixels=_jpeg_rows(arrays[3],lambda rows:np.rint(rows.astype(np.float32)/257).astype(np.uint8),
                              pool_stats=pool_stats)
            performance.note_pool("jpeg_rows.trail_pair",pool_stats)
            from . import resources
            resources.shadow_pool("py.jpeg_rows",actual=pool_stats.get("workers"),items=-(-shape[0]//128),
                                  width=shape[1])
            Image.fromarray(pixels).save(directory/"final.jpg",quality=TRAILS_JPEG_QUALITY if quality is None else quality,
                                         subsampling=0,icc_profile=icc)
            if layer and stats["residual_pixels"]:
                stats["trail_layer"]=_write_trail_layer(directory/LAYER_NAME,arrays[0],arrays[1],arrays[2],arrays[3],icc,meta)
            wf._verify_files(identities)
            return stats
        finally:
            for array in arrays:_close_map(array)


def _relocate_records(stage, target):
    """Keep the internal colour replay record usable after atomic rename."""
    old, new = str(stage), str(target)
    def visit(value):
        if isinstance(value,str) and value.startswith(old):
            return new+value[len(old):]
        if isinstance(value,list):return [visit(item) for item in value]
        if isinstance(value,dict):return {key:visit(item) for key,item in value.items()}
        return value
    for path in stage.rglob("*.json"):
        with hashing.writing(path):
            atomic_json(path,visit(json.loads(path.read_text(encoding="utf-8"))))


_KINDS = {".tif": "tiff", ".tiff": "tiff", ".jpg": "jpeg", ".png": "png", ".json": "json", ".txt": "text"}
LAYER_LABEL = "亮迹图层（16 位 RGBA，叠在基础成片上）"


def apply_review(workflow_dir, session_path, *, expected_revision=None, expected_draft_revision=None):
    """Apply one saved review batch and atomically publish a new delivery.

    Calling again with an applied draft reuses its published delivery. If a
    render failed after the child revision was saved, retry renders that same
    child revision without applying labels a second time. Besides the updated
    workflow the result reports kept_count/kept_ids, published (a new image
    folder was written), delivery_dir (that folder or None), identical_to_base
    and trail_layer_path.
    """
    directory=Path(workflow_dir).resolve(strict=True)
    with file_lock(directory/"edit.lock",timeout=30):
        if is_locked(directory/"worker.lock"):
            raise ValueError("工作流仍在运行，请完成后复核亮迹")
        return _apply_locked(directory,session_path,expected_revision=expected_revision,
                             expected_draft_revision=expected_draft_revision,automatic=False)


def apply_default_trails(workflow_dir):
    """Worker-only: add back the meteor default right after the base publication.

    The workflow worker still holds worker.lock and its record is running with
    trail_addback_pending, so no reviewer can apply concurrently. Only
    measured, trustworthy meteor candidates are added, with algorithm
    provenance; every other candidate stays for review.
    """
    from . import trail_review_session as review
    directory=Path(workflow_dir).resolve(strict=True)
    with file_lock(directory/"edit.lock",timeout=30):
        record=wf._read(directory)
        if record.get("trail_addback_pending") is not True or record.get("state")!="running":
            raise ValueError("工作流没有待自动加回的流星")
        if record["request"].get("trail_default")!="meteor":
            raise ValueError("此工作流未选择默认加回流星")
        profiling.switch(".review_open")
        session=review.open_review(record["stack_job_dir"])
        if session.get("application_policy_version")!=review.METEOR_POLICY_VERSION:
            raise ValueError("复核草稿不是流星默认保留规则")
        if session.get("decisions"):
            # A reviewer's saved draft is never applied under algorithm provenance.
            raise ValueError("复核草稿已有待应用的选择；自动加回让位于人工或 Agent 复核")
        return _apply_locked(directory,session["session_path"],expected_draft_revision=session["draft_revision"],
                             automatic=True)


def _trail_summary(job,kept_ids,*,policy,applied_by,reviewer,automatic):
    from .transient_stage import keep_source, read_record
    diag=((job.get("diagnostics") or {}).get("transients") or {})
    try:
        candidates=(read_record(Path(job["job_dir"])/"work",job=job) or {}).get("candidates") or []
    except ValueError:
        candidates=[]
    by_id={c.get("id"):c for c in candidates if isinstance(c,dict)}
    kept_by={"algorithm":0,"human":0,"agent":0}
    for cid in kept_ids:
        source=keep_source(by_id[cid]) if cid in by_id else None
        if source in kept_by:
            kept_by[source]+=1
    pending=len((((diag.get("summary") or {}).get("recall_review") or {}).get("candidate_ids")) or [])
    return {"state":"needs_review" if pending else "reviewed","requested":True,
            "automatic_addback":bool(kept_by["algorithm"] or (automatic and kept_ids)),
            "default_policy":policy,"applied_by":applied_by,"kept_count":len(kept_ids),"kept_ids":list(kept_ids),
            "kept_by":kept_by,"pending_count":pending,"job_dir":job["job_dir"],"revision":job.get("transient_revision"),
            "reviewer":reviewer,"scan":diag}


def _messages(summary,*,published,automatic):
    kept,pending=summary["kept_count"],summary["pending_count"]
    if published:
        text=(f"已自动加回 {kept} 条量测可靠的流星候选（算法建议，不认证身份）。" if automatic
              else f"已按复核加回 {kept} 条亮迹并另存成片。")
        message=(f"成片已保存，并自动加回 {kept} 条流星候选；不含亮迹的基础版另存保留。" if automatic
                 else "亮迹复核版本已另存；原母版和原成片保留")
    elif kept:
        text=message="保留的亮迹没有带来可见的像素变化；成片与基础版相同，未另存新文件夹。"
    elif automatic:
        text="未发现量测可靠的流星候选；成片未加入亮迹。"
        message="基础成片已保存；"+text
    else:
        text=message="本次没有保留任何亮迹；当前成片为不含亮迹的基础版，未另存新文件夹。"
    if pending:
        text+=f"另有 {pending} 条候选待复核；自动省略不代表已排除流星。"
    return text,message


def _finalize_worker(latest,directory):
    """End the worker's automatic add-back: the workflow is complete with or without trails."""
    latest.pop("trail_addback_pending",None)
    (directory/"cancel.request").unlink(missing_ok=True)
    latest.update(state="succeeded",stage="complete",progress=100,error=None)


def _base_artifacts(latest,base):
    if latest.get("base_artifacts"):
        return latest["base_artifacts"]
    root=Path(base)
    paths=[p for p in sorted(root.iterdir()) if p.is_file()]
    return [{**record,"path":str(root/p.name),"name":p.name,"kind":_KINDS.get(p.suffix.lower(),"file"),"label":p.name}
            for p,record in zip(paths,wf._files(paths,source="workflow_edit._base_artifacts"))]


def _show_base(latest,summary,*,automatic,directory):
    """No new image folder: the untouched base photograph is the current one."""
    text,message=_messages(summary,published=False,automatic=automatic)
    base=latest.get("base_delivery_dir",latest["delivery_dir"])
    if latest["delivery_dir"]!=base:
        latest.update(delivery_dir=base,artifacts=_base_artifacts(latest,base))
    latest["base_delivery_dir"]=base
    latest["extras"]=[item for item in latest.get("extras",[]) if item.get("role")!="trails_layer"]
    latest.update(trail_review={**summary,"message":text,"trail_layer":None,"identical_to_base":True},
                  needs_review=summary["pending_count"]>0,message=message)
    if automatic:
        _finalize_worker(latest,directory)


def _commit(directory,record,job,revision,update):
    """Apply update(latest) under the child writer lock and the metadata lock, after re-checking the revision.

    A concurrent advanced review cannot race the final identity check and
    leave the workflow claiming a newer child revision.
    """
    with file_lock(Path(job["job_dir"])/"worker.lock",timeout=30), file_lock(directory/"metadata.lock",timeout=30):
        if core.load_job(record["stack_job_dir"]).get("transient_revision")!=revision:
            raise ValueError("发布锁内亮迹修订发生改变，请重新读取候选")
        latest=wf._read(directory)
        if latest["signature"]!=record["signature"]:
            raise ValueError("发布前工作流参数改变")
        update(latest)
        latest["updated_at"]=core.now()
        atomic_json(directory/"workflow.json",latest)
    return latest


def _outcome(latest,result,delivery,summary,*,reused=False,published=False,identical=False,layer=None):
    return {"workflow":latest,"review":result,"delivery":delivery,"reused":reused,
            "kept_count":summary["kept_count"],"kept_ids":list(summary["kept_ids"]),"published":published,
            "delivery_dir":delivery["delivery_dir"] if published else None,
            "identical_to_base":identical,"trail_layer_path":layer}


def _slim(directory):
    """Drop the cached pixel copies of an externally published version; keep its JSON records."""
    for path in sorted(Path(directory).rglob("*"),reverse=True):
        try:
            if path.is_file() and path.suffix.lower()!=".json":
                path.unlink()
        except OSError:
            pass


def _apply_locked(directory,session_path,*,expected_revision=None,expected_draft_revision=None,automatic):
    # Diagnostics only: the worker's timer (when active) sees the same steps
    # under trail_addback.*; this apply's own timer is stored with the
    # review_deliveries entry of a published version (never in its receipt).
    profiling.switch(".entry_verify")
    timer=profiling.StageTimer(initial="entry_verify")
    started_at=core.now()
    # One hashing run scope per apply (inside the workflow worker it is the
    # worker's own scope, shared).
    with profiling.activate(timer), hashing.run_scope("workflow_edit.apply"):
        return _apply_body(directory,session_path,timer,started_at,expected_revision=expected_revision,
                           expected_draft_revision=expected_draft_revision,automatic=automatic)


def _apply_timings(timer,started_at):
    """Timestamps and stage timings of one apply, for its review_deliveries entry.

    Called after the irreversible publication rename, so it must never raise:
    diagnostics that cannot be collected are simply left out.
    """
    entry={"started_at":started_at,"finished_at":core.now()}
    try:
        hashing.note_scope_report()   # this apply's hash ledger use (reads, hits, shadow checks): never raises
        timings=timer.finish()
        timings.update(scope="trail_apply",state="published")
        entry["timings"]=timings
    except Exception:
        pass
    return entry


def _apply_body(directory,session_path,timer,started_at,*,expected_revision=None,expected_draft_revision=None,automatic):
    from . import trail_review_session as review
    # P1c §3.5: the pair and its JPEG run on the frame pool; the add-back-pending branch of the workflow worker skips
    # the admission check, so a malformed AESTER_FRAME_WORKERS / AESTER_OVERLAP is refused here, before any work
    frame_pool.validate_environment()
    record=wf._read(directory)
    external=record["request"].get("delivery_root") is not None
    if automatic:
        if record.get("trail_addback_pending") is not True or record["state"]!="running":
            raise ValueError("工作流没有待自动加回的流星")
    elif record["state"]!="succeeded":
        raise ValueError("只有已完成工作流可以保存亮迹版本")
    wf._verify(record)
    if not record["request"].get("trails_enabled"):
        raise ValueError("此工作流未开启亮迹复核")
    session=Path(session_path).resolve(strict=True)
    if not session.is_relative_to(Path(record["stack_job_dir"]).resolve()):
        raise ValueError("复核草稿不属于本工作流")
    stack=wf._cached(directory,"stack")
    if stack["colour"].get("contract")=="legacy_input_RGB":
        raise ValueError("当前亮迹成片复核仅支持已标定的相机 RAW 工作流；未标定 FITS 可保留候选记录")
    _base_render_params(stack)
    finished=_frozen_final(stack,record.get("finishing") or {},directory=directory)
    wf._local(finished["tiff"],directory)
    with profiling.stage(".recompose"):
        result=review.apply_review(str(session),expected_revision=expected_revision,
                                   expected_draft_revision=expected_draft_revision)
    profiling.switch(".reuse_check")
    value=result["session"]
    applied=value.get("state")=="applied"
    policy=value.get("application_policy_version") or review.EXPLICIT_POLICY_VERSION
    meteor=policy==review.METEOR_POLICY_VERSION
    options={**review.DEFAULT_OPTIONS,**(value.get("options") or {})}
    applied_by=("algorithm" if automatic else
                (value.get("application_policy") or {}).get("applied_by") or value.get("reviewer"))
    current=record.get("trail_review") or {}
    if not applied and not meteor and not automatic:
        # Explicit-keep-only drafts without a change leave the workflow as it is.
        base=record.get("base_delivery_dir",record.get("delivery_dir"))
        return _outcome(record,result,None,{"kept_count":int(current.get("kept_count") or 0),
                                             "kept_ids":list(current.get("kept_ids") or [])},
                        identical=record.get("delivery_dir")==base)
    job=result["job"]
    wf.validate_stack_binding(record,job)
    revision=job.get("transient_revision")
    diag=((job.get("diagnostics") or {}).get("transients") or {})
    composed=bool(diag.get("composite"))
    kept_ids=[item["id"] for item in diag.get("kept") or [] if isinstance(item,dict) and item.get("id")] if composed else []
    summary=_trail_summary(job,kept_ids,policy=policy,applied_by=applied_by,reviewer=value.get("reviewer"),automatic=automatic)
    key_source={"revision":revision,"signature":record["signature"]}
    if options["export_trail_layer"]:
        key_source["options"]=options
    key=wf._digest(key_source)
    for prior in record.get("review_deliveries",[]):
        if prior["key"]==key:
            if external:
                from .storage import delivery_file
                for item in prior["artifacts"]:
                    delivery_file(record,item["path"])
            wf._verify_files(prior["artifacts"])
            latest=record
            profiling.switch(".commit")
            if automatic:
                latest=_commit(directory,record,job,revision,lambda current_record:_finalize_worker(current_record,directory))
            return _outcome(latest,result,prior,summary,reused=True,layer=prior.get("trail_layer"))
    if not composed:
        if not applied and not automatic:
            base=record.get("base_delivery_dir",record.get("delivery_dir"))
            return _outcome(record,result,None,summary,identical=record.get("delivery_dir")==base)
        profiling.switch(".commit")
        latest=_commit(directory,record,job,revision,
                       lambda current_record:_show_base(current_record,summary,automatic=automatic,directory=directory))
        return _outcome(latest,result,None,summary,identical=True)
    token=uuid.uuid4().hex[:12]
    stage=directory/(".partial-reviewed-"+token)
    target=directory/("delivery-trails-%dkept-%s"%(len(kept_ids),token))
    from .photographic_job import _discard_failed, _discard_partial, _sweep_discarded
    _sweep_discarded(directory,".partial-reviewed-*")
    stage.mkdir()
    # P1c T-5c: only the worker's own automatic add-back can be stopped inside the pair; a manual GUI / MCP apply
    # takes no cancel point (a stale cancel.request must not fail it)
    check=(lambda:wf._cancel_check(directory)) if automatic else None
    try:
        profiling.switch(".render_fixed")
        rendered=_render_fixed(stage,stack,job)
        if rendered is None:
            raise ValueError("含亮迹合成缺失，不能生成亮迹版本")
        profiling.switch(".write_pair")
        quality=record["request"].get("jpeg_quality")    # frozen with the workflow; absent in older workflows
        if quality is not None:
            from .jpeg_estimate import validate_quality
            validate_quality(quality)
        stats=_write_pair(stage,stack,rendered,finished,layer=options["export_trail_layer"],check=check,quality=quality)
        if not stats["residual_pixels"]:
            # Kept trails that change no pixel must not produce an identical
            # copy that looks like a separate trails version.
            profiling.switch(".commit")
            shutil.rmtree(stage,ignore_errors=True)
            latest=_commit(directory,record,job,revision,
                           lambda current_record:_show_base(current_record,summary,automatic=automatic,directory=directory))
            return _outcome(latest,result,None,summary,identical=True)
        # Keep the original scene-linear mother, not an aesthetic or trail
        # derivative relabelled as a new camera master. External deliveries
        # read it from the frozen stack; only the cache-local layout copies it.
        profiling.switch(".stage_delivery")
        master_name="master_scene_linear.tif"
        sources={}
        extra_names={}
        if external:
            sources[master_name]=stack["colour"]["master"]["path"]
        else:
            _copy_verified(stack["colour"]["master"]["path"],stage/master_name)
        for field,name,kind in (("recipe","master_recipe.json","json"),("mask","camera_saturation_mask.png","png")):
            if stack["colour"]["master"].get(field):
                if external:
                    sources[name]=stack["colour"]["master"][field]
                else:
                    _copy_verified(stack["colour"]["master"][field],stage/name)
                extra_names[name]=(kind,name)
        layer=stats.get("trail_layer")
        if layer:
            layer={**layer,"sha256":wf._sha(stage/LAYER_NAME,"workflow_edit._apply_body"),"kept_ids":summary["kept_ids"],
                   "base_final_sha256":wf._file(finished["tiff"])["sha256"],
                   "trails_final_sha256":wf._sha(stage/"final_srgb16.tif","workflow_edit._apply_body"),
                   "provenance":{"applied_by":applied_by,"kept_by":summary["kept_by"],"application_policy_version":policy}}
        receipt={"version":VERSION,"key":key,"state":"succeeded","base_signature":record["signature"],
                 "source_revision":revision,"session_path":str(session),"reviewer":value.get("reviewer"),
                 "base_finishing":record.get("finishing"),"paired_statistics":stats,
                 "color_recipe":"frozen-base-display-transform","master_unchanged":True,
                 "geometric_scope":"same native stacked reference", "physical_flux_preserved":False,
                 "application_policy_version":policy,"applied_by":applied_by,"options":options,
                 "kept_ids":summary["kept_ids"],"kept_count":summary["kept_count"],"kept_by":summary["kept_by"],
                 "identity_certified":False,"delivery_name":target.name,"trail_layer":layer}
        if quality is not None:      # the workflow's frozen jpeg_quality; absent: final.jpg at the historical 97
            receipt.update(jpeg_quality=quality,jpeg_subsampling="4:4:4")
        if external:
            receipt["display_transform"]=_base_render_params(stack)[0].to_json()
            receipt["color_adjustments"]=record["request"].get("color_adjustments")
            if record["request"].get("delivery_format")=="aester-delivery-2":
                receipt["master_pixel_bits_unchanged"]=True
                receipt["master_container_repackaged"]=True
        atomic_json(stage/"review_delivery.json",receipt)
        profiling.switch(".verify")
        wf._verify(record)
        if core.load_job(record["stack_job_dir"]).get("transient_revision")!=revision:
            raise ValueError("发布前亮迹修订又发生改变，请重新读取候选")
        profiling.switch(".stage_delivery")
        names={master_name:("tiff","后期母版"),"final_srgb16.tif":("tiff","复核后 16 位成片"),
               "final.jpg":("jpeg","复核后 JPEG"),"review_delivery.json":("json","亮迹复核记录")}
        names.update(extra_names)
        names["finished_background_srgb16.tif"]=("tiff","与亮迹配对的处理后背景")
        if layer:
            names[LAYER_NAME]=("tiff",LAYER_LABEL)
        package=None
        if external:
            # Rendering stays in the cache (its fixed-colour intermediate
            # is not a deliverable). Only the finished files cross roots.
            from .storage import stage_delivery
            package=stage_delivery(record,{name:{"path":sources.get(name,str(stage/name)),"kind":kind,"label":label}
                                            for name,(kind,label) in names.items()},target.name,
                                   check=lambda:wf._cancel_check(directory))
            publish_stage,target,artifacts=package["stage"],package["target"],package["artifacts"]
            profiling.switch(".verify")
            wf._verify(record)
        else:
            _relocate_records(stage,target)
            records=wf._files([stage/name for name in names],source="workflow_edit._apply_body")
            artifacts=[{**record,"path":str(target/name),"name":name,"kind":kind,"label":label}
                       for record,(name,(kind,label)) in zip(records,names.items())]
            publish_stage=stage
        cached_receipt=bool(external and package.get("receipt"))
        records_dir=stage.with_name("reviewed-"+token)
        layer_path=str(target/LAYER_NAME) if layer else None
        delivery={"key":key,"delivery_dir":str(target),"artifacts":artifacts,
                  "receipt":str((records_dir if cached_receipt else target)/"review_delivery.json"),
                  "kept_ids":summary["kept_ids"],"kept_count":summary["kept_count"],"applied_by":applied_by,
                  "application_policy_version":policy,"options":options,"trail_layer":layer_path}
        text,message=_messages(summary,published=True,automatic=automatic)

        def publish(latest):
            if external:
                from .storage import publication_parent, verify_staged
                if publication_parent(latest)!=target.parent:
                    raise ValueError("作品发布位置改变")
                verify_staged(package)
                wf._cancel_check(directory)
            publish_stage.rename(target)
            if external:
                try:
                    stage.rename(records_dir)
                except OSError:
                    delivery["receipt"]=str(stage/"review_delivery.json") if cached_receipt else delivery["receipt"]
            if cached_receipt:
                latest.setdefault("publication_receipts",{})[str(target)]=package["receipt"]
            base=latest.get("base_delivery_dir",latest["delivery_dir"])
            if not latest.get("base_artifacts") and latest["delivery_dir"]==base:
                latest["base_artifacts"]=latest["artifacts"]
            extras=[item for item in latest.get("extras",[]) if item.get("role")!="trails_layer"]
            if layer_path:
                entry=next(item for item in artifacts if item["name"]==LAYER_NAME)
                extras.append({"role":"trails_layer","path":entry["path"],"sha256":entry["sha256"]})
            # Diagnostics in the workflow index only; the delivered receipt is already frozen.
            delivery.update(_apply_timings(timer,started_at))
            latest.update(base_delivery_dir=base,review_deliveries=[*latest.get("review_deliveries",[]),delivery],
                          delivery_dir=str(target),artifacts=artifacts,extras=extras,
                          trail_review={**summary,"message":text,"trail_layer":layer_path,"identical_to_base":False},
                          needs_review=summary["pending_count"]>0,message=message)
            if automatic:
                _finalize_worker(latest,directory)

        profiling.switch(".commit")
        latest=_commit(directory,record,job,revision,publish)
        if external:
            _slim(Path(delivery["receipt"]).parent if cached_receipt else (records_dir if records_dir.exists() else stage))
        return _outcome(latest,result,delivery,summary,published=True,layer=layer_path)
    except Exception as exc:
        if stage.exists():
            atomic_json(stage/"failure.json",{"state":"failed","error":str(exc)})
            # P1c §3.5: _write_pair has closed its maps and every pool has ended; a pair stopped by a block's error
            # may have written rows past it, so the staged pixels go (the records stay): one failed folder at any N
            _discard_failed(stage)
        raise
    except KeyboardInterrupt as exc:
        # WorkflowCancelled bypasses the clause above (P1c §3.5): mark the attempt failed as well, then remove the
        # staging folder -- every pool has ended and _write_pair has closed its maps -- so a stopped and resumed run
        # leaves the same tree as an uninterrupted one
        try:
            if stage.exists():atomic_json(stage/"failure.json",{"state":"cancelled","error":str(exc) or type(exc).__name__})
        except OSError:
            pass
        _discard_partial(stage)
        raise


def trail_rederive_scope(directory):
    """Whether trail layers may be re-derived inside the complete workflow that contains ``directory``.

    Returns None (allowed, or not inside a workflow) or a reason code: ``workflow_not_succeeded``,
    ``frozen_job_index`` (an older workflow froze the job index rather than an immutable snapshot) or
    ``snapshot_missing``. Read-only; the record is parsed here so front ends never read workflow.json.
    """
    directory=Path(directory)
    for parent in [directory, *directory.parents]:
        path = parent / "workflow.json"
        if not path.is_file():
            continue
        workflow = json.loads(path.read_text(encoding="utf8"))
        if workflow.get("state") != "succeeded":
            return "workflow_not_succeeded"
        checkpoint = (workflow.get("checkpoints") or {}).get("stack") or {}
        snapshot = (checkpoint.get("data") or {}).get("job_snapshot")
        files = checkpoint.get("files") or []
        matches = [item for item in files if Path(str(item.get("path", ""))).name.startswith("stack_snapshot-")
                   and Path(str(item.get("path", ""))).suffix == ".json"]
        if not matches or (snapshot and not any(Path(item["path"]) == Path(snapshot) for item in matches)):
            return "frozen_job_index"
        if not all(Path(item["path"]).is_file() for item in matches):
            return "snapshot_missing"
        return None
    return None

