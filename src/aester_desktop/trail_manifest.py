# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Wang Jian (王俭). See LICENSE.txt in this folder.
"""Composite list of a trails version: which exposure each added-back meteor came from.

Read only. The rows come from the open review session (``trail_review_session.open_review``), the
header from the stack job (``core.load_job``) and the workflow record the main window already holds.
The list is written as CSV and JSON to a place the user chooses, never into the works folder (its
delivery folders hold only the declared images). Files are named by file name only: no drive,
directory or user-profile path is written.
"""
from __future__ import annotations

import csv
import io
import json
import math
import re
from datetime import datetime
from pathlib import Path

from starstack import PRODUCT_VERSION
from starstack.i18n import tr

FORMAT = "aester-composite-list-1"
KINDS = ("algorithm", "human", "agent")
CLASSES = ("meteor", "satellite", "aircraft", "artifact", "unknown")
COLUMNS = ("id", "source_file", "input_sha256", "frame_index", "capture_time_camera_clock", "exposure_s",
           "x1", "y1", "x2", "y2", "source_x1", "source_y1", "source_x2", "source_y2",
           "identity", "identity_source", "keep_source")


def file_name(path) -> str:
    """The last component of a Windows or POSIX path, so no directory or drive survives."""
    return re.split(r"[\\/]", str(path))[-1] if path else ""


def _frame_number(stem):
    """light_00042 -> 42; None for anything else."""
    match = re.fullmatch(r"light_(\d+)", str(stem or ""))
    return int(match.group(1)) if match else None


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return value


def _point_pairs(points):
    if not isinstance(points, (list, tuple)) or len(points) != 2:
        return [None] * 4
    values = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            return [None] * 4
        values += [round(float(v), 1) if _number(v) is not None else None for v in point]
    return values


def published_keeps(session: dict, workflow: dict | None) -> list[str] | None:
    """The ids the published trails version added back, or None when this session has no such version.

    The workflow's trail review summary names them; it must describe the same candidate revision the
    session was opened on (a later apply makes a new revision and a new session).
    """
    review = (workflow or {}).get("trail_review") or {}
    kept = review.get("kept_ids")
    revision = (review.get("revision") or {}).get("id")
    base = (session.get("base_revision") or {}).get("id")
    if not isinstance(kept, list) or not revision or revision != base:
        return None
    delivery, base_folder = (workflow or {}).get("delivery_dir"), (workflow or {}).get("base_delivery_dir")
    if kept and (not delivery or delivery == base_folder):
        return None
    return [str(cid) for cid in kept]


def _published_identity(candidate):
    """(identity, source) in the published record; the session view layers saved drafts on other fields."""
    label = candidate.get("label")
    if label in CLASSES:
        source = candidate.get("label_source")
        return label, source if source in KINDS else "human"
    effective = candidate.get("effective_label")
    suggestion = (candidate.get("suggestion") or {}).get("class")
    return (effective if effective in CLASSES else suggestion if suggestion in CLASSES else "unknown"), "algorithm"


def _published_keep_source(candidate, drafted):
    if not drafted and candidate.get("keep_source") in KINDS:
        return candidate["keep_source"]
    if candidate.get("override") == "keep":
        source = candidate.get("override_source")
        return source if source in KINDS else "human"
    return _published_identity(candidate)[1]


def _not_scanned(job):
    """Frames left out of the trail scan, with the job's own reasons."""
    diagnostics = job.get("diagnostics") or {}
    inputs = job.get("inputs") or []
    detail = diagnostics.get("registration_detail") or {}
    gate = diagnostics.get("quality_gate") or {}
    reasons = {item.get("stem"): [str(r) for r in item.get("reasons") or []]
               for item in gate.get("excluded") or [] if isinstance(item, dict)}
    gated = set(detail.get("quality_excluded_frames") or gate.get("excluded_frames") or [])
    frames = []
    for stem in sorted(set(detail.get("unregistered_frames") or []) | gated):
        number = _frame_number(stem)
        source = inputs[number - 1].get("path") if number and 0 < number <= len(inputs) else None
        frames.append({"frame_index": number, "file": file_name(source),
                       "reason": "quality_gate" if stem in gated else "not_registered",
                       "details": reasons.get(stem, [])})
    transients = diagnostics.get("transients") or {}
    scanned = transients.get("frames_scanned")
    total = diagnostics.get("input_count") or len(inputs)
    result = {"count": (total - scanned) if isinstance(scanned, int) and isinstance(total, int) else None,
              "frames": frames}
    if isinstance(result["count"], int) and result["count"] != len(frames):
        result["unexplained"] = result["count"] - len(frames)
    budget = (transients.get("summary") or {}).get("detector_budget_exceeded_frames") or []
    if budget:
        result["scanned_without_candidates"] = [
            {"frame_index": _frame_number(stem), "reason": "detector_budget_exceeded"} for stem in budget]
    return result


def build(session: dict, job: dict, workflow: dict | None, *, now: datetime | None = None) -> dict:
    """{'header': ..., 'rows': [...]} for the published trails version of this session."""
    kept = published_keeps(session, workflow)
    if kept is None:
        raise ValueError(tr('没有与当前候选记录对应的含亮迹版本，不能导出合成清单。'))
    candidates = session.get("candidates") or []
    by_id = {c.get("id"): c for c in candidates}
    missing = [cid for cid in kept if cid not in by_id]
    if missing:
        raise ValueError(tr('加回记录里的候选不在当前复核数据中，不能导出合成清单。'))
    drafts = session.get("decisions") or {}
    rows = []
    for cid in kept:
        c = by_id[cid]
        identity, identity_source = _published_identity(c)
        image = _point_pairs(c.get("endpoints_native_reference"))
        source = _point_pairs(c.get("endpoints_native_source"))
        exposure = c.get("exptime")
        rows.append(dict(zip(COLUMNS, (
            cid, file_name(c.get("input")), c.get("input_sha256") or "", c.get("frame_index"),
            c.get("date_obs") or "", exposure if exposure not in (None, "") else "",
            *image, *source, identity, identity_source, _published_keep_source(c, bool(drafts.get(cid)))))))
    rows.sort(key=lambda row: (row["frame_index"] or 0, row["id"]))
    by = {kind: sum(row["keep_source"] == kind for row in rows) for kind in KINDS}

    diagnostics = job.get("diagnostics") or {}
    inputs = job.get("inputs") or []
    reference_number = _frame_number(diagnostics.get("reference_stem"))
    reference_time = next((c.get("date_obs") for c in candidates
                           if c.get("frame_index") == reference_number and c.get("date_obs")), None)
    timed = sorted({(c.get("date_obs"), c.get("frame_index")) for c in candidates if c.get("date_obs")})
    frames_timed = len({frame for _, frame in timed})
    exposure = diagnostics.get("exposure") or {}
    request = (workflow or {}).get("request") or {}
    stamp = (now or datetime.now().astimezone()).isoformat(timespec="seconds")
    header = {
        "format": FORMAT, "exported_at": stamp, "app_version": PRODUCT_VERSION,
        "trails_folder": file_name((workflow or {}).get("delivery_dir")),
        "base_folder": file_name((workflow or {}).get("base_delivery_dir")),
        "candidate_revision": (session.get("base_revision") or {}).get("id"),
        "added_back": len(rows), "added_back_by": by,
        "reference_frame": {
            "file": file_name(diagnostics.get("reference_input")), "frame_index": reference_number,
            "capture_time_camera_clock": reference_time,
            "capture_time_basis": "candidates_on_reference_frame" if reference_time else "unknown_no_candidate_on_frame"},
        "segment": {
            "frames": diagnostics.get("input_count") or len(inputs), "stacked": diagnostics.get("stacked_count"),
            "scanned": (diagnostics.get("transients") or {}).get("frames_scanned"),
            "first_capture_time_camera_clock": timed[0][0] if timed else None,
            "first_frame_index": timed[0][1] if timed else None,
            "last_capture_time_camera_clock": timed[-1][0] if timed else None,
            "last_frame_index": timed[-1][1] if timed else None,
            "frames_with_capture_time": frames_timed,
            "capture_time_basis": "candidates_only: frames without candidates carry no recorded time here"},
        "exposure": {"input_total_s": exposure.get("input_total_seconds"),
                     "stacked_total_s": exposure.get("included_total_seconds"),
                     "per_row": "filled only when the review data carries it"},
        "not_scanned": _not_scanned(job),
        "time_zone": "unknown",
        "coordinates": {"image": "x1..y2: output photograph pixels, zero-based pixel centres, origin top left"
                                 + ("" if not request.get("image_rotation") else ", before the requested rotation"),
                        "source": "source_x1..source_y2: the same trail in its own exposure (unaligned)",
                        "image_size": [(session.get("geometry") or {}).get("width"),
                                       (session.get("geometry") or {}).get("height")]},
        "image_rotation_deg": request.get("image_rotation") or 0,
        "notes": [
            tr('拍摄时刻是相机记下的时间，时区未知，不一定是北京时间或世界时。'),
            tr('含亮迹版本里的线性母版（master_scene_linear.tif）不含流星；流星只加在成片（final_srgb16.tif、final.jpg）里。'),
            tr('身份是算法建议或复核者的选择，不是认证；同一张曝光的一颗流星可能有多条候选。'),
            tr('未扫描的帧（质量门剔除或未能对齐）里即使有流星，也不在本清单中。'),
        ],
    }
    if not any(row["exposure_s"] != "" for row in rows):
        header["notes"].append(tr('单张曝光时长不在复核数据里，各行暂不填写；本段合计见 exposure。'))
    return {"header": header, "rows": rows}


def csv_text(manifest: dict) -> str:
    """Header as '# key: value' lines (pandas comment='#'), then one table row per added-back meteor."""
    out = io.StringIO()
    for key, value in manifest["header"].items():
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        out.write(f"# {key}: {text}\n")
    writer = csv.DictWriter(out, fieldnames=COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(manifest["rows"])
    return out.getvalue()


def write(manifest: dict, csv_path) -> tuple[Path, Path]:
    """Write <name>.csv (UTF-8 with BOM, for spreadsheets) and <name>.json beside it."""
    csv_path = Path(csv_path)
    if csv_path.suffix.lower() != ".csv":
        csv_path = csv_path.with_name(csv_path.name + ".csv")
    json_path = csv_path.with_suffix(".json")
    csv_path.write_text(csv_text(manifest), encoding="utf-8-sig", newline="")
    json_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    return csv_path, json_path


def inside_works(path, workflow: dict | None) -> bool:
    """Whether ``path`` (the chosen file) is in the works folder or one of its delivery folders."""
    folder = Path(path).resolve().parent
    roots = []
    request = (workflow or {}).get("request") or {}
    for value in (request.get("delivery_root"), (workflow or {}).get("delivery_dir"),
                  (workflow or {}).get("base_delivery_dir")):
        if value:
            roots.append(Path(value).resolve())
    delivery = (workflow or {}).get("delivery_dir")
    if delivery:
        roots.append(Path(delivery).resolve().parent)
    if any(folder == root or folder.is_relative_to(root) for root in roots):
        return True
    # Any other works folder: aester-<20 hex> holding delivery-<12 hex> / delivery-trails-<N>kept-<12 hex>.
    return any(_WORKS_PART.fullmatch(part) for part in folder.parts)


_WORKS_PART = re.compile(r"aester-[0-9a-f]{20}|delivery-[0-9a-f]{12}|delivery-trails-\d+kept-[0-9a-f]{12}")
