# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Wang Jian (王俭). See LICENSE.txt in this folder.
"""Composite list of a trails version: which exposure each added-back meteor came from.

Read only. The rows come from the open review session (``trail_review_session.open_review``), the
header from the stack job (``core.load_job``) and the workflow record the main window already holds.
The list is written as CSV and JSON to a place the user chooses, never into the works folder (its
delivery folders hold only the declared images). Files are named by file name only: no drive,
directory or user-profile path is written.

Capture times: a candidate's ``date_obs`` is the converted frame's FITS DATE-OBS, the observation start
in UT. For camera RAW, Siril converts the camera's clock time to UT with the processing computer's time
zone, so converting it back with this computer's zone gives the camera clock when the camera, the
processing computer and this computer use the same zone.

Exposure: ``exposure_s`` is the review view's ``exptime_s``, the converted frame's FITS EXPTIME in seconds
(empty when it was not recorded); the header keeps the segment totals.

Original files: when Aester converted camera RAW files to DNG working copies, the job stacked the DNGs.
The main window passes what ``raw_import.prepare_import`` returned in this session (see
``import_sources``) and its cache folder. For stacked inputs that map does not name (after a restart),
``with_cache_records`` asks ``raw_import.import_sources_for`` for the import records in the cache (read
only). Either way an original is named only when the record's DNG hash equals the hash the job recorded
for that input; otherwise the row names the stacked input. The header counts each basis.
"""
from __future__ import annotations

import csv
import io
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from starstack import PRODUCT_VERSION
from starstack.i18n import tr

FORMAT = "aester-composite-list-3"
KINDS = ("algorithm", "human", "agent")
CLASSES = ("meteor", "satellite", "aircraft", "artifact", "unknown")
COLUMNS = ("id", "source_file", "source_sha256", "frame_index", "capture_time_local", "capture_time_utc",
           "exposure_s", "x1", "y1", "x2", "y2", "source_x1", "source_y1", "source_x2", "source_y2",
           "identity", "identity_source", "keep_source", "stacked_input", "stacked_input_sha256")
TIME_BASIS = ("FITS DATE-OBS (UT). For camera RAW, Siril writes it from the camera's clock time using the "
              "processing computer's time zone; capture_time_local converts it back with this computer's time zone.")


def file_name(path) -> str:
    """The last component of a Windows or POSIX path, so no directory or drive survives."""
    return re.split(r"[\\/]", str(path))[-1] if path else ""


def _path_key(path) -> str:
    return os.path.normcase(os.path.abspath(str(path)))


def capture_time(date_obs, zone=None):
    """The aware local datetime of a DATE-OBS value (UT when it names no zone), or None.

    ``zone`` None: this computer's time zone.
    """
    text = str(date_obs or "").strip()
    if not text:
        return None
    if text[-1:] in ("Z", "z"):
        text = text[:-1] + "+00:00"
    try:
        value = datetime.fromisoformat(text)
    except ValueError:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(zone)


def utc_offset(value) -> str:
    """'UTC+08:00' for an aware datetime."""
    offset = value.utcoffset()
    minutes = int(offset.total_seconds() // 60) if offset is not None else 0
    sign = "+" if minutes >= 0 else "-"
    return f"UTC{sign}{abs(minutes) // 60:02d}:{abs(minutes) % 60:02d}"


def _times(date_obs, zone=None):
    """(utc, local) ISO texts of a DATE-OBS value; ('', '') when it cannot be read."""
    local = capture_time(date_obs, zone)
    if local is None:
        return "", ""
    utc = local.astimezone(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds") + "Z"
    return utc, local.isoformat(timespec="seconds")


def import_sources(result) -> dict:
    """{stacked input key: original file name and hashes} from what ``raw_import.prepare_import`` returned.

    Only conversions name an original different from the stacked input; the DNG's own hash is kept so a
    later job that stacked a changed file is not matched to the original.
    """
    if not isinstance(result, dict) or not result.get("converted"):
        return {}
    inputs, sources, outputs = result.get("input_files"), result.get("sources"), result.get("outputs")
    if not all(isinstance(value, list) for value in (inputs, sources, outputs)) or not (
            len(inputs) == len(sources) == len(outputs)):
        return {}
    found = {}
    for path, source, output in zip(inputs, sources, outputs):
        if isinstance(source, dict) and isinstance(output, dict) and source.get("path") and output.get("sha256"):
            found[_path_key(path)] = {"file": file_name(source["path"]), "sha256": source.get("sha256") or "",
                                      "input_sha256": output["sha256"]}
    return found


# How a row's source_file was found (header source_file_basis counts each).
SESSION_BASIS = "import_record_of_this_session"
CACHE_BASIS = "import_record_found_in_cache"
STACKED_BASIS = "stacked_input"


def with_cache_records(job: dict, originals: dict | None, cache_root) -> dict:
    """This session's ``originals`` plus, for stacked inputs it does not name, the import records in the cache.

    ``raw_import.import_sources_for`` reads those records only (no hashing, nothing written). An entry is
    added only when the DNG hash in the record equals the hash the job recorded for that input, and is
    marked as found in the cache. No ``cache_root``, or a lookup that fails, adds nothing.
    """
    merged = dict(originals or {})
    if not cache_root:
        return merged
    wanted = {}
    for entry in job.get("inputs") or []:
        if isinstance(entry, dict) and entry.get("path") and entry.get("sha256"):
            key = _path_key(entry["path"])
            if key not in merged:
                wanted[key] = (str(entry["path"]), entry["sha256"])
    if not wanted:
        return merged
    from starstack.raw_import import import_sources_for
    try:
        found = import_sources_for([path for path, _ in wanted.values()], cache_root)
    except (OSError, ValueError):
        return merged
    for key, entry in found.items():
        if key in wanted and isinstance(entry, dict) and entry.get("file") and entry.get("input_sha256") == wanted[key][1]:
            merged[key] = {**entry, "basis": CACHE_BASIS}
    return merged


def _original(path, sha256, originals):
    """(file, sha256, basis) of the photograph behind one stacked input."""
    entry = (originals or {}).get(_path_key(path)) if path else None
    if entry and sha256 and entry.get("input_sha256") == sha256:
        basis = CACHE_BASIS if entry.get("basis") == CACHE_BASIS else SESSION_BASIS
        return entry["file"], entry.get("sha256") or "", basis
    return file_name(path), sha256 or "", STACKED_BASIS


def _maybe_working_copy(path, sha256, originals) -> bool:
    """A stacked DNG without a matching import record: possibly a working copy of a camera RAW."""
    return _original(path, sha256, originals)[2] == STACKED_BASIS and file_name(path).lower().endswith(".dng")


def _frame_number(stem):
    """light_00042 -> 42; None for anything else."""
    match = re.fullmatch(r"light_(\d+)", str(stem or ""))
    return int(match.group(1)) if match else None


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return value


def _seconds(value):
    """A positive, finite exposure time in seconds, or None."""
    value = _number(value)
    return value if value is not None and value > 0 else None


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
    session was opened on (a later apply makes a new revision and a new session). A revision that added
    nothing back has no trails version to describe.
    """
    review = (workflow or {}).get("trail_review") or {}
    kept = review.get("kept_ids")
    revision = (review.get("revision") or {}).get("id")
    base = (session.get("base_revision") or {}).get("id")
    if not isinstance(kept, list) or not revision or revision != base:
        return None
    delivery, base_folder = (workflow or {}).get("delivery_dir"), (workflow or {}).get("base_delivery_dir")
    if not kept or not delivery or delivery == base_folder:
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


def _not_scanned(job, originals):
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
        entry = inputs[number - 1] if number and 0 < number <= len(inputs) else {}
        frames.append({"frame_index": number, "file": _original(entry.get("path"), entry.get("sha256"), originals)[0],
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


def _reference_input(job, number):
    """(path, sha256) of the reference frame's stacked input."""
    inputs = job.get("inputs") or []
    path = (job.get("diagnostics") or {}).get("reference_input")
    entry = inputs[number - 1] if number and 0 < number <= len(inputs) else {}
    if entry and (not path or _path_key(entry.get("path") or "") == _path_key(path)):
        return entry.get("path"), entry.get("sha256")
    return path, None


def build(session: dict, job: dict, workflow: dict | None, *, originals: dict | None = None,
          now: datetime | None = None, zone=None) -> dict:
    """{'header': ..., 'rows': [...]} for the published trails version of this session.

    ``originals``: ``import_sources`` of this session's imports, with ``with_cache_records`` added when the
    window knows its cache folder. ``zone``: the local time zone (None: this computer's).
    """
    kept = published_keeps(session, workflow)
    if kept is None:
        raise ValueError(tr('没有与当前候选记录对应的含亮迹版本，不能导出合成清单。'))
    candidates = session.get("candidates") or []
    by_id = {c.get("id"): c for c in candidates}
    missing = [cid for cid in kept if cid not in by_id]
    if missing:
        raise ValueError(tr('加回记录里的候选不在当前复核数据中，不能导出合成清单。'))
    drafts = session.get("decisions") or {}
    rows, bases = [], []
    for cid in kept:
        c = by_id[cid]
        identity, identity_source = _published_identity(c)
        image = _point_pairs(c.get("endpoints_native_reference"))
        source = _point_pairs(c.get("endpoints_native_source"))
        original, original_sha, basis = _original(c.get("input"), c.get("input_sha256"), originals)
        bases.append(basis)
        utc, local = _times(c.get("date_obs"), zone)
        rows.append(dict(zip(COLUMNS, (
            cid, original, original_sha, c.get("frame_index"), local, utc, _seconds(c.get("exptime_s")),
            *image, *source, identity, identity_source, _published_keep_source(c, bool(drafts.get(cid))),
            file_name(c.get("input")), c.get("input_sha256") or ""))))
    rows.sort(key=lambda row: (row["frame_index"] or 0, row["id"]))
    by = {kind: sum(row["keep_source"] == kind for row in rows) for kind in KINDS}

    diagnostics = job.get("diagnostics") or {}
    inputs = job.get("inputs") or []
    reference_number = _frame_number(diagnostics.get("reference_stem"))
    reference_time = next((c.get("date_obs") for c in candidates
                           if c.get("frame_index") == reference_number and capture_time(c.get("date_obs"))), None)
    reference_path, reference_sha = _reference_input(job, reference_number)
    reference_utc, reference_local = _times(reference_time, zone)
    timed = sorted((capture_time(c.get("date_obs")).timestamp(), c.get("frame_index") or 0, c.get("date_obs"))
                   for c in candidates if capture_time(c.get("date_obs")))
    first = _times(timed[0][2], zone) if timed else ("", "")
    last = _times(timed[-1][2], zone) if timed else ("", "")
    exposure = diagnostics.get("exposure") or {}
    request = (workflow or {}).get("request") or {}
    exported = now or datetime.now().astimezone(zone)
    shown_zone = capture_time(timed[0][2], zone) if timed else exported
    header = {
        "format": FORMAT, "exported_at": exported.isoformat(timespec="seconds"), "app_version": PRODUCT_VERSION,
        "trails_folder": file_name((workflow or {}).get("delivery_dir")),
        "base_folder": file_name((workflow or {}).get("base_delivery_dir")),
        "candidate_revision": (session.get("base_revision") or {}).get("id"),
        "added_back": len(rows), "added_back_by": by,
        "capture_time_basis": TIME_BASIS, "capture_time_local_zone": utc_offset(shown_zone),
        "source_file_basis": {basis: bases.count(basis) for basis in (SESSION_BASIS, CACHE_BASIS, STACKED_BASIS)},
        "reference_frame": {
            "file": _original(reference_path, reference_sha, originals)[0], "stacked_input": file_name(reference_path),
            "frame_index": reference_number,
            "capture_time_local": reference_local or None, "capture_time_utc": reference_utc or None,
            "capture_time_from": "candidates_on_reference_frame" if reference_time else "unknown_no_candidate_on_frame"},
        "segment": {
            "frames": diagnostics.get("input_count") or len(inputs), "stacked": diagnostics.get("stacked_count"),
            "scanned": (diagnostics.get("transients") or {}).get("frames_scanned"),
            "first_capture_time_local": first[1] or None, "first_capture_time_utc": first[0] or None,
            "first_frame_index": timed[0][1] if timed else None,
            "last_capture_time_local": last[1] or None, "last_capture_time_utc": last[0] or None,
            "last_frame_index": timed[-1][1] if timed else None,
            "frames_with_capture_time": len({frame for _, frame, _ in timed}),
            "capture_time_from": "candidates_only: frames without candidates carry no recorded time here"},
        "exposure": {"input_total_s": exposure.get("input_total_seconds"),
                     "stacked_total_s": exposure.get("included_total_seconds"),
                     "per_row": "exposure_s: the row's exposure time in seconds (FITS EXPTIME of its converted "
                                "frame); empty when it was not recorded"},
        "not_scanned": _not_scanned(job, originals),
        "coordinates": {"image": "x1..y2: output photograph pixels, zero-based pixel centres, origin top left"
                                 + ("" if not request.get("image_rotation") else ", before the requested rotation"),
                        "source": "source_x1..source_y2: the same trail in its own exposure (unaligned)",
                        "image_size": [(session.get("geometry") or {}).get("width"),
                                       (session.get("geometry") or {}).get("height")]},
        "image_rotation_deg": request.get("image_rotation") or 0,
        "notes": [
            tr('拍摄时刻：capture_time_utc 是世界时，capture_time_local 按这台电脑的时区换算。相机 RAW 的世界时由 Siril 按处理时电脑的时区从相机时间换算，所以相机和电脑设在同一时区时，本地时刻就是相机上的时间；相机设在别的时区时会差整小时。'),
            tr('含亮迹版本里的线性母版（master_scene_linear.tif）不含流星；流星只加在成片（final_srgb16.tif、final.jpg）里。'),
            tr('身份是算法建议或复核者的选择，不是认证；同一张曝光的一颗流星可能有多条候选。'),
            tr('未扫描的帧（质量门剔除或未能对齐）里即使有流星，也不在本清单中。'),
            tr('exposure_s 是每行那张曝光的时长（秒，取自转换后帧的 FITS EXPTIME），没有记录时留空；exposure 是本段合计。'),
        ],
    }
    named = [(by_id[cid].get("input"), by_id[cid].get("input_sha256")) for cid in kept]
    named.append((reference_path, reference_sha))
    skipped = {frame["frame_index"] for frame in header["not_scanned"]["frames"]}
    named += [(entry.get("path"), entry.get("sha256")) for number, entry in enumerate(inputs, 1) if number in skipped]
    if any(_maybe_working_copy(path, sha, originals) for path, sha in named if path):
        header["notes"].append(tr('若加入照片时相机 RAW 被转成了 DNG 工作副本，清单里写的是工作副本的文件名：原片文件名主干相同，哈希不同（没有找到与之相符的转换记录）。'))
    return {"header": header, "rows": rows}


def csv_text(manifest: dict) -> str:
    """Header as two-column rows ('# key', value), then one table row per added-back meteor.

    Spreadsheets show each header entry in two cells instead of splitting its JSON at the commas; pandas
    reads the table with comment='#'. The JSON beside the CSV holds the same header as structured data.
    """
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    for key, value in manifest["header"].items():
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        writer.writerow([f"# {key}", text])
    table = csv.DictWriter(out, fieldnames=COLUMNS, lineterminator="\n")
    table.writeheader()
    table.writerows(manifest["rows"])
    return out.getvalue()


def paths(csv_path) -> tuple[Path, Path]:
    """(<name>.csv, <name>.json) for a chosen file name."""
    csv_path = Path(csv_path)
    if csv_path.suffix.lower() != ".csv":
        csv_path = csv_path.with_name(csv_path.name + ".csv")
    return csv_path, csv_path.with_suffix(".json")


def write(manifest: dict, csv_path) -> tuple[Path, Path]:
    """Write <name>.csv (UTF-8 with BOM, for spreadsheets) and <name>.json beside it."""
    csv_path, json_path = paths(csv_path)
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
