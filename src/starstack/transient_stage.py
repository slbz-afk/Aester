"""Job-level bright-trail stage (S4 step 1, RUN_LOG §28).

Runs the detection of ``transient_trails`` over every frame the stack used
(``work/r_light_*.fit`` against the master on the same grid), attaches the
provenance of each candidate (input file and hash, frame index, DATE-OBS),
converts its geometry to native coordinates, adds the cross-frame evidence,
the suggested class and the decision under the job's policy, writes
``work/transients/candidates.json`` and one review PNG per candidate, and
builds the native layer boxes of the kept candidates for the compose stage.
Manual labels, overrides, policy changes and manual marks go through
``update_record`` / ``detect_marks`` (used by ``core.recompose_transients``).

1.2.0: frames are scanned by small owner-protected processes; each frame is
independent, results are assembled in frame order, so the numerical results
are the same whatever the worker count (timings differ).
Publication uses an immutable revision; same-source footprints are united
before warp-back so one pixel is contributed once. A worker's peak commit is about
0.55 GB for a 45.7 MP frame, at most about 1.64 GB (the measured worst case, a trail
across the whole frame: SCAN_UNIT_BYTES, scan_unit_bytes).

Efficiency P1c (docs/EFFICIENCY_P1C_PREREG_20261005.md §3.4), nothing recorded changes:

- T-2, the scan process count, first of: ``workers=``; ``STARSTACK_TRANSIENT_WORKERS`` (a positive integer;
  ``auto`` or empty passes on, anything else means automatic); ``AESTER_FRAME_WORKERS`` (1-16, or ``max`` =
  max(floor, ceiling)); automatic = max(floor, n_mem), n_mem <= ceiling, where the floor is today's min(4, usable
  logical processors - 1) (never charged memory), the ceiling min(SCAN_WORKER_CAP, physical cores in the affinity mask
  - 1) (``resources.scan_process_bounds``) and n_mem the children that fit the smaller of available physical memory
  and available commit at ``scan_unit_bytes`` each (a code-derived worst case, ``plan_scan_workers``). The overrides
  are not memory-checked (benches and development). Failures are reported in frame order (``transient_scan``).
- T-3, the review crops after the scan run on the frame pool (``_crop_phase``, cap TRAIL_CROP_CAP, 32 MiB a unit:
  T-3m reads only the sampled pixels); ``review_png`` is set in frame order and is each candidate's last key.
- ``frame_pool``, ``performance`` and ``resources`` are imported lazily: the scan children's import closure
  (tests/test_transient_scan_blas.py) is unchanged.
"""
from __future__ import annotations

import contextlib
import os
import time
from pathlib import Path

import numpy as np

from .transient_revision import read_record, record_path
from .transient_scan import scan_frames
from .transient_union import union_boxes
from . import fits_crop
from . import transient_trails as tt

__all__ = ["STAGE_VERSION", "TRANSIENTS_DIR", "CANDIDATES_FILE", "WORKERS_ENV", "grid_master_path", "frame_order", "default_workers",
           "detect_job", "detect_marks", "update_record", "native_boxes", "summary", "read_record", "record_path", "mark_to_grid",
           "METEOR_DEFAULT_RULE", "METEOR_DEFAULT_POLICY", "UNTRUSTWORTHY_FLAGS", "untrustworthy_reasons", "keep_source", "keep_view",
           "SCAN_WORKER_CAP", "SCAN_UNIT_BYTES", "scan_unit_bytes", "scan_override", "plan_scan_workers", "frame_pixels",
           "TRAIL_CROP_CAP", "TRAIL_CROP_UNIT_BYTES"]

STAGE_VERSION = "1.4.1"
TRANSIENTS_DIR = "transients"
CANDIDATES_FILE = tt.CANDIDATES_FILE
WORKERS_ENV = "STARSTACK_TRANSIENT_WORKERS"
FRAME_WORKERS_ENV = "AESTER_FRAME_WORKERS"          # frame_pool.ENV_WORKERS (not imported: the children's closure)
# P1c T-2 (prereg §3.4, §2.3): at most SCAN_WORKER_CAP scan processes by default, each charged the code-derived worst
# case of one child, scaled by pixels from the 45.7 MP reference: SCAN_BASE_BYTES (interpreter, numpy and the child's
# modules) + the rest per pixel. Measured by tests/test_p1c_site_transient_scan.py on the synthetic worst case (a trail
# across the whole frame's diagonal, found automatically and marked by hand: both measurements read the whole frame):
# peak commit 1.635 GB at 45.7 MP, 61.4-61.6 MB at 64 x 96 px (2026-10-06, Python 3.12, numpy 2.5.3); charged with
# a margin of about 10 % (the prereg's provisional 1.9 GB is replaced by the measurement).
SCAN_WORKER_CAP = 8
REFERENCE_PIXELS = 5520 * 8280
SCAN_UNIT_BYTES = 1_800_000_000
SCAN_BASE_BYTES = 70_000_000
# P1c T-3 (prereg §3.4, §2.3): the review crops of one frame per pool unit (T-3m: each crop reads only its sampled
# pixels, about 10-16 MB at most).
TRAIL_CROP_CAP = 8
TRAIL_CROP_UNIT_BYTES = 32 << 20
EVIDENCE_CHECK_EVERY = 8             # _finish: a cancel probe every 8 outer iterations of the pairwise loops
# 2026-09-27 user decision: meteors are the only class kept by default, and only
# with a trustworthy measurement. A default keep is an algorithm suggestion, not
# an identity certification; its provenance stays distinct from human/agent.
METEOR_DEFAULT_RULE = "meteor-default-keep-1"
METEOR_DEFAULT_POLICY = {"meteor": "keep", "satellite": "omit", "aircraft": "omit", "artifact": "omit", "unknown": "omit"}
UNTRUSTWORTHY_FLAGS = ("width_limit_reached", "profile_boundary_hit", "sideband_insufficient")
REVIEWERS = ("human", "agent")


def untrustworthy_reasons(candidate: dict) -> list[str]:
    """Why a measurement cannot support a default (unreviewed) keep; empty when trustworthy."""
    measurement = candidate.get("measurement") or {}
    if measurement.get("status") != "measured":
        return ["not_measured"]
    quality = measurement.get("quality") or {}
    return [flag for flag in UNTRUSTWORTHY_FLAGS if quality.get(flag)]


def keep_source(candidate: dict) -> str | None:
    """Who authorised a recorded keep: human, agent, or algorithm (a policy default on a suggested class).

    Labels/overrides written before sources were recorded were manual review
    input; they are reported as human.
    """
    if candidate.get("decision") != "keep":
        return None
    reason = candidate.get("requested_reason")
    if reason == "override":
        return candidate.get("override_source") or "human"
    if reason == "mark_protect":
        return candidate.get("mark_source") or "human"
    if candidate.get("label") is not None:
        return candidate.get("label_source") or "human"
    return "algorithm"


def keep_view(candidate: dict, *, meteor_default: bool, decision: dict | None = None, reviewer: str | None = None) -> dict:
    """Identity and keep intent of one candidate, with an unsaved/saved draft decision layered on the record.

    An explicit override wins; otherwise a measured, trustworthy meteor identity
    is kept by default under the meteor-default policy. Relabelling a meteor
    drops the default keep, relabelling as meteor adds it, ``omit`` removes it.
    """
    decision = decision or {}
    if "label" in decision:
        label = decision["label"]
        label_source = (reviewer or "human") if label is not None else None
    else:
        label = candidate.get("label")
        label_source = (candidate.get("label_source") or "human") if label is not None else None
    suggested = (candidate.get("suggestion") or {}).get("class")
    suggested = suggested if suggested in tt.CLASSES else "unknown"
    identity = label if label is not None else suggested
    identity_source = label_source if label is not None else "algorithm"
    if "override" in decision:
        override = decision["override"]
        override_source = (reviewer or "human") if override is not None else None
    else:
        override = candidate.get("override")
        override_source = (candidate.get("override_source") or "human") if override is not None else None
    reasons = untrustworthy_reasons(candidate)
    measured = (candidate.get("measurement") or {}).get("status") == "measured"
    default_keep = bool(meteor_default and identity == "meteor" and not reasons)
    if override is not None:
        requested, source = override == "keep", override_source
    elif (candidate.get("mark") or {}).get("protect") is True:
        requested, source = True, candidate.get("mark_source") or "human"
    else:
        requested, source = default_keep, identity_source
    effective = bool(requested and measured)
    return {"identity": identity, "identity_source": identity_source, "suggested_identity": suggested,
            "measurement_trustworthy": not reasons, "untrustworthy_reasons": reasons,
            "default_keep": default_keep, "effective_keep": effective, "keep_source": source if effective else None,
            "keep_blocked_reason": "not_measured" if requested and not measured else None}


def _resolve(record: dict) -> None:
    """Resolve intents, then withhold untrustworthy default keeps of a meteor-default record and record keep provenance."""
    tt.resolve_decisions(record)
    rule = (record.get("default_keep_rule") or {}).get("version")
    if rule == METEOR_DEFAULT_RULE:
        withheld = 0
        for c in record["candidates"]:
            if c.get("requested_reason") != "policy" or c.get("decision") != "keep":
                continue
            reasons = untrustworthy_reasons(c)
            if reasons:
                c["decision"] = "omit"
                c["decision_reason"] = "default keep withheld: untrustworthy measurement (" + ", ".join(reasons) + ")"
                c["protection_status"] = "needs_review"
                c["protection_reason"] = "untrustworthy_measurement:" + ",".join(reasons)
                withheld += 1
        intent = record.get("protection_intent_summary")
        if withheld and isinstance(intent, dict):
            intent["eligible_keep"] = intent.get("eligible_keep", 0) - withheld
            intent["needs_review"] = intent.get("needs_review", 0) + withheld
            intent["withheld_untrustworthy_default_keep"] = withheld
    for c in record["candidates"]:
        source = keep_source(c)
        if source is None:
            c.pop("keep_source", None)
        else:
            c["keep_source"] = source


def scan_unit_bytes(pixels: int | None = None) -> int:
    """One scan child's worst-case peak commit for a frame of ``pixels`` (None: the 45.7 MP reference)."""
    pixels = REFERENCE_PIXELS if not pixels else int(pixels)
    return int(SCAN_BASE_BYTES + (SCAN_UNIT_BYTES - SCAN_BASE_BYTES) * pixels / REFERENCE_PIXELS)


def frame_pixels(master: Path) -> int | None:
    """NAXIS1 x NAXIS2 of the grid master, None when it cannot be read (the planner then assumes 45.7 MP)."""
    try:
        header, _ = fits_crop.read_header(master)
        return int(header["NAXIS1"]) * int(header["NAXIS2"])
    except (OSError, ValueError, KeyError):
        return None


def scan_override() -> tuple[int | None, str | None]:
    """(processes, variable) of the first scan override set, in the T-2 order (prereg §3.4), or (None, None).

    ``STARSTACK_TRANSIENT_WORKERS``: a positive integer; ``auto`` or empty passes on to the next variable; anything else
    means automatic (as the governor's shadow reads it: the first one set decides, a malformed one included).
    ``AESTER_FRAME_WORKERS``: 1-16, or ``max`` = max(floor, ceiling) of ``resources.scan_process_bounds``; malformed
    raises ValueError (frame_pool.validate_environment refuses it at job start). Overrides are not memory-checked."""
    raw = os.environ.get(WORKERS_ENV, "").strip()
    if raw and raw.lower() != "auto":
        try:
            value = int(raw)
        except ValueError:
            return None, None
        return (value, WORKERS_ENV) if value >= 1 else (None, None)
    from . import frame_pool
    setting = frame_pool.worker_setting()
    if setting == "max":
        from . import resources
        return max(resources.scan_process_bounds(SCAN_WORKER_CAP)), FRAME_WORKERS_ENV
    if setting != "auto":
        return int(setting), FRAME_WORKERS_ENV
    return None, None


_MEASURE = object()


def plan_scan_workers(pixels: int | None = None, *, frames: int | None = None, available=_MEASURE,
                      reserved_cores: int = 0, floor: bool = True, decision: dict | None = None) -> int:
    """The automatic scan process count (no override): max(floor, n_mem) with n_mem <= the ceiling, where n_mem =
    ``frame_pool.default_workers(unit_bytes=scan_unit_bytes(pixels), available=..., blas=False)``.

    ``available`` (bytes) defaults to the smaller of ``performance.available_memory()`` and ``available_commit()``
    now, both net of in-process reservations (R20; an unknown commit is ignored); None (unknown) gives n_mem 1.
    ``floor=False`` (O-1's planner, prereg §3.6) drops the floor; ``reserved_cores`` lowers the ceiling. ``decision``
    (a dict) receives the bounds, for a timings note only."""
    from . import frame_pool, performance, resources
    lowest, ceiling = resources.scan_process_bounds(SCAN_WORKER_CAP, reserved_cores=reserved_cores)
    if available is _MEASURE:
        physical, commit = performance.available_memory(), performance.available_commit()
        available = None if physical is None else (physical if commit is None else min(physical, commit))
    unit = scan_unit_bytes(pixels)
    why = {}
    n_mem = frame_pool.default_workers(cap=ceiling, unit_bytes=unit, available_bytes=available,
                                       units=max(1, int(frames or ceiling)), blas=False, why=why)
    workers = max(lowest, n_mem) if floor else n_mem
    if decision is not None:
        decision.update(workers=workers, bound="floor" if floor and lowest > n_mem else why.get("bound"),
                        floor=lowest if floor else None, ceiling=ceiling, n_mem=n_mem, unit_bytes=unit,
                        pixels=pixels, available_bytes=None if available is None else int(available),
                        reserved_cores=int(reserved_cores))
    return workers


def default_workers(pixels: int | None = None, *, frames: int | None = None, decision: dict | None = None) -> int:
    """Scan processes for a frame of ``pixels`` (None: 45.7 MP): the first override set (``scan_override``), else
    ``plan_scan_workers``. ``decision`` (a dict) receives how it was decided (timings note only)."""
    override, variable = scan_override()
    if override is not None:
        if decision is not None:
            decision.update(workers=override, bound="override", variable=variable)
        return override
    return plan_scan_workers(pixels, frames=frames, decision=decision)


def grid_master_path(work: Path, own_registration: bool) -> Path:
    """The master on the grid the exported frames are in: the undistorted stack with the product's own registration,
    the native stack with Siril's two-pass fallback."""
    undistorted = work / "sky_master_undistorted.fit"
    return undistorted if own_registration and undistorted.is_file() else work / "sky_master.fit"


def frame_order(work: Path) -> list[str]:
    stems = sorted(p.stem[2:] for p in work.glob("r_light_*.fit"))
    sequence = work / "light_.seq"
    if sequence.is_file():
        from .siril_sequence import read_seq
        selected = {f["stem"] for f in read_seq(sequence)["images"] if f["selected"]}
        stems = [s for s in stems if s in selected]
    return stems


def _apply_h(h, points: np.ndarray) -> np.ndarray:
    h = np.asarray(h, dtype=np.float64)
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    hom = np.column_stack([pts, np.ones(len(pts))]) @ h.T
    return hom[:, :2] / hom[:, 2:3]


def mark_to_grid(mark: dict, h, model) -> dict:
    """A manual mark in the source frame's native top-down coordinates -> the stack grid (through the frame's registration)."""
    if h is None:
        raise ValueError("缺少配准矩阵，无法映射原片标记")
    pts = np.array([[mark["x0"], mark["y0"]], [mark["x1"], mark["y1"]]], dtype=np.float64)
    if model is not None:
        pts = model.apply(pts)
    q = _apply_h(h, pts)
    if not np.isfinite(q).all():
        raise ValueError("标记的配准坐标无效")
    return {**mark, "x0": float(q[0, 0]), "y0": float(q[0, 1]), "x1": float(q[1, 0]), "y1": float(q[1, 1]),
            "_source_mark": dict(mark), "corridor_mapping": "endpoints_only_width_not_transformed"}


def _native_endpoints(m: dict, h, model, *, own_registration=False) -> dict:
    q = np.array([[m["x0"], m["y0"]], [m["x1"], m["y1"]]], dtype=np.float64)
    known_grid = not own_registration or model is not None
    reference = model.inverse(q) if model is not None else q
    result = {"grid": q.tolist(), "native_reference": reference.tolist() if known_grid else None,
              "native_source": None, "native_source_status": "unavailable",
              "native_source_reason": "missing_registration_transform" if h is None else "missing_grid_model"}
    if h is not None and known_grid:
        try:
            source_grid = _apply_h(np.linalg.inv(np.asarray(h, dtype=np.float64)), q)
            source = model.inverse(source_grid) if model is not None else source_grid
            if not np.isfinite(source).all():
                raise ValueError("nonfinite source coordinates")
            result.update(native_source=source.tolist(), native_source_status="available", native_source_reason=None)
        except (ValueError, np.linalg.LinAlgError):
            result["native_source_reason"] = "invalid_registration_transform"
    return result


def _mark_key(mark):
    a = (mark["x0"], mark["y0"])
    b = (mark["x1"], mark["y1"])
    return (mark["stem"], *sorted((a, b)), mark.get("half_width_px", 30))


def _provenance(stem: str, inputs: list[dict], work: Path) -> dict:
    index = int(stem[-5:])
    entry = inputs[index - 1] if 0 < index <= len(inputs) else None
    header = {}
    converted = work / f"{stem}.fit"
    if converted.is_file():
        try:
            header, _ = fits_crop.read_header(converted)
        except (OSError, ValueError):
            header = {}
    return {"stem": stem, "frame_index": index, "input": entry, "date_obs": header.get("DATE-OBS"),
            "exptime": header.get("EXPTIME")}


def _candidate(stem: str, k: int, seg: dict, *, inputs, work, h, model, own_registration=False) -> dict:
    c = {"id": f"{stem}-{k}", "stem": stem, "source": seg["source"], "detection": seg["detection"], "measurement": seg["measurement"],
         "label": None, "override": None, "note": ""}
    if seg.get("mark") is not None:
        c["mark"] = dict(seg["mark"].get("_source_mark", seg["mark"]))
        c["mark_grid"] = {key: seg["mark"][key] for key in ("x0", "y0", "x1", "y1", "half_width_px") if key in seg["mark"]}
        c["mark_mapping"] = "endpoints_only_width_not_transformed"
    c["measurement"].setdefault("clipping_evidence", "unknown")
    c["provenance"] = _provenance(stem, inputs, work)
    if seg["measurement"].get("status") == "measured":
        c["geometry"] = _native_endpoints(seg["measurement"], h, model, own_registration=own_registration)
    return c


def _crop(work: Path, master, c: dict, frame=None, *, review_dir=None) -> None:
    crops = (review_dir if review_dir is not None else work / TRANSIENTS_DIR) / "crops"
    crops.mkdir(parents=True, exist_ok=True)
    info = tt.write_crop_png(frame if frame is not None else work / f"r_{c['stem']}.fit", master, c, crops / f"{c['id']}.png")
    c["review_png"] = str(crops / f"{c['id']}.png") if info.get("written") else None


def _scan_frame(frame_path: str, master_path: str, detection: dict, grid_marks: list[dict], stats_step: int) -> dict:
    """One frame's statistics and candidates (runs in a worker process; everything returned is plain JSON-like data)."""
    started = time.monotonic()
    frame, master = tt.Planes(frame_path), tt.Planes(master_path)
    stats = tt.frame_stats(frame, step=stats_step)
    result = tt.detect_frame(frame, master, params=detection, marks=grid_marks)
    return {"stats": stats, "result": result, "seconds": round(time.monotonic() - started, 1)}


def related_candidates(record: dict, *, check=None) -> None:
    """Review-only same-exposure pairs: no transitive merge or disposition propagation. ``check`` (P1c T-3): a cancel
    probe every EVIDENCE_CHECK_EVERY outer iterations, the order unchanged."""
    candidates = record["candidates"]
    for c in candidates:
        c["related_candidate_ids"] = []
        c["relation_conflicts"] = []
    groups = []
    ordered = sorted(candidates, key=lambda c: c["id"])
    for index, a in enumerate(ordered):
        if check is not None and index % EVIDENCE_CHECK_EVERY == 0:
            check()
        if a["measurement"].get("status") != "measured":
            continue
        for b in ordered[index + 1:]:
            if b["measurement"].get("status") != "measured" or a["stem"] != b["stem"] or {a.get("source"), b.get("source")} != {"manual", "auto"}:
                continue
            pa = (a.get("provenance") or {}).get("input") or {}
            pb = (b.get("provenance") or {}).get("input") or {}
            if pa.get("sha256") != pb.get("sha256"):
                continue
            forward = tt.marked_line_relation(a["measurement"], b["measurement"])
            reverse = tt.marked_line_relation(b["measurement"], a["measurement"])
            if not (forward["matches"] and reverse["matches"]):
                continue
            conflicts = []
            if a.get("requested_decision") != b.get("requested_decision"):
                conflicts.append("related_candidates_have_different_requested_decisions")
            if a.get("label") and b.get("label") and a["label"] != b["label"]:
                conflicts.append("related_candidates_have_different_manual_labels")
            ids = [a["id"], b["id"]]
            groups.append({"members": ids, "stem": a["stem"], "source_sha256": pa.get("sha256"),
                           "grid": record.get("grid"), "relation": "same_exposure_geometric_overlap",
                           "identity_status": "unconfirmed_no_merge", "evidence": [forward, reverse], "conflicts": conflicts})
            a["related_candidate_ids"].append(b["id"])
            b["related_candidate_ids"].append(a["id"])
            for c in (a, b):
                c["relation_conflicts"].extend(conflicts)
    record["related_candidate_groups"] = groups


def reset_application(record: dict) -> None:
    """Clear the preceding revision's application claim before a new composition."""
    for c in record["candidates"]:
        requested = c.get("requested_decision") == "keep"
        c["application"] = {"status": "pending" if c.get("decision") == "keep" else ("not_applied" if requested else "not_requested"),
                            "reasons": [c["protection_reason"]] if c.get("protection_reason") else [],
                            "complete_trail_recovery": "unverified", "clipping_evidence": "unknown"}
    record.pop("application_by_source", None)


def _finish(record: dict, work: Path, *, persist=True, check=None) -> dict:
    tt.add_evidence(record["candidates"], record["frame_order"], params=record["params"]["evidence"], check=check,
                    check_every=EVIDENCE_CHECK_EVERY)
    for c in record["candidates"]:
        c["suggestion"] = tt.suggest(c, weak_snr=record["params"]["detection"]["weak_snr"])
    _resolve(record)
    related_candidates(record, check=check)
    reset_application(record)
    record["summary"] = summary(record)
    if persist:
        tt.write_candidates(work / TRANSIENTS_DIR, record)
    return record


class _Untimed:
    """The timer of a caller without one (tests, tools): its operations time nothing."""

    @staticmethod
    @contextlib.contextmanager
    def operation(name):
        yield


_UNTIMED = _Untimed()


def _crop_unit_bytes(scanned: dict, stems: list[str], shape) -> int:
    """unit_bytes of the crop pool (R10/R17): TRAIL_CROP_UNIT_BYTES (T-3m's sampled reads) when every frame with
    candidates was scanned on the master's grid -- always in a job, the scan refuses another grid (detection_image) --
    else what the reference crop body holds for the frames' tallest box: frame and master rows at full width plus a
    conversion temporary (3 x 4 B a pixel), plus the crop-sized arrays."""
    height, width = (int(v) for v in shape)
    rows = 0
    for stem in stems:
        result = scanned[stem]["result"]
        if not result["segments"] or list(result.get("shape") or ()) == [height, width]:
            continue
        frame_height, frame_width = (list(result.get("shape") or ()) + [height, width])[:2]
        width = max(width, int(frame_width))
        for seg in result["segments"]:
            if seg["measurement"].get("status") == "measured":
                box = tt._crop_box(seg["measurement"])
                rows = max(rows, min(max(height, int(frame_height)), box[3]) - max(0, box[1]))
    return TRAIL_CROP_UNIT_BYTES if not rows else 3 * 4 * rows * width + TRAIL_CROP_UNIT_BYTES


def _crop_phase(record: dict, scanned: dict, stems: list[str], master_planes, crops_dir: Path, check, workers=None, *,
                work: Path, inputs: list[dict], transforms: dict, model, own_registration: bool,
                cap: int = TRAIL_CROP_CAP, timer=None, inflight: int | None = None) -> None:
    """detect_job after the scan (efficiency P1c T-3, prereg §3.4): the frames' records, candidates and review crops.

    Per frame, in frame order on the calling thread (``prepare``, rule R15: it never calls ``check``): the frame's
    statistics, then -- when it has candidates -- the crops folder and the opened frame, then its candidates, built and
    appended to the record. On the frame pool (``workers``, default ``performance.frame_workers`` with ``cap`` and
    unit_bytes from ``_crop_unit_bytes``; no BLAS): the frame's review crops in candidate order, each after a call of
    ``unit.check`` (prereg §3.4: on both paths; on the serial path, the caller's ``check``, so a frame's crops add one
    check per candidate to the pool's one per frame). The parallel path's probe is also called within each crop's
    sampled reads (every 32 sampled rows: a tall crop read from disk); the serial path's crops get no ``check`` (R18:
    a probe added for the parallel path does not change the serial path's calls). Per frame in frame order
    (``on_result``): each candidate's ``review_png`` (its last key) and ``frames_scanned``. All of a frame's candidates
    are built before its first crop (prereg §5 item 4). Diagnostics: the timings note ``frame_pool.trail_crops`` and
    the governor's shadow decision ``py.trail_crops``; the pool runs in the timer operation ``trail_crops``.
    ``detect_job`` passes ``work/transients/crops``; a checking tool may pass a folder of its own. ``inflight``: the
    pool's in-flight bound (tests; default 2 x workers). Payloads (a frame's memmaps and candidate references) and
    results (a list of flags) are negligible beside a unit's crops, so unit_bytes needs no in-flight share (R17).
    On a failure (R9) the record, which the caller discards, may hold entries of frames prepared after the failed one,
    and ``crops_dir`` crops of frames after it: each the clean run's bytes under the same name, which a rerun rewrites."""
    from . import frame_pool, performance, resources
    crops_dir = Path(crops_dir)
    unit_bytes = _crop_unit_bytes(scanned, stems, master_planes.shape)
    decision = {}
    if workers is None:
        workers = performance.frame_workers(len(stems), unit_bytes, cap, blas=False, decision=decision)
    made = {}                                   # frame index -> its candidates, from prepare to on_result

    def prepare(i, stem, own_check):
        h = transforms.get(stem)
        stats, result = scanned[stem]["stats"], scanned[stem]["result"]
        record["frame_stats"][stem] = stats
        record["per_frame"][stem] = {"sigma": result["sigma"], "saturated": result["saturated"],
                                     "detector_budget_exceeded": result["saturated"], "clipping_evidence": "unknown", "foreground_pixels": result["foreground_pixels"],
                                     "segments": len(result["segments"]), "seconds": scanned[stem]["seconds"]}
        frame_planes = None
        if result["segments"]:
            crops_dir.mkdir(parents=True, exist_ok=True)
            frame_planes = tt.Planes(work / f"r_{stem}.fit")
        candidates = made[i] = []
        for k, seg in enumerate(result["segments"], 1):
            c = _candidate(stem, k, seg, inputs=inputs, work=work, h=h, model=model, own_registration=own_registration)
            record["candidates"].append(c)
            candidates.append(c)
        return frame_planes, candidates, own_check

    def crop(payload, unit):
        frame_planes, candidates, coordinator = payload
        # The serial path (one worker, a budget or nesting fallback) hands prepare and the unit the same check, the
        # caller's own (frame_pool); the parallel path gives the unit an abort probe, which the sampled reads also call
        # (R18). Decided per run, not by frame_pool.in_pool(): a nested pool runs serially on a worker thread.
        rows_probe = None if unit.check is coordinator else unit.check
        written = []
        for c in candidates:
            unit.check()                                          # prereg §3.4: before each candidate, both paths
            info = tt.write_crop_png(frame_planes, master_planes, c, crops_dir / f"{c['id']}.png", check=rows_probe)
            written.append(bool(info.get("written")))
        return written

    def deliver(i, written):
        for c, ok in zip(made.pop(i), written):
            c["review_png"] = str(crops_dir / f"{c['id']}.png") if ok else None
        record["frames_scanned"] += 1

    timer = timer if timer is not None else _UNTIMED
    pool_stats = {"decision": decision} if decision else {}
    try:
        with timer.operation("trail_crops"):
            frame_pool.run_ordered(crop, stems, workers=workers, name="trail_crops", check=check, prepare=prepare,
                                   on_result=deliver, inflight=inflight, unit_bytes=unit_bytes, blas=False,
                                   stats=pool_stats)
    finally:
        performance.note_pool("trail_crops", pool_stats)
    resources.shadow_pool("py.trail_crops", actual=pool_stats.get("workers"), items=len(stems))


def detect_job(work: Path, *, own_registration: bool, transforms: dict, model, inputs: list[dict], reference_stem: str | None,
               marks: list[dict] | None = None, policy: dict | None = None, check=None, log=print, params: dict | None = None,
               workers: int | None = None, timer=None, crop_cap: int = TRAIL_CROP_CAP, scan_done=None) -> dict:
    """Detect and record the candidates of a finished stack (all exported frames), with provenance, evidence, suggestions
    and decisions; returns the written record.  ``workers`` (default ``default_workers``) frames are scanned at a time
    in worker processes; the record does not depend on the count. The review crops run on the frame pool
    (``_crop_phase``, at most ``crop_cap`` threads, in ``timer``'s operation ``trail_crops``). ``scan_done``: called
    with no argument on this thread once ``scan_frames`` has returned or raised (its processes have ended), e.g. to
    release a reservation of their memory (efficiency P1c O-1); it must not raise."""
    started = time.monotonic()
    master = grid_master_path(work, own_registration)
    stems = frame_order(work)
    detection = dict(tt.DEFAULT_DETECTION, **((params or {}).get("detection") or {}))
    plan = {}
    if not workers:
        workers = default_workers(frame_pixels(master), frames=len(stems), decision=plan)
    workers = max(1, min(int(workers), max(1, len(stems))))
    from . import profiling, resources
    if plan:
        profiling.note_append("transient_scan.plan", dict(plan, frames=len(stems), processes=workers))
    # P1b-2 governor, shadow only; at T-2's frame size when T-2 planned (else the scope's job)
    resources.shadow_pool("py.transient_scan", actual=workers, items=len(stems), frame_pixels=plan.get("pixels"))
    record = {"version": tt.TRANSIENT_VERSION, "stage_version": STAGE_VERSION, "grid": "undistorted" if own_registration else "native",
              "master": str(master), "frames_scanned": 0, "frame_order": stems, "reference_stem": reference_stem,
              "params": {"detection": detection, "evidence": dict(tt.DEFAULT_EVIDENCE, **((params or {}).get("evidence") or {})),
                         "layer": dict(tt.DEFAULT_LAYER)},
              "policy": dict(tt.DEFAULT_POLICY, **(policy or {})), "marks": list(marks or []), "candidates": [], "per_frame": {},
              "frame_stats": {}, "workers": workers, "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    marks_by_stem = {}
    for mark in marks or []:
        marks_by_stem.setdefault(mark["stem"], []).append(mark)
    missing = [stem for stem in marks_by_stem if stem not in stems or transforms.get(stem) is None]
    if missing or (marks_by_stem and own_registration and model is None):
        raise ValueError(f"亮迹标记缺少已选配准帧或坐标变换：{', '.join(missing) or '镜头模型缺失'}")
    stats_step = int(record["params"]["layer"]["stats_step"])
    jobs = {}
    for stem in stems:
        h = transforms.get(stem)
        grid_marks = [mark_to_grid(m, h, model) for m in marks_by_stem.get(stem, [])] if h is not None else []
        jobs[stem] = dict(frame_path=str(work / f"r_{stem}.fit"), master_path=str(master), detection=detection, grid_marks=grid_marks, stats_step=stats_step)
    try:
        scanned = scan_frames(jobs, workers=workers, output_dir=work / TRANSIENTS_DIR, check=check, log=log)
    finally:
        if scan_done is not None:
            scan_done()
    master_planes = tt.Planes(master)
    _crop_phase(record, scanned, stems, master_planes, work / TRANSIENTS_DIR / "crops", check, work=work, inputs=inputs,
                transforms=transforms, model=model, own_registration=own_registration, cap=crop_cap, timer=timer)
    record["seconds"] = round(time.monotonic() - started, 1)
    return _finish(record, work, check=check)


def _trail_crops_reference(record: dict, scanned: dict, stems: list[str], master_planes, crops_dir: Path, check, *,
                           work: Path, inputs: list[dict], transforms: dict, model, own_registration: bool) -> None:
    """The per-frame loop of detect_job before P1c T-3, kept as the tests' reference for ``_crop_phase``: build one
    candidate, crop it, build the next, the crops through ``tt._write_crop_png_reference`` (before T-3m)."""
    crops = Path(crops_dir)
    for stem in stems:
        if check:
            check()
        h = transforms.get(stem)
        stats, result = scanned[stem]["stats"], scanned[stem]["result"]
        record["frame_stats"][stem] = stats
        record["per_frame"][stem] = {"sigma": result["sigma"], "saturated": result["saturated"],
                                     "detector_budget_exceeded": result["saturated"], "clipping_evidence": "unknown", "foreground_pixels": result["foreground_pixels"],
                                     "segments": len(result["segments"]), "seconds": scanned[stem]["seconds"]}
        frame_planes = tt.Planes(work / f"r_{stem}.fit") if result["segments"] else None
        for k, seg in enumerate(result["segments"], 1):
            c = _candidate(stem, k, seg, inputs=inputs, work=work, h=h, model=model, own_registration=own_registration)
            record["candidates"].append(c)
            crops.mkdir(parents=True, exist_ok=True)
            info = tt._write_crop_png_reference(frame_planes, master_planes, c, crops / f"{c['id']}.png")
            c["review_png"] = str(crops / f"{c['id']}.png") if info.get("written") else None
        record["frames_scanned"] += 1


def detect_marks(work: Path, record: dict, marks: list[dict], *, own_registration: bool, transforms: dict, model, inputs: list[dict],
                 check=None, persist=True, review_dir=None) -> dict:
    """Add manual marks to an existing record (measured in their frames) and re-derive evidence, suggestions and decisions."""
    master = tt.Planes(grid_master_path(work, own_registration))
    detection = record["params"]["detection"]
    by_stem = {}
    known = {_mark_key(m): m for m in record.get("marks", [])}
    for mark in marks:
        key = _mark_key(mark)
        if key in known:
            if mark.get("protect") is True:
                matches = [c for c in record["candidates"] if c.get("mark") and _mark_key(c["mark"]) == key]
                # Stage 1.2 initial-job marks accidentally retained grid coordinates;
                # migrate only an exact match under this job's frozen transform.
                if not matches and transforms.get(mark["stem"]) is not None and (not own_registration or model is not None):
                    grid = mark_to_grid(mark, transforms[mark["stem"]], model)
                    matches = [c for c in record["candidates"] if c.get("mark") and _mark_key(c["mark"]) == _mark_key(grid)]
                    for candidate in matches:
                        candidate["mark_grid"] = {k: candidate["mark"][k] for k in ("x0", "y0", "x1", "y1", "half_width_px") if k in candidate["mark"]}
                        candidate["mark"] = dict(known[key])
                        candidate["mark_mapping"] = "legacy_grid_mark_recovered_from_exact_frozen_transform"
                if not matches:
                    raise ValueError("已有标记找不到对应候选，无法兑现保护升级；请查看旧候选记录")
                known[key]["protect"] = True
                for candidate in matches:
                    candidate["mark"]["protect"] = True
            continue
        known[key] = mark
        by_stem.setdefault(mark["stem"], []).append(mark)
    for stem, stem_marks in by_stem.items():
        if not (work / f"r_{stem}.fit").is_file():
            raise ValueError(f"帧 {stem} 没有导出的配准帧（未参与堆栈），不能标记")
        h = transforms.get(stem)
        if h is None:
            raise ValueError(f"帧 {stem} 没有配准矩阵，不能把标记映射到堆栈网格")
        frame = tt.Planes(work / f"r_{stem}.fit")
        grid_marks = [mark_to_grid(m, h, model) for m in stem_marks]
        det = tt.detection_image(frame, master, params=detection)
        existing = sum(1 for c in record["candidates"] if c["stem"] == stem)
        for k, (mark, grid_mark) in enumerate(zip(stem_marks, grid_marks), existing + 1):
            corridor = (grid_mark["x0"], grid_mark["y0"], grid_mark["x1"], grid_mark["y1"], float(grid_mark.get("half_width_px", 30)))
            segs = tt.find_segments(det, params=detection, threshold=detection["mark_threshold_sigma"], corridor=corridor)
            seg, m = tt.measure_mark(frame, master, grid_mark, segs, params=detection)
            c = _candidate(stem, k, {"source": "manual", "mark": grid_mark, "detection": seg, "measurement": m}, inputs=inputs, work=work, h=h, model=model, own_registration=own_registration)
            record["candidates"].append(c)
            _crop(work, master, c, frame, review_dir=review_dir)
            record["marks"].append(dict(mark))
        if check:
            check()
    return _finish(record, work, persist=persist)


def _set_source(candidate: dict, field: str, reviewer: str | None, value) -> None:
    if value is not None and reviewer is not None:
        candidate[field] = reviewer
    else:
        # A cleared value, or a change without a recorded reviewer, must not
        # inherit the provenance of an earlier revision's value.
        candidate.pop(field, None)


def update_record(work: Path, record: dict, *, labels: dict | None = None, overrides: dict | None = None, policy: dict | None = None,
                  notes: dict | None = None, layer_style: str | None = None, persist=True,
                  default_keep_rule: str | None = None, reviewer: str | None = None) -> dict:
    """Apply manual labels / per-candidate overrides / a policy change and re-resolve the decisions; writes the record.

    ``reviewer`` (human/agent) is recorded as the source of the labels and
    overrides changed here. ``default_keep_rule`` (meteor-default-keep-1)
    withholds policy keeps whose measurement is untrustworthy; once set it
    stays with the record's later revisions.
    """
    if reviewer is not None and reviewer not in REVIEWERS:
        raise ValueError("复核来源须明确为 human 或 agent")
    if default_keep_rule is not None and default_keep_rule != METEOR_DEFAULT_RULE:
        raise ValueError("未知默认保留规则")
    if layer_style is not None:
        if layer_style not in ("source_residual", "photographic_v3"):
            raise ValueError("亮迹渲染需为 source_residual 或 photographic_v3")
        record.setdefault("params", {}).setdefault("layer", {})["style"] = layer_style
    by_id = {c["id"]: c for c in record["candidates"]}
    for cid, label in (labels or {}).items():
        if cid not in by_id:
            raise ValueError(f"没有候选 {cid}")
        if label is not None and label not in tt.CLASSES:
            raise ValueError(f"标签必须是 {', '.join(tt.CLASSES)} 之一或空")
        by_id[cid]["label"] = label
        _set_source(by_id[cid], "label_source", reviewer, label)
    for cid, decision in (overrides or {}).items():
        if cid not in by_id:
            raise ValueError(f"没有候选 {cid}")
        if decision is not None and decision not in tt.DECISIONS:
            raise ValueError("覆盖必须是 keep、omit 或空（按策略）")
        by_id[cid]["override"] = decision
        _set_source(by_id[cid], "override_source", reviewer, decision)
    for cid, note in (notes or {}).items():
        if cid not in by_id:
            raise ValueError(f"没有候选 {cid}")
        by_id[cid]["note"] = str(note)[:4000]
    if policy:
        for cls, decision in policy.items():
            if cls not in tt.CLASSES or decision not in tt.DECISIONS:
                raise ValueError(f"策略项须为 类别 -> keep/omit（类别：{', '.join(tt.CLASSES)}）")
        record["policy"] = dict(record.get("policy") or tt.DEFAULT_POLICY, **policy)
    if default_keep_rule is not None:
        record["default_keep_rule"] = {"version": default_keep_rule, "requires": "measured",
                                       "untrustworthy_quality_flags": list(UNTRUSTWORTHY_FLAGS),
                                       "identity_certified": False,
                                       "default_keep_source": "algorithm"}
    _resolve(record)
    related_candidates(record)
    reset_application(record)
    record["summary"] = summary(record)
    if persist:
        tt.write_candidates(work / TRANSIENTS_DIR, record)
    return record


def native_boxes(work: Path, record: dict, *, own_registration: bool, model, native_shape: tuple[int, int], check=None) -> tuple[list[dict], list[dict]]:
    """Native layer boxes of every kept candidate (normalised residual of its source frame, warped like the master)."""
    style = record.get("params", {}).get("layer", {}).get("style", "source_residual")
    if style not in ("source_residual", "photographic_v3"):
        raise ValueError("未知亮迹渲染版本")
    renderer, unite = tt.layer_box, union_boxes
    if style == "photographic_v3":
        from . import trail_finish
        renderer, unite = trail_finish.layer_box, trail_finish.union_photographic_boxes
    master = tt.Planes(grid_master_path(work, own_registration))
    ref_stem = record.get("reference_stem")
    stats = record.get("frame_stats") or {}
    ref_stats = stats.get(ref_stem) if ref_stem else None
    boxes, kept = [], []
    by_stem = {}
    for c in record["candidates"]:
        if c.get("decision") == "keep" and c["measurement"].get("status") == "measured":
            by_stem.setdefault(c["stem"], []).append(c)
    for stem in sorted(by_stem):
        frame = tt.Planes(work / f"r_{stem}.fit")
        frame_stats = stats.get(stem) or tt.frame_stats(frame)
        if ref_stats:
            scale, ref_median = tt.normalisation(frame_stats, ref_stats), ref_stats["median"]
        else:
            scale, ref_median = [1.0, 1.0, 1.0], frame_stats["median"]
        frame_boxes = []
        for c in sorted(by_stem[stem], key=lambda c: c["id"]):
            if check:
                check()
            box = renderer(frame, master, c, scale, frame_stats["median"], ref_median, params=record["params"]["layer"])
            frame_boxes.append(box)
            c.setdefault("application", {}).update(box.get("application_evidence") or {})
            c["application"]["source_support"] = box.get("source_support")
            kept.append({"id": c["id"], "stem": stem, "footprint_pixels": box.get("footprint_pixels"),
                         "scale": [round(s, 4) for s in scale], "pixel_contribution": "union_with_same_source_frame"})
        for box in unite(frame_boxes):
            if check:
                check()
            native = tt.warp_box_to_native(box, model, native_shape, check=check)
            native["stem"] = stem
            boxes.append(native)
    return boxes, kept


def finish_application(record: dict, boxes: list[dict], *, weight=None, weight_known=True, native_warped=False,
                       composite_path: str | None = None) -> None:
    """Record actual nonzero layer operations, without asserting complete trail recovery.

    Counts belong to source-union boxes, may overlap after warp, and are not a
    partition among candidate aliases or final pixels after exposure summation.
    """
    sources = {}
    for box in boxes:
        data = box["data"]
        before = np.any(np.isfinite(data) & (data != 0), axis=0)
        x, y = box["x0"], box["y0"]
        h, w = before.shape
        weighted = data if weight is None else data * weight[y:y+h, x:x+w][None]
        after = np.any(np.isfinite(weighted) & (weighted != 0), axis=0)
        counts = sources.setdefault(box["stem"], {"native_nonzero_box_samples": 0, "composite_nonzero_box_samples": 0,
                                                   "fully_suppressed_box_samples": 0, "attenuated_box_samples": 0,
                                                   "native_support_status": "applied", "native_support_removed_box_samples": 0})
        support = box.get("native_support") or {}
        if support.get("status") != "applied":
            counts["native_support_status"] = "unavailable"
        counts["native_support_removed_box_samples"] += int(support.get("nonzero_pixels_removed", 0))
        counts["native_nonzero_box_samples"] += int(before.sum())
        counts["composite_nonzero_box_samples"] += int(after.sum())
        counts["fully_suppressed_box_samples"] += int(np.count_nonzero(before & ~after))
        if weight is not None:
            counts["attenuated_box_samples"] += int(np.count_nonzero(before & (weight[y:y+h, x:x+w] < 1)))
    record["application_by_source"] = sources
    kept_per_source = {}
    for candidate in record["candidates"]:
        if candidate.get("decision") == "keep":
            kept_per_source[candidate["stem"]] = kept_per_source.get(candidate["stem"], 0) + 1
    for c in record["candidates"]:
        app = c.setdefault("application", {})
        if c.get("decision") != "keep":
            continue
        source = sources.get(c["stem"], {})
        quality = c["measurement"].get("quality") or {}
        reasons = []
        if app.get("grid_zero_or_nonfinite_pixels", 0):
            reasons.append("grid_zero_or_nonfinite_samples")
        reasons.extend(key for key in ("width_limit_reached", "profile_boundary_hit", "sideband_insufficient") if quality.get(key))
        if quality.get("visibility_scope_version") == "2.0.0":
            if quality.get("selected_visible_runs", 0) > 1:
                reasons.append("disconnected_selected_visible_runs")
            if quality.get("unselected_significant_runs"):
                reasons.append("unselected_significant_visible_runs")
            app["visibility_quality_scope"] = "selected_extent_and_separate_unselected_significant_runs"
        else:
            # Frozen legacy records lack selected-extent evidence. Retain their
            # conservative reason, explicitly identify its search-window scope.
            if quality.get("disconnected_visible_runs"):
                reasons.append("disconnected_visible_runs")
            app["visibility_quality_scope"] = "legacy_search_extent_selected_extent_unverified"
        if source.get("attenuated_box_samples", 0):
            reasons.append("source_union_attenuated_by_final_sky_weight")
        if source.get("native_support_removed_box_samples", 0):
            reasons.append("source_union_incomplete_native_kernel_support")
        if not source.get("composite_nonzero_box_samples", 0) or not app.get("grid_nonzero_residual_pixels", 0):
            status = "not_applied"
            reasons.append("no_nonzero_composite_contribution")
        elif reasons:
            status = "partial"
        else:
            status = "unknown"
        # Measurable nonzero pixels are not proof of the whole requested trail.
        reasons.append("complete_requested_extent_unverified")
        if native_warped and source.get("native_support_status") != "applied":
            reasons.append("per_exposure_native_kernel_support_unverified")
        elif source.get("native_support_status") == "applied":
            reasons.append("registered_observation_support_not_raw_or_flux_completeness")
        if not weight_known:
            reasons.append("final_legacy_mask_weight_unavailable")
            if status != "not_applied":
                status = "unknown"
        attribution_known = kept_per_source.get(c["stem"], 0) == 1 and weight_known
        if not attribution_known:
            reasons.append("per_candidate_attribution_unavailable_after_same_source_union")
        app.update(status=status, reasons=reasons, source_union_evidence=source,
                   contribution_observed=(bool(app.get("grid_nonzero_residual_pixels", 0) and source.get("composite_nonzero_box_samples", 0)) if attribution_known else None),
                   source_union_contribution_observed=bool(source.get("composite_nonzero_box_samples", 0)) if weight_known else None,
                   contribution_scope="source_union_box_samples_before_exposure_sum_not_per_candidate_pixel_count",
                   complete_trail_recovery="unverified", clipping_evidence="unknown", composite_path=composite_path)
    record["summary"] = summary(record)


def summary(record: dict) -> dict:
    from .transient_review import summarize_review
    recall_review = summarize_review(record)
    counts = {cls: 0 for cls in tt.CLASSES}          # effective labels of the measured candidates (the reviewable ones)
    labelled = 0
    kept = []
    for c in record["candidates"]:
        if c["measurement"].get("status") == "measured":
            counts[c.get("effective_label", "unknown")] = counts.get(c.get("effective_label", "unknown"), 0) + 1
        labelled += 1 if c.get("label") else 0
        if c.get("decision") == "keep":
            kept.append(c["id"])
    statuses = {}
    for c in record["candidates"]:
        status = c["measurement"].get("status") or "unknown"
        statuses[status] = statuses.get(status, 0) + 1
    application = {}
    for c in record["candidates"]:
        state = (c.get("application") or {}).get("status", "unknown")
        application[state] = application.get(state, 0) + 1
    return {"candidates": len(record["candidates"]), "measured": statuses.get("measured", 0), "by_status": statuses,
            "by_effective_label": counts, "manually_labelled": labelled, "kept": kept, "omitted": len(record["candidates"]) - len(kept),
            "frames_scanned": record.get("frames_scanned"), "saturated_frames": [s for s, f in (record.get("per_frame") or {}).items() if f.get("saturated")],
            "detector_budget_exceeded_frames": [s for s, f in (record.get("per_frame") or {}).items() if f.get("detector_budget_exceeded", f.get("saturated"))],
            "saturated_frames_semantics": "legacy_name_for_detector_budget_exceeded_not_sensor_clipping",
            "protection_intent": record.get("protection_intent_summary"), "application": application,
            "related_candidate_pairs": len(record.get("related_candidate_groups") or []),
            "related_conflicts": sum(bool(group["conflicts"]) for group in record.get("related_candidate_groups") or []),
            "policy": record.get("policy"), "recall_review": recall_review}
