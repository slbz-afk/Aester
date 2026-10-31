"""Immutable candidate records published by a single job.json pointer update.

The fixed work record is a legacy/detection checkpoint. A published revision
is authoritative, and a damaged pointer must never fall back to that record.
"""
import json
from pathlib import Path

from . import hashing
from . import transient_trails as tt


def record_path(work: Path, *, job: dict | None = None) -> Path:
    work = Path(work).resolve()
    if job is None:
        meta = work.parent / "job.json"
        job = json.loads(meta.read_text(encoding="utf-8")) if meta.is_file() else {}
    revision = job.get("transient_revision")
    if revision is None:
        return work / "transients" / tt.CANDIDATES_FILE
    try:
        path = Path(revision["record"]).resolve()
        path.relative_to(work.parent.resolve())
        path.relative_to((work.parent / "products").resolve())
        if path.name != tt.CANDIDATES_FILE or not revision["id"] or not revision["sha256"]:
            raise ValueError("invalid revision")
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("亮迹修订指针无效或超出作业产品目录") from exc
    return path


def read_record(work: Path, *, job: dict | None = None) -> dict | None:
    work = Path(work)
    if job is None:
        meta = work.parent / "job.json"
        job = json.loads(meta.read_text(encoding="utf-8")) if meta.is_file() else {}
    if "transient_revision" in job and job["transient_revision"] is None:
        return None  # explicit absence of a publication, not a legacy job
    path = record_path(work, job=job)
    revision = job.get("transient_revision")
    if revision is None and not path.is_file():
        return None
    try:
        raw = path.read_bytes()
        if revision is not None and hashing.sha256_bytes(raw, source="transient_revision.read_record") != revision["sha256"]:
            raise ValueError("亮迹修订记录校验失败")
        record = json.loads(raw)
        if not isinstance(record, dict) or (revision is not None and record.get("revision_id") != revision["id"]):
            raise ValueError("亮迹修订记录身份不符")
        return record
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("亮迹记录缺失或损坏，请保留作业以供检查") from exc


def prepare(export_dir: Path, record: dict) -> dict:
    """Write into a fresh, unpublished export directory; caller commits job.json."""
    record["revision_id"] = export_dir.name
    path = tt.write_candidates(export_dir / "transients", record)
    return {"id": export_dir.name, "record": str(path.resolve()),
            "sha256": hashing.sha256_file(path, source="transient_revision.prepare")}
