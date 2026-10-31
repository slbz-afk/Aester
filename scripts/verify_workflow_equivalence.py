#!/usr/bin/env python3
"""Whole-workflow equivalence verifier (runtime efficiency plan P0 item 2).

Fingerprints a finished Aester workflow -- the workflow directory, its stack
job, the workflow's own delivery parent aester-<id> (the whole folder, so
unnamed versions and leftover .partial-* staging folders are seen) and every
other delivery folder the workflow record names -- into a manifest, and
compares two manifests file by file on canonical content, plus the directory
and link sets.  A bit-identical optimisation can then be accepted from stored
manifests (fingerprint, then delete the run) without keeping two full runs on
disk.

The tool only reads the workflow, its deliveries and the files its records
name.  It writes nothing but the requested manifest or report, and refuses to
write either inside a fingerprinted tree.

Commands
  fingerprint WORKFLOW_DIR -o MANIFEST [--workers N] [--root LABEL=PATH ...]
  compare A_MANIFEST B_MANIFEST [--allow GLOB ...] [--deep]
          [--tolerance GLOB=ABS ...] [--allow-field NAME_GLOB=PATH_REGEX ...]
          [--strict-ids | --no-strict-ids] [-o REPORT]
  self-test
  rules        print the exclusion list and the normalisation rules as JSON

Exit codes: 0 pass (or done), 1 a disallowed difference (or a self-test
failure), 2 a usage or input error.  A manifest ending in .gz is gzip-compressed.

What a manifest stores, per file (besides size and the raw SHA-256)
  FITS/.wcs  each HDU: header cards without DATE/CHECKSUM/DATASUM (dropped
             values kept for the report), and a digest of the data bytes with
             per-block digests
  TIFF       each page: a digest of the decoded pixels (C order, little
             endian) with per-block digests, every tag except strip/tile
             layout and DateTime; binary tags by digest (ICC without its
             creation time and profile ID); descriptions raw
  JPEG/PNG   a digest of the decoded pixels with blocks, format/mode/size and
             info (ICC as for TIFF, EXIF by digest, text chunks raw)
  NPY        dtype, shape, order and a digest of the data bytes with blocks
  JSON       the raw document (rules are applied when comparing)
  text       the raw text (.seq .lst .ssf .ini .md .txt .csv)
  logs, locks and transient-scan worker ledgers are listed, not compared
  anything else: a digest of the raw bytes with blocks
A "digest with blocks" is SHA-256 over consecutive block digests (4 MiB
blocks); a differing block locates the first differing byte range, which is
reported as an array index.  Large files are streamed; decoding is limited by
a memory budget and done by a bounded thread pool.

Canonicalisation, applied when comparing (so rules can change without
fingerprinting again)
  1. Roots.  Absolute paths under a recorded root become {LABEL}/rest with
     forward slashes: {WF} the workflow directory, {JOB} the stack job when it
     is outside the workflow, {DELIV} the workflow's own delivery parent
     aester-<sha256(workflow dir + NUL + signature)[:20]> (walked whole) or
     the parent of other named delivery folders (only those are walked;
     {DELIV1}, {DELIV2}... if several), {DELIVERY_ROOT}
     the requested delivery root, {INPUT} the input folder, {CACHE} the parent
     of the workflow directory, plus --root LABEL=PATH.  Longest root wins;
     matching ignores case and accepts / and \\ separators.
  2. Path ids.  A file stem or directory name ending in 8-40 lower-case hex
     characters after '-', '_' or '.' (or consisting of them) is a run id:
     workflow-<id>, job-<id>, finite-<id>, products/<rev>, delivery-<token>,
     delivery-trails-<n>kept-<token>, reviewed-<token>, aester-<id>,
     masks-<id>, milkyway-region-<id>, stack_snapshot-<id>, scan-runs/<uuid>,
     draft-<id>, separation-/star-layers-/star-reduction-/star-advice-<id>...
     The token becomes <kind#n>, kind = the prefix (or <parent>-hex<len> for a
     bare token; a token seen under several prefixes takes the smallest, a
     name in the tree winning over one only named by a record), n = order of
     creation (birth time, then path) within the kind.  Paths that records
     name under {WF} {JOB} {DELIV*} {CACHE} but that are not in the tree
     (staging folders that are gone, cache siblings such as reference-<uuid>)
     get the same rule, ordered after the tree; input names never do.  Every
     occurrence of the token anywhere (names, records, text, metadata) maps
     to the same placeholder.  64-hex tokens are never path ids.
  3. Record ids.  Values of VALUE_ALIAS_KEYS and VALUE_ALIAS_PATTERNS, the
     workflow signature (verified as the digest of the request), the stack job
     signature (verified as core.submit_job's digest of the job admission
     fields; those fields are compared) and verified trail review keys become
     placeholders the same way.  A request, a job admission field or a review
     key source that contains an excluded key (a timing, pid, ... field) fails
     the comparison: such fields must never feed a signature or key.
  4. File digests.  A 64-hex value equal to the SHA-256 of a fingerprinted
     file on the same side becomes <sha256-of:canonical name(s)>; a size next
     to it that equals that file's size is masked the same way.
  5. Exclusions (EXCLUDED_KEYS, EXCLUDED_KEY_PATTERNS; not KEPT_KEYS; *_at
     only for timestamp-like values) drop a key wherever it appears; derived
     digests are recomputed on each side and compared as verified (a failed
     recomputation fails the comparison); digests and sizes of files that no
     longer exist, and sizes of record files, are masked (DICT_RULES).  FITS
     commentary cards split at 72 characters are joined first.  Every
     exclusion that actually hid a difference is listed in the report.
     Exclusions hide value differences only: an excluded key (or dropped
     FITS card / TIFF tag) present on one side only fails the comparison
     ("one-sided excluded field"), unless its canonical file name and path
     match a DIAGNOSTIC_SLOTS entry (the instrumentation's timing receipts)
     or --allow-field NAME_GLOB=PATH_REGEX; allowed ones are listed.
  6. Deterministic ids (strict ids).  When the two manifests share their base
     roots ({WF} {CACHE} {INPUT} {DELIVERY_ROOT} and --root labels; {JOB} and
     {DELIV*} are derived from the signature), the runs are reruns in one
     slot with one idempotency key, and the ids of DETERMINISTIC_ID_KINDS
     (workflow folder, workflow signature, stack job folder, stack job
     signature, delivery parent) must be literally equal; a difference fails
     ("DETERMINISTIC IDS DIFFER").  --strict-ids forces this check,
     --no-strict-ids turns it off.
  7. Tree shape.  The canonical directory sets must be equal (a directory, even
     an empty one, on one side only fails unless --allow matches its name or
     name + '/'); links must not differ (the product never creates any).
  Per side, every {path, sha256} binding in a record is checked against the
  file (outside files are hashed too); a mismatch fails the comparison.

Everything not listed is compared exactly: floats by repr, integers and
strings literally, pixels bit for bit.  --allow GLOB marks canonical names
whose difference or absence is intended; --tolerance GLOB=ABS accepts a
numeric difference up to ABS in the data of matching files (it implies
--deep for them and needs both runs on disk); both are printed in the report.
"""
from __future__ import annotations

import argparse
import fnmatch
import gzip
import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

TOOL_VERSION = "1.1.0"
MANIFEST_SCHEMA = "aester-workflow-fingerprint-1"
REPORT_SCHEMA = "aester-workflow-equivalence-report-1"
DEFAULT_BLOCK = 4 * 1024 * 1024
READ_CHUNK = 4 * 1024 * 1024
TEXT_INLINE_LIMIT = 16 * 1024 * 1024
DEFAULT_MEMORY = 2 * 1024 ** 3
LIST_LIMIT = 200

# --------------------------------------------------------------------------
# Rules.  Everything not listed here is compared.  Each entry carries its
# reason; the report repeats every rule that hid an actual difference.
# --------------------------------------------------------------------------
EXCLUDED_KEYS = {
    "timings": "stage timer block (monotonic stage durations and the P0 per-stage CPU/memory/I/O measurements)",
    "timing_attempts": "timer blocks of earlier attempts of the same job",
    "seconds": "stage duration",
    "megapixels_per_second": "throughput derived from a duration",
    "worker_pid": "process id of the worker process",
    "pid": "process id",
    "recorded_pid_alive": "whether a recorded process was alive (process state)",
    "render_workers": "render worker count (scheduling; renders are bit-identical for any count by contract)",
    "workers": "worker count of a pool or of the transient scan (scheduling)",
    "implementation": "code-path label; an optimisation A/B changes it on purpose while its outputs are compared",
    "free_bytes_at_admission": "free disk space observed at admission (environment)",
    "export_before_sha256": "digest of the pre-zeroing Siril export: its bytes embed the FITS DATE card and the state "
                            "is overwritten in place; the post-state data is compared",
    "signed_trail_source_sha256": "digest of a staging fixed-colour render whose TIFF metadata embeds a random "
                                  "products/<rev> path; the staging file is not retained; final pixels are compared",
}
EXCLUDED_KEY_PATTERNS = (
    (r".+_at", "wall-clock timestamp (created_at, updated_at, started_at, applied_at, ...; "
               "only ISO-8601, epoch-number or null values)"),
    (r".+_seconds", "duration or timer value (..._seconds), except KEPT_KEYS"),
)
KEPT_KEYS = {
    "input_total_seconds": "total exposure time of the selected inputs (semantic)",
    "included_total_seconds": "total exposure time of the stacked frames (semantic)",
    "timeout_seconds": "configured timeout (a request parameter)",
}
LISTED_ONLY = (  # (glob on the root-relative path, lower case; class; reason) -- listed, never compared
    ("*.log", "log", "process or tool log (timings, pids, paths): listed, not compared"),
    ("*.lock", "lock", "lock file (owner and pid bookkeeping): listed, not compared"),
    ("*/scan-runs/*/worker-*.json", "ledger",
     "transient-scan worker ledger, one per worker process; the count follows the worker setting"),
)
LISTED_CLASSES = {cls for _, cls, _ in LISTED_ONLY}
FITS_DROPPED_CARDS = {
    "DATE": "file write time",
    "CHECKSUM": "header and data checksum; covers DATE",
    "DATASUM": "data checksum; redundant with the data digest compared here",
}
TIFF_DROPPED_TAGS = {
    "StripOffsets": "strip layout", "StripByteCounts": "strip layout", "RowsPerStrip": "strip layout",
    "TileOffsets": "tile layout", "TileByteCounts": "tile layout", "DateTime": "file write time",
}
ICC_DROPPED_FIELDS = {
    "date": "ICC header bytes 24-35: profile creation time (a profile built at run time carries the clock)",
    "profile_id": "ICC header bytes 84-99: MD5 profile ID, computed over a header that includes the creation time",
}
VALUE_ALIAS_KEYS = {
    "draft_revision": "trail review draft token (uuid4, trail_review_session)",
    "application_id": "trail review application token (uuid4, trail_review_session)",
}
VALUE_ALIAS_PATTERNS = (
    (r"(?P<kind>candidate|reference|render)-(?P<id>[0-9a-f]{32})", "uuid4 ids of finishing sessions"),
)
DERIVED_RULES = {
    "data_sha256": "workflow checkpoint digest of its sibling 'data' (covers ids and durations): recomputed and "
                   "compared as verified; 'data' itself is compared",
    "signature+request": "workflow signature = digest of the sibling 'request' (embeds run paths): recomputed, "
                         "compared as verified; the request is compared",
    "review key": "trail review delivery key = digest of {revision, signature[, options]}: recomputed in the "
                  "receipt; workflow.json review_deliveries keys must name a verified receipt key",
    "job signature": "stack job signature = core.submit_job's digest of the job admission fields (inputs, mask, "
                     "frame_masks, engine, transient_marks, support/range/engine contracts, admission_options), "
                     "which embed paths: recomputed where a record holds those fields, aliased consistently; "
                     "the fields are compared",
    "X_sha256+X": "a digest whose sibling X (object or list) is its preimage under one of the product's JSON "
                  "digest recipes: compared as verified; X is compared",
}
DICT_RULES = {
    "unretained file": "a {path, sha256, bytes} (or X, X_sha256, X_bytes) whose file no longer existed when "
                       "fingerprinted: digest and size masked (the state is gone; its path is still compared)",
    "record file size": "the byte size of a referenced record file (.json .txt .md .seq .ssf .ini .lst .log .csv) "
                        "is masked: records embed durations; their content is compared as a file",
    "fingerprinted file size": "a size recorded next to the digest of a fingerprinted file, equal to that file's size, "
                               "is masked like the digest: the file itself is compared, and its size changes with the "
                               "length of paths embedded in it",
}
# Places where the P0 instrumentation keeps its timing receipts (outside every signature, checkpoint data digest
# and delivery receipt).  An excluded key present on one side only is accepted here, and only here; anywhere else
# it fails.  (canonical file name globs, full-match regex on the field path, reason)
DIAGNOSTIC_SLOTS = (
    (("{WF}/workflow.json",), r"/(?:timings|timing_attempts)",
     "workflow worker timing receipt and those of earlier attempts (workflow._record_timings)"),
    (("{WF}/workflow.json",), r"/review_deliveries\[\d+\]/(?:started_at|finished_at|timings)",
     "timing of one trail apply, kept in the workflow index entry of a published version, never in its receipt "
     "(workflow_edit._apply_timings)"),
    (("{WF}/stack/job-<job#*>/job.json", "{JOB}/job.json"), r"/diagnostics/(?:timings|timing_attempts)",
     "stack job timing receipt and those of earlier attempts (core timed_diagnostics)"),
)
# Ids that a rerun in the same slot (same base roots, same idempotency key) reproduces exactly.
DETERMINISTIC_ID_KINDS = {
    "workflow": "workflow-<sha256(idempotency key)[:24]>: the workflow folder",
    "workflow-signature": "digest of the request (inputs, parameters, roots)",
    "job": "job-<sha256('workflow-stack-' + workflow signature)[:24]>: the stack job folder",
    "job-signature": "digest of the stack job admission fields",
    "aester": "aester-<sha256(workflow dir + NUL + signature)[:20]>: the workflow's delivery parent",
}
NOT_DETERMINISTIC_IDS = {
    "review-key": "digest of {transient revision, signature[, options]}; the revision id is the uuid4 products/<rev> "
                  "folder, so it changes per run: it is recomputed from that source, which is compared",
}
# Root label prefixes derived from the signature ({JOB}, {DELIV}, {DELIVn}; not {DELIVERY_ROOT}): named by ids
# that strict mode checks, so they are not part of the base roots that identify a slot.
BASE_ROOTS_EXCLUDED = ("{JOB}", "{DELIV")
RUN_ROOTS =("{WF}", "{JOB}", "{DELIV", "{DELIVERY_ROOT}", "{CACHE}")  # ids in paths under these are run ids
TEXT_SUFFIXES = {".seq", ".lst", ".ssf", ".ini", ".md", ".txt", ".csv"}
RECORD_SUFFIXES = {".json", ".txt", ".md", ".seq", ".ssf", ".ini", ".lst", ".log", ".csv"}
FITS_SUFFIXES = {".fit", ".fits", ".fts", ".wcs"}

_TIMESTAMP = re.compile(r"\d{4}-?\d{2}-?\d{2}(?:[T ]\d{2}:?\d{2}(?::?\d{2}(?:[.,]\d+)?)?)?(?:Z|[+-]\d{2}(?::?\d{2})?)?")


def _looks_like_time(value):
    if value is None or (isinstance(value, (int, float)) and not isinstance(value, bool)):
        return True
    return isinstance(value, str) and bool(_TIMESTAMP.fullmatch(value))


_KEY_PATTERNS = tuple((re.compile(pattern), reason, _looks_like_time if pattern == r".+_at" else None)
                      for pattern, reason in EXCLUDED_KEY_PATTERNS)
_VALUE_PATTERNS = tuple((re.compile(pattern), reason) for pattern, reason in VALUE_ALIAS_PATTERNS)
_TOKEN = re.compile(r"(?<![0-9A-Za-z])[0-9a-f]{8,64}(?![0-9A-Za-z])")
_SEGMENT_ID = re.compile(r"(?P<prefix>(?:.*[-_.])?)(?P<id>[0-9a-f]{8,40})")
_SHA = re.compile(r"[0-9a-f]{64}")
_ABS_WIN = re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\[^\\/])")
_NAME_CHARS = r"[\w\-.~$#@+=]"


def excluded_key_reason(key, value=None):
    """Reason when a JSON key (with this value) is excluded wherever it appears, else None."""
    if not isinstance(key, str) or key in KEPT_KEYS:
        return None
    if key in EXCLUDED_KEYS:
        return EXCLUDED_KEYS[key]
    for pattern, reason, accepts in _KEY_PATTERNS:
        if pattern.fullmatch(key) and (accepts is None or accepts(value)):
            return reason
    return None


def rules():
    """Every rule with its reason (printed by 'rules' and embedded in each report)."""
    return {"excluded_keys": EXCLUDED_KEYS, "excluded_key_patterns": dict(EXCLUDED_KEY_PATTERNS),
            "kept_keys": KEPT_KEYS, "listed_only_files": [list(x) for x in LISTED_ONLY],
            "fits_dropped_cards": FITS_DROPPED_CARDS, "tiff_dropped_tags": TIFF_DROPPED_TAGS,
            "icc_dropped_fields": ICC_DROPPED_FIELDS,
            "value_alias_keys": VALUE_ALIAS_KEYS, "value_alias_patterns": dict(VALUE_ALIAS_PATTERNS),
            "derived_digests": DERIVED_RULES, "dict_rules": DICT_RULES,
            "one_sided_excluded_fields": "an excluded key, dropped FITS card or TIFF tag present on one side only "
                                         "fails, except in these diagnostic slots or --allow-field",
            "diagnostic_slots": [[list(names), path, reason] for names, path, reason in DIAGNOSTIC_SLOTS],
            "deterministic_id_kinds": DETERMINISTIC_ID_KINDS, "not_deterministic_ids": NOT_DETERMINISTIC_IDS,
            "strict_ids": "on when the base roots are equal (all roots but {JOB} and {DELIV*}), or with "
                          "--strict-ids: the ids of deterministic_id_kinds must be literally equal",
            "tree_shape": "canonical directory sets must be equal (--allow NAME or NAME/ accepts one); links must "
                          "not differ",
            "roots": "{WF} workflow dir; {JOB} stack job outside it; {DELIV}/{DELIVn} delivery parents (the "
                     "workflow's own aester-<id> walked whole); {DELIVERY_ROOT}; {INPUT}; {CACHE} parent of the "
                     "workflow dir; --root LABEL=PATH",
            "path_ids": "stem/dir name ending in 8-40 lower-case hex after -/_/. -> <kind#n> by creation order"}


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def _io(path):
    """Path string usable for I/O, also beyond MAX_PATH on Windows."""
    text = str(path)
    if os.name == "nt" and len(text) >= 240 and not text.startswith("\\\\?\\"):
        text = os.path.abspath(text)
        return "\\\\?\\UNC\\" + text[2:] if text.startswith("\\\\") else "\\\\?\\" + text
    return text


def _is_abs_path(text):
    if not isinstance(text, str) or not text or len(text) > 1024 or "\n" in text:
        return False
    if _ABS_WIN.match(text):
        return True
    return os.name != "nt" and text.startswith("/") and "//" not in text


def _norm_key(path):
    return os.path.normcase(os.path.normpath(str(path)))


def _inside(path, root):
    path, root = _norm_key(path), _norm_key(root)
    return path == root or path.startswith(root.rstrip("\\/") + os.sep)


def _birth_ns(st):
    value = getattr(st, "st_birthtime_ns", None)
    return int(value) if value is not None else int(st.st_mtime_ns)


def _plain(value):
    """A JSON-safe copy of a metadata value (no normalisation)."""
    import enum
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return int(value)  # IntEnum tag values are stored as their number
    if isinstance(value, float):
        return value if math.isfinite(value) else {"$float": repr(value)}
    if isinstance(value, enum.Enum):
        return value.name
    if isinstance(value, (bytes, bytearray, memoryview)):
        data = bytes(value)
        return {"$bytes_sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, np.ndarray):
        if value.size <= 64:
            return {"$array": _plain(value.tolist()), "dtype": value.dtype.str}
        data = np.ascontiguousarray(value).tobytes()
        return {"$array_sha256": hashlib.sha256(data).hexdigest(), "dtype": value.dtype.str, "shape": list(value.shape)}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if type(value).__name__ == "Undefined":
        return None
    return str(value)


def _icc(data):
    """Digest of an ICC profile without its creation time and MD5 profile ID (kept aside for the report)."""
    data = bytes(data)
    if len(data) < 128 or data[36:40] != b"acsp":
        return _plain(data)
    masked = data[:24] + bytes(12) + data[36:84] + bytes(16) + data[100:]
    return {"$icc_sha256": hashlib.sha256(masked).hexdigest(), "bytes": len(data),
            "$volatile": {"date": data[24:36].hex(), "profile_id": data[84:100].hex()}}


def _json_digest(value, *, ensure_ascii, compact):
    text = json.dumps(value, sort_keys=True, ensure_ascii=ensure_ascii, allow_nan=False,
                      separators=(",", ":") if compact else None)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _digest_matches(value, digest):
    """True when digest is the SHA-256 of value under one of the product's JSON digest recipes."""
    for ensure_ascii in (False, True):
        for compact in (False, True):
            try:
                if _json_digest(value, ensure_ascii=ensure_ascii, compact=compact) == digest:
                    return True
            except (TypeError, ValueError):
                return False
    return False


JOB_SIGNATURE_FIELDS = ("inputs", "mask", "frame_masks", "engine", "transient_marks", "support_mode",
                        "support_version", "fits_input", "stack_range_contract", "stack_engine", "admission_options")


def _job_signature_preimage(job):
    """core.submit_job's signature preimage rebuilt from a stack job record (None without its admission fields)."""
    if not isinstance(job.get("inputs"), list) or "engine" not in job:
        return None
    preimage = {"files": job["inputs"], "mask": job.get("mask"), "frame_masks": job.get("frame_masks"),
                "engine": job["engine"], "transient_marks": job.get("transient_marks")}
    if job.get("support_mode") not in (None, "legacy"):
        preimage.update(support_mode=job["support_mode"], support_version=job.get("support_version"))
    for key in ("fits_input", "stack_range_contract", "admission_options"):
        if job.get(key):
            preimage[key] = job[key]
    if "stack_engine" in job:  # present exactly when the finite stack engine was admitted
        preimage["stack_engine"] = job["stack_engine"]
    return preimage


def _short(value, limit=240):
    text = json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[:limit] + "..."


def _read_json_file(path):
    raw = Path(path).read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return json.loads(raw.decode("utf-8-sig"))


def _write_json_file(path, value, *, overwrite=False):
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path} (use --overwrite)")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if path.suffix.lower() == ".gz":
        data = gzip.compress(data, compresslevel=6, mtime=0)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_bytes(data)
    os.replace(temporary, path)


class _Budget:
    """Bounds the memory of concurrent decodes (a weighted semaphore)."""

    def __init__(self, limit):
        self.limit = max(1, int(limit))
        self.used = 0
        self.cond = threading.Condition()

    def acquire(self, amount):
        amount = min(max(1, int(amount)), self.limit)
        with self.cond:
            while self.used + amount > self.limit:
                self.cond.wait()
            self.used += amount
        return amount

    def release(self, amount):
        with self.cond:
            self.used -= amount
            self.cond.notify_all()


class _BlockDigest:
    """SHA-256 of consecutive fixed-size blocks; the whole digest is SHA-256 over the block digests."""

    def __init__(self, block_bytes):
        self.block_bytes = block_bytes
        self.current = hashlib.sha256()
        self.fill = 0
        self.size = 0
        self.digests = []

    def update(self, data):
        view = memoryview(data).cast("B")
        self.size += len(view)
        while len(view):
            take = min(self.block_bytes - self.fill, len(view))
            self.current.update(view[:take])
            self.fill += take
            view = view[take:]
            if self.fill == self.block_bytes:
                self.digests.append(self.current.digest())
                self.current = hashlib.sha256()
                self.fill = 0

    def finish(self):
        if self.fill or not self.digests:
            self.digests.append(self.current.digest())
        whole = hashlib.sha256(self.size.to_bytes(8, "little") + b"".join(self.digests)).hexdigest()
        return {"digest": whole, "bytes": self.size, "blocks": [d.hex()[:32] for d in self.digests]}


def _stream_file(path, ranges, block_bytes):
    """Raw SHA-256 of the whole file plus block digests of each (start, length) range, in one pass."""
    digests = [_BlockDigest(block_bytes) for _ in ranges]
    order = sorted(range(len(ranges)), key=lambda i: ranges[i][0])
    whole = hashlib.sha256()
    buffer = bytearray(READ_CHUNK)
    view = memoryview(buffer)
    position = 0
    with open(_io(path), "rb", buffering=0) as handle:
        while True:
            count = handle.readinto(buffer)
            if not count:
                break
            chunk = view[:count]
            whole.update(chunk)
            end = position + count
            for i in order:
                start, length = ranges[i]
                lo, hi = max(start, position), min(start + length, end)
                if lo < hi:
                    digests[i].update(chunk[lo - position:hi - position])
            position = end
    results = []
    for (start, length), digest in zip(ranges, digests):
        result = digest.finish()
        if result["bytes"] != length:
            raise ValueError(f"file ends inside a data range ({result['bytes']} of {length} bytes at {start})")
        results.append(result)
    return whole.hexdigest(), position, results


def _array_digest(array, block_bytes):
    array = np.ascontiguousarray(array)
    if array.dtype.byteorder == ">" or (array.dtype.byteorder == "=" and sys.byteorder == "big"):
        array = array.astype(array.dtype.newbyteorder("<"))
    digest = _BlockDigest(block_bytes)
    flat = array.reshape(-1).view(np.uint8) if array.size else np.zeros(0, np.uint8)
    for start in range(0, flat.size, READ_CHUNK):
        digest.update(flat[start:start + READ_CHUNK])
    return digest.finish()


def _canonical_dtype(dtype):
    dtype = np.dtype(dtype)
    return dtype.newbyteorder("<").str if dtype.itemsize > 1 else dtype.str


# --------------------------------------------------------------------------
# Per-file fingerprints
# --------------------------------------------------------------------------
def classify(rel):
    low = rel.lower()
    for pattern, cls, _ in LISTED_ONLY:
        if fnmatch.fnmatchcase(low, pattern):
            return cls
    suffix = os.path.splitext(low)[1]
    if suffix in FITS_SUFFIXES:
        return "fits"
    if suffix in (".tif", ".tiff"):
        return "tiff"
    if suffix in (".jpg", ".jpeg", ".png"):
        return "image"
    if suffix == ".npy":
        return "npy"
    if suffix == ".json":
        return "json"
    if suffix in TEXT_SUFFIXES:
        return "text"
    return "bytes"


def _fits_data_bytes(header):
    naxis = int(header.get("NAXIS", 0) or 0)
    if not naxis:
        return 0, None, None
    bitpix = int(header["BITPIX"])
    dims = [int(header[f"NAXIS{i}"]) for i in range(1, naxis + 1)]
    groups = bool(header.get("GROUPS")) and dims and dims[0] == 0
    if groups:
        dims = dims[1:]
    count = int(np.prod(dims, dtype=np.int64)) if dims else 0
    pcount, gcount = int(header.get("PCOUNT", 0) or 0), int(header.get("GCOUNT", 1) or 1)
    nbytes = abs(bitpix) // 8 * gcount * (pcount + count)
    image = not groups and pcount == 0 and gcount == 1 and header.get("XTENSION", "IMAGE") in ("IMAGE", None)
    dtype = {8: "u1", 16: ">i2", 32: ">i4", 64: ">i8", -32: ">f4", -64: ">f8"}.get(bitpix)
    if image and dtype:
        return nbytes, dtype, list(reversed(dims))
    return nbytes, "u1", [nbytes]


def _fits(path, block_bytes, budget):
    from astropy.io import fits
    hdus = []
    with fits.open(_io(path), mode="readonly", memmap=True, lazy_load_hdus=True,
                   do_not_scale_image_data=True) as hdul:
        for hdu in hdul:
            cards, dropped = [], []
            for card in hdu.header.cards:
                if card.keyword in FITS_DROPPED_CARDS:
                    dropped.append([card.keyword, _plain(card.value)])
                    continue
                cards.append([card.keyword, _plain(card.value), str(card.comment or "")])
            nbytes, dtype, shape = _fits_data_bytes(hdu.header)
            hdus.append({"cards": cards, "dropped": dropped, "offset": int(hdu.fileinfo()["datLoc"]),
                         "data_bytes": nbytes, "dtype": dtype, "shape": shape})
    ranges = [(h["offset"], h["data_bytes"]) for h in hdus if h["data_bytes"]]
    raw, size, results = _stream_file(path, ranges, block_bytes)
    it = iter(results)
    for h in hdus:
        h["data"] = next(it) if h["data_bytes"] else None
    return raw, size, {"hdus": hdus}


def _tiff_stream_range(tf, page):
    try:
        if page.is_tiled or not page.is_contiguous or int(page.compression) != 1:
            return None
        if int(getattr(page, "predictor", 1) or 1) != 1 or int(getattr(page, "fillorder", 1) or 1) != 1:
            return None
        if int(page.photometric) == 6:  # YCbCr subsampling
            return None
        dtype = page.dtype
        if dtype is None or dtype.kind not in "biuf" or (dtype.itemsize > 1 and tf.byteorder != "<"):
            return None
        bits = page.bitspersample if isinstance(page.bitspersample, int) else page.bitspersample[0]
        if bits != dtype.itemsize * 8:
            return None
        nbytes = int(np.prod(page.shape, dtype=np.int64)) * dtype.itemsize
        position = int(page.dataoffsets[0])
        for offset, count in zip(page.dataoffsets, page.databytecounts):
            if int(offset) != position:
                return None
            position += int(count)
        if position - int(page.dataoffsets[0]) != nbytes:
            return None
        return int(page.dataoffsets[0]), nbytes
    except Exception:
        return None


def _tiff(path, block_bytes, budget):
    import tifffile
    pages, ranges = [], []
    with tifffile.TiffFile(_io(path)) as tf:
        for index, page in enumerate(tf.pages):
            tags, dropped = [], {}
            for tag in page.tags:
                if tag.name in TIFF_DROPPED_TAGS:
                    dropped[tag.name] = _plain(tag.value) if not isinstance(tag.value, (tuple, list)) \
                        else {"$value_sha256": hashlib.sha256(repr(tag.value).encode()).hexdigest()}
                    continue
                value = _icc(tag.value) if tag.code == 34675 and isinstance(tag.value, (bytes, bytearray)) \
                    else _plain(tag.value)
                tags.append([int(tag.code), tag.name, value])
            entry = {"tags": tags, "dropped": dropped}
            stream = _tiff_stream_range(tf, page)
            if stream:
                entry.update(shape=list(page.shape), dtype=_canonical_dtype(page.dtype), stream=list(stream))
                ranges.append((index, stream))
            else:
                estimate = 3 * int(np.prod(page.shape, dtype=np.int64)) * max(1, (page.dtype or np.dtype("u1")).itemsize)
                held = budget.acquire(estimate)
                try:
                    array = page.asarray(maxworkers=1)
                    entry.update(shape=list(array.shape), dtype=_canonical_dtype(array.dtype),
                                 pixels=_array_digest(array, block_bytes))
                    del array
                finally:
                    budget.release(held)
            pages.append(entry)
    raw, size, results = _stream_file(path, [r for _, r in ranges], block_bytes)
    for (index, _), result in zip(ranges, results):
        pages[index]["pixels"] = result
    return raw, size, {"pages": pages}


def _image(path, block_bytes, budget):
    from PIL import Image
    raw, size, _ = _stream_file(path, [], block_bytes)
    with Image.open(_io(path)) as im:
        bands = len(im.getbands())
        per = 4 if im.mode in ("I", "F", "RGBA", "CMYK") else 2 if "16" in im.mode else 1
        held = budget.acquire(3 * im.size[0] * im.size[1] * bands * per)
        try:
            im.load()
            info = {str(k): _icc(v) if k == "icc_profile" and isinstance(v, (bytes, bytearray)) else _plain(v)
                    for k, v in im.info.items()}
            palette = im.getpalette() if im.mode in ("P", "PA") else None
            array = np.asarray(im)
            content = {"format": im.format, "mode": im.mode, "size": list(im.size), "info": info,
                       "frames": int(getattr(im, "n_frames", 1) or 1), "shape": list(array.shape),
                       "dtype": _canonical_dtype(array.dtype), "pixels": _array_digest(array, block_bytes)}
            if palette is not None:
                content["palette_sha256"] = hashlib.sha256(bytes(palette)).hexdigest()
            del array
        finally:
            budget.release(held)
    return raw, size, content


def _npy(path, block_bytes, budget):
    with open(_io(path), "rb") as handle:
        version = np.lib.format.read_magic(handle)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(handle)
        else:
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(handle)
        offset = handle.tell()
    if dtype.hasobject:
        raise ValueError("object arrays are not supported")
    nbytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    raw, size, (data,) = _stream_file(path, [(offset, nbytes)], block_bytes)
    return raw, size, {"dtype": dtype.str, "shape": list(shape), "fortran_order": bool(fortran),
                       "offset": offset, "data": data}


def _json_file(path, block_bytes, budget):
    raw = Path(_io(path)).read_bytes()
    content = {}
    try:
        content["doc"] = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, ValueError) as exc:
        content = {"json_error": f"{type(exc).__name__}: {exc}", "text": raw.decode("utf-8", errors="replace")}
    return hashlib.sha256(raw).hexdigest(), len(raw), content


def _text(path, block_bytes, budget):
    size = os.stat(_io(path)).st_size
    if size > TEXT_INLINE_LIMIT:
        raw, size, (data,) = _stream_file(path, [(0, size)], block_bytes)
        return raw, size, {"too_large": True, "data": data}
    raw = Path(_io(path)).read_bytes()
    return hashlib.sha256(raw).hexdigest(), len(raw), {"text": raw.decode("utf-8", errors="replace")}


def _bytes(path, block_bytes, budget):
    size = os.stat(_io(path)).st_size
    raw, size, (data,) = _stream_file(path, [(0, size)], block_bytes)
    return raw, size, {"data": data}


def _listed(path, block_bytes, budget):
    raw, size, _ = _stream_file(path, [], block_bytes)
    return raw, size, None


_HANDLERS = {"fits": _fits, "tiff": _tiff, "image": _image, "npy": _npy, "json": _json_file,
             "text": _text, "bytes": _bytes, "log": _listed, "lock": _listed, "ledger": _listed}


def fingerprint_file(path, rel, block_bytes=DEFAULT_BLOCK, budget=None):
    """One manifest entry.  Decoding errors are recorded (and fail any comparison), never raised."""
    budget = budget or _Budget(DEFAULT_MEMORY)
    cls = classify(rel)
    st = os.stat(_io(path))
    entry = {"path": str(path), "class": cls, "bytes": st.st_size, "mtime_ns": st.st_mtime_ns,
             "birth_ns": _birth_ns(st)}
    if st.st_nlink > 1:
        entry["nlink"] = st.st_nlink
    try:
        raw, size, content = _HANDLERS[cls](path, block_bytes, budget)
        entry["sha256"] = raw
        entry["bytes"] = size
        if content is not None:
            entry["content"] = content
    except Exception as exc:
        entry["error"] = f"{type(exc).__name__}: {exc}"
        try:
            entry["sha256"] = _stream_file(path, [], block_bytes)[0]
        except OSError:
            pass
    return entry


# --------------------------------------------------------------------------
# Fingerprint a workflow
# --------------------------------------------------------------------------
def _usable_root(path):
    """A root must be specific enough that substituting it cannot hit unrelated paths."""
    path = Path(path)
    return len(path.parts) >= 3


def own_delivery_parent(record, wf):
    """The workflow's own delivery parent (storage.publication_parent), when it exists outside the workflow."""
    request = record.get("request") if isinstance(record.get("request"), dict) else {}
    root, signature = request.get("delivery_root"), record.get("signature")
    if not (isinstance(root, str) and root and isinstance(signature, str) and signature):
        return None
    for directory in dict.fromkeys(x for x in (str(wf), record.get("workflow_dir")) if isinstance(x, str) and x):
        parent = Path(root) / ("aester-" + hashlib.sha256((directory + "\0" + signature).encode()).hexdigest()[:20])
        if parent.is_dir() and not _inside(parent, wf):
            return parent.resolve()
    return None


def discover(workflow_dir, extra_roots=None):
    """(record, roots {label: path}, walked [(label, dir)], missing named delivery folders).

    The workflow's own delivery parent is walked whole (it belongs to this workflow alone, so any unnamed
    version or leftover staging folder in it is part of the run); other named delivery folders one by one."""
    wf = Path(workflow_dir).expanduser().resolve(strict=True)
    record_path = wf / "workflow.json"
    if not record_path.is_file():
        raise ValueError(f"{wf} has no workflow.json")
    record = _read_json_file(record_path)
    request = record.get("request") if isinstance(record.get("request"), dict) else {}
    roots = {"{WF}": str(wf)}
    walked = [("{WF}", wf)]
    job = record.get("stack_job_dir")
    if isinstance(job, str) and job:
        job_path = Path(job)
        if job_path.is_dir() and not _inside(job_path, wf):
            roots["{JOB}"] = str(job_path.resolve())
            walked.append(("{JOB}", job_path.resolve()))
    named = []
    for key in ("delivery_dir", "base_delivery_dir"):
        if isinstance(record.get(key), str) and record[key]:
            named.append(record[key])
    for item in record.get("review_deliveries") or []:
        if isinstance(item, dict) and isinstance(item.get("delivery_dir"), str):
            named.append(item["delivery_dir"])
    if isinstance(record.get("publication_receipts"), dict):
        named.extend(k for k in record["publication_receipts"] if isinstance(k, str))
    missing, outside = [], {}
    own = own_delivery_parent(record, wf)
    if own is not None and not any(_inside(own, r) for _, r in walked):
        outside[str(own)] = []
    for text in dict.fromkeys(named):
        path = Path(text)
        if not path.is_dir():
            missing.append(text)
        elif not _inside(path, wf) and not any(_inside(path, r) for _, r in walked):
            parent = path.resolve().parent
            key = next((p for p in outside if _norm_key(p) == _norm_key(parent)), str(parent))
            outside.setdefault(key, []).append(path.resolve())
    parents = sorted(outside)
    for number, parent in enumerate(parents, 1):
        label = "{DELIV}" if len(parents) == 1 else "{DELIV%d}" % number
        roots[label] = parent
        if own is not None and _norm_key(parent) == _norm_key(own):
            walked.append((label, Path(parent)))  # whole folder
            continue
        for folder in sorted(outside[parent], key=str):
            walked.append((label, folder))
    for label, key in (("{DELIVERY_ROOT}", "delivery_root"), ("{INPUT}", "input_dir")):
        value = request.get(key)
        if isinstance(value, str) and value and _usable_root(value):
            roots[label] = os.path.normpath(value)
    if _usable_root(wf.parent):
        roots["{CACHE}"] = str(wf.parent)
    for label, value in (extra_roots or {}).items():
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", label):
            raise ValueError(f"root label must be upper case letters/digits/_: {label}")
        if "{" + label + "}" in roots:
            raise ValueError(f"root label {label} is reserved")
        if not _usable_root(value):
            raise ValueError(f"root {value} is too short to substitute safely")
        roots["{" + label + "}"] = os.path.normpath(str(Path(value).expanduser().resolve()))
    return record, roots, walked, missing


def _walk(label, top, root_path):
    """Yield (kind, rel, real path, birth) for every directory and file below top."""
    prefix = "\\\\?\\" if os.name == "nt" else ""
    start = prefix + str(top) if prefix and not str(top).startswith("\\\\") else str(top)
    base = str(root_path)
    for dirpath, dirnames, filenames in os.walk(start, followlinks=False):
        real_dir = dirpath[len(prefix):] if prefix and dirpath.startswith(prefix) else dirpath
        keep = []
        for name in sorted(dirnames):
            full = os.path.join(real_dir, name)
            if os.path.islink(_io(full)) or (hasattr(os.path, "isjunction") and os.path.isjunction(_io(full))):
                yield "link", _rel(label, base, full), full, None
                continue
            keep.append(name)
            yield "dir", _rel(label, base, full), full, _birth_ns(os.stat(_io(full)))
        dirnames[:] = keep
        for name in sorted(filenames):
            full = os.path.join(real_dir, name)
            if os.path.islink(_io(full)):
                yield "link", _rel(label, base, full), full, None
                continue
            yield "file", _rel(label, base, full), full, None


def _rel(label, base, full):
    rest = os.path.relpath(full, base)
    return label + "/" + rest.replace("\\", "/")


def _strings_with_path(value):
    if isinstance(value, dict):
        for k, v in value.items():
            if _is_abs_path(k):
                yield k
            yield from _strings_with_path(v)
    elif isinstance(value, list):
        for v in value:
            yield from _strings_with_path(v)
    elif _is_abs_path(value):
        yield value


def _embedded_docs(entry):
    """(label, JSON document) pairs inside a manifest entry: records, TIFF descriptions, PNG text."""
    content = entry.get("content") or {}
    if entry["class"] == "json" and "doc" in content:
        yield "", content["doc"]
    elif entry["class"] == "tiff":
        for i, page in enumerate(content.get("pages", [])):
            for code, name, value in page.get("tags", []):
                doc = _maybe_json(value)
                if doc is not None:
                    yield f"#page{i}:{name}", doc
    elif entry["class"] == "image":
        for key, value in (content.get("info") or {}).items():
            doc = _maybe_json(value)
            if doc is not None:
                yield f"#info:{key}", doc


def _maybe_json(value):
    if isinstance(value, str) and value[:1] in "{[":
        try:
            doc = json.loads(value)
        except ValueError:
            return None
        return doc if isinstance(doc, (dict, list)) else None
    return None


def _bindings_of(value, where, out):
    """(where, path, digest key, digest, size key, size) for each file binding in a record."""
    if isinstance(value, dict):
        path_keys = [k for k, v in value.items() if isinstance(k, str) and _is_abs_path(v)]
        for k in path_keys:
            digest_key = "sha256" if k in ("path", "file") or (len(path_keys) == 1 and k + "_sha256" not in value) \
                else k + "_sha256"
            size_key = ("bytes" if "bytes" in value else "size") if k in ("path", "file") else k + "_bytes"
            digest = value.get(digest_key)
            if isinstance(digest, str) and _SHA.fullmatch(digest):
                size = value.get(size_key)
                out.append((where, value[k], digest_key, digest, size_key,
                            size if isinstance(size, int) and not isinstance(size, bool) else None))
        for v in value.values():
            _bindings_of(v, where, out)
    elif isinstance(value, list):
        for v in value:
            _bindings_of(v, where, out)


def fingerprint(workflow_dir, *, workers=None, extra_roots=None, block_bytes=DEFAULT_BLOCK,
                memory_bytes=DEFAULT_MEMORY, external_bindings=True, progress=None):
    """Manifest of a finished workflow (read-only)."""
    started = time.monotonic()
    record, roots, walked, missing = discover(workflow_dir, extra_roots)
    workers = max(1, int(workers or min(8, os.cpu_count() or 1)))
    budget = _Budget(memory_bytes)
    files, dirs, links = {}, {}, {}
    delivery_parents = {}
    for label, top in walked:
        base = roots[label]
        if label.startswith("{DELIV") and label != "{DELIVERY_ROOT}":
            whole = _norm_key(top) == _norm_key(base)
            delivery_parents[label] = "whole" if whole else "named"
            if not whole:
                dirs[_rel(label, base, str(top))] = _birth_ns(os.stat(_io(top)))
        for kind, rel, full, birth in _walk(label, top, base):
            if kind == "dir":
                dirs[rel] = birth
            elif kind == "link":
                links[rel] = os.readlink(_io(full)) if os.path.islink(_io(full)) else "<junction>"
            else:
                files[rel] = full
    order = sorted(files, key=lambda r: -os.stat(_io(files[r])).st_size)
    entries, done, lock = {}, [0], threading.Lock()
    total_bytes = sum(os.stat(_io(files[r])).st_size for r in files)

    def one(rel):
        entry = fingerprint_file(files[rel], rel, block_bytes, budget)
        with lock:
            done[0] += 1
            if progress and (done[0] % 25 == 0 or done[0] == len(order)):
                progress(f"fingerprinted {done[0]}/{len(order)} files")
        return rel, entry

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for rel, entry in pool.map(one, order):
            entries[rel] = entry
    # Paths named by records: which exist, and do their recorded digests match?
    tree = {_norm_key(e["path"]): e.get("sha256") for e in entries.values()}
    existing, candidates, bindings = set(), set(), []
    for rel in sorted(entries):
        for where, doc in _embedded_docs(entries[rel]):
            for text in _strings_with_path(doc):
                candidates.add(text)
            if entries[rel]["class"] == "json":
                _bindings_of(doc, rel + where, bindings)
    for text in candidates:
        key = _norm_key(text)
        if key in tree or os.path.isfile(_io(text)):
            existing.add(key)
    external = sorted({_norm_key(b[1]) for b in bindings if _norm_key(b[1]) not in tree and _norm_key(b[1]) in existing})
    external_digest = {}
    if external_bindings and external:
        if progress:
            progress(f"hashing {len(external)} external files named by records")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for key, digest in pool.map(lambda k: (k, _stream_file(k, [], block_bytes)[0]), external):
                external_digest[key] = digest
    result = {"verified": 0, "mismatch": [], "missing": [], "not_checked": 0}
    for where, path, digest_key, digest, size_key, size in bindings:
        key = _norm_key(path)
        actual = tree.get(key) or external_digest.get(key)
        if actual is None:
            if key in existing:
                result["not_checked"] += 1
            else:
                result["missing"].append([where, path, digest_key])
        elif actual != digest:
            result["mismatch"].append([where, path, digest_key, digest, actual])
        else:
            result["verified"] += 1
    manifest = {
        "schema": MANIFEST_SCHEMA, "tool_version": TOOL_VERSION, "environment": _environment(),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "block_bytes": block_bytes, "workers": workers, "workflow_dir": roots["{WF}"],
        "workflow_state": record.get("state"), "roots": roots,
        "walked": [[label, str(top)] for label, top in walked], "delivery_parents": delivery_parents,
        "missing_deliveries": missing,
        "files": {rel: entries[rel] for rel in sorted(entries)}, "dirs": dict(sorted(dirs.items())),
        "links": links, "existing_paths": sorted(existing), "referenced_paths": sorted(candidates),
        "bindings": result,
        "bytes_fingerprinted": total_bytes,
    }
    manifest["fingerprint_seconds"] = round(time.monotonic() - started, 3)
    return manifest


def _environment():
    versions = {"python": sys.version.split()[0], "numpy": np.__version__, "platform": sys.platform}
    for module in ("astropy", "tifffile", "PIL"):
        try:
            versions[module] = __import__(module).__version__
        except Exception:
            versions[module] = None
    return versions


# --------------------------------------------------------------------------
# Canonicalisation (compare time)
# --------------------------------------------------------------------------
class Canon:
    """Root replacement, id aliases and file-digest roles for one side."""

    def __init__(self, roots):
        forms = {}
        for label, root in roots.items():
            if not root:
                continue
            text = os.path.normpath(str(root))
            for form in (text, text.replace("\\", "/"), text.replace("\\", "\\\\"), text.replace("/", "\\")):
                forms.setdefault(form.lower(), (form, label))
        ordered = sorted(forms.values(), key=lambda f: -len(f[0]))
        self._labels = {form.lower(): label for form, label in ordered}
        self._root_re = re.compile("(?:" + "|".join(re.escape(form) for form, _ in ordered) + ")(?!" + _NAME_CHARS + ")",
                                   re.IGNORECASE) if ordered else None
        self.alias = {}
        self.digest_role = {}

    def _root(self, match):
        return self._labels[match.group(0).lower()]

    def _token(self, match):
        token = match.group(0)
        alias = self.alias.get(token)
        if alias is not None:
            return alias
        if len(token) == 64:
            return self.digest_role.get(token, token)
        return token

    def string(self, text):
        if self._root_re is not None:
            replaced, count = self._root_re.subn(self._root, text)
            if count:
                text = replaced.replace("\\\\", "/").replace("\\", "/")
        return _TOKEN.sub(self._token, text)


class Side:
    """One manifest with its canonical names and lazily normalised content."""

    def __init__(self, manifest, name):
        self.m = manifest
        self.name = name
        self.files = manifest["files"]
        self.canon = Canon(manifest["roots"])
        self.alias_sources = {}
        self._path_aliases()
        self.names = {rel: self.canon.string(rel) for rel in self.files}
        self.by_name = {}
        for rel, canonical in self.names.items():
            self.by_name.setdefault(canonical, []).append(rel)
        groups = {}
        for rel, entry in self.files.items():
            if entry.get("sha256") and entry["class"] not in LISTED_CLASSES:
                groups.setdefault(entry["sha256"], []).append(self.names[rel])
        self.canon.digest_role = {d: "<sha256-of:" + "|".join(sorted(n)) + ">" for d, n in groups.items()}
        self.size_by_digest = {e["sha256"]: e["bytes"] for e in self.files.values()
                               if e.get("sha256") in self.canon.digest_role}
        self.existing = set(manifest.get("existing_paths", []))
        self.derived = {}
        self.verified_review_keys = set()
        self.preimage_exclusions = []
        self._prepass()
        self._cache = {}

    # -- aliases -----------------------------------------------------------
    def _add_alias(self, token, kind):
        if token not in self.canon.alias:
            n = sum(1 for v in self.canon.alias.values() if v.startswith("<" + kind + "#")) + 1
            self.canon.alias[token] = f"<{kind}#{n}>"
            self.alias_sources[self.canon.alias[token]] = token
        return self.canon.alias[token]

    def _path_aliases(self):
        seen = {}
        births = dict(self.m.get("dirs", {}))
        for rel, entry in self.files.items():
            births[rel] = entry.get("birth_ns", 0)
        for label, root in self.m["roots"].items():
            match = _SEGMENT_ID.fullmatch(os.path.basename(os.path.normpath(root)))
            if match:
                kind = (_TOKEN.sub("#", match.group("prefix").rstrip("-_."))
                        or label.strip("{}").lower() + f"-hex{len(match.group('id'))}")
                seen.setdefault(match.group("id"), []).append((kind, -1, label))
        items = [(rel, birth or 0, rel in self.files) for rel, birth in births.items()]
        # Paths that records name outside the fingerprinted trees (staging folders that are gone, cache
        # siblings such as reference-<uuid>): same rule under the run roots, ordered after the tree.
        late = 2 ** 63
        for text in self.m.get("referenced_paths", []):
            rel = self.canon.string(text)
            if not rel.startswith(RUN_ROOTS) or rel in births:
                continue
            parts = rel.split("/")
            for i in range(2, len(parts) + 1):
                items.append(("/".join(parts[:i]), late, i == len(parts) and "." in parts[-1][1:]))
        for rel, birth, is_file in items:
            parts = rel.split("/")
            segment = parts[-1]
            stem = Path(segment).stem if is_file and "." in segment[1:] else segment
            match = _SEGMENT_ID.fullmatch(stem) if len(parts) > 1 else None
            if not match:
                continue
            token = match.group("id")
            parent = _TOKEN.sub("#", parts[-2]) if len(parts) > 2 else parts[0].strip("{}").lower()
            # a prefix holding another run id (.partial-delivery-<token>-<uuid8>) must not leak it into the kind
            kind = _TOKEN.sub("#", match.group("prefix").rstrip("-_.")) or f"{parent}-hex{len(token)}"
            seen.setdefault(token, []).append((kind, birth, _TOKEN.sub("#", rel)))
        by_kind = {}
        for token, occurrences in seen.items():
            # a kind seen in the tree wins over one only named by a record; staging (.partial-) names last
            kind = min(occurrences, key=lambda o: (o[1] == late, o[0].startswith("."), o[0]))[0]
            key = (min(b for _, b, _ in occurrences), min(t for _, _, t in occurrences), token)
            by_kind.setdefault(kind, []).append(key)
        for kind in sorted(by_kind):
            for n, (_, _, token) in enumerate(sorted(by_kind[kind]), 1):
                self.canon.alias[token] = f"<{kind}#{n}>"
                self.alias_sources[f"<{kind}#{n}>"] = token

    def _docs(self):
        for rel in sorted(self.files, key=lambda r: self.names[r]):
            entry = self.files[rel]
            if entry["class"] == "json" and "doc" in (entry.get("content") or {}):
                yield self.names[rel], entry["content"]["doc"]

    def _prepass(self):
        """Register record ids and verified derived digests before any document is normalised."""
        for name, doc in self._docs():
            self._scan(doc, name, "")

    def _check(self, doc, obj, key, rule, ok, path):
        """Record one derived-digest check of key in the raw object obj (identified by id, reported by path)."""
        self.derived[(doc, id(obj), key, rule)] = (bool(ok), path)

    def _checked(self, doc, obj, key, rule):
        found = self.derived.get((doc, id(obj), key, rule))
        return bool(found and found[0])

    def _preimage_keys(self, value, doc, path, preimage):
        """Record every excluded key inside a signature or key preimage (timing fields must never feed one)."""
        if isinstance(value, dict):
            for k, v in value.items():
                p = f"{path}/{k}"
                reason = excluded_key_reason(k, v)
                if reason:
                    self.preimage_exclusions.append({"doc": doc, "path": p, "preimage": preimage, "rule": reason})
                self._preimage_keys(v, doc, p, preimage)
        elif isinstance(value, list):
            for i, v in enumerate(value):
                self._preimage_keys(v, doc, f"{path}[{i}]", preimage)

    def _scan(self, value, doc, path):
        if isinstance(value, dict):
            for key in VALUE_ALIAS_KEYS:
                token = value.get(key)
                if isinstance(token, str) and re.fullmatch(r"[0-9a-f]{8,64}", token):
                    self._add_alias(token, key)
            signature = value.get("signature")
            if isinstance(signature, str) and _SHA.fullmatch(signature):
                if "request" in value:
                    ok = _digest_matches(value["request"], signature)
                    self._check(doc, value, "signature", "signature+request", ok, path + "/signature")
                    self._preimage_keys(value["request"], doc, path + "/request", "workflow signature")
                    if ok:
                        self._add_alias(signature, "workflow-signature")
                elif "job_dir" in value:
                    preimage = _job_signature_preimage(value)
                    if preimage is not None:
                        ok = _digest_matches(preimage, signature)
                        self._check(doc, value, "signature", "job signature", ok, path + "/signature")
                        for field in JOB_SIGNATURE_FIELDS:
                            if field in value:
                                self._preimage_keys(value[field], doc, f"{path}/{field}", "job signature")
                    self._add_alias(signature, "job-signature")
            key = value.get("key")
            if isinstance(key, str) and _SHA.fullmatch(key) and "source_revision" in value and "base_signature" in value:
                source = {"revision": value["source_revision"], "signature": value["base_signature"]}
                if isinstance(value.get("options"), dict) and value["options"].get("export_trail_layer"):
                    source["options"] = value["options"]
                self._preimage_keys(value["source_revision"], doc, path + "/source_revision", "review key")
                if "options" in source:
                    self._preimage_keys(source["options"], doc, path + "/options", "review key")
                ok = _digest_matches(source, key)
                self._check(doc, value, "key", "review key", ok, path + "/key")
                if ok:
                    self.verified_review_keys.add(key)
                    self._add_alias(key, "review-key")
            for k, v in value.items():
                self._scan(v, doc, f"{path}/{k}")
        elif isinstance(value, list):
            for i, v in enumerate(value):
                self._scan(v, doc, f"{path}[{i}]")
        elif isinstance(value, str):
            for pattern, _ in _VALUE_PATTERNS:
                for match in pattern.finditer(value):
                    self._add_alias(match.group("id"), match.group("kind"))

    # -- normalisation -----------------------------------------------------
    def exists(self, path):
        return _norm_key(path) in self.existing

    def _masks(self, value, doc, path):
        """Replacements for keys of one object (derived digests, unretained files, record sizes)."""
        masks = {}
        digest = value.get("data_sha256")
        if isinstance(digest, str) and _SHA.fullmatch(digest) and "data" in value:
            ok = _digest_matches(value["data"], digest)
            self._check(doc, value, "data_sha256", "data_sha256", ok, path + "/data_sha256")
            if ok:
                masks["data_sha256"] = ("<derived-digest of data: verified>", DERIVED_RULES["data_sha256"])
        if "request" in value and self._checked(doc, value, "signature", "signature+request"):
            masks["signature"] = ("<derived-digest of request: verified>", DERIVED_RULES["signature+request"])
        key = value.get("key")
        if isinstance(key, str) and _SHA.fullmatch(key):
            if "source_revision" in value and self._checked(doc, value, "key", "review key"):
                masks["key"] = ("<review-key: verified>", DERIVED_RULES["review key"])
            elif "source_revision" not in value and "delivery_dir" in value:
                ok = key in self.verified_review_keys
                self._check(doc, value, "key", "review key names a verified receipt", ok, path + "/key")
                if ok:
                    masks["key"] = ("<review-key: verified receipt>", DERIVED_RULES["review key"])
        for k, v in value.items():
            if isinstance(k, str) and k.endswith("_sha256") and k not in masks and isinstance(v, str) \
                    and _SHA.fullmatch(v) and isinstance(value.get(k[:-7]), (dict, list)) \
                    and _digest_matches(value[k[:-7]], v):
                masks[k] = (f"<derived-digest of {k[:-7]}: verified>", DERIVED_RULES["X_sha256+X"])
        for k, v in value.items():
            if isinstance(k, str) and (k == "sha256" or k.endswith("_sha256")) and isinstance(v, str) \
                    and v in self.size_by_digest:
                for sk in (("bytes", "size") if k == "sha256" else (k[:-7] + "_bytes",)):
                    size = value.get(sk)
                    if sk not in masks and isinstance(size, int) and not isinstance(size, bool) \
                            and size == self.size_by_digest[v]:
                        masks[sk] = ("<size-of-fingerprinted-file>", DICT_RULES["fingerprinted file size"])
        for k, v in value.items():
            if not (isinstance(k, str) and _is_abs_path(v)):
                continue
            generic = k in ("path", "file")
            digest_keys = ("sha256",) if generic else (k + "_sha256",)
            size_keys = ("bytes", "size") if generic else (k + "_bytes",)
            if not self.exists(v):
                for dk in digest_keys:
                    if dk not in masks and isinstance(value.get(dk), str) and _SHA.fullmatch(value[dk]):
                        masks[dk] = ("<digest-of-unretained-file>", DICT_RULES["unretained file"])
                for sk in size_keys:
                    if isinstance(value.get(sk), int) and not isinstance(value.get(sk), bool):
                        masks[sk] = ("<size-of-unretained-file>", DICT_RULES["unretained file"])
            elif os.path.splitext(v)[1].lower() in RECORD_SUFFIXES:
                for sk in size_keys:
                    if isinstance(value.get(sk), int) and not isinstance(value.get(sk), bool):
                        masks[sk] = ("<size-of-record-file>", DICT_RULES["record file size"])
        return masks

    def normalise_doc(self, value, doc, path, hidden):
        """Canonical copy of a JSON value; hidden[path] = (rule, raw value, dropped) for everything excluded
        (dropped: the key is absent from the copy) or masked (kept as a placeholder)."""
        if isinstance(value, dict):
            masks = self._masks(value, doc, path)
            out = {}
            for k, v in value.items():
                ck = self.canon.string(k) if isinstance(k, str) else str(k)
                p = f"{path}/{ck}" if "/" not in ck else f'{path}["{ck}"]'
                reason = excluded_key_reason(k, v)
                if reason:
                    hidden[p] = (reason, v, True)
                    continue
                if k in masks:
                    out[ck] = masks[k][0]
                    hidden[p] = (masks[k][1], v, False)
                    continue
                out[ck] = self.normalise_doc(v, doc, p, hidden)
            return out
        if isinstance(value, list):
            return [self.normalise_doc(v, doc, f"{path}[{i}]", hidden) for i, v in enumerate(value)]
        if isinstance(value, str):
            return self.canon.string(value)
        if isinstance(value, float) and not math.isfinite(value):
            return {"$float": repr(value)}
        return value

    def _meta(self, value, doc, hidden, where):
        """Normalise a stored metadata value: JSON strings get the record rules, other strings the canon.
        where labels the item in hidden paths, e.g. <page0:ImageDescription>."""
        parsed = _maybe_json(value)
        if parsed is not None:
            return {"$json": self.normalise_doc(parsed, doc, where, hidden)}
        if isinstance(value, str):
            return self.canon.string(value)
        if isinstance(value, list):
            return [self._meta(v, doc, hidden, where) for v in value]
        if isinstance(value, dict):
            if "$icc_sha256" in value:
                for field, raw in (value.get("$volatile") or {}).items():
                    hidden[f"{where}/icc/{field}"] = (ICC_DROPPED_FIELDS.get(field, "ICC header field"), raw, True)
                return {k: v for k, v in value.items() if k != "$volatile"}
            return {k: self._meta(v, doc, hidden, where) for k, v in value.items()}
        return value

    def content(self, rel):
        """(normalised content, hidden fields) of one entry; cached."""
        if rel in self._cache:
            return self._cache[rel]
        entry, name = self.files[rel], self.names[rel]
        raw = entry.get("content") or {}
        hidden = {}
        cls = entry["class"]
        if cls == "json":
            if "doc" in raw:
                out = {"json": self.normalise_doc(raw["doc"], name, "", hidden)}
            else:
                out = {"json_error": raw.get("json_error"), "text": self.canon.string(raw.get("text", ""))}
        elif cls == "text":
            out = {"data": raw["data"]["digest"]} if raw.get("too_large") else {"text": self.canon.string(raw["text"])}
        elif cls == "fits":
            out = {"hdus": []}
            for i, hdu in enumerate(raw["hdus"]):
                cards = [[kw, self._meta(v, f"{name}#hdu{i}:{kw}", hidden, f"<hdu{i}:{kw}>"), self.canon.string(c)]
                         for kw, v, c in _merge_commentary(hdu["cards"])]
                for kw, v in hdu.get("dropped", []):
                    hidden[f"hdu{i}/{kw}"] = (FITS_DROPPED_CARDS.get(kw, "dropped card"), v, True)
                out["hdus"].append({"cards": cards, "dtype": hdu["dtype"], "shape": hdu["shape"],
                                    "data": (hdu.get("data") or {}).get("digest")})
        elif cls == "tiff":
            out = {"pages": []}
            for i, page in enumerate(raw["pages"]):
                tags = [[code, tag, self._meta(v, f"{name}#page{i}:{tag}", hidden, f"<page{i}:{tag}>")]
                        for code, tag, v in page["tags"]]
                for tag, v in page.get("dropped", {}).items():
                    hidden[f"page{i}/{tag}"] = (TIFF_DROPPED_TAGS.get(tag, "dropped tag"), v, True)
                out["pages"].append({"shape": page["shape"], "dtype": page["dtype"], "tags": tags,
                                     "pixels": page["pixels"]["digest"]})
        elif cls == "image":
            out = {k: raw.get(k) for k in ("format", "mode", "size", "frames", "shape", "dtype", "palette_sha256")}
            out["info"] = {k: self._meta(v, f"{name}#info:{k}", hidden, f"<info:{k}>")
                           for k, v in (raw.get("info") or {}).items()}
            out["pixels"] = raw["pixels"]["digest"]
        elif cls == "npy":
            out = {k: raw[k] for k in ("dtype", "shape", "fortran_order")}
            out["data"] = raw["data"]["digest"]
        elif cls == "bytes":
            out = {"data": raw["data"]["digest"]}
        else:
            out = None
        self._cache[rel] = (out, hidden)
        return out, hidden

    def derived_failures(self):
        return sorted(({"doc": d, "path": path, "rule": r} for (d, _, _, r), (ok, path) in self.derived.items() if not ok),
                      key=lambda f: (f["doc"], f["path"], f["rule"]))

    def normalise_records(self):
        """Normalise every JSON record once, so the derived digests of every record are checked on each side."""
        for rel, entry in self.files.items():
            if entry["class"] == "json" and not entry.get("error"):
                self.content(rel)


def _merge_commentary(cards):
    """Join runs of HISTORY/COMMENT/blank cards: writers split a long value at 72 characters, so where
    the split falls depends on the length of any path inside it (and therefore on the run's root)."""
    out = []
    for keyword, value, comment in cards:
        if keyword in ("HISTORY", "COMMENT", "") and out and out[-1][0] == keyword \
                and isinstance(value, str) and isinstance(out[-1][1], str):
            out[-1] = [keyword, out[-1][1] + value, out[-1][2]]
        else:
            out.append([keyword, value, comment])
    return out


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------
def _json_diff(a, b, path, out, limit):
    if len(out) >= limit:
        return
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            p = f"{path}/{k}" if "/" not in str(k) else f'{path}["{k}"]'
            if k not in a:
                out.append({"path": p, "a": "<absent>", "b": _short(b[k])})
            elif k not in b:
                out.append({"path": p, "a": _short(a[k]), "b": "<absent>"})
            else:
                _json_diff(a[k], b[k], p, out, limit)
            if len(out) >= limit:
                return
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append({"path": path or "/", "a": f"<list of {len(a)}>", "b": f"<list of {len(b)}>"})
        for i, (x, y) in enumerate(zip(a, b)):
            _json_diff(x, y, f"{path}[{i}]", out, limit)
            if len(out) >= limit:
                return
    elif type(a) is not type(b) or a != b or (isinstance(a, float) and repr(a) != repr(b)):
        out.append({"path": path or "/", "a": _short(a), "b": _short(b)})


def _diagnostic_slot(name, path, slots):
    """Reason of the diagnostic slot (or --allow-field) that accepts a one-sided field, else None."""
    for names, pattern, reason in slots:
        if any(fnmatch.fnmatchcase(name, glob) for glob in names) and re.fullmatch(pattern, path):
            return reason
    return None


def _hidden_differences(ha, hb, name="", slots=()):
    """(hidden, one_sided): excluded or masked fields whose raw values differ between the sides, and excluded
    fields present on one side only.  A one-sided field in a diagnostic slot goes to hidden, marked with it."""
    hidden, one_sided = [], []
    for path in sorted(set(ha) | set(hb)):
        a, b = ha.get(path), hb.get(path)
        if (a is None) != (b is None) and (a or b)[2]:
            item = {"path": path, "rule": (a or b)[0], "only_in": "a" if a else "b"}
            slot = _diagnostic_slot(name, path, slots)
            if slot:
                hidden.append({**item, "diagnostic_slot": slot})
            else:
                one_sided.append(item)
            continue
        va, vb = (a[1] if a else "<absent>"), (b[1] if b else "<absent>")
        if json.dumps(va, sort_keys=True, default=str) != json.dumps(vb, sort_keys=True, default=str):
            hidden.append({"path": path, "rule": (a or b)[0]})
    return hidden, one_sided


def _locate(da, db, dtype, shape, block_bytes, order="C"):
    """First differing block of two digests-with-blocks, as byte range and array index."""
    out = {"bytes": [da["bytes"], db["bytes"]]} if da["bytes"] != db["bytes"] else {}
    ba, bb = da["blocks"], db["blocks"]
    different = [i for i, (x, y) in enumerate(zip(ba, bb)) if x != y]
    if len(ba) != len(bb) and not different:
        different = [min(len(ba), len(bb))]
    out.update(blocks_total=[len(ba), len(bb)] if len(ba) != len(bb) else len(ba),
               blocks_different=len(different) + abs(len(ba) - len(bb)), first_blocks=different[:10],
               block_bytes=block_bytes)
    if different:
        start = different[0] * block_bytes
        end = min(start + block_bytes, max(da["bytes"], db["bytes"]))
        out["first_byte_range"] = [start, end]
        if dtype:
            itemsize = np.dtype(dtype).itemsize
            first = start // itemsize
            out["first_element_range"] = [first, (end + itemsize - 1) // itemsize]
            if shape and first < int(np.prod(shape, dtype=np.int64)):
                out["block_starts_at_index"] = [int(i) for i in np.unravel_index(first, shape, order=order)]
    return out


def _text_diff(a, b):
    la, lb = a.splitlines(), b.splitlines()
    lines = [{"line": i + 1, "a": x[:200], "b": y[:200]} for i, (x, y) in enumerate(zip(la, lb)) if x != y][:20]
    if len(la) != len(lb) and len(lines) < 20:
        i = min(len(la), len(lb))
        lines.append({"line": i + 1, "a": la[i][:200] if i < len(la) else "<end>",
                      "b": lb[i][:200] if i < len(lb) else "<end>"})
    return {"lines": [len(la), len(lb)], "first_different_lines": lines}


def compare_entry(sa, ra, sb, rb, block_bytes, slots=DIAGNOSTIC_SLOTS):
    """Verdict for one file present on both sides."""
    ea, eb = sa.files[ra], sb.files[rb]
    if ea["class"] != eb["class"]:
        return {"verdict": "DIFFERENT", "why": f"class {ea['class']} != {eb['class']}"}
    cls = ea["class"]
    if cls in LISTED_CLASSES:
        return {"verdict": "EXCLUDED", "class": cls, "bytes_equal": ea.get("sha256") == eb.get("sha256")}
    if ea.get("sha256") and ea.get("sha256") == eb.get("sha256"):
        return {"verdict": "IDENTICAL_BYTES"}
    if ea.get("error") or eb.get("error"):
        return {"verdict": "ERROR", "a_error": ea.get("error"), "b_error": eb.get("error")}
    ca, ha = sa.content(ra)
    cb, hb = sb.content(rb)
    raw_a, raw_b = ea["content"], eb["content"]
    diffs = {}
    if cls == "json":
        out = []
        _json_diff(ca, cb, "", out, LIST_LIMIT)
        if out:
            diffs["json"] = out
    elif cls == "text":
        if ca != cb:
            diffs["text"] = _text_diff(ca["text"], cb["text"]) if "text" in ca and "text" in cb else {"digest": [ca, cb]}
    elif cls == "fits":
        if len(ca["hdus"]) != len(cb["hdus"]):
            diffs["hdus"] = [len(ca["hdus"]), len(cb["hdus"])]
        for i, (x, y) in enumerate(zip(ca["hdus"], cb["hdus"])):
            header = []
            _json_diff({c[0] + (f"#{n}" if n else ""): c[1:] for n, c in _numbered(x["cards"])},
                       {c[0] + (f"#{n}" if n else ""): c[1:] for n, c in _numbered(y["cards"])}, "", header, 40)
            if header:
                diffs.setdefault("header", []).append({"hdu": i, "cards": header})
            if (x["dtype"], x["shape"]) != (y["dtype"], y["shape"]):
                diffs.setdefault("data", []).append({"hdu": i, "dtype": [x["dtype"], y["dtype"]],
                                                     "shape": [x["shape"], y["shape"]]})
            elif x["data"] != y["data"]:
                diffs.setdefault("data", []).append(
                    {"hdu": i, "dtype": x["dtype"], "shape": x["shape"],
                     **_locate(raw_a["hdus"][i]["data"], raw_b["hdus"][i]["data"], x["dtype"], x["shape"], block_bytes)})
    elif cls == "tiff":
        if len(ca["pages"]) != len(cb["pages"]):
            diffs["pages"] = [len(ca["pages"]), len(cb["pages"])]
        for i, (x, y) in enumerate(zip(ca["pages"], cb["pages"])):
            tags = []
            _json_diff({t[1] + (f"#{n}" if n else ""): t[2] for n, t in _numbered(x["tags"], 1)},
                       {t[1] + (f"#{n}" if n else ""): t[2] for n, t in _numbered(y["tags"], 1)}, "", tags, 40)
            if tags:
                diffs.setdefault("tags", []).append({"page": i, "tags": tags})
            if (x["dtype"], x["shape"]) != (y["dtype"], y["shape"]):
                diffs.setdefault("pixels", []).append({"page": i, "dtype": [x["dtype"], y["dtype"]],
                                                       "shape": [x["shape"], y["shape"]]})
            elif x["pixels"] != y["pixels"]:
                diffs.setdefault("pixels", []).append(
                    {"page": i, "dtype": x["dtype"], "shape": x["shape"],
                     **_locate(raw_a["pages"][i]["pixels"], raw_b["pages"][i]["pixels"], x["dtype"], x["shape"], block_bytes)})
    elif cls == "image":
        meta = []
        _json_diff({k: v for k, v in ca.items() if k != "pixels"}, {k: v for k, v in cb.items() if k != "pixels"},
                   "", meta, 40)
        if meta:
            diffs["metadata"] = meta
        if ca["pixels"] != cb["pixels"] and (ca["shape"], ca["dtype"]) == (cb["shape"], cb["dtype"]):
            diffs["pixels"] = {"dtype": ca["dtype"], "shape": ca["shape"],
                               **_locate(raw_a["pixels"], raw_b["pixels"], ca["dtype"], ca["shape"], block_bytes)}
    elif cls == "npy":
        meta = []
        _json_diff({k: v for k, v in ca.items() if k != "data"}, {k: v for k, v in cb.items() if k != "data"}, "", meta, 20)
        if meta:
            diffs["metadata"] = meta
        elif ca["data"] != cb["data"]:
            diffs["data"] = {"dtype": ca["dtype"], "shape": ca["shape"],
                             **_locate(raw_a["data"], raw_b["data"], ca["dtype"], ca["shape"], block_bytes,
                                       "F" if ca["fortran_order"] else "C")}
    else:
        if ca != cb:
            diffs["bytes"] = _locate(raw_a["data"], raw_b["data"], None, None, block_bytes)
    hidden, one_sided = _hidden_differences(ha, hb, sa.names[ra], slots)
    if one_sided:  # an exclusion hides value differences, never a field that exists on one side only
        diffs["one_sided_fields"] = one_sided[:LIST_LIMIT]
    if diffs:
        slotted = [item for item in hidden if "diagnostic_slot" in item]
        if slotted:
            diffs["one_sided_diagnostic_fields"] = slotted[:LIST_LIMIT]
        return {"verdict": "DIFFERENT", "class": cls, **diffs}
    result = {"verdict": "IDENTICAL_CONTENT", "class": cls}
    if hidden:
        result["hidden_differences"] = hidden[:LIST_LIMIT]
    else:
        result["bytes_differ_in"] = {"json": "formatting, roots, ids or digests of run-specific files",
                                     "text": "roots, ids or digests of run-specific files",
                                     "image": "encoded bytes only (decoded pixels and metadata equal)"}.get(
            cls, "layout or encoding only (canonical content equal)")
    return result


def _numbered(items, name_index=0):
    """(occurrence number, item) so repeated keywords/tags keep their order in a mapping."""
    count = {}
    for item in items:
        key = item[name_index]
        n = count.get(key, 0)
        count[key] = n + 1
        yield n, item


# ---- deep numeric comparison (both runs still on disk) --------------------
class _DiffStats:
    def __init__(self, dtype, tolerance=None):
        self.dtype = np.dtype(dtype)
        self.tolerance = tolerance
        self.values = 0
        self.different = 0
        self.nonfinite_changes = 0
        self.max_abs = 0.0
        self.over_tolerance = 0
        self.first = None

    def update(self, a, b, offset):
        if self.dtype.kind == "f":
            bits = a.view(f"u{a.dtype.itemsize}") != b.view(f"u{b.dtype.itemsize}")
        else:
            bits = a != b
        index = np.flatnonzero(bits)
        if not index.size:
            return
        self.different += int(index.size)
        xa, xb = a[index].astype(np.float64), b[index].astype(np.float64)
        if self.first is None:
            self.first = [int(offset + index[0]), _plain(a[index[0]]), _plain(b[index[0]])]
        if self.dtype.kind == "f":
            finite = np.isfinite(xa) & np.isfinite(xb)
            same_special = (np.isnan(xa) & np.isnan(xb)) | (~finite & (xa == xb))
            self.nonfinite_changes += int(np.count_nonzero(~finite & ~same_special))
            delta = np.abs(xa[finite] - xb[finite])
        else:
            delta = np.abs(xa - xb)
        if delta.size:
            self.max_abs = max(self.max_abs, float(delta.max()))
            if self.tolerance is not None:
                self.over_tolerance += int(np.count_nonzero(delta > self.tolerance))

    def result(self, shape, order="C"):
        out = {"values_compared": self.values, "values_different": self.different,
               "max_abs_diff": self.max_abs, "nonfinite_pattern_changes": self.nonfinite_changes}
        if self.tolerance is not None:
            out.update(tolerance=self.tolerance, values_over_tolerance=self.over_tolerance)
        if self.first is not None:
            out["first_difference"] = {"element": self.first[0], "a": self.first[1], "b": self.first[2]}
            if shape:
                out["first_difference"]["index"] = [int(i) for i in np.unravel_index(self.first[0], shape, order=order)]
        return out


def _read_exact(handle, count):
    data = handle.read(count)
    if len(data) != count:
        raise ValueError("file shorter than recorded")
    return data


def _deep_ranges(pa, oa, pb, ob, nbytes, dtype, shape, tolerance, order="C"):
    dtype = np.dtype(dtype)
    stats = _DiffStats(dtype, tolerance)
    step = max(1, READ_CHUNK // dtype.itemsize) * dtype.itemsize
    with open(_io(pa), "rb") as fa, open(_io(pb), "rb") as fb:
        fa.seek(oa)
        fb.seek(ob)
        done = 0
        while done < nbytes:
            n = min(step, nbytes - done)
            xa, xb = _read_exact(fa, n), _read_exact(fb, n)
            if xa != xb:
                stats.update(np.frombuffer(xa, dtype), np.frombuffer(xb, dtype), done // dtype.itemsize)
            stats.values += n // dtype.itemsize
            done += n
    return stats.result(shape, order)


def _deep_arrays(a, b, tolerance):
    stats = _DiffStats(a.dtype, tolerance)
    fa, fb = np.ascontiguousarray(a).reshape(-1), np.ascontiguousarray(b).reshape(-1)
    step = max(1, READ_CHUNK // a.dtype.itemsize)
    for start in range(0, fa.size, step):
        xa, xb = fa[start:start + step], fb[start:start + step]
        if not np.array_equal(xa.view(np.uint8), xb.view(np.uint8)):
            stats.update(xa, xb, start)
        stats.values += xa.size
    return stats.result(list(a.shape))


def _available(entry):
    path = entry.get("path")
    try:
        st = os.stat(_io(path))
    except (OSError, TypeError):
        return False
    return st.st_size == entry["bytes"] and st.st_mtime_ns == entry.get("mtime_ns")


def deep_compare(ea, eb, tolerance=None):
    """Numeric statistics of the differing arrays of two files still on disk (both unchanged)."""
    if not (_available(ea) and _available(eb)):
        return {"unavailable": "a file is missing or changed since it was fingerprinted"}
    cls = ea["class"]
    ca, cb = ea["content"], eb["content"]
    try:
        if cls == "fits":
            out = []
            for i, (x, y) in enumerate(zip(ca["hdus"], cb["hdus"])):
                if x.get("data") and y.get("data") and x["data"]["digest"] != y["data"]["digest"] \
                        and (x["dtype"], x["shape"]) == (y["dtype"], y["shape"]) and x["data_bytes"] == y["data_bytes"]:
                    out.append({"hdu": i, **_deep_ranges(ea["path"], x["offset"], eb["path"], y["offset"],
                                                         x["data_bytes"], x["dtype"], x["shape"], tolerance)})
            return {"arrays": out}
        if cls == "npy":
            if ca["data"]["digest"] == cb["data"]["digest"] or (ca["dtype"], ca["shape"]) != (cb["dtype"], cb["shape"]):
                return {"arrays": []}
            return {"arrays": [_deep_ranges(ea["path"], ca["offset"], eb["path"], cb["offset"], ca["data"]["bytes"],
                                            ca["dtype"], ca["shape"], tolerance,
                                            "F" if ca["fortran_order"] else "C")]}
        if cls == "tiff":
            import tifffile
            out = []
            for i, (x, y) in enumerate(zip(ca["pages"], cb["pages"])):
                if x["pixels"]["digest"] == y["pixels"]["digest"] or (x["dtype"], x["shape"]) != (y["dtype"], y["shape"]):
                    continue
                if x.get("stream") and y.get("stream"):
                    out.append({"page": i, **_deep_ranges(ea["path"], x["stream"][0], eb["path"], y["stream"][0],
                                                          x["stream"][1], x["dtype"], x["shape"], tolerance)})
                else:
                    with tifffile.TiffFile(_io(ea["path"])) as ta, tifffile.TiffFile(_io(eb["path"])) as tb:
                        out.append({"page": i, **_deep_arrays(ta.pages[i].asarray(maxworkers=1),
                                                              tb.pages[i].asarray(maxworkers=1), tolerance)})
            return {"arrays": out}
        if cls == "image":
            from PIL import Image
            if ca["pixels"]["digest"] == cb["pixels"]["digest"] or (ca["shape"], ca["dtype"]) != (cb["shape"], cb["dtype"]):
                return {"arrays": []}
            with Image.open(_io(ea["path"])) as ia, Image.open(_io(eb["path"])) as ib:
                return {"arrays": [_deep_arrays(np.asarray(ia), np.asarray(ib), tolerance)]}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {"not_numeric": cls}


def _within(detail, deep, tolerance):
    """True when every difference of a file is numeric data within the declared tolerance."""
    if set(detail) - {"verdict", "class", "data", "pixels", "deep", "one_sided_diagnostic_fields"}:
        return False
    for key in ("data", "pixels"):
        items = detail.get(key)
        for item in ([items] if isinstance(items, dict) else items or []):
            if "blocks_different" not in item:  # dtype or shape changed: not a numeric difference
                return False
    arrays = deep.get("arrays") if isinstance(deep, dict) else None
    if not arrays:
        return False
    return all(a.get("nonfinite_pattern_changes") == 0 and a.get("values_over_tolerance") == 0
               and a.get("max_abs_diff", math.inf) <= tolerance for a in arrays)


def _parse_tolerances(items):
    out = []
    for item in items or ():
        pattern, sep, value = item.rpartition("=")
        if not sep or not pattern:
            raise ValueError(f"--tolerance needs GLOB=ABS, got {item!r}")
        tolerance = float(value)
        if not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError(f"tolerance must be finite and >= 0: {item!r}")
        out.append((pattern, tolerance))
    return out


def _parse_allow_fields(items):
    """--allow-field NAME_GLOB=PATH_REGEX items as diagnostic slots."""
    out = []
    for item in items or ():
        if isinstance(item, (tuple, list)):
            glob, pattern = item
        else:
            glob, sep, pattern = str(item).partition("=")
            if not sep or not glob or not pattern:
                raise ValueError(f"--allow-field needs NAME_GLOB=PATH_REGEX, got {item!r}")
        try:
            re.compile(pattern)
        except re.error as exc:
            raise ValueError(f"--allow-field path regex {pattern!r}: {exc}") from None
        out.append(((glob,), pattern, f"--allow-field {glob}={pattern}"))
    return out


def _base_roots(manifest):
    """Roots that a rerun in the same slot keeps ({JOB} and {DELIV*} are named after the signature)."""
    return {label: _norm_key(path) for label, path in manifest["roots"].items()
            if label == "{DELIVERY_ROOT}" or not label.startswith(BASE_ROOTS_EXCLUDED)}


def _deterministic_id_differences(sa, sb):
    out = []
    for kind in DETERMINISTIC_ID_KINDS:
        prefix = "<" + kind + "#"
        for placeholder in sorted({p for s in (sa, sb) for p in s.alias_sources if p.startswith(prefix)}):
            a, b = sa.alias_sources.get(placeholder), sb.alias_sources.get(placeholder)
            if a != b:
                out.append({"placeholder": placeholder, "kind": kind, "a": a or "<absent>", "b": b or "<absent>"})
    return out


def _tree_shape(report, sa, sb, allowed):
    """Directory and link sets (canonical names); differences fail unless --allow matches."""
    dirs = {s.name: {s.canon.string(rel) for rel in s.m.get("dirs", {})} for s in (sa, sb)}
    for side, other in (("a", "b"), ("b", "a")):
        for name in sorted(dirs[side] - dirs[other]):
            by = allowed(name) or allowed(name + "/")
            if by:
                verdict = "ALLOWED_DIR_ONLY_IN_" + side.upper()
                report["allowed"][name + "/"] = {"verdict": verdict, "allowed_by": by}
                report["counts"][verdict] = report["counts"].get(verdict, 0) + 1
            else:
                report["dirs_only_in_" + side].append(name + "/")
    links = {s.name: {s.canon.string(rel): s.canon.string(str(target)) for rel, target in s.m.get("links", {}).items()}
             for s in (sa, sb)}
    for name in sorted(set(links["a"]) | set(links["b"])):
        a, b = links["a"].get(name), links["b"].get(name)
        if a != b and not allowed(name):
            report["links_different"].append({"name": name, "a": a or "<absent>", "b": b or "<absent>"})
    report["links"] = {"a": len(links["a"]), "b": len(links["b"])}


def compare(ma, mb, *, allow=(), tolerances=(), deep=False, allow_fields=(), strict_ids=None):
    """Report comparing two manifests; report['pass'] is the verdict.

    strict_ids: None = on when the base roots are equal (a rerun in the same slot), True/False forces it."""
    for m, side in ((ma, "A"), (mb, "B")):
        if m.get("schema") != MANIFEST_SCHEMA:
            raise ValueError(f"manifest {side} is not a {MANIFEST_SCHEMA} manifest")
    if ma["block_bytes"] != mb["block_bytes"]:
        raise ValueError(f"block sizes differ ({ma['block_bytes']} vs {mb['block_bytes']}); fingerprint both alike")
    tolerances = _parse_tolerances(tolerances) if tolerances and isinstance(next(iter(tolerances)), str) else list(tolerances)
    allow = list(allow)
    slots = DIAGNOSTIC_SLOTS + tuple(_parse_allow_fields(allow_fields))
    block = ma["block_bytes"]
    sa, sb = Side(ma, "a"), Side(mb, "b")
    allowed = lambda name: next((p for p in allow if fnmatch.fnmatchcase(name, p)), None)
    tolerated = lambda name: next((t for p, t in tolerances if fnmatch.fnmatchcase(name, p)), None)
    same_slot = _base_roots(ma) == _base_roots(mb)
    strict = same_slot if strict_ids is None else bool(strict_ids)
    report = {"schema": REPORT_SCHEMA, "tool_version": TOOL_VERSION, "pass": False,
              "a": _summary(ma), "b": _summary(mb), "allow_patterns": allow,
              "allow_fields": [[list(n), p] for n, p, _ in slots[len(DIAGNOSTIC_SLOTS):]],
              "tolerances": [[p, t] for p, t in tolerances], "deep": bool(deep),
              "counts": {}, "only_in_a": [], "only_in_b": [], "only_in_listed": {"a": [], "b": []},
              "dirs_only_in_a": [], "dirs_only_in_b": [], "links_different": [],
              "allowed": {}, "different": {}, "tolerated": {}, "errors": {}, "identical_content": {},
              "excluded": {}, "one_sided_fields": {}, "one_sided_diagnostic_fields": [], "warnings": []}
    report["strict_ids"] = {"on": strict, "mode": "auto" if strict_ids is None else "forced",
                            "same_base_roots": same_slot, "kinds": list(DETERMINISTIC_ID_KINDS),
                            "differ": _deterministic_id_differences(sa, sb) if strict else []}
    for key in ("tifffile", "PIL", "astropy", "numpy"):
        if ma.get("environment", {}).get(key) != mb.get("environment", {}).get(key):
            report["warnings"].append(f"{key} version differs between the manifests: decoded content may differ "
                                      f"({ma.get('environment', {}).get(key)} vs {mb.get('environment', {}).get(key)})")
    walks = [m.get("delivery_parents", {label: "named (tool 1.0)" for label in m["roots"]
                                         if label.startswith("{DELIV") and label != "{DELIVERY_ROOT}"})
             for m in (ma, mb)]
    if walks[0] != walks[1]:
        report["warnings"].append(f"delivery folders were walked differently ({walks[0]} vs {walks[1]}): files of "
                                  f"a folder walked whole on one side only show up as only in that side")
    for name, rels in list(sa.by_name.items()) + list(sb.by_name.items()):
        if len(rels) > 1:
            report["warnings"].append(f"several files map to {name}: {rels}")
    names_a, names_b = set(sa.by_name), set(sb.by_name)
    for name in sorted(names_a - names_b):
        _only(report, "a", name, sa.files[sa.by_name[name][0]], allowed(name))
    for name in sorted(names_b - names_a):
        _only(report, "b", name, sb.files[sb.by_name[name][0]], allowed(name))
    counts = report["counts"]
    bytes_compared = 0
    for name in sorted(names_a & names_b):
        ra, rb = sa.by_name[name][0], sb.by_name[name][0]
        bytes_compared += sa.files[ra]["bytes"]
        result = compare_entry(sa, ra, sb, rb, block, slots)
        verdict = result["verdict"]
        if verdict == "DIFFERENT":
            tolerance = tolerated(name)
            if deep or tolerance is not None:
                result["deep"] = deep_compare(sa.files[ra], sb.files[rb], tolerance)
            if tolerance is not None and _within(result, result["deep"], tolerance):
                verdict = result["verdict"] = "TOLERATED"
                result["tolerance"] = tolerance
            elif allowed(name):
                verdict = result["verdict"] = "ALLOWED_DIFFERENT"
                result["allowed_by"] = allowed(name)
        counts[verdict] = counts.get(verdict, 0) + 1
        target = {"DIFFERENT": "different", "ALLOWED_DIFFERENT": "allowed", "TOLERATED": "tolerated",
                  "ERROR": "errors", "IDENTICAL_CONTENT": "identical_content", "EXCLUDED": "excluded"}.get(verdict)
        if target:
            report[target][name] = result
    # rules that actually hid a difference, grouped by rule
    hidden_by_rule = {}
    for name, result in report["identical_content"].items():
        for item in result.get("hidden_differences", []):
            bucket = hidden_by_rule.setdefault(item["rule"], {"count": 0, "examples": []})
            bucket["count"] += 1
            if len(bucket["examples"]) < 5:
                bucket["examples"].append(name + " " + item["path"])
    report["hidden_by_rule"] = hidden_by_rule
    for section in ("identical_content", "different", "allowed", "tolerated"):
        for name, result in report[section].items():
            for item in result.get("hidden_differences", []) + result.get("one_sided_diagnostic_fields", []):
                if "diagnostic_slot" in item:
                    report["one_sided_diagnostic_fields"].append(
                        {"name": name, "path": item["path"], "only_in": item["only_in"], "slot": item["diagnostic_slot"]})
            if section == "different" and result.get("one_sided_fields"):
                report["one_sided_fields"][name] = result["one_sided_fields"]
    _tree_shape(report, sa, sb, allowed)
    missing = {"a": ma.get("missing_deliveries", []), "b": mb.get("missing_deliveries", [])}
    report["missing_deliveries"] = missing
    for side in (sa, sb):
        side.normalise_records()
    report["derived_digests"] = {s.name: {"checked": len(s.derived), "failed": s.derived_failures()} for s in (sa, sb)}
    report["excluded_keys_in_preimages"] = {s.name: s.preimage_exclusions[:LIST_LIMIT] for s in (sa, sb)}
    report["bindings"] = {s: {"verified": m["bindings"]["verified"], "mismatch": m["bindings"]["mismatch"][:LIST_LIMIT],
                              "missing": len(m["bindings"]["missing"]), "not_checked": m["bindings"].get("not_checked", 0)}
                          for s, m in (("a", ma), ("b", mb))}
    report["aliases"] = {"a": sa.alias_sources, "b": sb.alias_sources}
    report["bytes_compared"] = bytes_compared
    report["rules"] = rules()
    hardlinks = [n for s in (sa, sb) for r, n in s.names.items() if s.files[r].get("nlink", 1) > 1]
    if hardlinks:
        report["warnings"].append(f"{len(hardlinks)} fingerprinted files have several hard links: {hardlinks[:10]}")
    report["pass"] = not (report["only_in_a"] or report["only_in_b"] or report["different"] or report["errors"]
                          or report["dirs_only_in_a"] or report["dirs_only_in_b"] or report["links_different"]
                          or report["strict_ids"]["differ"]
                          or report["excluded_keys_in_preimages"]["a"] or report["excluded_keys_in_preimages"]["b"]
                          or missing["a"] or missing["b"] or report["derived_digests"]["a"]["failed"]
                          or report["derived_digests"]["b"]["failed"] or ma["bindings"]["mismatch"]
                          or mb["bindings"]["mismatch"])
    return report


def _only(report, side, name, entry, allowed_by):
    if entry["class"] in LISTED_CLASSES:
        report["only_in_listed"][side].append(name)
    elif allowed_by:
        report["allowed"][name] = {"verdict": "ALLOWED_ONLY_IN_" + side.upper(), "allowed_by": allowed_by}
        report["counts"]["ALLOWED_ONLY_IN_" + side.upper()] = report["counts"].get("ALLOWED_ONLY_IN_" + side.upper(), 0) + 1
    else:
        report["only_in_" + side].append(name)


def _summary(m):
    return {"workflow_dir": m.get("workflow_dir"), "state": m.get("workflow_state"), "files": len(m["files"]),
            "bytes": m.get("bytes_fingerprinted"), "created_utc": m.get("created_utc"),
            "tool_version": m.get("tool_version"), "environment": m.get("environment")}


def format_report(report, limit=40):
    """Human-readable summary of a comparison report."""
    lines = []
    add = lines.append
    add(f"Workflow equivalence: {'PASS' if report['pass'] else 'FAIL'}")
    add(f"  A: {report['a']['workflow_dir']}  ({report['a']['files']} files)")
    add(f"  B: {report['b']['workflow_dir']}  ({report['b']['files']} files)")
    add(f"  bytes compared: {report['bytes_compared']:,}")
    add("  " + " | ".join(f"{k} {v}" for k, v in sorted(report["counts"].items())))
    strict = report.get("strict_ids") or {}
    add(f"  strict ids: {'on' if strict.get('on') else 'off'} ({strict.get('mode')}; base roots "
        f"{'equal' if strict.get('same_base_roots') else 'differ'})")
    for warning in report["warnings"]:
        add(f"  warning: {warning}")

    def section(title, items):
        if items:
            add(title)
            for item in list(items)[:limit]:
                add("  " + item)
            if len(items) > limit:
                add(f"  ... {len(items) - limit} more (see the JSON report)")

    section("ONLY IN A (fails):", report["only_in_a"])
    section("ONLY IN B (fails):", report["only_in_b"])
    section("DIRECTORIES ONLY IN A (fails):", report.get("dirs_only_in_a", []))
    section("DIRECTORIES ONLY IN B (fails):", report.get("dirs_only_in_b", []))
    section("LINKS DIFFER (fails):", [f"{d['name']}: {d['a']} != {d['b']}" for d in report.get("links_different", [])])
    if report["missing_deliveries"]["a"] or report["missing_deliveries"]["b"]:
        section("NAMED DELIVERY FOLDERS MISSING (fails):",
                [f"{s}: {p}" for s in "ab" for p in report["missing_deliveries"][s]])
    section("DETERMINISTIC IDS DIFFER (fails; same slot, so these must be literally equal):",
            [f"{d['placeholder']}: {d['a']} != {d['b']}" for d in strict.get("differ", [])])
    for side in "ab":
        section(f"EXCLUDED KEYS INSIDE SIGNATURE/KEY PREIMAGES IN {side.upper()} (fails):",
                [f"{e['doc']} {e['path']} (feeds the {e['preimage']}; {e['rule']})"
                 for e in (report.get("excluded_keys_in_preimages") or {}).get(side, [])])
    section("ONE-SIDED EXCLUDED FIELDS (fails; exclusions hide value differences only):",
            [f"{n} {i['path']}  only in {i['only_in'].upper()}  [{i['rule']}]"
             for n, items in (report.get("one_sided_fields") or {}).items() for i in items])
    section("DIFFERENT (fails):", [f"{n}\n" + _describe(r) for n, r in report["different"].items()])
    section("ERROR (fails):", [f"{n}: {r}" for n, r in report["errors"].items()])
    for side in "ab":
        failed = report["derived_digests"][side]["failed"]
        section(f"DERIVED DIGESTS NOT VERIFIED IN {side.upper()} (fails):", [f"{f['doc']} {f['path']} ({f['rule']})" for f in failed])
        section(f"BINDING MISMATCH IN {side.upper()} (fails):",
                [f"{m[0]}: {m[1]} {m[2]}" for m in report["bindings"][side]["mismatch"]])
    section("ONE-SIDED FIELDS IN DIAGNOSTIC SLOTS (informational):",
            [f"{d['name']} {d['path']}  only in {d['only_in'].upper()}" for d in report.get("one_sided_diagnostic_fields", [])])
    section("ALLOWED (by --allow, not failing):",
            [f"{n}  [{r['allowed_by']}]" + ("\n" + _describe(r) if r["verdict"] == "ALLOWED_DIFFERENT" else f"  {r['verdict']}")
             for n, r in report["allowed"].items()])
    section("TOLERATED (by --tolerance, not failing):", [f"{n}\n" + _describe(r) for n, r in report["tolerated"].items()])
    listed = [f"{s}: {n}" for s in "ab" for n in report["only_in_listed"][s]]
    section("ONLY ON ONE SIDE, LISTED-ONLY CLASSES (informational):", listed)
    if report["hidden_by_rule"]:
        add("EXCLUSIONS THAT HID A DIFFERENCE (by rule):")
        for rule, bucket in sorted(report["hidden_by_rule"].items(), key=lambda x: -x[1]["count"]):
            add(f"  {bucket['count']:5d}  {rule}")
            for example in bucket["examples"][:2]:
                add(f"         e.g. {example}")
    for side in "ab":
        b = report["bindings"][side]
        add(f"{side.upper()}: derived digests {report['derived_digests'][side]['checked'] - len(report['derived_digests'][side]['failed'])}"
            f"/{report['derived_digests'][side]['checked']} verified; bindings {b['verified']} verified, "
            f"{len(b['mismatch'])} mismatch, {b['missing']} to unretained files")
    return "\n".join(lines)


def _describe(result):
    out = []
    for key in ("why", "hdus", "pages"):
        if key in result:
            out.append(f"    {key}: {result[key]}")
    for key in ("json", "metadata"):
        for item in (result.get(key) or [])[:10]:
            out.append(f"    {item['path']}: {item['a']} != {item['b']}")
    for item in (result.get("one_sided_fields") or [])[:10]:
        out.append(f"    {item['path']}: excluded field only in {item['only_in'].upper()} ({item['rule']})")
    for item in result.get("header", []) + result.get("tags", []):
        for d in (item.get("cards") or item.get("tags") or [])[:10]:
            where = f"hdu {item['hdu']}" if "hdu" in item else f"page {item['page']}"
            out.append(f"    {where} {d['path']}: {d['a']} != {d['b']}")
    if "text" in result:
        text = result["text"]
        out.append(f"    lines {text.get('lines')}")
        for d in text.get("first_different_lines", [])[:5]:
            out.append(f"    line {d['line']}: {d['a']!r} != {d['b']!r}")
    for key in ("data", "pixels", "bytes"):
        items = result.get(key)
        for item in ([items] if isinstance(items, dict) else items or []):
            where = f"hdu {item['hdu']} " if "hdu" in item else f"page {item['page']} " if "page" in item else ""
            if "blocks_different" in item:
                text = (f"    {where}{key}: {item['blocks_different']}/{item['blocks_total']} blocks differ; "
                        f"first block {item['first_blocks'][0] if item['first_blocks'] else '?'}")
                if "first_byte_range" in item:
                    text += f" = bytes [{item['first_byte_range'][0]}, {item['first_byte_range'][1]})"
                if "block_starts_at_index" in item:
                    text += f", from index {tuple(item['block_starts_at_index'])} ({item.get('dtype')} {item.get('shape')})"
                out.append(text)
            else:
                out.append(f"    {where}{key}: dtype {item.get('dtype')} shape {item.get('shape')}")
    deep = result.get("deep")
    if isinstance(deep, dict):
        for arr in deep.get("arrays", []):
            where = f"hdu {arr['hdu']} " if "hdu" in arr else f"page {arr['page']} " if "page" in arr else ""
            first = arr.get("first_difference", {})
            out.append(f"    deep {where}{arr['values_different']}/{arr['values_compared']} values differ, "
                       f"max |a-b| {arr['max_abs_diff']:.6g}, non-finite pattern changes {arr['nonfinite_pattern_changes']}"
                       + (f", first at {tuple(first['index'])}: {first['a']} vs {first['b']}" if "index" in first else ""))
        for key in ("unavailable", "error", "not_numeric"):
            if key in deep:
                out.append(f"    deep: {key} {deep[key]}")
    return "\n".join(out) if out else "    (see the JSON report)"


# --------------------------------------------------------------------------
# Synthetic workflow (self-test and unit tests)
# --------------------------------------------------------------------------
def _workflow_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def _sha_file(path):
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _binding(path):
    path = Path(path)
    return {"path": str(path), "sha256": _sha_file(path), "bytes": path.stat().st_size}


VARIANTS = ("fits_pixel", "fits_ulp", "fits_header", "tiff_pixel", "tiff_zlib_pixel", "tiff_compression",
            "tiff_description", "jpeg_pixel", "png_pixel", "npy_element", "json_semantic", "json_new_field",
            "text_line", "missing_ledger", "extra_file", "broken_derived", "broken_binding",
            # timing or diagnostic fields where they must never go, and their allowed slots
            "request_timing", "job_signature", "checkpoint_timing", "receipt_timing", "description_timing",
            "diagnostic_slots",
            # tree shape
            "extra_dir", "extra_delivery", "staging_leftover")


def build_synthetic_workflow(base, *, seed, variants=(), inputs=None, idempotency_key=None, height=40, width=48):
    """A small tree with the layout and record structure of a real workflow.

    Pixel content is fixed; ids, timestamps, durations, pids, worker counts and the root differ with
    ``seed``/``base`` exactly like two runs of the same job.  As in the product, the workflow id is random
    unless an idempotency key is given, and the job and delivery-parent ids derive from the signature.
    ``variants`` (see VARIANTS) inject one semantic change each.
    Returns {"workflow_dir": Path, "paths": {role: Path}}.
    """
    from astropy.io import fits
    import tifffile
    from PIL import Image
    unknown = set(variants) - set(VARIANTS)
    if unknown:
        raise ValueError(f"unknown variants {sorted(unknown)}")
    rnd = random.Random(seed)
    hexid = lambda n: "".join(rnd.choice("0123456789abcdef") for _ in range(n))
    stamp = lambda: f"2026-09-28T{rnd.randrange(24):02d}:{rnd.randrange(60):02d}:{rnd.randrange(60):02d}.{rnd.randrange(10**6):06d}+00:00"
    base = Path(base)
    inputs = Path(inputs) if inputs else base.parent / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    frames = []
    for i in range(2):
        frame = inputs / f"frame{i + 1:05d}.dng"
        if not frame.exists():
            frame.write_bytes(bytes([i + 1]) * 4099)
        frames.append(frame)
    cache, works = base / "cache", base / "works"
    request = {"input_dir": str(inputs), "input_files": [str(f) for f in frames],
               "files": [_binding(f) for f in frames], "delivery_root": str(works),
               "delivery_format": "aester-delivery-2", "config": {"background_strength": 0.7, "trail_default": "meteor"},
               "color_adjustments": {"warmth": -5, "tint": -6, "exposure_ev": 0.3}}
    if "request_timing" in variants:
        request["timings"] = {"total_seconds": 12.5}
    signature = _workflow_digest(request)
    ident = hashlib.sha256(idempotency_key.encode()).hexdigest()[:24] if idempotency_key else hexid(16)
    wf = cache / f"workflow-{ident}"
    job = wf / "stack" / ("job-" + hashlib.sha256(("workflow-stack-" + signature).encode()).hexdigest()[:24])
    work, finite = job / "work", job / f"finite-{hexid(12)}"
    rev = job / "products" / hexid(12)
    scan = work / "transients" / "scan-runs" / hexid(32)
    masks = wf / f"masks-{hexid(10)}"
    parent = works / ("aester-" + hashlib.sha256((str(wf) + "\0" + signature).encode()).hexdigest()[:20])
    base_token, trail_token = hexid(12), hexid(12)
    base_dir, trail_dir = parent / f"delivery-{base_token}", parent / f"delivery-trails-1kept-{trail_token}"
    reviewed = wf / f"reviewed-{trail_token}"
    for folder in (work / "transients" / "crops", finite, rev, scan, masks, base_dir, trail_dir, reviewed,
                   job / "review", wf / "publication-receipts"):
        folder.mkdir(parents=True, exist_ok=True)
    paths = {}
    g = np.random.default_rng(12345)
    h, w = height, width
    # FITS
    sky = (g.random((3, h, w), dtype=np.float32) * 0.25).astype(np.float32)
    if "fits_pixel" in variants:
        sky[1, 5, 6] += np.float32(1.0)
    if "fits_ulp" in variants:
        sky[1, 5, 6] = np.nextafter(sky[1, 5, 6], np.float32(1))
    hdu = fits.PrimaryHDU(sky)
    hdu.header["DATE"] = stamp()[:19]
    hdu.header["EXPTIME"] = (11.0 if "fits_header" in variants else 10.0, "exposure")
    hdu.header["HISTORY"] = f"stacked in {work}"
    paths["sky_master"] = work / "sky_master.fit"
    hdu.writeto(paths["sky_master"])
    light = (g.random((3, h, w)) * 60000).astype(np.uint16)
    hdu = fits.PrimaryHDU(light)
    hdu.header["DATE"] = stamp()[:19]
    paths["light"] = work / "light_00001.fit"
    hdu.writeto(paths["light"])
    finite_master = finite / "finite_master.fit"
    fits.PrimaryHDU(sky * 2).writeto(finite_master)
    # NPY
    support = g.random((h, w)) > 0.1
    if "npy_element" in variants:
        support[0, 0] = not support[0, 0]
    paths["support"] = work / "frame_support_light_00001.npy"
    np.save(paths["support"], support)
    valid = finite / "stack_range_valid.npy"
    np.save(valid, g.random((h, w)) > 0.05)
    np.save(work / "counts.npy", (g.random((h, w)) * 48).astype(np.uint16))
    # TIFF / JPEG / PNG
    display = (g.random((h, w, 3)) * 65535).astype(np.uint16)
    if "tiff_pixel" in variants:
        display[3, 3, 0] ^= 1
    linear = g.random((h, w, 3), dtype=np.float32)
    if "tiff_zlib_pixel" in variants:
        linear[2, 2, 1] += np.float32(0.5)
    icc = hashlib.sha256(b"synthetic icc").digest() * 8
    description = {"created_at": stamp(), "render_seconds": rnd.random(), "source": str(rev / "master_linear.fit"),
                   "statistics": {"rounded_to_zero": 6 if "tiff_description" in variants else 5},
                   "signed_trail_source_sha256": hexid(64)}
    if "description_timing" in variants:
        description["encode_seconds"] = rnd.random()
    paths["display"] = rev / "natural_display_16bit.tif"
    tifffile.imwrite(paths["display"], display, photometric="rgb", iccprofile=icc,
                     description=json.dumps(description), compression="zlib" if "tiff_compression" in variants else None,
                     metadata=None)
    paths["linear"] = rev / "master_scene_linear.tif"
    tifffile.imwrite(paths["linear"], linear, photometric="rgb", compression="zlib", metadata=None)
    jpeg_source = (display >> 8).astype(np.uint8)
    if "jpeg_pixel" in variants:
        jpeg_source[8:16, 8:16] = 255 - jpeg_source[8:16, 8:16]
    paths["jpeg"] = rev / "natural.jpg"
    Image.fromarray(jpeg_source).save(paths["jpeg"], quality=95, subsampling=0)
    flags = np.zeros((h, w), np.uint8)
    flags[10:20, 10:20] = 128
    if "png_pixel" in variants:
        flags[0, 0] = 255
    paths["png"] = rev / "render_flags.png"
    Image.fromarray(flags).save(paths["png"])
    Image.fromarray((display[:16, :16] >> 8).astype(np.uint8)).save(work / "transients" / "crops" / "light_00001-1.png")
    # masks and region evidence (two ids of one kind, created in order)
    tifffile.imwrite(masks / "sky_weight.tiff", g.random((h, w), dtype=np.float32), metadata=None)
    for n in (1, 2):
        (wf / f"milkyway-evidence-{hexid(24)}.json").write_text(json.dumps({"evidence": n, "created_at": stamp()}),
                                                              encoding="utf-8")
        time.sleep(0.002)
    # records
    render = rev / "photo_render.json"
    render_record = {"tiff": str(paths["display"]), "tiff_sha256": _sha_file(paths["display"]),
                     "jpeg": str(paths["jpeg"]), "jpeg_sha256": _sha_file(paths["jpeg"]),
                     "render_seconds": rnd.random(), "render_workers": (1, 2, 4, 8, 16)[seed % 5],
                     "metadata": {"statistics": {"rounded_to_zero": 5}}}
    if "broken_binding" in variants:
        render_record["tiff_sha256"] = "0" * 64
    render.write_text(json.dumps(render_record), encoding="utf-8")
    candidates = rev / "candidates.json"
    candidates.write_text(json.dumps({"revision_id": rev.name, "candidates": [{"id": "light_00001-1", "class": "meteor"}],
                                      "seconds": rnd.random()}), encoding="utf-8")
    ledgers = 1 if "missing_ledger" in variants else 2
    for n in range(ledgers):
        (scan / f"worker-{n:03d}.json").write_text(json.dumps({"pid": rnd.randrange(99999), "frames": [n]}), encoding="utf-8")
        (scan / f"worker-{n:03d}.log").write_text("", encoding="utf-8")
    (scan / "result-000000.json").write_text(json.dumps({"frame": "light_00001", "segments": 3}), encoding="utf-8")
    (finite / "receipt.json").write_text(json.dumps({
        "state": "succeeded", "attempt": finite.name, "valid": _binding(valid), "master": _binding(finite_master),
        "seconds": rnd.random(), "implementation": "synthetic-" + str(seed)}), encoding="utf-8")
    (work / "interpolation_support.json").write_text(json.dumps({"frames": [{
        "frame": "light_00001", "path": str(paths["light"]), "export_before_sha256": hexid(64),
        "export_after_sha256": _sha_file(paths["light"]), "supported": 1234}],
        "filter_seconds": rnd.random(), "implementation": "numpy_anchor_axis_tiles_1"}), encoding="utf-8")
    paths["seq"] = work / "light_.seq"
    seq = "S 'light_' 1 2 2 0\nL -1\nI 1 1\nI 2 1\n" if "text_line" not in variants else "S 'light_' 1 2 1 0\nL -1\nI 1 1\nI 2 0\n"
    paths["seq"].write_text(seq, encoding="utf-8")
    (job / "01_convert.ssf").write_text(f'requires 1.4.4\ncd "{work}"\nconvert light_ -out="{work}"\n', encoding="utf-8")
    (job / "01_convert.log").write_text(f"took {rnd.random()} s\n", encoding="utf-8")
    (job / "siril.ini").write_text("[core]\nmem_mode=1\nmem_amount=16\n", encoding="utf-8")
    (job / "worker.lock").write_text(str(rnd.randrange(99999)), encoding="utf-8")
    (wf / "worker.log").write_text(f"pid {rnd.randrange(99999)}\n", encoding="utf-8")
    (wf / "metadata.lock").write_text("", encoding="utf-8")
    if "extra_file" in variants:
        (work / "extra.txt").write_text("extra\n", encoding="utf-8")
    if "extra_dir" in variants:
        (wf / "left-over").mkdir()
    # stack job admission fields and core.submit_job's signature over them; as in the product, the sky mask is a
    # region asset in the cache root (named after the region project revision), so the signature follows the root
    sky = cache / "region-assets-5eed0123456789ab" / "sky.png"
    sky.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((h, w), 255, np.uint8)).save(sky)
    admission = {"inputs": [{"path": str(f), "size": f.stat().st_size, "sha256": _sha_file(f)} for f in frames],
                 "mask": {"path": str(sky), "sha256": _sha_file(sky)}, "frame_masks": None, "transient_marks": None,
                 "engine": {"available": True, "executable": str(inputs / "siril-cli.exe"), "version": "1.4.4"},
                 "support_mode": "legacy", "support_version": None, "fits_input": None, "stack_range_contract": None}
    job_signature = "e" * 64 if "job_signature" in variants else hashlib.sha256(
        json.dumps(_job_signature_preimage(admission), sort_keys=True).encode()).hexdigest()
    draft = job / "review" / f"draft-{hexid(24)}.json"
    draft.write_text(json.dumps({"draft_revision": hexid(32), "binding": {"job_signature": job_signature},
                                 "decisions": {"light_00001-1": "keep"}, "updated_at": stamp()}), encoding="utf-8")
    revision = {"id": rev.name, "record": str(candidates), "sha256": _sha_file(candidates)}
    job_record = {"schema_version": 2, "job_dir": str(job), "state": "succeeded", "created_at": stamp(),
                  "updated_at": stamp(), "worker_pid": rnd.randrange(99999), "signature": job_signature,
                  "idempotency_key": "workflow-stack-" + signature, **admission,
                  "disk_estimate":{"free_bytes_at_admission": rnd.randrange(10**12), "required_bytes": 123456},
                  "diagnostics": {"timings": {"version": "monotonic-stages-1", "phases_seconds": {"setup": rnd.random()}},
                                  "timing_attempts": [],
                                  "support": {"implementation": "numpy_anchor_axis_tiles_1", "filter_seconds": rnd.random(),
                                              "zeroed": 18 if "json_semantic" in variants else 17},
                                  "transients": {"status": "needs_review", "summary": {"candidates": 1},
                                                 "seconds": rnd.random(), "workers": (1, 2, 4, 8, 16)[seed % 5]},
                                  "exposure": {"input_total_seconds": 20.0, "included_total_seconds": 20.0}},
                  "transient_revision": revision}
    if "json_new_field" in variants:
        job_record["diagnostics"]["support"]["new_field"] = 1
    if "diagnostic_slots" in variants:  # an older job receipt without timings (one side only: a diagnostic slot)
        del job_record["diagnostics"]["timings"], job_record["diagnostics"]["timing_attempts"]
    (job / "job.json").write_text(json.dumps(job_record, indent=1), encoding="utf-8")
    snapshot = wf / f"stack_snapshot-{hexid(12)}.json"
    snapshot.write_text(json.dumps({"job_dir": str(job), "created_at": stamp(),
                                    "disk_estimate": {"free_bytes_at_admission": rnd.randrange(10**12)}}), encoding="utf-8")
    for source, target in ((paths["display"], base_dir / "final_srgb16.tif"), (paths["jpeg"], base_dir / "final.jpg"),
                           (paths["linear"], base_dir / "master_scene_linear.tif"),
                           (paths["linear"], trail_dir / "master_scene_linear.tif")):
        shutil.copyfile(source, target)
    trail = display.copy()
    trail[20:22, 5:40] = 65535
    tifffile.imwrite(trail_dir / "final_srgb16.tif", trail, photometric="rgb", iccprofile=icc, metadata=None)
    Image.fromarray((trail >> 8).astype(np.uint8)).save(trail_dir / "final.jpg", quality=95, subsampling=0)
    key = _workflow_digest({"revision": revision, "signature": signature})
    receipt_path = reviewed / "review_delivery.json"
    receipt_path.write_text(json.dumps({
        "version": "workflow-review-delivery-1", "key": key, "state": "succeeded", "base_signature": signature,
        "source_revision": revision, "session_path": str(draft), "options": {"export_trail_layer": False},
        "applied_at": stamp(), "delivery_name": trail_dir.name, "kept_ids": ["light_00001-1"]}), encoding="utf-8")
    receipts = {}
    for folder in (base_dir, trail_dir):
        receipt = {"version": "aester-delivery-2", "workflow_signature": signature, "delivery_name": folder.name,
                   "created_at": stamp(), "files": [{"name": p.name, "sha256": _sha_file(p)} for p in sorted(folder.iterdir())],
                   "source_copies": [{"source": _binding(receipt_path if folder == trail_dir else render)}]}
        if "receipt_timing" in variants:
            receipt.update(recorded_at=stamp(), timings={"phases_seconds": {"copy": rnd.random()}})
        receipts[str(folder)] = receipt
        (wf / "publication-receipts" / f"{folder.name}.json").write_text(json.dumps(receipt), encoding="utf-8")
    checkpoints = {}
    for name, data, files in (
            ("stack", {"job_dir": str(job), "colour": {"photo": {"tiff": str(paths["display"])}}, "seconds": rnd.random(),
                       "snapshot": snapshot.name}, [paths["display"], job / "job.json"]),
            ("masks", {"masks_dir": str(masks), "id": masks.name, "seconds": rnd.random()}, [masks / "sky_weight.tiff"])):
        if name == "masks" and "checkpoint_timing" in variants:
            data["timings"] = {"phases_seconds": {"masks": rnd.random()}}
        checkpoints[name] = {"data": data, "data_sha256": _workflow_digest(data), "files": [_binding(p) for p in files]}
    if "broken_derived" in variants:
        checkpoints["masks"]["data_sha256"] = "0" * 64
    artifacts = [dict(_binding(p), name=p.name) for p in sorted(trail_dir.iterdir())]
    record = {"version": "workflow-1", "workflow_dir": str(wf), "state": "succeeded", "stage": "complete",
              "stack_job_dir": str(job), "worker_pid": rnd.randrange(99999), "request": request,
              "config": request["config"], "signature": signature, "created_at": stamp(), "updated_at": stamp(),
              "checkpoints": checkpoints, "artifacts": artifacts,
              "base_artifacts": [dict(_binding(p), name=p.name) for p in sorted(base_dir.iterdir())],
              "delivery_dir": str(trail_dir), "base_delivery_dir": str(base_dir), "publication_receipts": receipts,
              "review_deliveries": [{"key": key, "delivery_dir": str(trail_dir), "receipt": str(receipt_path),
                                     "artifacts": artifacts}],
              "trail_review": {"state": "applied", "kept_ids": ["light_00001-1"], "job_dir": str(job)},
              "staging": {"path": str(wf / f".partial-delivery-{hexid(12)}" / "final_srgb16.tif"),
                          "sha256": hexid(64), "bytes": rnd.randrange(10**6)}}
    if "diagnostic_slots" in variants:  # the instrumented worker's receipts (one side only: diagnostic slots)
        record.update(timings={"version": "monotonic-stages-1", "recorded_at": stamp()}, timing_attempts=[])
        record["review_deliveries"][0].update(started_at=stamp(), finished_at=stamp(),
                                              timings={"scope": "trail_apply", "phases_seconds": {"x": rnd.random()}})
    (wf / "workflow.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    if "extra_delivery" in variants:  # a version no record names, in the workflow's own delivery parent
        (parent / "delivery-feedfacecafe").mkdir()
        shutil.copyfile(base_dir / "final.jpg", parent / "delivery-feedfacecafe" / "final.jpg")
    if "staging_leftover" in variants:  # storage.stage_delivery's .partial-<target>-<uuid8>, never renamed
        leftover = parent / f".partial-{trail_dir.name}-{hexid(8)}"
        leftover.mkdir()
        (leftover / "failure.json").write_text("{}", encoding="utf-8")
    paths.update(workflow_record=wf / "workflow.json", job_record=job / "job.json", trail_final=trail_dir / "final_srgb16.tif",
                 base_final=base_dir / "final_srgb16.tif")
    return {"workflow_dir": wf, "paths": paths}


def self_test(verbose=True):
    """Build synthetic runs, check that equal runs pass and that every injected change is reported."""
    checks = {}
    with tempfile.TemporaryDirectory(prefix="wfeq-self-test-") as tmp:
        tmp = Path(tmp)
        block = 1024

        def run(label, seed, variants=(), **kwargs):
            tree = build_synthetic_workflow(tmp / label, seed=seed, variants=variants, inputs=tmp / "inputs", **kwargs)
            return fingerprint(tree["workflow_dir"], workers=4, block_bytes=block), tree

        base, _ = run("base", 1)
        same, _ = run("same", 2)
        report = compare(base, same)
        checks["equal runs in different roots pass"] = report["pass"]
        checks["excluded fields that differed are listed"] = bool(report["hidden_by_rule"])
        expected = {"fits_pixel": "sky_master.fit", "fits_header": "sky_master.fit", "tiff_pixel": "natural_display_16bit.tif",
                    "tiff_zlib_pixel": "master_scene_linear.tif", "tiff_compression": "natural_display_16bit.tif",
                    "tiff_description": "natural_display_16bit.tif", "jpeg_pixel": "natural.jpg",
                    "png_pixel": "render_flags.png", "npy_element": "frame_support_light_00001.npy",
                    "json_semantic": "job.json", "json_new_field": "job.json", "text_line": "light_.seq"}
        for variant, suffix in expected.items():
            other, _ = run(variant, 3, (variant,))
            report = compare(base, other)
            checks[f"{variant} reported"] = (not report["pass"]) and any(n.endswith(suffix) for n in report["different"])
        other, _ = run("ledger", 4, ("missing_ledger",))
        checks["missing worker ledger is informational"] = compare(base, other)["pass"]
        other, _ = run("extra", 5, ("extra_file",))
        report = compare(base, other)
        checks["extra file fails"] = not report["pass"] and any(n.endswith("extra.txt") for n in report["only_in_b"])
        checks["extra file passes when allowed"] = compare(base, other, allow=["*/extra.txt"])["pass"]
        other, _ = run("derived", 6, ("broken_derived",))
        checks["broken derived digest fails"] = not compare(base, other)["pass"]
        other, _ = run("binding", 7, ("broken_binding",))
        checks["broken binding fails"] = bool(other["bindings"]["mismatch"]) and not compare(base, other)["pass"]
        other, _ = run("ulp", 8, ("fits_ulp",))
        report = compare(base, other, deep=True)
        diff = report["different"].get(next((n for n in report["different"] if n.endswith("sky_master.fit")), ""), {})
        arrays = (diff.get("deep") or {}).get("arrays") or [{}]
        checks["deep reports max abs diff without changing the verdict"] = (not report["pass"]) and \
            arrays[0].get("values_different") == 1 and 0 < arrays[0].get("max_abs_diff", 0) < 1e-6
        # the change propagates to finite_master.fit, which is built from the same array
        checks["declared tolerance accepts the ulp"] = compare(base, other, tolerances=["*master.fit=1e-6"])["pass"]
        checks["too small tolerance still fails"] = not compare(base, other, tolerances=["*master.fit=1e-12"])["pass"]
        # timing fields where they must never go, and the slots where they may
        for variant in ("request_timing", "checkpoint_timing", "receipt_timing", "description_timing",
                        "extra_dir", "extra_delivery", "staging_leftover"):
            other, _ = run(variant, 9, (variant,))
            checks[f"{variant} fails"] = not compare(base, other)["pass"]
        other, _ = run("slots", 10, ("diagnostic_slots",))
        checks["one-sided timing receipts in diagnostic slots pass"] = compare(base, other)["pass"]
        # reruns in one slot: deterministic ids are compared literally
        first, _ = run("slot", 11, idempotency_key="self-test")
        shutil.rmtree(tmp / "slot")
        second, _ = run("slot", 12, idempotency_key="self-test")
        report = compare(first, second)
        checks["same-slot rerun passes with strict ids"] = report["pass"] and report["strict_ids"]["on"]
        for variant in ("request_timing", "job_signature"):
            shutil.rmtree(tmp / "slot")
            changed, _ = run("slot", 13, (variant,), idempotency_key="self-test")
            report = compare(first, changed)
            checks[f"same-slot {variant} fails on deterministic ids"] = (not report["pass"]) and \
                bool(report["strict_ids"]["differ"])
    ok = all(checks.values())
    if verbose:
        print(json.dumps({"self_test": checks, "pass": ok}, indent=1))
    return ok


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _parse_roots(items):
    roots = {}
    for item in items or ():
        label, sep, value = item.partition("=")
        if not sep or not label or not value:
            raise ValueError(f"--root needs LABEL=PATH, got {item!r}")
        roots[label] = value
    return roots


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     epilog="See the module docstring for the full rules; 'rules' prints them.",
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fingerprint", help="fingerprint a finished workflow")
    f.add_argument("workflow")
    f.add_argument("-o", "--out", required=True, help="manifest path (.json or .json.gz); never inside the workflow")
    f.add_argument("--workers", type=int, default=None, help="parallel files (default min(8, CPUs))")
    f.add_argument("--root", action="append", default=[], help="extra root LABEL=PATH, shown as {LABEL}")
    f.add_argument("--memory-mb", type=int, default=DEFAULT_MEMORY // 2 ** 20, help="budget for concurrent decodes")
    f.add_argument("--no-external-bindings", action="store_true", help="do not hash files outside the tree that records bind")
    f.add_argument("--overwrite", action="store_true")
    f.add_argument("--quiet", action="store_true")
    c = sub.add_parser("compare", help="compare two manifests")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("-o", "--out", help="write the JSON report here")
    c.add_argument("--allow", action="append", default=[], help="glob of canonical names expected to differ")
    c.add_argument("--deep", action="store_true", help="numeric statistics for differing arrays (both runs on disk)")
    c.add_argument("--tolerance", action="append", default=[], help="GLOB=ABS: accept numeric data differences up to ABS")
    c.add_argument("--allow-field", action="append", default=[],
                   help="NAME_GLOB=PATH_REGEX: accept an excluded field present on one side only (beyond the "
                        "built-in diagnostic slots; see 'rules')")
    strict = c.add_mutually_exclusive_group()
    strict.add_argument("--strict-ids", dest="strict_ids", action="store_const", const=True, default=None,
                        help="require equal deterministic ids even when the base roots differ")
    strict.add_argument("--no-strict-ids", dest="strict_ids", action="store_const", const=False,
                        help="do not compare deterministic ids literally (default: on when the base roots are equal)")
    c.add_argument("--overwrite", action="store_true")
    c.add_argument("--limit", type=int, default=40, help="items per section in the printed summary")
    sub.add_parser("self-test", help="synthetic runs with known differences")
    sub.add_parser("rules", help="print the exclusion list and normalisation rules")
    args = parser.parse_args(argv)
    try:
        if args.command == "self-test":
            return 0 if self_test() else 1
        if args.command == "rules":
            print(json.dumps(rules(), indent=1, ensure_ascii=False))
            return 0
        if args.command == "fingerprint":
            out = Path(args.out).expanduser().resolve()
            record, roots, walked, _ = discover(args.workflow, _parse_roots(args.root))
            if any(_inside(out, top) for _, top in walked):
                raise ValueError("the manifest must not be written inside a fingerprinted folder")
            if out.exists() and not args.overwrite:
                raise FileExistsError(f"refusing to overwrite {out} (use --overwrite)")
            log = None if args.quiet else (lambda text: print(text, file=sys.stderr, flush=True))
            manifest = fingerprint(args.workflow, workers=args.workers, extra_roots=_parse_roots(args.root),
                                   memory_bytes=args.memory_mb * 2 ** 20,
                                   external_bindings=not args.no_external_bindings, progress=log)
            _write_json_file(out, manifest, overwrite=args.overwrite)
            errors = [r for r, e in manifest["files"].items() if e.get("error")]
            print(json.dumps({"out": str(out), "files": len(manifest["files"]),
                              "bytes": manifest["bytes_fingerprinted"], "seconds": manifest["fingerprint_seconds"],
                              "bindings_verified": manifest["bindings"]["verified"],
                              "binding_mismatch": len(manifest["bindings"]["mismatch"]),
                              "missing_deliveries": len(manifest["missing_deliveries"]),
                              "file_errors": errors[:20]}, ensure_ascii=False))
            return 0
        ma, mb = _read_json_file(args.a), _read_json_file(args.b)
        report = compare(ma, mb, allow=args.allow, tolerances=args.tolerance, deep=args.deep,
                         allow_fields=args.allow_field, strict_ids=args.strict_ids)
        if args.out:
            _write_json_file(Path(args.out), report, overwrite=args.overwrite)
        print(format_report(report, limit=args.limit))
        return 0 if report["pass"] else 1
    except (ValueError, FileExistsError, FileNotFoundError, KeyError) as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
