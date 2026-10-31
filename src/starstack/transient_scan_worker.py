"""Private JSON-file entry point for an owner-protected transient scanner.

A frame whose scan raises gets an error record (efficiency P1c T-2): ``error-<index>.json`` in the run folder, written
atomically, with the version, the frame index, the stem, the exception type and the first 4000 characters of its
message; then the exception is re-raised (also when the record cannot be written), so the child exits non-zero with
its traceback in its log and publishes nothing further. The owner reports the failure in frame order
(``transient_scan``).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

ERROR_MESSAGE_CHARS = 4000      # transient_scan.ERROR_MESSAGE_CHARS (not imported: the child's closure stays as it is)


def _inside(path, request: Path, what: str) -> Path:
    path = Path(path).resolve()
    if path.parent != request.parent:
        raise ValueError(f"scanner {what} must stay inside their new run directory")
    return path


def _publish(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".pending")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        raise ValueError("expected a scanner request file")
    request = Path(argv[0]).resolve()
    payload = json.loads(request.read_text(encoding="utf-8"))
    if payload.get("version") != 1 or not isinstance(payload.get("jobs"), list):
        raise ValueError("invalid scanner request")
    # Import only after startup under the owner-held process guard.
    from .transient_stage import _scan_frame
    for job in payload["jobs"]:
        result = _inside(job["result"], request, "results")
        error = _inside(job["error"], request, "error records") if job.get("error") is not None else None
        try:
            scanned = _scan_frame(**job["args"])
        except Exception as exc:
            if error is not None:
                try:
                    _publish(error, {"version": 1, "index": job.get("index"), "stem": job["stem"],
                                     "type": type(exc).__name__, "message": str(exc)[:ERROR_MESSAGE_CHARS]})
                except Exception as failed:      # never replaces the frame's own exception
                    print(f"transient scanner: could not write the error record of {job['stem']}: "
                          f"{type(failed).__name__}: {failed}", file=sys.stderr, flush=True)
            raise
        _publish(result, {"version": 1, "stem": job["stem"], "scanned": scanned})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
