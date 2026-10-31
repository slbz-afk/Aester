"""Atomic local JSON replacement; failures preserve the previous document."""
import errno
import json
import os
from pathlib import Path
import time
import uuid


def atomic_json(path: Path, value, *, default=None) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, default=default, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        deadline = time.monotonic() + 2
        while True:
            try:
                os.replace(tmp, path)
                break
            except PermissionError as exc:
                if getattr(exc, "winerror", None) not in {5, 32, 33} or time.monotonic() >= deadline:
                    raise
                time.sleep(.02)
    finally:
        tmp.unlink(missing_ok=True)


READ_RETRY_SECONDS = 2.0


def read_json(path: Path, *, retry_seconds: float = READ_RETRY_SECONDS):
    """Read a JSON record that another process may be replacing with atomic_json.

    On Windows an open that lands inside another process's os.replace of the
    same name is refused (ERROR_ACCESS_DENIED / sharing violation, sometimes
    surfaced only as errno EACCES). Only those denials are retried, until a
    fixed deadline; any other error, and a denial that outlasts the deadline,
    is raised unchanged. The bytes read are not altered.
    """
    path = Path(path)
    deadline = time.monotonic() + retry_seconds
    while True:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except PermissionError as exc:
            winerror = getattr(exc, "winerror", None)
            transient = winerror in {5, 32, 33} or (winerror is None and exc.errno == errno.EACCES)
            if not transient or time.monotonic() >= deadline:
                raise
            time.sleep(.02)
