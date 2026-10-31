"""Recorded blend weights bound to a particular immutable base composite."""
from pathlib import Path
import io
import numpy as np
import tifffile

from . import hashing


def sha256(path):
    return hashing.sha256_file(Path(path), source="compose_evidence.sha256")


def write_weight(export_dir: Path, weight, composite: Path) -> dict:
    """Save actual TOP-DOWN weights without deriving them again or changing dtype."""
    a = np.asarray(weight)
    if a.ndim != 2 or a.dtype.kind != "f" or not np.isfinite(a).all() or np.any(a < 0) or np.any(a > 1):
        raise ValueError("invalid actual composition weight")
    path = Path(export_dir) / "sky_weight.tiff"
    tifffile.imwrite(path, a, photometric="minisblack", compression="deflate", maxworkers=1,
                     metadata={"axes": "YX", "roworder": "TOP-DOWN", "meaning": "actual sky blend weight"})
    return {"version": "1.0.0", "source": "recorded_at_composition", "path": str(path.resolve()),
            "sha256": sha256(path), "shape": list(a.shape), "dtype": str(a.dtype), "roworder": "TOP-DOWN",
            "composite": str(Path(composite).resolve()), "composite_sha256": sha256(composite)}


def read_weight(job_dir: Path, descriptor: dict, composite: Path):
    """Verify the binding before decoding. Missing evidence is handled by caller."""
    root = Path(job_dir).resolve()
    try:
        if (descriptor.get("version") != "1.0.0" or descriptor.get("roworder") != "TOP-DOWN"
                or descriptor.get("source") != "recorded_at_composition"):
            raise ValueError("unknown composition-weight format or row order")
        path = Path(descriptor["path"]).resolve()
        bound = Path(descriptor["composite"]).resolve()
        path.relative_to(root)
        bound.relative_to(root)
        if bound != Path(composite).resolve():
            raise ValueError("composition weight belongs to another composite")
        raw = path.read_bytes()   # the decoded bytes are the hashed bytes (in memory, no ledger)
        if (hashing.sha256_bytes(raw, source="compose_evidence.read_weight") != descriptor["sha256"]
                or sha256(bound) != descriptor["composite_sha256"]):
            raise ValueError("composition evidence hash mismatch")
        weight = tifffile.imread(io.BytesIO(raw))
        if list(weight.shape) != descriptor["shape"] or str(weight.dtype) != descriptor["dtype"]:
            raise ValueError("composition evidence shape/dtype mismatch")
        if weight.ndim != 2 or weight.dtype.kind != "f" or not np.isfinite(weight).all() or np.any(weight < 0) or np.any(weight > 1):
            raise ValueError("invalid recorded composition weight")
        return weight
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("malformed composition evidence descriptor") from exc
