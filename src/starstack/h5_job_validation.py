"""Read-only H5 validation with explicit missing evidence and a pinned job snapshot."""
import json
import io
from pathlib import Path

from . import hashing, sky_validity as sv, horizon_acceptance as ha
from .compose_evidence import read_weight, sha256
from .acceptance_status import summarize_checks


def _one(status, reason, name="inputs"):
    return summarize_checks([{"name": name, "status": status, "samples": 0, "reason": reason}])


def _bounded(path, root):
    target = Path(path).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError("H5 asset path escapes the job directory") from exc
    return target


def rgb_top_down(path, *, expected_sha256=None, evidence=None):
    from .imaging import read_rgb
    from astropy.io import fits
    # Product linear FITS are BOTTOM-UP. Missing ROWORDER follows the existing
    # product BOTTOM-UP default; other values are rejected in this verifier.
    raw = Path(path).read_bytes()
    digest = hashing.sha256_bytes(raw, source="h5_job_validation.rgb_top_down")
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("composite snapshot hash mismatch")
    header = fits.getheader(io.BytesIO(raw))
    if header.get("ROWORDER", "BOTTOM-UP") != "BOTTOM-UP":
        raise ValueError("H5 requires BOTTOM-UP product FITS")
    if evidence is not None:
        evidence[str(Path(path).resolve())] = digest
    return read_rgb(io.BytesIO(raw))[::-1]


def evaluate_job(job_dir: Path, *, job=None, sample_step=1, mask_path=None) -> dict:
    """Only validates a finished base composite; does not run/recompose a job."""
    root = Path(job_dir).resolve()
    out = {"horizon_acceptance_version": ha.HORIZON_ACCEPTANCE_VERSION, "job": str(root),
           "scope": "base composite applies recorded effective weights; visibility and interpolation not certified"}
    try:
        if job is None:
            raw = (root / "job.json").read_bytes()
            out["job_json_sha256"] = hashing.sha256_bytes(raw, source="h5_job_validation.job")
            job = json.loads(raw)
        if not isinstance(job, dict):
            raise ValueError("job metadata must be an object")
        if job.get("state") != "succeeded":
            out["judgement"] = _one("insufficient_data", "job has no completed publication")
            return out
        if not job.get("mask"):
            out["judgement"] = _one("not_applicable", "job has no sky/ground mask; horizon composition not requested")
            return out
        artifacts = job.get("artifacts", [])
        diagnostics = job.get("diagnostics")
        diagnostics = {} if diagnostics is None else diagnostics
        if not isinstance(artifacts, list) or any(not isinstance(a, dict) for a in artifacts) or not isinstance(diagnostics, dict):
            raise ValueError("malformed job artifacts or diagnostics")
        validity = diagnostics.get("sky_validity")
        validity = {} if validity is None else validity
        if not isinstance(validity, dict):
            raise ValueError("malformed sky validity metadata")
        composite = next((a.get("path") for a in artifacts if a.get("label") == "合成线性母版"), None)
        descriptor = validity.get("compose_weight")
        if not composite or descriptor is None:
            out["judgement"] = _one("insufficient_data", "base composite or recorded effective weight is missing", "actual_weight")
            return out
        composite = _bounded(composite, root)
        mask = _bounded(root / "sky_mask.png", root)
        mask_bytes = mask.read_bytes()
        mask_hash = hashing.sha256_bytes(mask_bytes, source="h5_job_validation.mask")
        if mask_path is not None and sha256(mask_path) != mask_hash:
            raise ValueError("requested mask differs from the job's actual sky mask")
        sky, ground = (_bounded(root / "work" / name, root) for name in ("sky_master.fit", "foreground_master.fit"))
        for path in (composite, sky, ground, mask):
            if not path.is_file():
                raise FileNotFoundError(f"required H5 asset missing: {path.name}")
        weight = read_weight(root, descriptor, composite)
        height, width = weight.shape
        mk = sv.load_mask_top_down(io.BytesIO(mask_bytes), width, height)
        evidence = {str(mask): mask_hash}
        metrics = ha.seam(rgb_top_down(composite, expected_sha256=descriptor["composite_sha256"], evidence=evidence),
                          rgb_top_down(sky, evidence=evidence), rgb_top_down(ground, evidence=evidence), mk,
                          effective_weight=weight, sample_step=sample_step, tolerance=ha.CRITERIA["H5"]["compose_tolerance"])
        out.update(metrics=metrics, input_snapshot_sha256=evidence, weight_evidence=descriptor,
                   judgement=ha.judge_seam(metrics, ha.CRITERIA["H5"]))
    except (FileNotFoundError, PermissionError) as exc:
        out["judgement"] = _one("insufficient_data", f"{type(exc).__name__}: {exc}")
    except (ValueError, TypeError, KeyError, OSError) as exc:
        out["judgement"] = _one("failed", f"{type(exc).__name__}: {exc}")
    return out
