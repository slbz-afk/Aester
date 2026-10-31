"""Full-resolution checks of same-location linear composition, TOP-DOWN RGB.

This validates application of a recorded weight, not the optical correctness
of that weight, the interpolation support, or photographic seam appearance.
"""
from __future__ import annotations

import math
import numpy as np

METRIC_VERSION = "2.0.0"
REGIONS = ("all_pixels", "sky_only", "ground_only", "unobserved_sky", "feather")


def _positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _array(value, shape, name, *, bounded=False):
    if value is None:
        raise ValueError(f"{name} is missing")
    a = np.asarray(value)
    if a.shape != shape or a.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be a real numeric array of shape {shape}")
    # Bounded row blocks avoid full-size boolean temporaries on a 46 MP image.
    for start in range(0, shape[0], 128):
        block = a[start:start+128]
        if not np.isfinite(block).all():
            raise ValueError(f"{name} contains non-finite values")
        if bounded and (np.any(block < 0) or np.any(block > 1)):
            raise ValueError(f"{name} must be in [0, 1]")
    return a


def _observations(composite, mask, rim_rows):
    """Historical cross-location ratios, retained only as visual-review hints."""
    height, width = mask.shape
    rim, band = [], []
    for x in range(0, width, 8):
        below = mask[:, x] < 1
        positive = mask[:, x] > 0
        if not below.any():
            continue
        edge = int(below.argmax())
        if edge >= max(40, rim_rows) and edge < height:
            ref = float(np.median(composite[edge-40:edge-20, x, 1]))
            if ref > 0:
                ratio = float(np.min(composite[edge-rim_rows:edge, x, 1])) / ref
                if math.isfinite(ratio):
                    rim.append(ratio)
        if not positive.any():
            continue
        bottom = height-1-int(positive[::-1].argmax())
        if bottom < edge or edge < 20 or bottom+20 >= height:
            continue
        ref = min(float(np.median(composite[edge-20:edge, x, 1])),
                  float(np.median(composite[bottom+1:bottom+21, x, 1])))
        if ref > 0:
            ratio = float(np.min(composite[edge:bottom+1, x, 1])) / ref
            if math.isfinite(ratio):
                band.append(ratio)
    return {"used_for_verdict": False, "reason": "different locations may have different natural brightness",
            "rim_columns": len(rim), "band_columns": len(band),
            "rim_min_ratio_p05": float(np.percentile(rim, 5)) if rim else None,
            "band_min_ratio_p05": float(np.percentile(band, 5)) if band else None}


def seam(composite, sky, ground, mask, coverage=None, step=4, *, effective_weight=None,
         sample_step=1, rim_rows=12, tolerance=1e-5) -> dict:
    """Measure composition against supplied actual weights; never infer them from coverage.

    ``sample_step`` defaults to one. A coarser sample can prove a failure,
    but unexamined eligible pixels preclude a full-region pass. ``step`` is
    only the old coverage-map stride and does not control composition sampling.
    """
    _positive_int(step, "step")
    _positive_int(sample_step, "sample_step")
    _positive_int(rim_rows, "rim_rows")
    if type(tolerance) not in (float, int) or not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and non-negative")
    if mask is None or np.asarray(mask).ndim != 2:
        raise ValueError("mask must be two-dimensional")
    shape = np.asarray(mask).shape
    mask = _array(mask, shape, "mask", bounded=True)
    composite = _array(composite, (*shape, 3), "composite")
    sky = _array(sky, (*shape, 3), "sky")
    ground = _array(ground, (*shape, 3), "ground")
    if coverage is not None:
        cov_shape = tuple((n+step-1)//step for n in shape)
        cov = _array(coverage, cov_shape, "coverage")
        if np.any(cov < 0) or not np.equal(cov, np.floor(cov)).all():
            raise ValueError("coverage must contain non-negative integer counts")
    result = {"metric_version": METRIC_VERSION, "shape": list(shape), "sample_step": sample_step,
              "coverage_step": step, "weight_source": "supplied" if effective_weight is not None else "missing",
              "mask_feather_pixels": int(np.count_nonzero((mask > 0) & (mask < 1))), "regions": {},
              "observations": _observations(composite, mask, rim_rows)}
    if effective_weight is None:
        result["missing_reason"] = "actual effective weight unavailable; sampled coverage is not a substitute"
        return result
    weight = _array(effective_weight, shape, "effective_weight", bounded=True)
    regions = {name: {"eligible_samples": 0, "samples": 0, "finite_samples": 0, "max_abs_diff": None,
                      "max_error_position": None} for name in REGIONS}
    for start in range(0, shape[0], 128):
        stop = min(start+128, shape[0])
        w, mk = weight[start:stop], mask[start:stop]
        areas = {"all_pixels": np.ones(w.shape, bool), "sky_only": w == 1,
                 "ground_only": (mk == 0) & (w == 0), "unobserved_sky": (mk > 0) & (w == 0),
                 "feather": (w > 0) & (w < 1)}
        offset = (-start) % sample_step
        rows, cols = slice(offset, None, sample_step), slice(None, None, sample_step)
        alpha = np.asarray(w[rows, cols], dtype=np.float64)[..., None]
        a = np.asarray(sky[start:stop][rows, cols], dtype=np.float64)
        b = np.asarray(ground[start:stop][rows, cols], dtype=np.float64)
        actual = np.asarray(composite[start:stop][rows, cols], dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            expected = alpha*a+(1-alpha)*b
            delta = np.abs(actual-expected)
        if not np.isfinite(delta).all():
            raise ValueError("composition difference overflows finite arithmetic")
        for name, area in areas.items():
            region = regions[name]
            region["eligible_samples"] += int(np.count_nonzero(area))
            sampled = area[rows, cols]
            count = int(np.count_nonzero(sampled))
            region["samples"] += count
            region["finite_samples"] += count
            if count:
                values = delta[sampled]
                at = np.unravel_index(np.argmax(values), values.shape)
                maximum = float(values[at])
                if region["max_abs_diff"] is None or maximum > region["max_abs_diff"]:
                    rr, cc = np.nonzero(sampled)
                    region.update(max_abs_diff=maximum, max_error_position=[int(start+offset+rr[at[0]]*sample_step),
                                  int(cc[at[0]]*sample_step), int(at[1])])
    result["regions"] = regions
    return result


def judge_seam(result: dict, criteria: dict) -> dict:
    """Four-state H5 verdict; missing/old metrics do not imply zero error."""
    from .acceptance_status import summarize_checks
    tol = criteria["compose_tolerance"]
    if type(tol) not in (float, int) or not math.isfinite(tol) or tol < 0:
        raise ValueError("compose_tolerance must be finite and non-negative")
    if not isinstance(result, dict):
        raise ValueError("seam result must be an object")
    if result.get("metric_version") != METRIC_VERSION or result.get("weight_source") != "supplied":
        return summarize_checks([{"name": "actual_weight", "status": "insufficient_data", "samples": 0,
                                  "reason": result.get("missing_reason") or "missing current metrics or actual weight"}])
    if result.get("regions") is not None and not isinstance(result["regions"], dict):
        raise ValueError("seam regions must be an object")
    checks = []
    for name in REGIONS:
        r = (result.get("regions") or {}).get(name)
        status, reason, samples = "insufficient_data", "required region metrics missing", 0
        if isinstance(r, dict):
            eligible, samples, finite = (r.get(k) for k in ("eligible_samples", "samples", "finite_samples"))
            valid_counts = all(type(v) is int and v >= 0 for v in (eligible, samples, finite))
            if not valid_counts or not (finite <= samples <= eligible):
                status, reason, samples = "failed", "invalid sample counts", 0
            elif finite != samples:
                status, reason = "failed", "non-finite measured samples"
            elif eligible == 0:
                status = "insufficient_data" if name == "all_pixels" else "not_applicable"
                reason = "empty image" if name == "all_pixels" else "no eligible pixels in this region"
            elif samples == 0:
                reason = "eligible pixels exist but none were measured"
            else:
                error = r.get("max_abs_diff")
                if error is None:
                    reason = "composition error evidence missing"
                elif type(error) not in (float, int) or not math.isfinite(error) or error < 0:
                    status, reason = "failed", "invalid finite composition error"
                elif error > tol:
                    status, reason = "failed", f"same-location composition error {error:.6g} exceeds {tol:g}"
                elif samples < eligible:
                    reason = "eligible pixels remain unmeasured"
                else:
                    status, reason = "passed", f"all eligible pixels agree within {tol:g}"
        checks.append({"name": name, "status": status, "samples": samples, "reason": reason,
                       "metrics": {k: r.get(k) for k in ("eligible_samples", "finite_samples", "max_abs_diff", "max_error_position")}
                                  if isinstance(r, dict) else {}})
    # Invalid incoming records must not put NaN into persisted report metadata.
    for check in checks:
        for key, value in list(check["metrics"].items()):
            if isinstance(value, float) and not math.isfinite(value):
                check["metrics"][key] = None
    return summarize_checks(checks)
