"""Real-image train/freeze/test diagnostic; never promotes conditional matches to truth.

Usage: python scripts/verify_image_holdout.py --plan PLAN.json --output-root DIR
The v1 plan fixes converted FITS identities, native geometry, comparison frame,
held-out rectangles, two guards, and the independent test association radius.
Siril two-pass references remain automatic within each isolated pixel view.
No existing source, cache, sequence, or product configuration is modified.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from astropy.io import fits

from starstack import fits_crop, registration_adapter as adapter
from starstack import registration_calibration as calibration
from starstack import registration_distortion as rd
from starstack import siril, siril_sequence
from starstack.registration_models import apply_matrix

VERSION = "1.0.0"
PROBE_COMMANDS = ["setext fit", "set32bits", "setcompress 0", "cd work",
                  "register light -2pass -transf=homography"]
REGIONS = ("center", "top_left", "top_right", "bottom_left", "bottom_right", "other")
# Explicit transitive project imports, including lazy local imports in these
# modules. Unrelated core/GUI/S4 workers are not imported by this runner.
SOURCE_MODULES = ("__init__", "install_lifecycle", "fits_crop", "registration_calibration", "registration_distortion",
                  "registration_models", "registration_adapter", "registration_diagnostics", "resample",
                  "siril", "siril_sequence", "process_guard", "sip_header", "holdout_association",
                  "registered_support", "siril_support_map", "support_anchor", "support_bounds",
                  "interpolation_support", "records", "sky_validity",
                  "finite_stack", "stack_range", "range_tiles",
                  "hashing", "profiling",   # P1 hashing (docs/EFFICIENCY_P1_HASHING_PREREG_20260929.md)
                  "support_kernel",         # P1a strict support (registered_support imports it)
                  "frame_pool")             # P1b-1 frame pool (sky_validity imports it; standard library only)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def write_json(path, value):
    with Path(path).open("xb") as stream:
        stream.write(canonical(value) + b"\n")


def _local_module_names(source_text):
    names = set()
    for node in ast.walk(ast.parse(source_text)):
        if isinstance(node, ast.ImportFrom):
            if node.level:
                if node.level != 1:
                    raise ValueError("unexpected parent-package import in frozen dependency")
                names.update([node.module.split(".")[0]] if node.module else [a.name.split(".")[0] for a in node.names])
            elif node.module == "starstack":
                names.update(a.name.split(".")[0] for a in node.names)
            elif node.module and node.module.startswith("starstack."):
                names.add(node.module.split(".")[1])
        elif isinstance(node, ast.Import):
            names.update(a.name.split(".")[1] for a in node.names if a.name.startswith("starstack."))
    return names


def source_manifest():
    # Validate the entire declared import closure on every snapshot. A newly
    # introduced project dependency fails closed until reviewed and added.
    paths = [Path(__file__).resolve(), *(ROOT / "src/starstack" / (name + ".py") for name in SOURCE_MODULES)]
    for path in paths:
        missing = _local_module_names(path.read_text(encoding="utf-8")) - set(SOURCE_MODULES)
        if missing:
            raise ValueError(f"unfrozen local imports in {path.name}: {sorted(missing)}")
    return {str(p.relative_to(ROOT)): digest(p) for p in paths}


def calibration_recipe():
    return {"radius": 8.0, "max_fit_frames": 24, "min_frames": 3, "min_pairs": 200,
            "required_improvement": .2, "reassociation_radii": [20., 10., 5., 3.],
            "registration_max_rms_px": 1.0, "sip_inverse_order": calibration.DEFAULT_INVERSE_ORDER,
            "distortion_direction": "u_to_p", "quality_gate_thresholds": dict(calibration.DEFAULT_QUALITY_GATE)}


def validate_plan(plan):
    required = {"version", "width", "height", "reference_index", "inputs", "test_rectangles_xyxy",
                "exclusion_guard_px", "detection_guard_px", "association_radius_px"}
    if type(plan) is not dict or set(plan) != required or plan["version"] != VERSION:
        raise ValueError("expected exactly the v1 image-holdout plan fields")
    for key in ("width", "height"):
        if type(plan[key]) is not int or not 2 <= plan[key] < 32767:
            raise ValueError(f"{key} must be an integer in [2,32767)")
    for key in ("exclusion_guard_px", "detection_guard_px"):
        if type(plan[key]) is not int or plan[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    radius = plan["association_radius_px"]
    if type(radius) not in (float, int) or not np.isfinite(radius) or radius <= 0:
        raise ValueError("association radius must be positive and finite")
    inputs = plan["inputs"]
    if type(inputs) is not list or len(inputs) < 2:
        raise ValueError("at least two input frames are required")
    ref = plan["reference_index"]
    if type(ref) is not int or not 0 <= ref < len(inputs):
        raise ValueError("reference_index must be a valid zero-based input index")
    ids, paths = set(), set()
    for frame in inputs:
        if type(frame) is not dict or set(frame) != {"id", "path", "sha256"}:
            raise ValueError("input entries require id, path and sha256")
        if type(frame["id"]) is not str or not frame["id"].strip() or frame["id"] in ids:
            raise ValueError("input IDs must be nonempty and unique")
        if type(frame["path"]) is not str or not Path(frame["path"]).is_absolute():
            raise ValueError("input paths must be absolute")
        path = str(Path(frame["path"]).resolve())
        if path.casefold() in paths:
            raise ValueError("duplicate input path")
        if type(frame["sha256"]) is not str or re.fullmatch(r"[0-9a-f]{64}", frame["sha256"]) is None:
            raise ValueError("input sha256 must be lowercase hexadecimal")
        ids.add(frame["id"])
        paths.add(path.casefold())
    rectangles = plan["test_rectangles_xyxy"]
    if type(rectangles) is not list or not rectangles:
        raise ValueError("at least one fixed test rectangle is required")
    for rect in rectangles:
        if type(rect) is not list or len(rect) != 4 or any(type(v) is not int for v in rect):
            raise ValueError("rectangles must have four integer half-open bounds")
        x0, y0, x1, y1 = rect
        if not (0 <= x0 < x1 <= plan["width"] and 0 <= y0 < y1 <= plan["height"]):
            raise ValueError("rectangle outside native geometry")
    return json.loads(canonical(plan))


def role_masks(plan):
    height, width = plan["height"], plan["width"]
    test = np.zeros((height, width), dtype=bool)
    train = np.ones_like(test)
    guard = plan["exclusion_guard_px"]
    for x0, y0, x1, y1 in plan["test_rectangles_xyxy"]:
        test[y0:y1, x0:x1] = True
        train[max(0, y0-guard):min(height, y1+guard), max(0, x0-guard):min(width, x1+guard)] = False
    if not train.any() or not test.any() or np.any(train & test):
        raise ValueError("plan must leave nonempty disjoint train and test pixel sets")
    return train, test


def materialize_view(source, destination, allowed, *, rows=128):
    """Write fresh geometry-only FITS; preserve allowed float32 storage bits.

    Image-derived source header metadata is deliberately not copied into the
    training view. Unsigned-16 FITS uses one fixed float32 (int16+32768)/65535
    mapping, never per-image normalization. That branch is a diagnostic scale
    conversion, not a byte-identical copy of the original. The trusted splitter
    reads source bytes; the trainer sees only its output. No image statistics.
    """
    header, offset = fits_crop.read_header(source)
    width, height, planes, bitpix, order = fits_crop._geometry(header)
    if (height, width) != allowed.shape or planes != 3 or bitpix not in (-32, 16):
        raise ValueError("inputs must be full-size float32 or unsigned-16 RGB FITS matching the plan")
    if bitpix == -32 and (float(header.get("BSCALE", 1)) != 1 or float(header.get("BZERO", 0)) != 0):
        raise ValueError("scaled FITS input is not supported by this diagnostic")
    if type(rows) is not int or rows < 1:
        raise ValueError("rows must be a positive integer")
    storage_mask = allowed[::-1] if order == "BOTTOM-UP" else allowed
    fresh = fits.Header()
    for key, value in (("SIMPLE", True), ("BITPIX", -32), ("NAXIS", 3), ("NAXIS1", width),
                       ("NAXIS2", height), ("NAXIS3", 3), ("ROWORDER", order)):
        fresh[key] = value
    mapped = np.memmap(source, mode="r", dtype=">f4" if bitpix == -32 else ">i2", offset=offset, shape=(3, height, width))
    try:
        with Path(destination).open("xb") as stream:
            stream.write(fresh.tostring(endcard=True, padding=True).encode("ascii"))
            for channel in range(3):
                for y in range(0, height, rows):
                    if bitpix == -32:
                        tile = np.array(mapped[channel, y:y+rows], dtype=">f4", copy=True)
                    else:
                        tile = mapped[channel, y:y+rows].astype(np.float32)
                        tile += np.float32(32768.)
                        tile /= np.float32(65535.)
                        tile = tile.astype(">f4")
                    tile[~storage_mask[y:y+rows]] = np.float32(0)
                    stream.write(tile.tobytes(order="C"))
            stream.write(b"\0" * ((-3 * height * width * 4) % 2880))
    finally:
        del mapped
    return digest(destination)


def prepare_phase(run, plan, role, mask):
    phase = run / role
    work = phase / "work"
    work.mkdir(parents=True, exist_ok=False)
    records = []
    for index, source in enumerate(plan["inputs"]):
        if digest(source["path"]) != source["sha256"]:
            raise ValueError(f"source identity changed: {source['id']}")
        stem = f"light_{index+1:05d}"
        path = work / (stem + ".fit")
        header, _ = fits_crop.read_header(source["path"])
        bitpix = int(header["BITPIX"])
        records.append({"source_id": source["id"], "stem": stem, "sha256": materialize_view(source["path"], path, mask),
                        "input_bitpix": bitpix,
                        "pixel_mapping": "preserve_float32_bits" if bitpix == -32 else "float32(int16+32768)/float32(65535)"})
    n = len(records)
    lines = [f"S 'light_' 1 {n} {n} 5 {plan['reference_index']} 6 0 0 0", "L 3"]
    lines += [f"I {index+1} 1" for index in range(n)]
    (work / "light_.seq").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(phase / "inputs.json", records)
    return phase, records


def run_probe(executable, phase):
    siril.run_script(executable, phase, "probe", PROBE_COMMANDS, lambda: None)


def sequence_matrices(sequence, height):
    records = sequence["regdata"].get(1, [])
    result = {}
    for image, record in zip(sequence["images"], records):
        h = np.asarray(record["H"], dtype=np.float64)
        if image.get("selected", True) and h.shape == (3, 3) and np.isfinite(h).all() and abs(np.linalg.det(h)) > 1e-12:
            result[image["stem"]] = adapter.convert_matrix(adapter.HYPOTHESES[adapter.DEFAULT_HYPOTHESIS], h, height, height)
    return result


def rebase(matrices, comparison):
    if comparison not in matrices:
        return {}
    inv = np.linalg.inv(np.asarray(matrices[comparison], dtype=np.float64))
    result = {}
    for stem, matrix in matrices.items():
        h = inv @ np.asarray(matrix, dtype=np.float64)
        if np.isfinite(h).all() and abs(h[2, 2]) > 1e-12:
            result[stem] = (h / h[2, 2]).tolist()
    return result


def freeze_models(plan, sequence, calibrated, training_records):
    comparison = f"light_{plan['reference_index']+1:05d}"
    base = rebase(sequence_matrices(sequence, plan["height"]), comparison)
    candidates = {"homography": {"frames": base, "distortion": None}}
    registration = calibrated.get("registration") or {}
    radial = {stem: value["H_top_down_magnified"] for stem, value in registration.get("frames", {}).items()
              if value.get("status") == "registered" and value.get("H_top_down_magnified")}
    if calibrated.get("parameters") and radial:
        candidates["radial"] = {"frames": rebase(radial, comparison), "distortion": calibrated["parameters"]}
    model_bytes = canonical(candidates)
    return {"version": VERSION, "comparison_reference": comparison,
            "training_reference_position": sequence["header"]["reference_image"],
            "training_input_sha256": hashlib.sha256(canonical(training_records)).hexdigest(),
            "models_sha256": hashlib.sha256(model_bytes).hexdigest(), "models": candidates,
            "calibration_status": calibrated.get("status"), "calibration_reason": calibrated.get("reason"),
            "calibration_recipe": calibration_recipe(),
            "selection": "both predetermined diagnostics; production choice recorded before opening test view",
            "all_training_readable_frames": [r["stem"] for r in training_records]}


def load_detections(work, stems, height):
    result, sources = {}, {}
    hypothesis = adapter.HYPOTHESES[adapter.DEFAULT_HYPOTHESIS]
    for stem in stems:
        path = work / "cache" / (stem + ".lst")
        if not path.is_file():
            result[stem] = np.empty((0, 2), dtype=np.float64)
            sources[stem] = {"state": "missing"}
            continue
        stars = siril_sequence.read_lst(path)["stars"]
        xy = np.array([adapter.convert_point(hypothesis, s["x"], s["y"], height) for s in stars], dtype=np.float64).reshape(-1, 2)
        if not np.isfinite(xy).all():
            raise ValueError("nonfinite test detection")
        result[stem] = xy
        sources[stem] = {"state": "present", "sha256": digest(path), "count": len(xy)}
    return result, sources


def detection_inside(plan, xy, *, guard):
    result = np.zeros(len(xy), dtype=bool)
    for x0, y0, x1, y1 in plan["test_rectangles_xyxy"]:
        result |= ((xy[:, 0] >= x0+guard) & (xy[:, 0] < x1-guard)
                   & (xy[:, 1] >= y0+guard) & (xy[:, 1] < y1-guard))
    return result


def spatial_region(x, y, width, height):
    if .3*width <= x <= .7*width and .3*height <= y <= .7*height:
        return "center"
    if (x < .3*width or x > .7*width) and (y < .3*height or y > .7*height):
        return ("top_" if y < .3*height else "bottom_") + ("left" if x < .3*width else "right")
    return "other"


def _associate(reference_xy, predicted_xy, radius):
    from starstack.holdout_association import associate_holdout
    return associate_holdout(reference_xy, predicted_xy, radius=radius)


def associate_test_tracks(plan, detections, test_matrices):
    """No candidate transform is accepted by this interface."""
    stems = [f"light_{i+1:05d}" for i in range(len(plan["inputs"]))]
    reference = stems[plan["reference_index"]]
    ref_xy = detections[reference]
    core = detection_inside(plan, ref_xy, guard=plan["detection_guard_px"])
    track = {stem: [] for stem in stems}
    for stem in stems:
        source = detections[stem]
        source_core = detection_inside(plan, source, guard=plan["detection_guard_px"])
        source_in_region = detection_inside(plan, source, guard=0)
        if stem == reference:
            records = [{"reference_index": i, "source_index": i, "status": "matched",
                        "candidate_count": 1, "distance_px": 0.} for i in range(len(ref_xy))]
        elif stem not in test_matrices:
            records = [{"reference_index": i, "source_index": None, "status": "frame_unavailable",
                        "candidate_count": 0, "distance_px": None} for i in range(len(ref_xy))]
        else:
            predicted = apply_matrix(np.asarray(test_matrices[stem]), source)
            finite = np.isfinite(predicted).all(axis=1)
            indices = np.flatnonzero(finite)
            records = _associate(ref_xy, predicted[finite], plan["association_radius_px"])
            for entry in records:
                if entry["source_index"] is not None:
                    entry["source_index"] = int(indices[entry["source_index"]])
        if len(records) != len(ref_xy) or [r["reference_index"] for r in records] != list(range(len(ref_xy))):
            raise ValueError("association helper did not preserve the reference denominator")
        for entry in records:
            i, j = entry["reference_index"], entry["source_index"]
            entry["association_status"] = entry["status"]
            entry["reference_xy"] = ref_xy[i].tolist()
            entry["source_xy"] = source[j].tolist() if j is not None else None
            if not core[i]:
                entry["status"] = "reference_guard"
            elif entry["status"] == "matched" and not source_core[j]:
                entry["status"] = "source_guard" if source_in_region[j] else "left_region"
            entry["reference_id"] = f"test-reference-{i:06d}"
        track[stem] = records
    complete = [i for i in range(len(ref_xy)) if all(track[s][i]["status"] == "matched" for s in stems)]
    for records in track.values():
        for entry in records:
            entry["global_track_isolation"] = "conditional_on_test_identity" if entry["reference_index"] in complete else "unknown"
    return {"reference": reference, "reference_count": len(ref_xy), "reference_core_count": int(core.sum()),
            "globally_hidden_reference_indices": complete, "frames": track,
            "identity_basis": "independent_test_only_matcher_not_physical_truth",
            "global_isolation_basis": "all training-readable frames matched inside fixed test guards, conditional on inferred identity",
            "selection_bias": "test detection and finite association gate remain; complete-track subset is a survivor subset"}


def summarize_errors(records, denominator):
    values = np.array([r["error_px"] for r in records], dtype=np.float64)
    return {"reference_denominator": denominator, "measured": len(values), "unmeasured": denominator-len(values),
            "rms_px": float(np.sqrt(np.mean(values**2))) if len(values) else None,
            "median_px": float(np.median(values)) if len(values) else None,
            "p95_px": float(np.percentile(values, 95)) if len(values) else None,
            "max_px": float(values.max()) if len(values) else None}


def evaluate_candidate(plan, frozen, tracks):
    reference = tracks["reference"]
    refs = tracks["frames"][reference]
    regions = {i: spatial_region(*entry["reference_xy"], plan["width"], plan["height"]) for i, entry in enumerate(refs)}
    results = {}
    for model_name, model in frozen["models"].items():
        distortion = rd.RadialDistortion(**model["distortion"]) if model["distortion"] is not None else None
        results[model_name] = {}
        for stem, entries in tracks["frames"].items():
            if stem == reference:
                continue
            h = model["frames"].get(stem)
            rows, complete_rows = [], []
            missing = []
            for entry in entries:
                reason = entry["status"] if entry["status"] != "matched" else ("model_unavailable" if h is None else None)
                if reason:
                    missing.append({"reference_id": entry["reference_id"], "reason": reason})
                    continue
                xy = np.array([entry["source_xy"]], dtype=np.float64)
                pred = apply_matrix(np.asarray(h), xy if distortion is None else distortion.apply(xy))
                if distortion is not None:
                    pred = distortion.inverse(pred)
                if not np.isfinite(pred).all():
                    missing.append({"reference_id": entry["reference_id"], "reason": "nonfinite_model_prediction"})
                    continue
                delta = pred[0] - np.asarray(entry["reference_xy"])
                row = {"reference_id": entry["reference_id"], "region": regions[entry["reference_index"]],
                       "residual_xy": delta.tolist(), "error_px": float(np.hypot(*delta)),
                       "global_track_isolation": entry["global_track_isolation"]}
                rows.append(row)
                if entry["global_track_isolation"] == "conditional_on_test_identity":
                    complete_rows.append(row)
            report = {"all_conditionally_matched": summarize_errors(rows, len(refs)),
                      "complete_hidden_tracks": summarize_errors(complete_rows, len(refs)),
                      "residuals": rows, "missing": missing, "regions": {}}
            for region in REGIONS:
                n = sum(value == region for value in regions.values())
                selected = [row for row in complete_rows if row["region"] == region]
                report["regions"][region] = {**summarize_errors(selected, n),
                                             "status": "measured_only" if selected else "insufficient_data"}
            results[model_name][stem] = report
    return {"models": results, "reference_self_evaluated": False,
            "judgement": {"status": "insufficient_data", "reason": "operational identities and match-conditioned sample; no real-image precision pass declared"}}


def run_plan(plan_path, output_root, *, executable=None):
    plan_path = Path(plan_path)
    raw_plan = plan_path.read_bytes()
    plan = validate_plan(json.loads(raw_plan))
    train_mask, test_mask = role_masks(plan)
    run = Path(output_root).resolve() / ("image-holdout-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:6])
    run.mkdir(parents=True, exist_ok=False)
    (run / "PLAN.json").write_bytes(raw_plan)
    sources_before = source_manifest()
    write_json(run / "sources-before.json", sources_before)
    result = {"version": VERSION, "run": str(run), "state": "failed", "precision_passed": False,
              "events": [], "stage_errors": [], "plan_sha256": hashlib.sha256(raw_plan).hexdigest()}

    def event(name):
        item = {"stage": name, "at": datetime.now(timezone.utc).isoformat()}
        result["events"].append(item)
        with (run / "events.jsonl").open("ab") as stream:
            stream.write(canonical(item) + b"\n")

    try:
        engine = siril.doctor(executable)
        if not engine.get("available"):
            raise RuntimeError(engine.get("message", "Siril unavailable"))
        from starstack.registered_support import require_engine
        require_engine(engine)
        write_json(run / "engine.json", engine)
        write_json(run / "resolved-contract.json", {"version": VERSION, "calibration": calibration_recipe(),
                   "probe_commands": PROBE_COMMANDS, "reference_policy": "two-pass auto per isolated view; comparison index fixed by plan; matrices rebased without rewriting seq",
                   "mask_train_sha256": hashlib.sha256(train_mask.tobytes()).hexdigest(),
                   "mask_test_sha256": hashlib.sha256(test_mask.tobytes()).hexdigest(),
                   "input_scale": "float32 preserved; unsigned16 fixed float32(value)/65535; never adaptive min/max",
                   "source_freeze_scope": {"modules": list(SOURCE_MODULES),
                       "validation": "complete explicit starstack import closure, including lazy imports, checked before and after",
                       "exclusion_reason": "core, GUI and transient/S4 modules are neither imported nor called by this runner; their independent edits do not invalidate this experiment",
                       "external_environment": {"python": sys.version.split()[0], "numpy": np.__version__}},
                   "point_coordinates": "TOP-DOWN zero-based native pixel centres"})
        event("training_views_begin")
        train, records = prepare_phase(run, plan, "train", train_mask)
        event("training_probe_begin")
        try:
            run_probe(engine["executable"], train)
        except Exception as exc:
            result["stage_errors"].append({"stage": "training_probe", "error": str(exc)})
        sequence = siril_sequence.read_seq(train / "work/light_.seq")
        event("training_calibration_begin")
        with (train / "calibration.log").open("w", encoding="utf-8") as log:
            calibrated = calibration.calibrate(train / "work", **calibration_recipe(), log=lambda message: log.write(message + "\n"))
        for item in records:
            if digest(train / "work" / (item["stem"] + ".fit")) != item["sha256"]:
                raise ValueError("training image changed during probe/calibration")
        frozen = freeze_models(plan, sequence, calibrated, records)
        write_json(run / "candidate.json", frozen)
        frozen_sha = digest(run / "candidate.json")
        write_json(run / "freeze.json", {"candidate_sha256": frozen_sha, "models_sha256": frozen["models_sha256"],
                   "sources": sources_before, "training_inputs": records})
        event("candidate_frozen")
        # This is deliberately the first test-view construction/detection call.
        test, test_records = prepare_phase(run, plan, "test", test_mask)
        event("test_views_ready")
        try:
            run_probe(engine["executable"], test)
        except Exception as exc:
            result["stage_errors"].append({"stage": "test_probe", "error": str(exc)})
        test_sequence = siril_sequence.read_seq(test / "work/light_.seq")
        test_h = rebase(sequence_matrices(test_sequence, plan["height"]), frozen["comparison_reference"])
        detections, detection_sources = load_detections(test / "work", [r["stem"] for r in records], plan["height"])
        write_json(run / "test-detections.json", {"frames": {k: v.tolist() for k, v in detections.items()}, "sources": detection_sources,
                   "test_internal_reference_position": test_sequence["header"]["reference_image"], "test_only_comparison_matrices": test_h})
        if digest(run / "candidate.json") != frozen_sha:
            raise ValueError("frozen candidate changed before evaluation")
        tracks = associate_test_tracks(plan, detections, test_h)
        write_json(run / "test-tracks.json", tracks)
        evaluation = evaluate_candidate(plan, frozen, tracks)
        write_json(run / "evaluation.json", evaluation)
        for item in test_records:
            if digest(test / "work" / (item["stem"] + ".fit")) != item["sha256"]:
                raise ValueError("test image changed during probe")
        if any(digest(item["path"]) != item["sha256"] for item in plan["inputs"]):
            raise ValueError("original input changed during diagnostic")
        after = source_manifest()
        write_json(run / "sources-after.json", after)
        if after != sources_before:
            raise ValueError("diagnostic sources changed during execution")
        event("evaluation_complete")
        result.update(state="partial" if result["stage_errors"] else "completed", candidate_sha256=frozen_sha,
                      models_sha256=frozen["models_sha256"], source_files_unchanged=True, original_inputs_unchanged=True,
                      reference_detections=tracks["reference_count"], globally_hidden_tracks=len(tracks["globally_hidden_reference_indices"]),
                      identity_basis=tracks["identity_basis"], judgement=evaluation["judgement"])
    except Exception as exc:
        result.update(error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    write_json(run / "RESULT.json", result)
    (run / "REPORT.md").write_text("# 实拍图像隔离诊断\n\n"
        + f"执行状态：{result['state']}。实拍精度通过：未声明。\n\n"
        + "训练图实际检测／匹配与共享拟合先完成并冻结，测试视图随后独立运行。两遍参考各自由隔离像素选择；固定中段只用于评价坐标。\n\n"
        + "测试身份来自独立的有限门匹配，完整隐藏轨迹仍以该身份假设为条件；未匹配、歧义和迁出见 test-tracks.json。"
        + "主报告分母保留全部参考测试检出，不按候选残差剔点；单帧匹配与完整隐藏轨迹分别统计。\n\n"
        + f"全部参考检出：{result.get('reference_detections', '未完成')}；完整隐藏轨迹：{result.get('globally_hidden_tracks', '未完成')}。\n\n"
        + "该实验不替代原生产配准，不证明 RAW 全流程隔离、物理身份真值或全体星点无偏精度。\n"
        + (f"\n执行错误：{result['error']}\n" if result.get("error") else ""), encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--siril", default=None)
    args = parser.parse_args(argv)
    result = run_plan(args.plan, args.output_root, executable=args.siril)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0 if result["state"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
