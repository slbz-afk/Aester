"""Known-kernel dependency proofs and bounded, real Siril edge-response experiments.

Does not change production coverage. Siril's kernel is observed, never assumed
to be the Keys cubic used for the independently checked second resampling.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np

VERSION = "1.0.0"
NUMERIC_TOLERANCE = 2e-6
FLAT_TOLERANCE = 1e-6
HEIGHT, WIDTH, LEVEL = 64, 80, .25


def scalar_weight(t, kernel="cubic"):
    """Independent scalar definition; no production kernel helpers."""
    z = abs(float(t))
    if kernel == "bilinear":
        return max(1-z, 0.)
    if z < 1:
        return 1.5*z**3-2.5*z**2+1
    if z < 2:
        return -.5*z**3+2.5*z**2-4*z+2
    return 0.


def scalar_taps(x, y, shape, kernel="cubic", boundary="clamp"):
    """None = unsupported geometry, otherwise exact nonzero (y,x,weight)."""
    h, w = shape
    if not (-.5 <= x < w-.5 and -.5 <= y < h-.5):
        return None
    offsets = (0, 1) if kernel == "bilinear" else (-1, 0, 1, 2)
    ix, iy = math.floor(x), math.floor(y)
    taps = []
    for oy in offsets:
        for ox in offsets:
            coefficient = scalar_weight(x-ix-ox, kernel)*scalar_weight(y-iy-oy, kernel)
            if coefficient == 0:
                continue
            sx, sy = ix+ox, iy+oy
            if boundary == "strict" and not (0 <= sx < w and 0 <= sy < h):
                return None
            taps.append((min(max(sy, 0), h-1), min(max(sx, 0), w-1), coefficient))
    return taps


def scalar_support(validity, points, kernel="cubic", boundary="clamp"):
    output = []
    for x, y in points:
        taps = scalar_taps(x, y, validity.shape, kernel, boundary)
        output.append(taps is not None and all(validity[y0, x0] for y0, x0, _ in taps))
    return np.array(output, dtype=bool)


def scalar_warp(image, points, shape):
    values = []
    for x, y in points:
        taps = scalar_taps(x, y, image.shape)
        if taps is None:
            values.append(0.)
        else:
            values.append(sum(float(image[y0, x0])*c for y0, x0, c in taps)/sum(c for _, _, c in taps))
    return np.asarray(values, dtype=np.float32).reshape(shape)


def grid(shape):
    y, x = np.indices(shape)
    return np.column_stack((x.ravel(), y.ravel())).astype(np.float64)


def matrix_map(matrix, points):
    mapped = np.column_stack((points, np.ones(len(points)))) @ np.asarray(matrix).T
    return mapped[:, :2]/mapped[:, 2, None]


def nearest(validity, points):
    h, w = validity.shape
    inside = ((points[:, 0] >= -.5) & (points[:, 0] < w-.5)
              & (points[:, 1] >= -.5) & (points[:, 1] < h-.5))
    result = np.zeros(len(points), dtype=bool)
    xy = np.rint(points[inside]).astype(int)
    result[inside] = validity[np.clip(xy[:, 1], 0, h-1), np.clip(xy[:, 0], 0, w-1)]
    return result


def comparison(image, flat, scope):
    diff = np.abs(image.astype(np.float64)-flat.astype(np.float64))
    bad = scope & (diff > FLAT_TOLERANCE)
    examples = []
    for y, x in np.argwhere(bad)[:12]:
        examples.append({"x": int(x), "y": int(y), "image": float(image[y, x]),
                         "same_position_full_flat": float(flat[y, x]), "abs_error": float(diff[y, x])})
    return {"samples": int(scope.sum()), "invalid_count": int(bad.sum()),
            "max_same_flat_error": float(diff[scope].max()) if scope.any() else None,
            "examples": examples}


def pure_experiments():
    from starstack.interpolation_support import sample_support, SUPPORT_VERSION
    from starstack import resample
    shape = (24, 32)
    p = grid(shape)
    y, x = np.indices(shape)
    validities = {"vertical": x < 16, "diagonal": x+y < 29,
                  "thin_invalid_line": x != 13, "one_pixel_hole": ~((x == 13) & (y == 11)),
                  "all_valid": np.ones(shape, dtype=bool)}
    theta = math.radians(2.)
    rotation = np.array([[math.cos(theta), -math.sin(theta), .25],
                         [math.sin(theta), math.cos(theta), -.25], [0., 0., 1.]])
    maps = {"identity": p, "quarter_xy": p+[.25, .25], "half_x": p+[.5, 0.],
            "rotation_2deg": matrix_map(rotation, p)}
    cases = []
    for name, valid in validities.items():
        for map_name, coords in maps.items():
            for kernel in ("bilinear", "cubic"):
                for boundary in ("clamp", "strict"):
                    oracle = scalar_support(valid, coords, kernel, boundary)
                    actual = sample_support(valid, coords, kernel=kernel, boundary=boundary)
                    center = nearest(valid, coords)
                    cases.append({"mask": name, "mapping": map_name, "kernel": kernel, "boundary": boundary,
                                  "samples": len(p), "oracle_mismatches": int((oracle != actual).sum()),
                                  "invalid_count": int((~actual).sum()),
                                  "nearest_valid_full_unsupported": int((center & ~actual).sum()),
                                  "full_supported": int(actual.sum())})
    valid = validities["vertical"]
    first_points, second_points = p+[.25, 0.], p+[-.25, 0.]
    stage1 = sample_support(valid, first_points).reshape(shape)
    final = sample_support(stage1, second_points).reshape(shape)
    oracle1 = scalar_support(valid, first_points).reshape(shape)
    oracle2 = scalar_support(oracle1, second_points).reshape(shape)
    im1 = resample.warp_image(valid.astype(np.float32), lambda q: q+[.25, 0.], fill=0.)
    im2 = resample.warp_image(im1, lambda q: q+[-.25, 0.], fill=0.)
    true1 = scalar_warp(valid.astype(float), first_points, shape)
    true2 = scalar_warp(true1, second_points, shape)
    two = {"stage1_supported": int(stage1.sum()), "final_supported": int(final.sum()),
           "stage1_invalid_count": int((~stage1).sum()), "final_invalid_count": int((~final).sum()),
           "stage1_oracle_mismatches": int((stage1 != oracle1).sum()),
           "final_oracle_mismatches": int((final != oracle2).sum()),
           "net_identity_false_support": int((valid & ~final).sum()),
           "numeric_max_abs_error": float(np.max(np.abs(im2-true2))),
           "boundary_value_expected": .9044189453125, "boundary_value_actual": float(im2[12, 15]),
           "stage1_same_flat_under_nearest": comparison(im1, np.ones(shape, dtype=np.float32), nearest(valid, first_points).reshape(shape)),
           "same_flat_under_net_identity": comparison(im2, np.ones(shape, dtype=np.float32), valid)}
    return {"support_version": SUPPORT_VERSION, "cases": cases, "two_stage": two,
            "passed": all(c["oracle_mismatches"] == 0 for c in cases)
                      and two["stage1_oracle_mismatches"] == two["final_oracle_mismatches"] == 0
                      and two["numeric_max_abs_error"] <= NUMERIC_TOLERANCE
                      and two["boundary_value_actual"] == two["boundary_value_expected"]}


def declared_transforms():
    from starstack.synthetic_starfield import rotation_scale_about
    answer = [("identity", np.eye(3))]
    for phase in (.25, .5, .75):
        answer.append((f"translation_{phase}", np.array([[1., 0., phase], [0., 1., phase], [0., 0., 1.]])))
    answer.append(("rotation_2deg", rotation_scale_about(2., 1., (WIDTH-1)/2, (HEIGHT-1)/2, .25, -.25)))
    return answer


def real_engine(run, executable=None):
    from astropy.io import fits
    from starstack import siril, siril_nodata_proof, siril_sequence, registration_adapter, resample
    from starstack.registration_distortion import RadialDistortion
    from starstack.registration_calibration import native_from_undistorted
    from starstack.sip_header import sip_cards, write_wcs_header_file
    from starstack.synthetic_starfield import write_fits_rgb32
    status = siril.doctor(executable)
    if not status.get("available"):
        return {"status": "insufficient_data", "reason": status, "cases": []}
    points = grid((HEIGHT, WIDTH))
    yy, xx = np.indices((HEIGHT, WIDTH))
    validity = (xx < WIDTH//2) & ~((xx == 13) & (yy == 21))
    original = np.where(validity, LEVEL, 0.).astype(np.float32)
    flat = np.full((HEIGHT, WIDTH), LEVEL, dtype=np.float32)
    # Fixed model and all measurement constants are written in plan.json before running.
    model = RadialDistortion(k1=.04, k2=0., cx=(WIDTH-1)/2, cy=(HEIGHT-1)/2,
                             scale=50., magnification=.97, direction="u_to_p")
    transforms = declared_transforms()
    result = {"status": "measured", "engine": status, "frames": len(transforms), "commands": [],
              "actual_first_kernel": "unknown; measured through engine", "cases": [], "sip_model": model.as_dict()}
    for geometry in ("homography", "sip"):
        outputs = {}
        for kind, image in (("full", flat), ("masked", original)):
            target = run/f"{geometry}_{kind}"
            target.mkdir()
            files = []
            for i, _ in enumerate(transforms, 1):
                path = target/f"light_{i:05d}.fit"
                digest = write_fits_rgb32(path, np.repeat(image[None], 3, axis=0), comments=["synthetic interpolation support experiment"])
                files.append({"name": path.name, "sha256": digest})
            truth = {"height": HEIGHT, "frames": [{"shift_px": [0., 0.]} for _ in transforms]}
            commands = [f"cd {target.name}", "seqheader light NAXIS1"]
            result["commands"].append(commands)
            siril.run_script(status["executable"], run, target.name+"_sequence", commands, lambda: None,
                             total_timeout=90, progress_timeout=60)
            siril_nodata_proof.write_translation_sequence(target, truth)
            hypothesis = registration_adapter.HYPOTHESES[registration_adapter.DEFAULT_HYPOTHESIS]
            matrices = {i: registration_adapter.to_siril_matrix(hypothesis, h, HEIGHT, HEIGHT)
                        for i, (_, h) in enumerate(transforms)}
            siril_sequence.write_registration(target/"light_.seq", matrices, layer=1, reference_position=0)
            if geometry == "sip":
                write_wcs_header_file(target/"distortion.wcs", sip_cards(model, WIDTH, HEIGHT), WIDTH, HEIGHT)
                lines = (target/"light_.seq").read_text(encoding="utf-8").splitlines()
                first_registration = next(j for j, line in enumerate(lines) if line.startswith("R1 "))
                lines.insert(first_registration, "D1 2 distortion.wcs")
                (target/"light_.seq").write_text("\n".join(lines)+"\n", encoding="utf-8")
            (target/"sequence_before_export.txt").write_bytes((target/"light_.seq").read_bytes())
            commands = [f"cd {target.name}", "seqapplyreg light -filter-included"]
            result["commands"].append(commands)
            siril.run_script(status["executable"], run, target.name+"_export", commands, lambda: None,
                             total_timeout=90, progress_timeout=60)
            arrays = []
            for i in range(1, len(transforms)+1):
                path = target/f"r_light_{i:05d}.fit"
                with fits.open(path, memmap=False) as hdul:
                    if hdul[0].header.get("ROWORDER") != "BOTTOM-UP":
                        raise ValueError("Unexpected Siril row order")
                    if int(hdul[0].header["BITPIX"]) != -32:
                        raise ValueError("Expected float32 Siril output")
                    arrays.append(np.asarray(hdul[0].data[0, ::-1], dtype=np.float32))
            outputs[kind] = arrays
            result.setdefault("inputs", {})[target.name] = files
        for i, (name, h) in enumerate(transforms):
            # These exact-coordinate center checks are diagnostics, not Siril kernel claims.
            src = matrix_map(np.linalg.inv(h), points)
            if geometry == "sip":
                src = model.inverse(src)
            nn = nearest(validity, src).reshape(HEIGHT, WIDTH)
            full1, masked1 = outputs["full"][i], outputs["masked"][i]
            interior = (xx >= 8) & (xx < WIDTH-8) & (yy >= 8) & (yy < HEIGHT-8)
            if geometry == "homography":
                second_points = matrix_map(h, points)
                mapping = lambda q, matrix=h: matrix_map(matrix, q)
                full2 = resample.warp_image(full1, mapping, fill=0.)
                masked2 = resample.warp_image(masked1, mapping, fill=0.)
                final_native = validity
            else:
                second_points = model.apply(points)
                bottom = np.repeat(masked1[::-1, :, None], 3, axis=2)
                masked2 = native_from_undistorted(bottom, model.as_dict())[::-1, :, 0]
                bottom = np.repeat(full1[::-1, :, None], 3, axis=2)
                full2 = native_from_undistorted(bottom, model.as_dict())[::-1, :, 0]
                native_sources = model.inverse(matrix_map(np.linalg.inv(h), second_points))
                final_native = nearest(validity, native_sources).reshape(HEIGHT, WIDTH)
            oracle2 = scalar_warp(masked1, second_points, (HEIGHT, WIDTH))
            error = float(np.max(np.abs(masked2.astype(float)-oracle2.astype(float))))
            entry = {"geometry": geometry, "transform": name, "H_top_down": h.tolist(),
                     "first_stage": comparison(masked1, full1, nn & interior),
                     "final_stage": comparison(masked2, full2, final_native & interior),
                     "second_stage_scalar_oracle_max_abs_error": error,
                     "second_stage_oracle_passed": error <= NUMERIC_TOLERANCE,
                     "first_stage_full_support": "unknown: actual Siril kernel not identified",
                     "final_full_support": "unknown: first-stage support not known"}
            entry["full_flat_control_max_error"] = float(np.max(np.abs(full2[interior]-LEVEL)))
            result["cases"].append(entry)
            write_fits_rgb32(run/f"support_{geometry}_{i+1}_native.fit", np.repeat(masked2[None], 3, axis=0))
        (run/"support-engine-partial.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    result["second_stage_oracles_passed"] = all(c["second_stage_oracle_passed"] for c in result["cases"])
    result["full_flat_controls_passed"] = all(c["full_flat_control_max_error"] <= FLAT_TOLERANCE for c in result["cases"])
    result["nearest_counterexample_observed"] = any(c["final_stage"]["invalid_count"] > 0 for c in result["cases"])
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT/"outputs/a7-support-20260919")
    parser.add_argument("--real-siril", action="store_true")
    parser.add_argument("--siril-path")
    args = parser.parse_args(argv)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run = args.output_root.resolve()/f"support-{stamp}-{uuid.uuid4().hex[:6]}"
    run.mkdir(parents=True)
    plan = {"version": VERSION, "numeric_tolerance": NUMERIC_TOLERANCE, "flat_tolerance": FLAT_TOLERANCE,
            "shape": [HEIGHT, WIDTH], "level": LEVEL, "border_exclusion_px": 8,
            "transforms": [{"name": name, "H": h.tolist()} for name, h in declared_transforms()],
            "sip": {"k1": .04, "k2": 0., "cx": (WIDTH-1)/2, "cy": (HEIGHT-1)/2, "scale": 50.,
                    "magnification": .97, "direction": "u_to_p"},
            "scope": "Numerical diagnostics; no photographic threshold or production coverage change"}
    (run/"plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    report = {"version": VERSION, "run": str(run), "environment": {"python": sys.version, "numpy": np.__version__,
              "platform": platform.platform()}, "plan_sha256": hashlib.sha256((run/"plan.json").read_bytes()).hexdigest(),
              "pure": pure_experiments()}
    if args.real_siril:
        try:
            report["siril"] = real_engine(run, args.siril_path)
        except Exception as exc:
            report["siril"] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                               "note": "Partial scripts/logs retained in run; no measured success claimed"}
    else:
        report["siril"] = {"status": "not_run"}
    (run/"support.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# 插值支持实验", "", f"运行：{run.name}", "", f"纯核 oracle：{report['pure']['passed']}",
             f"真实 Siril：{report['siril']['status']}", "", "Siril 首段完整核支持尚未知；下表仅报告实际观测及第二段独立数值核对。", "",
             "| 几何 | 变换 | 首段中心有效但平场差异 | 最终中心有效但平场差异 | 第二段 oracle 最大差 |",
             "|---|---|---:|---:|---:|"]
    for c in report["siril"].get("cases", []):
        lines.append(f"| {c['geometry']} | {c['transform']} | {c['first_stage']['invalid_count']} | {c['final_stage']['invalid_count']} | {c['second_stage_scalar_oracle_max_abs_error']:.8g} |")
    lines += ["", "数值容差预先固定，不作为摄影画质门槛。未运行拒绝／归一化堆栈；不证明真实场景接边已修复。"]
    if report["siril"].get("error"):
        lines += ["", report["siril"]["error"]]
    (run/"SUPPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({"run": str(run), "pure_passed": report["pure"]["passed"], "siril": report["siril"]["status"]}))
    if not report["pure"]["passed"] or report["siril"]["status"] == "failed":
        return 2
    if args.real_siril and report["siril"]["status"] != "measured":
        return 3
    if args.real_siril and not (report["siril"]["second_stage_oracles_passed"]
                               and report["siril"]["full_flat_controls_passed"]
                               and report["siril"]["nearest_counterexample_observed"]):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
