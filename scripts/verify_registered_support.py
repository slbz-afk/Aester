"""Bounded real-Siril integration proof for conservative registered support.

Masked and full inputs share signal, transforms and the SAME export eligibility
before stacking. This fixes the actual contributor sets and avoids mistaking a
different stack/normalization population for interpolation leakage. No RAWs used.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from astropy.io import fits
from starstack import registered_support as support
from starstack import siril, siril_nodata_proof, siril_sequence
from starstack.registration_calibration import native_from_undistorted
from starstack.registration_distortion import RadialDistortion
from starstack.sip_header import sip_cards, write_wcs_header_file
from starstack.siril_support_map import make_export_mapping
from starstack.synthetic_starfield import write_fits_rgb32, rotation_scale_about

WIDTH, HEIGHT, FRAMES = 128, 96, 5
TOLERANCE = 2e-6
PATTERNS = ("flat", "step", "random_thin_structure")


def source_hashes():
    names = ("registered_support.py", "support_bounds.py", "support_anchor.py", "support_kernel.py",
             "interpolation_support.py", "siril_support_map.py", "resample.py", "sip_header.py")
    result = {}
    for name in names:
        path = ROOT / "src/starstack" / name
        if path.exists():
            result[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def engine_hashes(engine):
    executable = Path(engine["executable"])
    paths = [executable] + [executable.parent/name for name in support.INTERPOLATION_DLLS]
    return {path.name: support.sha256(path) for path in paths}


def read_cube(path):
    with fits.open(path, memmap=False) as hdus:
        if hdus[0].header["BITPIX"] != -32 or hdus[0].data.shape != (3, HEIGHT, WIDTH):
            raise ValueError("unexpected test FITS type or geometry")
        image = np.asarray(hdus[0].data, dtype=np.float32)
        if hdus[0].header["ROWORDER"] == "BOTTOM-UP":
            image = image[:, ::-1]
        elif hdus[0].header["ROWORDER"] != "TOP-DOWN":
            raise ValueError("unknown FITS row order")
        return np.ascontiguousarray(image)


def stats(a, b, mask):
    if mask.shape != a.shape[-2:] or a.shape != b.shape:
        raise ValueError("comparison geometry mismatch")
    diff = np.asarray(a, float)[:, mask] - np.asarray(b, float)[:, mask]
    finite = np.isfinite(diff)
    return {"pixels": int(mask.sum()), "finite_values": int(finite.sum()),
            "nonfinite_values": int((~finite).sum()),
            "max_abs_diff": float(np.max(np.abs(diff[finite]))) if finite.any() else None,
            "rms_diff": float(np.sqrt(np.mean(diff[finite]**2))) if finite.any() else None,
            "above_tolerance_values": int((np.abs(diff) > TOLERANCE).sum())}


def acceptable(result):
    return (result["pixels"] >= 100 and result["nonfinite_values"] == 0
            and result["max_abs_diff"] is not None and result["max_abs_diff"] <= TOLERANCE)


def pattern(name):
    y, x = np.indices((HEIGHT, WIDTH))
    if name == "flat":
        plane = np.full((HEIGHT, WIDTH), .3)
    else:
        rng = np.random.default_rng(83470)
        plane = .25 + rng.normal(0, .004, (HEIGHT, WIDTH))
        if name == "step":
            plane += .2 * (x >= WIDTH // 2)
        elif name == "random_thin_structure":
            plane += .08 * (x % 13 == 0) + .06 * (y % 17 == 0)
            for xx, yy, amp in zip(rng.integers(10, WIDTH-10, 20), rng.integers(10, HEIGHT-10, 20), rng.uniform(.05, .2, 20)):
                plane += amp * np.exp(-((x-xx)**2 + (y-yy)**2)/.8)
        else:
            raise ValueError(name)
    return np.asarray(plane[None] * np.array([1.1, 1., .85])[:, None, None], np.float32)


def validities():
    yy, xx = np.indices((HEIGHT, WIDTH))
    result = {}
    for i in range(FRAMES):
        valid = (yy < 76-i) & (xx > 10+i)
        valid &= ~((xx == 58+i) & (yy >= 25) & (yy < 48))
        valid &= ~((xx == 94-i) & (yy == 35+i))
        result[f"light_{i+1:05d}"] = valid
    return result


class Sources:
    def __init__(self, frames):
        self.frames = frames

    def for_frame(self, stem):
        return self.frames[stem]


def matrices():
    return {
        0: np.eye(3),
        1: np.array([[1., 0., .25], [0., 1., .5], [0., 0., 1.]]),
        2: np.array([[1., 0., -.5], [0., 1., .75], [0., 0., 1.]]),
        3: rotation_scale_about(2., 1., (WIDTH-1)/2, (HEIGHT-1)/2, .25, -.25),
        4: np.array([[1.002, .004, -.7], [-.003, .998, .8], [.00013, -.00007, 1.]]),
    }


def prepare_case(run, name, cube, validity, h, model, engine):
    work = run / name
    work.mkdir()
    records = []
    for i in range(FRAMES):
        stem = f"light_{i+1:05d}"
        # Both copies use identical per-frame signal; only masked loses source data.
        image = cube.copy()
        if name.endswith("_masked"):
            image[:, ~validity[stem]] = 0
        path = work / (stem + ".fit")
        records.append({"file": path.name, "sha256": write_fits_rgb32(path, image)})
    siril.run_script(engine, run, name+"_discover", [f"cd {name}", "seqheader light NAXIS1"], lambda: None,
                     total_timeout=90, progress_timeout=60)
    siril_nodata_proof.write_translation_sequence(work, {"height": HEIGHT, "frames": [{"shift_px": [0., 0.]} for _ in range(FRAMES)]})
    siril_sequence.write_registration(work/"light_.seq", h, layer=1, reference_position=0)
    if model is not None:
        write_wcs_header_file(work/"distortion.wcs", sip_cards(model, WIDTH, HEIGHT), WIDTH, HEIGHT)
        lines = (work/"light_.seq").read_text(encoding="utf-8").splitlines()
        i = next(i for i, line in enumerate(lines) if line.startswith("R1 "))
        lines.insert(i, "D1 2 distortion.wcs")
        (work/"light_.seq").write_text("\n".join(lines)+"\n", encoding="utf-8")
    shutil.copyfile(work/"light_.seq", work/support.EXPORT_SEQUENCE)
    siril.run_script(engine, run, name+"_export", [f"cd {name}", support.EXPORT_COMMAND], lambda: None,
                     total_timeout=90, progress_timeout=60)
    return work, records


def native_image(cube, model):
    if model is None:
        return cube.copy()
    bottom_up_rgb = np.moveaxis(cube[:, ::-1], 0, -1)
    native = native_from_undistorted(bottom_up_rgb, model.as_dict(), kernel="cubic")
    return np.ascontiguousarray(np.moveaxis(native[::-1], -1, 0))


def run_case(run, geometry, pattern_name, engine):
    model = None if geometry == "homography" else RadialDistortion(
        k1=.06, k2=.004, cx=61.75, cy=49.25, scale=70., magnification=.97, direction="u_to_p")
    source = Sources(validities())
    cube = pattern(pattern_name)
    h = matrices()
    works, inputs = {}, {}
    for kind in ("full", "masked"):
        works[kind], inputs[kind] = prepare_case(run, f"{geometry}_{pattern_name}_{kind}", cube, source.frames, h, model, engine)
    frame_results = []
    stored = support.export_transforms(works["masked"]/support.EXPORT_SEQUENCE)
    wcs_header = fits.getheader(works["masked"]/"distortion.wcs") if model is not None else None
    for stem, matrix in stored.items():
        full = read_cube(works["full"]/f"r_{stem}.fit")
        masked = read_cube(works["masked"]/f"r_{stem}.fit")
        mapping = make_export_mapping(matrix, stored["light_00001"], WIDTH, HEIGHT, wcs_header)
        valid = support.registered_mask(source.for_frame(stem), matrix, model, mapping=mapping)
        on = stats(full, masked, valid)
        off = stats(full, masked, ~valid)
        frame_results.append({"stem": stem, "supported": on, "unsupported_diagnostic": off,
                              "pass": acceptable(on)})
    (run/f"{geometry}_{pattern_name}_before_filter.json").write_text(json.dumps(frame_results, indent=2), encoding="utf-8")
    outputs, records = {}, {}
    # Flat uses no normalization because a perfectly uniform field has no scale.
    # Other patterns exercise the production addscale with exactly matched valid pixels.
    norm = "-nonorm" if pattern_name == "flat" else "-norm=addscale"
    for kind, work in works.items():
        support.prepare_exports(work, source, model)
        commands = [f"cd {work.name}", "seqclean r_light -stat",
                    f"stack r_light rej w 3 3 {norm} -32b -filter-included -out=sky_master.fit"]
        siril.run_script(engine, run, work.name+"_stack", commands, lambda: None,
                         total_timeout=90, progress_timeout=60)
        records[kind] = support.finish_stack(work, work/"sky_master.fit", model)
        outputs[kind] = read_cube(work/"sky_master.fit")
    # This bounded fixture retains the undistorted master rather than publishing
    # a product job's native master; use the descriptor before that publish step.
    masks = {kind: support.load_array(work, records[kind]["native_valid"]) for kind, work in works.items()}
    native = {kind: native_image(cube, model) for kind, cube in outputs.items()}
    for kind, image in native.items():
        write_fits_rgb32(works[kind]/"native_for_comparison.fit", image)
    stack_mask = support.load_array(works["masked"], records["masked"]["stack_valid"])
    stack_comparison = stats(outputs["full"], outputs["masked"], stack_mask)
    native_comparison = stats(native["full"], native["masked"], masks["masked"])
    all_pixels = stats(outputs["full"], outputs["masked"], np.ones((HEIGHT, WIDTH), dtype=bool))
    result = {"geometry": geometry, "pattern": pattern_name, "normalization": norm,
              "model": None if model is None else model.as_dict(), "inputs": inputs,
              "frames": frame_results, "stack_supported_comparison": stack_comparison,
              "stack_all_pixels_comparison": all_pixels, "native_supported_comparison": native_comparison,
              "same_native_support": bool(np.array_equal(masks["full"], masks["masked"])),
              "counts": {k: {name: records[k][name] for name in ("stack_supported_pixels", "native_supported_pixels", "ambiguous_zero_or_nonfinite_pixels")} for k in records},
              "counterexample_present_outside_certified_support": any(f["unsupported_diagnostic"]["above_tolerance_values"] > 0 for f in frame_results)}
    result["pass"] = (all(f["pass"] for f in frame_results) and acceptable(stack_comparison)
                      and acceptable(native_comparison) and acceptable(all_pixels)
                      and result["same_native_support"] and result["counterexample_present_outside_certified_support"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT/"outputs/support-integration-20260919")
    args = parser.parse_args()
    engine = siril.doctor()
    if not engine.get("available"):
        raise RuntimeError(engine)
    support.require_engine(engine)
    run = args.out.resolve()/("registered-support-"+dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"-"+uuid.uuid4().hex[:6])
    run.mkdir(parents=True, exist_ok=False)
    before_hashes = source_hashes()
    before_engine_hashes = engine_hashes(engine)
    plan = {"geometry": ["homography", "sip"], "patterns": PATTERNS, "width": WIDTH,
            "height": HEIGHT, "frames_per_copy": FRAMES, "tolerance": TOLERANCE,
            "support_version": support.SUPPORT_VERSION, "source_sha256": before_hashes,
            "export_command": support.EXPORT_COMMAND, "actual_engine_sha256": before_engine_hashes,
            "expected_engine_sha256": {Path(engine["executable"]).name: support.SIRIL_SHA256, **support.INTERPOLATION_DLLS},
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "comparison": "same signal and per-frame post-export eligibility in both copies; flat nonorm, structured images production addscale",
            "scope": "bounded synthetic proof using actual engine; no RAW, no photographic quality or flux-recovery claim"}
    (run/"PLAN.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    report = {"plan": plan, "engine": engine, "cases": []}
    for geometry in plan["geometry"]:
        for name in PATTERNS:
            result = run_case(run, geometry, name, engine["executable"])
            report["cases"].append(result)
            print(json.dumps({"case": geometry+"/"+name, "pass": result["pass"],
                              "native_diff": result["native_supported_comparison"]["max_abs_diff"]}), flush=True)
            (run/"RESULT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["source_unchanged_during_run"] = before_hashes == source_hashes()
    report["engine_unchanged_during_run"] = before_engine_hashes == engine_hashes(engine)
    report["pass"] = (all(c["pass"] for c in report["cases"]) and report["source_unchanged_during_run"]
                      and report["engine_unchanged_during_run"])
    (run/"RESULT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# Registered support integration", "", "Actual Siril with fixed masked/full eligibility; this is a numerical support proof.", "",
             "| Geometry | Pattern | Native supported pixels | Native max abs diff | Pass |", "| --- | --- | ---: | ---: | --- |"]
    for case in report["cases"]:
        lines.append(f"| {case['geometry']} | {case['pattern']} | {case['native_supported_comparison']['pixels']} | {case['native_supported_comparison']['max_abs_diff']:.8g} | {case['pass']} |")
    lines.extend(["", f"Source unchanged during run: {report['source_unchanged_during_run']}",
                  f"Engine executable and pinned DLLs unchanged during run: {report['engine_unchanged_during_run']}",
                  f"Export command: {support.EXPORT_COMMAND}", f"Overall pass: {report['pass']}", "",
                  "Flat tests use -nonorm because a perfectly flat image has no normalization scale. Step and random thin structure use the production -norm=addscale.",
                  "No equal-count or same-frame-for-all-native-neighbours assumption is made; support is aggregated before native propagation."])
    (run/"REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({"run": str(run), "pass": report["pass"]}), flush=True)
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
