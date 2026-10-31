"""Does Siril 1.4.4 stacking treat zero pixels as "no data"?  (S3 horizon band, RUN_LOG §24)

``run`` writes 8 synthetic frames with zero and dark regions in known frame
subsets (see ``starstack.siril_nodata_proof``), executes the stacking variants
on copies (the product's ``rej w 3 3 -norm=addscale``, no-normalisation
rejection, a plain mean, and the product's own-registration path
``seqapplyreg`` + stack with translation matrices we write) and analyzes the
masters; ``analyze`` re-reads an existing run; ``simulate`` writes fake
masters for both possible behaviours so the analysis is exercised off-Windows
(never a proof).

    <output-root>/nodata-proof-<UTC stamp>-<6hex>/
        work/            light_0000k.fit, truth.json
        v_<variant>/     copies + Siril products
        *.ssf / *.log    Siril scripts and logs
        nodata_proof.json, NODATA_PROOF.md

Example (Windows, product venv):

    .venv\\Scripts\\python.exe scripts\\s3_siril_nodata_proof.py run --output-root outputs\\fable-s2-20260917
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from starstack import siril_nodata_proof as npf  # noqa: E402


def _stamp() -> str:
    import datetime as _dt
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _copy_frames(work: Path, target: Path) -> None:
    target.mkdir(exist_ok=False)
    for path in sorted(work.glob("light_*.fit")):
        shutil.copyfile(path, target / path.name)


def run_variants(run: Path, siril_path: str | None, variants: list[str]) -> dict:
    from starstack import siril
    status = siril.doctor(siril_path)
    if not status.get("available"):
        raise SystemExit(f"Siril not available: {status.get('message')}")
    work = run / "work"
    truth = json.loads((work / "truth.json").read_text(encoding="utf-8"))

    def no_cancel():
        return None

    outcome = {"siril": status, "variants": {}}
    for name in variants:
        target = run / f"v_{name}"
        _copy_frames(work, target)
        entry = {"commands": []}
        try:
            if name in npf.APPLYREG_VARIANTS:
                # let Siril create the sequence file first (harmless header read), then put our matrices into it
                step_a = [f"cd v_{name}", "seqheader light NAXIS1"]
                entry["commands"].append(step_a)
                try:
                    siril.run_script(status["executable"], run, f"{name}_a_seqheader", step_a, no_cancel)
                except Exception as exc:   # the hand-written layout is the fallback
                    entry["seqheader_error"] = str(exc)[-1500:]
                entry["sequence"] = npf.write_translation_sequence(target, truth, deselect=npf.DESELECTED_FRAME if name in npf.DESELECTED_VARIANTS else None)
                shutil.copyfile(target / "light_.seq", target / "light_written_by_us.seq")
            step = [f"cd v_{name}", *npf.VARIANTS[name]]
            entry["commands"].append(step)
            siril.run_script(status["executable"], run, f"{name}", step, no_cancel)
            entry["status"] = "ran"
        except Exception as exc:
            entry["status"] = "failed"
            entry["error"] = str(exc)[-2000:]
        entry["files"] = sorted(p.name for p in target.iterdir())
        outcome["variants"][name] = entry
    return outcome


def analyze(run: Path, *, siril_version: str | None = None) -> dict:
    truth = json.loads((run / "work" / "truth.json").read_text(encoding="utf-8"))
    results = {"proof_version": npf.NODATA_PROOF_VERSION, "run": run.name, "siril_version": siril_version, "variants": {},
               "environment": {"platform": platform.platform(), "python": sys.version.split()[0], "numpy": np.__version__, "timestamp": npf.now()}}
    for target in sorted(run.glob("v_*")):
        name = target.name[2:]
        try:
            results["variants"][name] = npf.analyze_variant(truth, target, variant=name)
        except Exception as exc:
            results["variants"][name] = {"status": "analysis_error", "error": f"{type(exc).__name__}: {exc}"}
    return results


def simulate(run: Path) -> None:
    """Fake masters: 'excluded' behaviour in v_rej_addscale, 'included' in v_mean_nonorm (analysis exercise only)."""
    from starstack.synthetic_starfield import write_fits_rgb16
    truth = json.loads((run / "work" / "truth.json").read_text(encoding="utf-8"))
    ref = truth["frames"][0]["sky_level"]
    for name, excluded in (("rej_addscale", True), ("mean_nonorm", False)):
        target = run / f"v_{name}"
        target.mkdir(exist_ok=True)
        rng = np.random.default_rng(1)
        img = rng.normal(ref if excluded else float(np.mean([f["sky_level"] for f in truth["frames"]])), npf.NOISE / 2, size=(npf.HEIGHT, npf.WIDTH))
        for rname, r in truth["regions"].items():
            n_aff, n = len(r["frames"]), len(truth["frames"])
            if r["kind"] == "zero":
                value = (ref if n_aff < n else 0.0) if excluded else ref * (n - n_aff) / n
            else:
                value = ref if n_aff <= 3 else npf.GROUND
            img[r["y0"]:r["y1"], r["x0"]:r["x1"]] = value
        write_fits_rgb16(target / "stack.fit", np.clip(np.round(img * npf.FULL), 0, npf.FULL).astype(np.uint16), comments=["SIMULATED master, not Siril output"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    p_run = sub.add_parser("run")
    p_run.add_argument("--output-root", required=True)
    p_run.add_argument("--siril", default=None)
    p_run.add_argument("--variants", nargs="*", default=list(npf.VARIANTS))
    p_an = sub.add_parser("analyze")
    p_an.add_argument("--run", required=True)
    p_sim = sub.add_parser("simulate")
    p_sim.add_argument("--output-root", required=True)
    args = ap.parse_args(argv)
    if args.command == "run":
        # absolute: Siril is started with -d <run dir> and then resolves the -s script path from there
        run = (Path(args.output_root) / f"nodata-proof-{_stamp()}-{uuid.uuid4().hex[:6]}").resolve()
        run.mkdir(parents=True, exist_ok=False)
        npf.generate_dataset(run / "work")
        outcome = run_variants(run, args.siril, args.variants)
        (run / "run_outcome.json").write_text(json.dumps(outcome, indent=1, ensure_ascii=False), encoding="utf-8")
        results = analyze(run, siril_version=outcome["siril"].get("version"))
    elif args.command == "simulate":
        run = (Path(args.output_root) / f"nodata-proof-SIMULATED-{_stamp()}-{uuid.uuid4().hex[:6]}").resolve()
        run.mkdir(parents=True, exist_ok=False)
        npf.generate_dataset(run / "work")
        simulate(run)
        results = analyze(run, siril_version="SIMULATED")
    else:
        run = Path(args.run).resolve()
        results = analyze(run)
    (run / "nodata_proof.json").write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    (run / "NODATA_PROOF.md").write_text(npf.render(results), encoding="utf-8")
    summary = {name: (v.get("verdicts", {}).get("deselected_frame_skipped", v.get("verdicts", {}).get("zeros")) if name in npf.DESELECTED_VARIANTS
                      else v.get("verdicts", {}).get("zeros", v.get("status"))) for name, v in results["variants"].items()}
    print(f"{summary}: {run / 'NODATA_PROOF.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
