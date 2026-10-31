"""Compare isolated Siril CFA conversion with rawpy on two existing DNGs.

Run from the project root with .venv/Scripts/python.exe scripts/validate_siril_cfa.py.
Each run keeps fresh per-file directories and logs. This compares wrappers of
LibRaw, not independent decoders or the original camera ADC measurements.
"""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import sys
import time
import uuid

import numpy as np
import rawpy
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from starstack.siril import conversion_failure_message, doctor, run_script
from development_assets import INPUT_DIR

OUTPUT = ROOT / "outputs" / "raw-foundation-20260916"
REPORT = ROOT / "docs" / "diagnostics" / "siril-cfa-20260925.json"
SOURCES = (
    OUTPUT / "synthetic-rational.dng",
    INPUT_DIR / "frame00001.dng",  # DSC_2206, the first retained 48-frame input.
)


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(record: dict) -> None:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


def siril_libraw(executable: Path) -> dict:
    """Query only the known installed engine's version export; no image call."""
    candidates = sorted(executable.parent.glob("libraw-*.dll")) if os.name == "nt" else []
    if len(candidates) != 1:
        return {"version": None, "status": "not_queried", "candidate_count": len(candidates)}
    dll = candidates[0]
    with os.add_dll_directory(str(dll.parent)):
        library = ctypes.CDLL(str(dll))
        library.libraw_version.restype = ctypes.c_char_p
        value = library.libraw_version().decode("ascii")
    return {"path": str(dll), "sha256": sha(dll), "version": value, "status": "version_export_only"}


def compare_arrays(actual: np.ndarray, expected: np.ndarray) -> dict:
    if actual.shape != expected.shape:
        raise ValueError(f"CFA shape mismatch: {actual.shape} != {expected.shape}")
    actual_digest, expected_digest = hashlib.sha256(), hashlib.sha256()
    different, total_abs, diff_min, diff_max = 0, 0, None, None
    examples = []
    different_rows = np.zeros(actual.shape[0], dtype=bool)
    different_columns = np.zeros(actual.shape[1], dtype=bool)
    for y in range(0, actual.shape[0], 128):
        a, b = actual[y:y + 128], expected[y:y + 128]
        actual_digest.update(np.asarray(a, dtype="<u2", order="C").tobytes())
        expected_digest.update(np.asarray(b, dtype="<u2", order="C").tobytes())
        delta = a.astype(np.int32) - b.astype(np.int32)
        nonzero = delta != 0
        different += int(np.count_nonzero(nonzero))
        total_abs += int(np.abs(delta).sum(dtype=np.int64))
        low, high = int(delta.min()), int(delta.max())
        diff_min = low if diff_min is None else min(diff_min, low)
        diff_max = high if diff_max is None else max(diff_max, high)
        different_rows[y:y + len(a)] = np.any(nonzero, axis=1)
        different_columns |= np.any(nonzero, axis=0)
        if len(examples) < 12:
            for dy, x in np.argwhere(nonzero)[:12 - len(examples)]:
                examples.append({"row": int(y + dy), "column": int(x),
                                 "siril": int(a[dy, x]), "rawpy_mapped": int(b[dy, x]),
                                 "difference": int(delta[dy, x])})
    return {"sample_count": int(actual.size), "different_samples": different,
            "equal_fraction": float(1 - different / actual.size),
            "difference_minimum": diff_min, "difference_maximum": diff_max,
            "maximum_absolute_difference": max(abs(diff_min), abs(diff_max)),
            "mean_absolute_difference": float(total_abs / actual.size),
            "different_row_count": int(np.count_nonzero(different_rows)),
            "different_column_count": int(np.count_nonzero(different_columns)),
            "examples_in_fits_array_coordinates": examples,
            "canonical_hash_format": "C-order little-endian uint16 physical samples, FITS BOTTOM-UP array order",
            "siril_samples_sha256": actual_digest.hexdigest(),
            "rawpy_mapped_samples_sha256": expected_digest.hexdigest(),
            "exactly_equal": different == 0}


def validate_source(source: Path, executable: str, timeout: float) -> dict:
    source = source.resolve(strict=True)
    directory = OUTPUT / f"siril-cfa-{uuid.uuid4().hex}"
    stage = directory / "stage"
    stage.mkdir(parents=True)
    staged = stage / source.name
    log = directory / "convert-cfa.log"
    result = {"state": "running", "source_path": str(source), "directory": str(directory),
              "source_sha256_before": sha(source), "source_size_bytes": source.stat().st_size,
              "staged_path": str(staged), "log_path": str(log), "error": None}
    started = time.monotonic()
    try:
        shutil.copyfile(source, staged)
        result["staged_sha256_before"] = sha(staged)
        if result["staged_sha256_before"] != result["source_sha256_before"]:
            raise ValueError("Staged copy differs from original before conversion")
        commands = ["setext fit", "set16bits", "setcompress 0", "cd stage", "convertraw cfa -out=../cfa"]
        result["commands"] = commands
        deadline = time.monotonic() + timeout

        def deadline_check() -> None:
            if time.monotonic() > deadline:
                raise TimeoutError("Bounded Siril CFA conversion deadline exceeded")

        run_script(executable, directory, "convert-cfa", commands, deadline_check)
        converted = sorted((directory / "cfa").glob("cfa_*.fit"))
        result["converted_count"] = len(converted)
        if len(converted) != 1:
            raise RuntimeError(conversion_failure_message(log, f"Expected one CFA FITS, got {len(converted)}"))
        fit_path = converted[0]
        with fits.open(fit_path, memmap=False, uint=True) as hdul:
            header = hdul[0].header.copy()
            actual = np.array(hdul[0].data, copy=True)
        result["fits"] = {"path": str(fit_path), "sha256": sha(fit_path), "dtype": str(actual.dtype),
            "shape": list(actual.shape), "minimum": int(actual.min()), "maximum": int(actual.max()),
            "header": {key: header.get(key) for key in ("BITPIX", "BZERO", "BSCALE", "NAXIS", "NAXIS1", "NAXIS2", "NAXIS3", "BAYERPAT", "ROWORDER", "XBAYROFF", "YBAYROFF")},
            "fits_reader": "astropy with uint=True, memmap=False; BZERO/BSCALE applied"}
        if actual.ndim != 2 or header.get("NAXIS") != 2:
            raise ValueError("No-debayer conversion returned a non-CFA / non-2D image")
        if header.get("BAYERPAT") not in {"RGGB", "BGGR", "GBRG", "GRBG"}:
            raise ValueError("Expected a recorded 2x2 RGB Bayer pattern")
        if header.get("ROWORDER") != "BOTTOM-UP":
            raise ValueError("Unexpected row order; do not guess a mapping")
        if actual.dtype.kind != "u" or actual.dtype.itemsize != 2:
            raise ValueError("Expected uint16 physical FITS samples for exact code comparison")
        with rawpy.imread(str(source)) as raw:
            full = raw.raw_image.copy()
            sizes = {key: (float(value) if isinstance(value, float) else int(value))
                     for key, value in raw.sizes._asdict().items()}
            if full.ndim != 2 or full.dtype != np.uint16:
                raise ValueError("Expected rawpy uint16 full CFA")
            width, height = sizes["iwidth"], sizes["iheight"]
            left, top = sizes["left_margin"], sizes["top_margin"]
            if top < 0 or left < 0 or top + height > full.shape[0] or left + width > full.shape[1]:
                raise ValueError("Rawpy size metadata cannot map onto its CFA array")
            expected = full[top:top + height, left:left + width][::-1, :]
            result["rawpy"] = {"dtype": str(full.dtype), "full_shape": list(full.shape), "sizes": sizes,
                "full_minimum": int(full.min()), "full_maximum": int(full.max()),
                "reported_pattern": raw.raw_pattern.tolist() if raw.raw_pattern is not None else None,
                "reported_black_level_per_channel": list(raw.black_level_per_channel),
                "reported_white_level": int(raw.white_level)}
        result["mapping"] = {"operation": "crop by LibRaw top/left margin and iheight/iwidth, then reverse row order",
            "slice": {"top": top, "left": left, "height": height, "width": width},
            "rotation": "none", "normalization": "none", "black_subtraction": "none",
            "source": "Siril 1.4.4 readraw_in_cfa normal non-Fuji branch",
            "source_url": "https://gitlab.com/free-astro/siril/-/raw/1.4.4/src/io/image_formats_libraries.c"}
        result["comparison"] = compare_arrays(actual, expected)
        if not result["comparison"]["exactly_equal"]:
            raise ValueError("Siril CFA and mapped rawpy CFA differ; see retained quantitative comparison")
        result["state"] = "succeeded"
    except Exception as exc:
        result.update(state="failed", error=str(exc), error_type=type(exc).__name__)
    finally:
        result["elapsed_seconds"] = round(time.monotonic() - started, 3)
        result["source_sha256_after"] = sha(source)
        result["source_unchanged"] = result["source_sha256_after"] == result["source_sha256_before"]
        if staged.is_file():
            result["staged_sha256_after"] = sha(staged)
            result["staged_unchanged"] = result["staged_sha256_after"] == result.get("staged_sha256_before")
        if not result["source_unchanged"] or not result.get("staged_unchanged"):
            result.update(state="failed", error="An original or staged hash changed")
        for name in ("siril.ini", "convert-cfa.ssf", "convert-cfa.log"):
            path = directory / name
            if path.is_file():
                result[name] = {"path": str(path), "sha256": sha(path)}
        (directory / "comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--siril", default=None)
    parser.add_argument("--timeout", type=float, default=180, help="Maximum seconds per Siril conversion")
    args = parser.parse_args()
    if not 1 <= args.timeout <= 600:
        parser.error("--timeout must be between 1 and 600 seconds")
    report = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(), "state": "running",
              "script_path": str(Path(__file__).resolve()), "script_sha256": sha(Path(__file__)),
              "cases": [], "versions": {"python": sys.version.split()[0], "rawpy": rawpy.__version__,
                  "rawpy_libraw": list(rawpy.libraw_version), "rawpy_flags": rawpy.flags,
                  "numpy": np.__version__, "astropy": version("astropy")},
              "limits": ["Siril and rawpy are wrappers of LibRaw; this is not an independent-decoder validation.",
                  "Only the two named DNG files are tested, with the recorded versions and mapping.",
                  "Exact equality is not proof that source NEF, converted DNG and camera ADC are equivalent.",
                  "Unpack format-specific processing and in-camera processing are not reversed or independently verified.",
                  "This no-debayer test does not validate debayered RGB, white balance, color matrices or final stacked output."]}
    save(report)
    try:
        engine = doctor(args.siril)
        report["engine"] = engine
        if not engine.get("available"):
            raise RuntimeError(engine.get("message"))
        report["versions"]["siril_libraw"] = siril_libraw(Path(engine["executable"]))
        for source in SOURCES:
            result = validate_source(source, engine["executable"], args.timeout)
            report["cases"].append(result)
            save(report)
            print(json.dumps({"source": source.name, "state": result["state"],
                              "different_samples": result.get("comparison", {}).get("different_samples"),
                              "error": result.get("error")}, ensure_ascii=False), flush=True)
        report["state"] = "succeeded" if all(case["state"] == "succeeded" for case in report["cases"]) else "failed"
    except Exception as exc:
        report.update(state="failed", error=str(exc), error_type=type(exc).__name__)
    report["completed_at"] = datetime.now(timezone.utc).isoformat()
    save(report)
    print(f"Report: {REPORT}", flush=True)
    return 0 if report["state"] == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
