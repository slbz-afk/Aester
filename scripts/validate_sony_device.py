"""Sony α7R V single-frame evidence: E0 and B1–B11 of docs/SONY_ADAPTATION_PREREG_20260925.md.

Run from the project root (Windows venv):
    .venv\\Scripts\\python.exe -X utf8 scripts\\validate_sony_device.py --make-freeze outputs\\sony-a7rv-20260925\\freeze-stage1.json
    .venv\\Scripts\\python.exe -X utf8 scripts\\validate_sony_device.py --freeze outputs\\sony-a7rv-20260925\\freeze-stage1.json --output outputs\\sony-a7rv-20260925

Everything is written into one new ``<output>/stage1-<UTC>/`` directory.  Originals under raw_2_sony are only read or
copied (never linked).  Criteria and thresholds are the pre-registered ones; the script never adapts them to what it
measures.  The run refuses to start when any file recorded in the freeze receipt (including this script) changed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

PREREG = ROOT / "docs" / "SONY_ADAPTATION_PREREG_20260925.md"
MANIFEST = ROOT / "docs" / "testdata" / "RAW2_SONY_SCENES_20260925.json"
CATALOG = ROOT / "docs" / "RAW2_SONY_ASSET_CATALOG_20260925.md"
INVENTORY = ROOT / "outputs" / "sony-assets-20260925" / "raw2_sony_inventory.json"
SOURCES_REF = ROOT / "outputs" / "sony-color-refs-20260925" / "SOURCES.json"
RAW_ROOT = ROOT / "raw_2_sony"
SIRIL = ROOT / "tools" / "siril" / "bin" / "siril-cli.exe"
LIBRAW_DLL = ROOT / "tools" / "siril" / "bin" / "libraw-25.dll"
ADOBE = ROOT / ".cache" / "z8-test" / "dng-portable" / "{app}" / "Adobe DNG Converter.exe"
FROZEN_EXISTING = {  # E0: pre-change versions used in stage 1 (full hash recorded; prefix must match)
    "scripts/c1_accept_color_contract.py": "b7848ee45198d6d0",
    "scripts/validate_siril_cfa.py": "326649a80b81fdfd",
    "scripts/development_assets.py": None,
    "src/starstack/devices.py": "a9258e538aff0934",
    "src/starstack/devices_registry.py": "048d9950b939529a",
    "src/starstack/color_products.py": "1921f887c851757f",
}
E0_EXPECTED = {"siril_version": "1.4.4", "siril_sha_prefix": "132c735ecb623559", "libraw_sha_prefix": "f6509adf691fc469",
               "libraw_version": "0.22.1-Release", "rawpy": "0.27.1", "rawpy_libraw": [0, 22, 1], "adobe_version": "18.6"}
FREEZE_FILES = [PREREG, MANIFEST, CATALOG, INVENTORY, Path(__file__).resolve()]
CHECK_FRAMES = ["1/DSC07337.ARW", "1/DSC07376.ARW", "1/DSC07414.ARW", "2/DSC07606.ARW", "2/DSC07620.ARW", "2/DSC07634.ARW"]
MAIN_SEGMENT_2 = [f"DSC0{n}.ARW" for n in range(7606, 7635)]
CAMCONST_ILCE_7RM5 = [8200, -2976, -719, -4296, 12053, 2532, -429, 1282, 5774]
PDAF_PATTERN = [0, 12, 18, 36, 42, 60, 66, 72, 78, 96, 108, 120, 126, 138, 156, 168, 180, 186, 192, 198, 210, 222, 228, 240,
                246, 252, 270, 276, 282, 288, 306, 312, 318, 330, 336, 348, 360, 366, 372, 378, 390, 396, 408, 420]
PDAF_OFFSET = 1
PLATEAU_MIN_PIXELS = 20          # B6
PLATEAU_LOWER_FRACTION = 0.97    # B6, = master_export.SATURATION_FRACTION
EDGE_SCORE_LIMIT = 5.0           # B10 (report)

LOG_PATTERN = re.compile(r"error in libraw|corrupt|unexpected end", re.IGNORECASE)
ENV_CLEAR = ("AESTER_DNG_CONVERTER",)


def sha256(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def rel(path: Path) -> str:
    resolved = Path(path).resolve()
    return resolved.relative_to(ROOT).as_posix() if resolved.is_relative_to(ROOT) else str(resolved)


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, bytes):
        return value.decode("ascii", "replace")
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=1, allow_nan=True), encoding="utf-8")


# ---------------------------------------------------------------- freeze


def make_freeze(target: Path) -> int:
    target = Path(target)
    if target.exists():
        print(f"refusing to overwrite existing freeze receipt {target}")
        return 2
    target.parent.mkdir(parents=True, exist_ok=True)
    receipt = {"schema": "sony-prereg-freeze-1", "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "purpose": "stage 0.5 freeze before any Sony pixel decode (prereg §五)",
               "files": {rel(p): sha256(p) for p in FREEZE_FILES}}
    write_json(target, receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=1))
    return 0


def check_freeze(receipt_path: Path) -> dict:
    receipt = json.loads(Path(receipt_path).read_text(encoding="utf-8"))
    mismatches = {}
    for name, digest in receipt["files"].items():
        actual = sha256(ROOT / name)
        if actual != digest:
            mismatches[name] = {"frozen": digest, "actual": actual}
    missing = [rel(p) for p in FREEZE_FILES if rel(p) not in receipt["files"]]
    return {"receipt": str(receipt_path), "receipt_sha256": sha256(receipt_path), "created_utc": receipt.get("created_utc"),
            "mismatches": mismatches, "not_frozen": missing, "ok": not mismatches and not missing}


# ---------------------------------------------------------------- E0


def e0_engine() -> dict:
    import rawpy
    from starstack.siril import doctor
    import validate_siril_cfa as vsc
    out = {"expected": E0_EXPECTED}
    doc = doctor(str(SIRIL))
    out["siril"] = {"path": str(SIRIL), "sha256": sha256(SIRIL), "doctor": doc}
    out["libraw_dll"] = vsc.siril_libraw(SIRIL)
    rawpy_dir = Path(rawpy.__file__).parent
    binaries = sorted(list(rawpy_dir.glob("_rawpy*.pyd")) + list(rawpy_dir.glob("raw_r*.dll")) + list(rawpy_dir.glob("libraw*.dll")))
    out["rawpy"] = {"version": rawpy.__version__, "libraw_version": list(rawpy.libraw_version),
                    "binaries": {b.name: sha256(b) for b in binaries}}
    version = None
    if ADOBE.is_file():
        run = subprocess.run(["powershell", "-NoProfile", "-Command", f"(Get-Item -LiteralPath '{ADOBE}').VersionInfo.FileVersion"],
                             capture_output=True, text=True, timeout=60)
        version = run.stdout.strip() or None
    out["adobe"] = {"path": str(ADOBE), "exists": ADOBE.is_file(), "sha256": sha256(ADOBE) if ADOBE.is_file() else None,
                    "file_version": version}
    frozen = {}
    for name, prefix in FROZEN_EXISTING.items():
        digest = sha256(ROOT / name)
        frozen[name] = {"sha256": digest, "expected_prefix": prefix, "ok": prefix is None or digest.startswith(prefix)}
    out["frozen_existing"] = frozen
    checks = {
        "siril_version": doc.get("version") == E0_EXPECTED["siril_version"],
        "siril_sha": out["siril"]["sha256"].startswith(E0_EXPECTED["siril_sha_prefix"]),
        "libraw_dll_sha": str(out["libraw_dll"].get("sha256", "")).startswith(E0_EXPECTED["libraw_sha_prefix"])
                          and Path(out["libraw_dll"].get("path", "")).resolve() == LIBRAW_DLL.resolve(),
        "libraw_dll_version": out["libraw_dll"].get("version") == E0_EXPECTED["libraw_version"],
        "rawpy": rawpy.__version__ == E0_EXPECTED["rawpy"] and list(rawpy.libraw_version) == E0_EXPECTED["rawpy_libraw"],
        "adobe_version": bool(version) and version.startswith(E0_EXPECTED["adobe_version"]),
        "frozen_existing": all(v["ok"] for v in frozen.values()),
    }
    out["checks"] = checks
    out["pass"] = all(checks.values())
    return out


# ---------------------------------------------------------------- originals


def listing(root: Path) -> list:
    rows = []
    for p in sorted(root.rglob("*")):
        st = p.lstat()
        rows.append([p.relative_to(root).as_posix(), "dir" if p.is_dir() else "file", st.st_size if p.is_file() else None,
                     st.st_mtime_ns if p.is_file() else None, p.is_symlink()])
    return rows


def originals_snapshot(files: list[Path]) -> dict:
    return {"listing": listing(RAW_ROOT), "sha256": {f.relative_to(RAW_ROOT).as_posix(): sha256(f) for f in files}}


# ---------------------------------------------------------------- per-file LibRaw facts


def cfa_channels(raw) -> dict:
    """{name: (py, px)} over the visible area, names R/G1/G2/B from raw_colors_visible + color_desc."""
    colors = raw.raw_colors_visible[:2, :2]
    desc = raw.color_desc.decode("ascii") if isinstance(raw.color_desc, bytes) else str(raw.color_desc)
    names, seen_g = {}, 0
    for py in range(2):
        for px in range(2):
            letter = desc[int(colors[py, px])]
            if letter == "G":
                seen_g += 1
                letter = f"G{seen_g}"
            names[letter] = (py, px)
    return names


def per_file_facts(path: Path) -> dict:
    import rawpy
    from starstack import master_export as me
    from starstack.orientation import source_orientation
    from starstack.raw_metadata import read_raw_metadata
    facts = {"path": rel(path)}
    with rawpy.imread(str(path)) as raw:
        full = raw.raw_image
        facts["raw_image"] = {"shape": list(full.shape), "dtype": str(full.dtype)}
        facts["sizes"] = {k: (float(v) if isinstance(v, float) else int(v)) for k, v in raw.sizes._asdict().items()}
        facts["raw_pattern"] = raw.raw_pattern.tolist()
        facts["color_desc"] = raw.color_desc.decode("ascii")
        facts["visible_colors_2x2"] = raw.raw_colors_visible[:2, :2].tolist()
        facts["black_level_per_channel"] = [int(v) for v in raw.black_level_per_channel]
        facts["white_level"] = int(raw.white_level)
        cwl = raw.camera_white_level_per_channel
        facts["camera_white_level_per_channel"] = None if cwl is None else [int(v) for v in cwl]
        facts["camera_whitebalance"] = [float(v) for v in raw.camera_whitebalance]
        facts["daylight_whitebalance"] = [float(v) for v in raw.daylight_whitebalance]
        facts["rgb_xyz_matrix"] = np.asarray(raw.rgb_xyz_matrix, dtype=np.float64).tolist()
        visible = raw.raw_image_visible
        channels = cfa_channels(raw)
        stats = {}
        for name, (py, px) in channels.items():
            plane = visible[py::2, px::2]
            vmax = int(plane.max())
            stats[name] = {"phase": [py, px], "max": vmax, "count_at_max": int(np.count_nonzero(plane == vmax)),
                           "median_sub4": float(np.median(plane[::4, ::4]))}   # 事后 rename: 1/16 subsample, report only
        facts["cfa_stats"] = stats
    meta = read_raw_metadata(path)
    facts["metadata_status"] = meta.get("status")
    try:
        basis = me.colour_basis_from_raw(path, metadata=meta)
        facts["basis"] = {"state": "ok", "black_DN": basis["black_DN"], "white_DN": basis["white_DN"],
                          "as_shot_gains_rgb": basis["as_shot_gains_rgb"], "daylight_gains_rgb": basis["daylight_gains_rgb"],
                          "source_kind": basis["source"]["kind"]}
    except Exception as exc:
        facts["basis"] = {"state": "error", "error": f"{type(exc).__name__}: {exc}"}
    facts["orientation"] = source_orientation(path)
    return facts


# ---------------------------------------------------------------- B3 Adobe independent decode


def find_cfa_page(tif):
    pages = list(tif.pages)
    for page in list(pages):
        if page.pages is not None:
            pages.extend(page.pages)
    ifd0 = tif.pages[0]
    for page in pages:
        tags = page.tags
        photometric = tags.get("PhotometricInterpretation")
        subfile = tags.get("NewSubfileType")
        if photometric is not None and int(photometric.value) == 32803 and (subfile is None or int(subfile.value) == 0):
            return page, ifd0
    return None, ifd0


RATIONAL_TYPES = (5, 10)  # TIFF RATIONAL / SRATIONAL


def tag_value(page, code):
    """Tag value as numbers: RATIONAL/SRATIONAL decoded to floats by tag type, everything else as returned."""
    tag = page.tags.get(code)
    if tag is None:
        return None
    value = tag.value
    if int(tag.dtype) in RATIONAL_TYPES:
        flat = np.asarray(value, dtype=np.float64).reshape(-1)
        if flat.size % 2:
            raise ValueError(f"rational tag {code} has an odd number of integers")
        pairs = flat.reshape(-1, 2)
        return (pairs[:, 0] / pairs[:, 1]).tolist()
    if isinstance(value, bytes):
        # 事后 (erratum, 2026-09-26): BYTE/UNDEFINED tags (e.g. CFAPattern) are numbers, not text; only ASCII is decoded,
        # and only trailing NULs are dropped (the stage-1 records lost CFAPattern's leading 0 through strip()).
        if int(tag.dtype) == 2:
            return value.decode("ascii", "replace").rstrip("\x00")
        return list(value)
    return value


def rational_list(value):
    """Tag values are already decoded by type in ``tag_value``; this only flattens to a float list."""
    if value is None:
        return None
    return np.asarray(value, dtype=np.float64).reshape(-1).tolist()


def adobe_decode(frame: Path, stage: Path, facts: dict) -> dict:
    import rawpy
    import tifffile
    result = {"frame": rel(frame)}
    copy_dir = stage / "dng-stage"
    out_dir = stage / "dng" / frame.stem
    copy_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=False)
    staged = copy_dir / frame.name
    shutil.copyfile(frame, staged)
    result["staged_sha256"] = sha256(staged)
    command = [str(ADOBE), "-u", "-p0", "-d", str(out_dir), str(staged)]
    result["command"] = command
    started = time.monotonic()
    run = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True, timeout=300,
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    result["seconds"] = round(time.monotonic() - started, 3)
    result["returncode"] = run.returncode
    produced = sorted(out_dir.glob("*.dng")) + sorted(out_dir.glob("*.DNG"))
    produced = sorted(set(produced))
    result["produced"] = [p.name for p in produced]
    if run.returncode != 0 or len(produced) != 1:
        result.update(state="failed", error="Adobe conversion did not produce exactly one DNG")
        return result
    dng = produced[0]
    result["dng"] = {"path": rel(dng), "sha256": sha256(dng), "bytes": dng.stat().st_size}
    with tifffile.TiffFile(dng) as tif:
        page, ifd0 = find_cfa_page(tif)
        if page is None:
            result.update(state="failed", error="no CFA (PhotometricInterpretation 32803) main IFD in DNG")
            return result
        cfa = page.asarray()
        tags = {
            "compression": int(tag_value(page, "Compression") or 0),
            "shape": list(cfa.shape), "dtype": str(cfa.dtype),
            "ActiveArea": tag_value(page, 50829), "DefaultCropOrigin": tag_value(page, 50719),
            "DefaultCropSize": tag_value(page, 50720), "BlackLevel": tag_value(page, 50714),
            "BlackLevelRepeatDim": tag_value(page, 50713), "WhiteLevel": tag_value(page, 50717),
            "LinearizationTable_len": None if tag_value(page, 50712) is None else len(tag_value(page, 50712)),
            "CFAPattern": tag_value(page, 33422), "CFARepeatPatternDim": tag_value(page, 33421),
            "UniqueCameraModel": tag_value(ifd0, 50708), "ColorMatrix1": tag_value(ifd0, 50721),
            "ColorMatrix2": tag_value(ifd0, 50722), "CalibrationIlluminant1": tag_value(ifd0, 50778),
            "CalibrationIlluminant2": tag_value(ifd0, 50779), "AsShotNeutral": tag_value(ifd0, 50728),
            "BaselineExposure": tag_value(ifd0, 50730),
        }
        lin = tag_value(page, 50712)
    result["dng_tags"] = jsonable(tags)
    if lin is not None:
        table = np.asarray(lin, dtype=np.int64)
        cfa_cmp = table[np.clip(cfa.astype(np.int64), 0, table.size - 1)]
        result["linearization_applied"] = True
    else:
        cfa_cmp = cfa.astype(np.int64)
        result["linearization_applied"] = False
    with rawpy.imread(str(frame)) as raw:
        full = raw.raw_image.astype(np.int64)
    top, left = facts["sizes"]["top_margin"], facts["sizes"]["left_margin"]
    candidates = {"origin_0_0": (0, 0), "libraw_margins": (top, left)}
    aa = tags["ActiveArea"]
    if aa is not None and len(aa) == 4:
        candidates["activearea_vs_margins"] = (top - int(aa[0]), left - int(aa[1]))
    comparisons = {}
    for name, (oy, ox) in candidates.items():
        # D[y, x] <-> R[y + oy, x + ox]
        y0, x0 = max(0, -oy), max(0, -ox)
        y1 = min(cfa_cmp.shape[0], full.shape[0] - oy)
        x1 = min(cfa_cmp.shape[1], full.shape[1] - ox)
        if y1 - y0 <= 0 or x1 - x0 <= 0:
            comparisons[name] = {"offset": [oy, ox], "overlap": [0, 0], "different": None}
            continue
        d = cfa_cmp[y0:y1, x0:x1]
        r = full[y0 + oy:y1 + oy, x0 + ox:x1 + ox]
        diff = d - r
        different = int(np.count_nonzero(diff))
        comparisons[name] = {"offset": [oy, ox], "overlap": [int(y1 - y0), int(x1 - x0)], "different": different,
                             "max_abs_difference": int(np.abs(diff).max()),
                             "overlap_meets_9504x6336": (y1 - y0) >= 6336 and (x1 - x0) >= 9504}
    result["comparisons"] = comparisons
    winners = [n for n, c in comparisons.items() if c.get("different") == 0 and c.get("overlap_meets_9504x6336")]
    result["zero_difference_candidates"] = winners
    result["pass"] = bool(winners)
    result["state"] = "succeeded"
    result["staged_unchanged"] = sha256(staged) == result["staged_sha256"]
    return result


# ---------------------------------------------------------------- B4 Siril parity


def expected_bayerpat_bottom_up(facts: dict) -> dict:
    """Pattern of the BOTTOM-UP stored visible CFA under both plausible keyword semantics.

    Uses the colour indices of the visible area's top-left 2x2 (``raw_colors_visible``), so no assumption is made about
    which origin ``raw_pattern`` refers to."""
    colors = np.asarray(facts["visible_colors_2x2"])
    desc = facts["color_desc"]
    height = facts["sizes"]["iheight"]
    letters = lambda rows: "".join(desc[int(colors[r % 2, c])] for r in rows for c in (0, 1))
    top_down = letters([0, 1])
    stored_first = letters([(height - 1) % 2, (height - 2) % 2])
    return {"top_down_visible": top_down, "stored_row0_bottom_up": stored_first, "height": height,
            "semantics_agree": top_down == stored_first}


def siril_parity(frame: Path, stage: Path, facts: dict) -> dict:
    import validate_siril_cfa as vsc
    vsc.OUTPUT = stage / "siril-cfa"          # prereg B4: redirect the output root without touching the frozen file
    vsc.OUTPUT.mkdir(parents=True, exist_ok=True)
    result = vsc.validate_source(frame, str(SIRIL), 300)
    out = {"state": result.get("state"), "error": result.get("error"), "directory": result.get("directory"),
           "comparison": {k: result.get("comparison", {}).get(k) for k in ("exactly_equal", "different_samples", "sample_count",
                                                                          "maximum_absolute_difference")},
           "fits_header": result.get("fits", {}).get("header"), "rawpy_sizes": result.get("rawpy", {}).get("sizes"),
           "source_unchanged": result.get("source_unchanged")}
    header = out["fits_header"] or {}
    expected = expected_bayerpat_bottom_up(facts)
    out["bayerpat_expected"] = expected
    # Siril 1.4.4 writes BAYERPAT for the top-down visible image even with ROWORDER=BOTTOM-UP (established on a Z8
    # lossless NEF dry run before any Sony decode: height 5520, BAYERPAT RGGB = top-down visible pattern).
    out["bayerpat_semantics"] = "top_down_visible (Z8 dry-run 2026-09-25, pre-Sony)"
    out["bayerpat_ok"] = header.get("BAYERPAT") == expected["top_down_visible"]
    out["dims_ok"] = (header.get("NAXIS1") == facts["sizes"]["iwidth"] and header.get("NAXIS2") == facts["sizes"]["iheight"])
    log = Path(result.get("directory", "")) / "convert-cfa.log"
    hits = []
    if log.is_file():
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            if LOG_PATTERN.search(line):
                hits.append(line.strip())
    out["log_path"] = str(log)
    out["log_hits"] = hits
    out["pass"] = (out["state"] == "succeeded" and out["comparison"]["exactly_equal"] is True and out["dims_ok"]
                   and out["bayerpat_ok"] is True and header.get("ROWORDER") == "BOTTOM-UP" and not hits
                   and out["source_unchanged"] is True)
    return out


# ---------------------------------------------------------------- B10 / PDAF (report)


def robust_sigma(values: np.ndarray) -> float:
    med = np.median(values)
    return float(1.4826 * np.median(np.abs(values - med)))


def edge_report(frame: Path, facts: dict) -> dict:
    """B10 (report): per CFA channel plane, the outer strip vs the adjacent 64-px band (32 plane px), in noise units.

    Strips in visible (top-down LibRaw) coordinates: right 4 columns / bottom 2 rows (ART-flagged), left 4 / top 2 as
    controls.  On a channel plane (every other pixel) those are 2 columns or 1 row."""
    import rawpy
    black = facts["black_level_per_channel"]
    with rawpy.imread(str(frame)) as raw:
        visible = raw.raw_image_visible.copy()
        channels = cfa_channels(raw)
        colors = raw.raw_colors_visible[:2, :2].copy()
    edges = {"right_4_cols": (lambda P: P[:, -2:], lambda P: P[:, -34:-2]),
             "bottom_2_rows": (lambda P: P[-1:, :], lambda P: P[-33:-1, :]),
             "left_4_cols": (lambda P: P[:, :2], lambda P: P[:, 2:34]),
             "top_2_rows": (lambda P: P[:1, :], lambda P: P[1:33, :])}
    report = {}
    for edge, (strip_of, inner_of) in edges.items():
        per = {}
        for name, (py, px) in channels.items():
            plane = visible[py::2, px::2]
            strip = strip_of(plane).astype(np.float64).reshape(-1)
            inner = inner_of(plane).astype(np.float64).reshape(-1)
            if strip.size == 0 or inner.size == 0:
                per[name] = {"state": "empty"}
                continue
            blk = black[int(colors[py, px])]
            sigma = robust_sigma(inner)
            med_s, med_i = float(np.median(strip)), float(np.median(inner))
            score = abs(med_s - med_i) / sigma if sigma > 0 else float("inf")
            state = "anomalous" if score > EDGE_SCORE_LIMIT or float(strip.std()) == 0.0 else "ok"
            per[name] = {"strip_median": med_s, "inner_median": med_i, "inner_sigma": sigma, "score": score,
                         "strip_std": float(strip.std()), "inner_signal_over_black_sigma": (med_i - blk) / sigma if sigma > 0 else None,
                         "state": state}
        states = {v.get("state") for v in per.values()}
        report[edge] = {"channels": per, "state": "anomalous" if "anomalous" in states else "ok"}
    return report


def pdaf_report(frame: Path, facts: dict, group: str) -> dict:
    import rawpy
    with rawpy.imread(str(frame)) as raw:
        visible = raw.raw_image_visible.copy()
        channels = cfa_channels(raw)
    h, w = visible.shape
    rows = slice(0, int(0.6 * h)) if group == "1" else slice(0, h)
    cols = slice(0, w) if group == "1" else slice(int(0.65 * w), w)
    green_col_phase = {channels[n][0]: channels[n][1] for n in ("G1", "G2")}  # row parity -> green column parity
    y_idx = np.arange(rows.start or 0, rows.stop)
    c0 = cols.start or 0
    row_med = np.empty(y_idx.size, dtype=np.float64)
    for parity, px in green_col_phase.items():
        sel = (y_idx % 2) == parity
        first = px if (c0 % 2) == 0 else 1 - px          # column offset inside the region with the green phase
        block = visible[y_idx[sel], c0:cols.stop][:, first::2]
        row_med[sel] = np.median(block, axis=1)
    k = 31
    pad = np.pad(row_med, (k // 2, k // 2), mode="edge")
    trend = np.array([np.median(pad[i:i + k]) for i in range(row_med.size)])
    resid = row_med - trend
    y0 = rows.start or 0
    period = PDAF_PATTERN[-1]
    members = set(PDAF_PATTERN[:-1])
    out = {"region": {"rows": [y0, rows.stop], "cols": [cols.start or 0, cols.stop]}, "period": period}
    for origin, shift in (("visible_origin", 0), ("raw_origin", facts["sizes"]["top_margin"])):
        ys = np.arange(row_med.size) + y0 + shift
        is_pdaf = np.array([((y - PDAF_OFFSET) % period) in members for y in ys])
        a, b = resid[is_pdaf], resid[~is_pdaf]
        se = b.std() / np.sqrt(max(1, a.size))
        out[origin] = {"pdaf_rows": int(a.size), "mean_pdaf": float(a.mean()), "mean_other": float(b.mean()),
                       "t_statistic": float((a.mean() - b.mean()) / se) if se > 0 else None}
    return out


# ---------------------------------------------------------------- main run


def verdict(ok) -> str:
    return "PASS" if ok is True else ("FAIL" if ok is False else "UNDECIDABLE")


def run(freeze: Path, output: Path) -> int:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    # 事后 fix (program error, prereg §四.2): Siril resolves its script path from the stage directory; a relative
    # --output made B4 fail before any conversion (first run stage1-20260925T161516Z kept).
    stage = Path(output).resolve() / f"stage1-{stamp}"
    stage.mkdir(parents=True, exist_ok=False)
    record = {"schema": "sony-stage1-1", "started_utc": stamp, "prereg": rel(PREREG), "stage_dir": rel(stage)}
    started = time.monotonic()
    for name in ENV_CLEAR:
        if name in os.environ:
            record.setdefault("env_cleared", {})[name] = os.environ.pop(name)

    def save():
        record["elapsed_seconds"] = round(time.monotonic() - started, 1)
        write_json(stage / "RESULT.json", record)

    record["freeze"] = check_freeze(freeze)
    if not record["freeze"]["ok"]:
        record["state"] = "refused_freeze_mismatch"
        save()
        print("freeze mismatch; refusing to run")
        return 2
    record["E0"] = e0_engine()
    save()
    if not record["E0"]["pass"]:
        record["state"] = "stopped_E0"
        save()
        print("E0 failed; stopping")
        return 3

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    inventory = {f"{r['group']}/{r['name']}": r for r in json.loads(INVENTORY.read_text(encoding="utf-8"))["files"]}
    expected_sha = {}
    for scene in manifest["scenes"]:
        group = scene["source_directory"].split("/")[-1]
        for name, _bytes, digest in scene["source_identities"]:
            expected_sha[f"{group}/{name}"] = digest
    files = [RAW_ROOT / key for key in expected_sha]
    record["originals_before"] = originals_snapshot(files)
    save()

    # B1
    from starstack.devices import identify_camera
    b1 = {}
    for key in expected_sha:
        ident = identify_camera(RAW_ROOT / key)
        b1[key] = {"make": ident["make"], "model": ident["model"], "type": ident["sony_raw_file_type"],
                   "sha_ok": record["originals_before"]["sha256"][key] == expected_sha[key]}
    b1_ok = all(v["make"] == "SONY" and v["model"] == "ILCE-7RM5" and v["type"] == 4 and v["sha_ok"] for v in b1.values())
    record["B1"] = {"pass": b1_ok, "note": "结果事先已知（一致性复核）", "failures": {k: v for k, v in b1.items()
                    if not (v["make"] == "SONY" and v["model"] == "ILCE-7RM5" and v["type"] == 4 and v["sha_ok"])}}
    save()

    # B2 product import, fresh caches
    from starstack.raw_import import prepare_import
    b2 = {}
    for group in ("1", "2"):
        group_files = [str(RAW_ROOT / k) for k in expected_sha if k.startswith(group + "/")]
        for setting, discovery in (("a_discovery_off", False), ("b_discovery_on", True)):
            cache = stage / f"b2-{setting}-g{group}"
            cache.mkdir()
            t0 = time.monotonic()
            try:
                res = prepare_import(group_files, cache, converter_path=None, allow_converter_discovery=discovery)
                returned = [Path(p).resolve() for p in res.get("input_files", [])]
                entry = {"state": res.get("state"), "probe": res.get("probe"), "converted": res.get("converted"),
                         "count": len(returned), "same_as_originals": returned == [Path(p).resolve() for p in group_files]}
            except Exception as exc:
                entry = {"state": "error", "error": f"{type(exc).__name__}: {exc}"}
            probes = list(cache.rglob("probes/*.json"))
            supported = []
            for p in probes:
                try:
                    supported.append(json.loads(p.read_text("utf-8")).get("supported"))
                except (OSError, ValueError):
                    supported.append(None)
            entry.update(seconds=round(time.monotonic() - t0, 1), probe_records=len(probes),
                         probe_supported_all=bool(probes) and all(s is True for s in supported),
                         cache_hits_possible=False, note="新空缓存：全部为未命中")
            b2[f"g{group}-{setting}"] = entry
            save()
    b2_ok = all(v.get("state") == "ready" and v.get("probe") == "native_decode" and v.get("converted") is False
                and v.get("same_as_originals") for v in b2.values())

    # per-file facts (118)
    facts = {}
    for key in expected_sha:
        try:
            facts[key] = per_file_facts(RAW_ROOT / key)
        except Exception as exc:
            facts[key] = {"error": f"{type(exc).__name__}: {exc}"}
        if len(facts) % 10 == 0:
            print(f"facts {len(facts)}/{len(expected_sha)}", flush=True)
    write_json(stage / "per_file_facts.json", facts)
    unpack_ok = all("error" not in f for f in facts.values())
    sizes_set = {json.dumps(f.get("sizes"), sort_keys=True) for f in facts.values()}
    record["B2"] = {"pass": b2_ok and unpack_ok and len(sizes_set) == 1, "imports": b2, "unpack_all_ok": unpack_ok,
                    "sizes_identical": len(sizes_set) == 1, "sizes": json.loads(next(iter(sizes_set))) if len(sizes_set) == 1 else None}
    # 事后 (reported separately, the literal verdict above is kept): `flip` is the orientation field judged by B9.
    no_flip = {json.dumps({k: v for k, v in (f.get("sizes") or {}).items() if k != "flip"}, sort_keys=True) for f in facts.values()}
    flips = {}
    for key, f in facts.items():
        flips.setdefault(key.split("/")[0], set()).add((f.get("sizes") or {}).get("flip"))
    record["B2_posthoc_excluding_flip"] = {
        "pass": b2_ok and unpack_ok and len(no_flip) == 1, "sizes_identical_excluding_flip": len(no_flip) == 1,
        "flip_by_group": {g: sorted(v) for g, v in flips.items()},
        "sizes_excluding_flip": json.loads(next(iter(no_flip))) if len(no_flip) == 1 else None,
        "note": "事后：原判据字面包含 flip，两组方向不同必然不等；flip 由 B9 判定"}
    save()

    # B3 / B4 on the 6 frames
    b3, b4 = {}, {}
    for key in CHECK_FRAMES:
        frame = RAW_ROOT / key
        try:
            b3[key] = adobe_decode(frame, stage, facts[key])
        except Exception as exc:
            b3[key] = {"state": "error", "error": f"{type(exc).__name__}: {exc}", "pass": False}
        save()
        try:
            b4[key] = siril_parity(frame, stage, facts[key])
        except Exception as exc:
            b4[key] = {"state": "error", "error": f"{type(exc).__name__}: {exc}", "pass": False}
        record["B3"] = {"pass": all(v.get("pass") is True for v in b3.values()) and len(b3) == len(CHECK_FRAMES), "frames": b3}
        record["B4"] = {"pass": all(v.get("pass") is True for v in b4.values()) and len(b4) == len(CHECK_FRAMES), "frames": b4}
        save()

    # B5 black
    def adobe_scalar_list(value):
        if value is None:
            return None
        return [float(v) for v in rational_list(value)]
    b5_rows = {}
    for key, f in facts.items():
        inv = inventory[key]["raw_ifd"]["tag_7310_black"]
        basis = f.get("basis", {})
        b5_rows[key] = {"rawpy": f.get("black_level_per_channel"), "container_0x7310": inv,
                        "basis_black_DN": basis.get("black_DN"),
                        "ok": f.get("black_level_per_channel") == [512] * 4 and inv == [512] * 4
                              and basis.get("black_DN") == [512.0, 512.0, 512.0]}
    adobe_black = {k: adobe_scalar_list(v.get("dng_tags", {}).get("BlackLevel")) for k, v in b3.items()}
    adobe_black_ok = all(v is not None and all(abs(x - 512) < 1e-9 for x in v) for v in adobe_black.values())
    record["B5"] = {"pass": all(r["ok"] for r in b5_rows.values()) and adobe_black_ok, "adobe_black": adobe_black,
                    "failures": {k: r for k, r in b5_rows.items() if not r["ok"]}}
    save()

    # B6 white-level consistency
    white_sources = {"rawpy_white_level": sorted({f.get("white_level") for f in facts.values()}),
                     "rawpy_camera_white_level_per_channel": sorted({json.dumps(f.get("camera_white_level_per_channel")) for f in facts.values()}),
                     "container_WhiteLevel": sorted({inventory[k]["raw_ifd"]["white_level"] for k in facts}),
                     "adobe_WhiteLevel": {k: adobe_scalar_list(v.get("dng_tags", {}).get("WhiteLevel")) for k, v in b3.items()},
                     "basis_white_DN": sorted({f.get("basis", {}).get("white_DN") for f in facts.values()})}
    plateaus = []
    for key, f in facts.items():
        white = f.get("basis", {}).get("white_DN")
        for ch, s in (f.get("cfa_stats") or {}).items():
            if s["count_at_max"] >= PLATEAU_MIN_PIXELS:
                ok = white is not None and PLATEAU_LOWER_FRACTION * white <= s["max"] <= white
                plateaus.append({"frame": key, "channel": ch, "value": s["max"], "count": s["count_at_max"],
                                 "white_DN": white, "lower": None if white is None else PLATEAU_LOWER_FRACTION * white, "ok": ok})
    b6_state = None if not plateaus else all(p["ok"] for p in plateaus)
    record["B6"] = {"pass": b6_state, "verdict": verdict(b6_state), "white_sources": white_sources,
                    "plateau_count": len(plateaus), "plateaus_failing": [p for p in plateaus if not p["ok"]],
                    "plateau_values": sorted({(p["channel"], p["value"]) for p in plateaus}),
                    "note": "单帧必要条件；产品饱和掩膜在母版上计算"}
    write_json(stage / "B6_plateaus.json", plateaus)
    save()

    # B7 matrix
    target = np.asarray(CAMCONST_ILCE_7RM5, dtype=np.float64).reshape(3, 3)
    b7_rows = {}
    for key, f in facts.items():
        m = np.asarray(f.get("rgb_xyz_matrix"), dtype=np.float64)
        ok = m.shape == (4, 3) and np.all(np.abs(m[:3] * 10000 - target) <= 1.0) and np.all(m[3] == 0)
        b7_rows[key] = bool(ok)
    adobe_cm2 = {}
    first = facts[CHECK_FRAMES[0]]
    lib = np.asarray(first["rgb_xyz_matrix"], dtype=np.float64)[:3]
    for key, v in b3.items():
        cm2 = v.get("dng_tags", {}).get("ColorMatrix2")
        if cm2 is not None:
            arr = np.asarray(rational_list(cm2), dtype=np.float64).reshape(3, 3)
            adobe_cm2[key] = {"ColorMatrix2": arr.tolist(), "illuminant2": v["dng_tags"].get("CalibrationIlluminant2"),
                              "max_abs_diff_vs_libraw": float(np.abs(arr - lib).max())}
    record["B7"] = {"pass": all(b7_rows.values()), "libraw_matrix_first_frame": lib.tolist(),
                    "failures": [k for k, ok in b7_rows.items() if not ok], "adobe_report": adobe_cm2}
    save()

    # B8 as-shot WB
    b8_rows, main_cwb = {}, set()
    for key, f in facts.items():
        cwb = f.get("camera_whitebalance")
        tag = inventory[key]["raw_ifd"]["tag_7313_wb_rggb_levels"]
        rg, bg = cwb[0] / cwb[1], cwb[2] / cwb[1]
        trg, tbg = tag[0] / tag[1], tag[3] / tag[1]
        ok = abs(rg / trg - 1) <= 1e-6 and abs(bg / tbg - 1) <= 1e-6 and (cwb[3] == cwb[1] or cwb[3] == 0)
        b8_rows[key] = {"camera_whitebalance": cwb, "tag_0x7313": tag, "ok": ok}
        if key.startswith("2/") and key.split("/")[1] in MAIN_SEGMENT_2:
            main_cwb.add(json.dumps(cwb))
    adobe_asn = {}
    for key, v in b3.items():
        asn = v.get("dng_tags", {}).get("AsShotNeutral")
        if asn is not None:
            n = np.asarray(rational_list(asn), dtype=np.float64)
            cwb = facts[key]["camera_whitebalance"]
            adobe_asn[key] = {"AsShotNeutral": n.tolist(), "rel_diff_R_over_G": float((n[1] / n[0]) / (cwb[0] / cwb[1]) - 1),
                              "rel_diff_B_over_G": float((n[1] / n[2]) / (cwb[2] / cwb[1]) - 1)}
    record["B8"] = {"pass": all(r["ok"] for r in b8_rows.values()) and len(main_cwb) == 1,
                    "main_segment_identical": len(main_cwb) == 1, "note": "主段一致性结果事先已知",
                    "failures": {k: r for k, r in b8_rows.items() if not r["ok"]}, "adobe_report": adobe_asn}
    save()

    # B9 orientation
    b9_rows = {}
    for key, f in facts.items():
        o = f.get("orientation", {})
        exif = inventory[key]["orientation"]
        want = (0, 0) if key.startswith("1/") else (270, 5)
        exif_rot = {1: 0, 8: 270}.get(exif)
        b9_rows[key] = {"rotation": o.get("rotation"), "value": o.get("value"), "status": o.get("status"), "exif": exif,
                        "ok": (o.get("rotation"), o.get("value")) == want and o.get("status") == "known" and exif_rot == want[0]}
    record["B9"] = {"pass": all(r["ok"] for r in b9_rows.values()), "failures": {k: r for k, r in b9_rows.items() if not r["ok"]}}
    save()

    # B10 + PDAF (report)
    record["B10_report"] = {}
    record["PDAF_report"] = {}
    for key in CHECK_FRAMES:
        try:
            record["B10_report"][key] = edge_report(RAW_ROOT / key, facts[key])
        except Exception as exc:
            record["B10_report"][key] = {"error": f"{type(exc).__name__}: {exc}"}
        try:
            record["PDAF_report"][key] = pdaf_report(RAW_ROOT / key, facts[key], key.split("/")[0])
        except Exception as exc:
            record["PDAF_report"][key] = {"error": f"{type(exc).__name__}: {exc}"}
        save()

    # B11
    after = originals_snapshot(files)
    record["originals_after_sha_equal"] = after["sha256"] == record["originals_before"]["sha256"]
    record["originals_after_listing_equal"] = after["listing"] == record["originals_before"]["listing"]
    record["B11"] = {"pass": record["originals_after_sha_equal"] and record["originals_after_listing_equal"]}
    record.pop("originals_before", None)
    write_json(stage / "originals_before_after.json", {"after": after})

    gates = ["B1", "B2", "B3", "B4", "B5", "B7", "B8", "B9", "B11"]
    record["summary"] = {k: verdict(record[k]["pass"]) for k in gates}
    record["summary"]["B6"] = record["B6"]["verdict"]
    record["summary"]["B2_posthoc_excluding_flip（事后）"] = verdict(record["B2_posthoc_excluding_flip"]["pass"])
    record["stage1_pass"] = all(record[k]["pass"] is True for k in gates) and record["B6"]["pass"] in (True, None)
    record["stage1_pass_with_B2_posthoc（事后）"] = (all(record[k]["pass"] is True for k in gates if k != "B2")
                                                  and record["B2_posthoc_excluding_flip"]["pass"] is True
                                                  and record["B6"]["pass"] in (True, None))
    record["tier_after"] = "B（用户 2026-09-25 决定：保持 B）"
    record["state"] = "completed"
    save()
    write_markdown(stage, record)
    print(json.dumps(record["summary"], ensure_ascii=False), "stage1_pass=", record["stage1_pass"])
    return 0 if record["stage1_pass"] else 1


def write_markdown(stage: Path, record: dict) -> None:
    lines = [f"# 索尼 α7R V 阶段 1 单帧证据（{record['started_utc']}）", "",
             f"预登记：`{record['prereg']}`；冻结回执：`{record['freeze']['receipt']}`（一致：{record['freeze']['ok']}）。", "",
             f"**阶段 1 结论（按原判据字面）：{'通过' if record.get('stage1_pass') else '未通过'}**；"
             f"按事后解释（B2 不含 flip）：{'通过' if record.get('stage1_pass_with_B2_posthoc（事后）') else '未通过'}。分级保持 B（用户决定）。", "",
             "| 判据 | 结果 |", "| --- | --- |"]
    for k, v in record.get("summary", {}).items():
        lines.append(f"| {k} | {v} |")
    lines += ["", f"用时 {record.get('elapsed_seconds')} 秒。详细数据见 RESULT.json、per_file_facts.json、B6_plateaus.json。"]
    (stage / "RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--make-freeze")
    parser.add_argument("--freeze")
    parser.add_argument("--output", default=str(ROOT / "outputs" / "sony-a7rv-20260925"))
    args = parser.parse_args(argv)
    if args.make_freeze:
        return make_freeze(Path(args.make_freeze))
    if not args.freeze:
        parser.error("--freeze is required for a run")
    return run(Path(args.freeze), Path(args.output))


if __name__ == "__main__":
    raise SystemExit(main())
