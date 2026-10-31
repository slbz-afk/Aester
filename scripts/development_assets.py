"""Resolve the curated 48-frame development inputs after the 144-frame retirement.

This registry is for development scripts, not product input discovery. Historical
job records and their hashes remain unchanged. It never recreates missing assets.
"""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "docs/testdata/DEVELOPMENT_BASELINE_48_20260925.json"
BASELINE = json.loads(REGISTRY.read_text(encoding="utf-8"))
INPUT_DIR = ROOT / BASELINE["input_dir"]
JOB_DIR = ROOT / BASELINE["job_path"]


def dng_for_original(name: str) -> Path:
    """Return a retained input by original NEF/DNG name; missing is an error."""
    original = Path(name).with_suffix(".NEF").name
    records = [x for x in BASELINE["frames"] if x["original_name"] == original]
    if len(records) != 1:
        raise ValueError(f"{name} is not in the current 48-frame development set")
    record = records[0]
    path = ROOT / record["dng_path"]
    if not path.is_file() or path.stat().st_size != record["bytes"]:
        raise FileNotFoundError(f"Retained development input missing or changed: {path}")
    return path
