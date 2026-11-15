# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Wang Jian (王俭). See LICENSE.txt in this folder.
"""Cache space a job will need, worked out before it starts, the same way the engine admits it.

The engine checks free space once, when the job is submitted (``core.submit_job``): it inspects the selected
files (``core.inspect_sources``) and estimates with ``disk_budget.estimate`` in the job's stack mode, passing the
highlight-range contract's attempt count. This module repeats exactly those calls, so the figure shown beside
「开始合成」 equals the ``disk_estimate.required_bytes`` later written into ``job.json`` for the same files and
options. Nothing is written; only file sizes and the first frame's metadata are read.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from starstack.core import inspect_sources
from starstack.disk_budget import estimate
from starstack.stack_range import contract as range_contract

LEGACY = "legacy"
RANGE = "positive_power2_v1"
FINITE = "finite_validity_v1"
MAX_FRAMES = 300            # the engine's per-job limit (core.inspect_sources)
LONG_SEQUENCE = 100         # from here on, the extra cache of F/R mode is worth a warning


def stack_mode(stack_engine_path: str | None, protect_highlights: bool) -> str:
    """The stack mode the workflow will submit: an explicit engine means F mode, else the highlight checkbox."""
    if stack_engine_path:
        return FINITE
    return RANGE if protect_highlights else LEGACY


def free_bytes(cache_root: str | Path) -> int:
    """Free space on the drive of the cache folder (its nearest existing parent when it does not exist yet)."""
    path = Path(cache_root).expanduser().absolute()
    for candidate in (path, *path.parents):
        if candidate.exists():
            return int(shutil.disk_usage(candidate).free)
    raise FileNotFoundError(str(path))


def required_for(record: dict, frames: int) -> int:
    """``disk_budget.estimate``'s figure for ``frames`` frames like the record's (same geometry, mode and
    average file size); equal to ``record['required_bytes']`` at the record's own frame count."""
    count = max(1, int(record["frames"]))
    source = record["source_bytes"] if frames == count else record["source_bytes"] * frames // count
    if record.get("basis") == "frames_x_frame_volume":
        raw_total = (frames * record["per_frame_bytes"] + record["fixed_bytes"] + source
                     + frames * record["registered_frame_bytes"] * int(record.get("range_attempts") or 0))
        return int(raw_total * record["safety_factor"] + record["margin_bytes"])
    return int(source * record["legacy_factor"] + record["margin_bytes"])


def fitting_frames(record: dict, free: int) -> int:
    """The largest frame count (0–300) whose estimate fits in ``free`` bytes, from the record's own terms."""
    low, high = 0, MAX_FRAMES
    while low < high:                               # required_for grows with the frame count
        middle = (low + high + 1) // 2
        if required_for(record, middle) <= free:
            low = middle
        else:
            high = middle - 1
    return low


def job_estimate(input_files: list[str], mode: str, cache_root: str | Path | None) -> dict:
    """{'record', 'legacy_required', 'free_bytes', 'fits'} for these files in this stack mode.

    ``record`` is ``disk_budget.estimate``'s record exactly as admission computes it. ``legacy_required`` is the
    same files' figure in the default L mode (equal to ``record['required_bytes']`` in L mode). ``free_bytes`` is
    None when the cache folder cannot be read.
    """
    sources = inspect_sources(None, input_files=list(input_files))
    contract = sources.get("fits_input")
    attempts = range_contract()["max_attempts"]
    record = estimate(sources, contract, stack_range_mode=mode, range_attempts=attempts)
    legacy = record if mode == LEGACY else estimate(sources, contract, stack_range_mode=LEGACY, range_attempts=attempts)
    try:
        free = free_bytes(cache_root) if cache_root else None
    except OSError:
        free = None
    return {"record": record, "legacy_required": int(legacy["required_bytes"]), "free_bytes": free,
            "fits": None if free is None else free >= record["required_bytes"]}
