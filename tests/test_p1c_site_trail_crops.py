"""Efficiency P1c site tests for the review crops (prereg §3.4 T-3m and T-3, §6.2 A5 (a)-(h)).

T-3m: ``transient_trails.write_crop_png`` reads only the sampled pixels; ``_write_crop_png_reference`` (the former
body) is the reference. T-3: ``transient_stage._crop_phase`` runs the per-frame part of detect_job after the scan on the
frame pool; ``transient_stage._trail_crops_reference`` (the former loop, with the former crop body) is the reference.

(a) synthetic: the 13-frame injection data set of tests/test_transient_trails.py over the prereg's matrix; the record
    (key order included) and every PNG byte; PYTHONHASHSEED reruns and the numeric environments of (g);
(b) real: the F48 reference job (read only; crops into basetemp): the crop phase plus ``_finish`` rebuild the job's own
    candidates.json and its 299 review PNGs byte for byte, at workers 1, default and 16, and the reference agrees;
(c) failures at frame k in {0, 1, middle, last} of the frames with candidates, in a crop and in a candidate's
    construction, and the documented within-frame order (prereg §5 item 4); a rerun after a failure is the clean run;
(d) cancellation within 1 s at 1, default, 8 and 16 workers (slow synthetic crops and real-size units), units that
    only the check before each candidate (both paths) or the probe within a crop's sampled reads (parallel path) can
    stop in time, and the checks each path passes on (pinned per call);
(e) the context audit is on in every pooled run (pool_conditions);
(f) the pool's memory against granted x unit_bytes, the worker decision under memory, and one crop's peak (T-3m
    against the reference);
(h) the static guards list the site (tests/test_p1c_static_guards.py).
"""
from __future__ import annotations

import contextlib
import copy
import json
import os
import shutil
import threading
import time
from pathlib import Path

import numpy as np
import pytest

import p1c_site_harness as h
import p1c_trail_fixtures as fx
from starstack import frame_pool, performance
from starstack import transient_stage as ts
from starstack import transient_trails as tt

GiB = 1 << 30


# ================================================================== T-3m: the sampled reader

def test_numpy_float32_arcsinh_rint_isfinite_have_baseline_loops_only():
    """The bit-identity argument of T-3m (prereg §3.4): the element-wise ops a crop applies are add / subtract /
    divide (correctly rounded on every SIMD path) and arcsinh, rint, isfinite, which in numpy 2.5.3 have no SIMD
    dispatch for float32. A numpy upgrade that adds one fails here and asks for the argument to be redone."""
    from numpy.lib.introspect import opt_func_info
    for name in ("arcsinh", "rint", "isfinite"):
        info = opt_func_info(func_name=f"^{name}$", signature="float32")[name]
        assert info, name
        for signature, targets in info.items():
            available = targets["available"].split()
            assert len(available) == 1 and available[0].startswith("baseline("), (name, signature, targets)


def _planes_of(path):
    return tt.Planes(path)


def _random_plane_file(path, rng, height, width, *, bitpix, roworder, zeros=True, nans=False):
    data = rng.uniform(0.0, 1.0, (3, height, width))
    if zeros:
        data[:, rng.integers(0, height, 3), :] = 0.0                 # no-data rows and columns
        data[:, :, rng.integers(0, width, 3)] = 0.0
    if nans and bitpix == -32:
        data[1, rng.integers(0, height, 4), rng.integers(0, width, 4)] = np.nan
    fx.write_fits(path, height, width, lambda c, y0, y1: data[c, y0:y1], bitpix=bitpix, roworder=roworder, block=7)
    return path


FORMATS = [(16, "BOTTOM-UP"), (-32, "BOTTOM-UP"), (16, "TOP-DOWN"), (-32, "TOP-DOWN")]


@pytest.mark.parametrize("frame_format", FORMATS, ids=lambda f: f"frame{f[0]}{f[1][0]}")
@pytest.mark.parametrize("master_format", FORMATS, ids=lambda f: f"master{f[0]}{f[1][0]}")
def test_sampled_rows_equal_the_full_rows_sliced(tmp_path, frame_format, master_format):
    """``_sampled`` = ``_rows(...)[:, x0:x1][::step, ::step]`` bit for bit, each file by its own BITPIX and row order,
    over random in-image boxes and steps, including y1 == H (the smallest BOTTOM-UP start index)."""
    rng = np.random.default_rng([frame_format[0] & 0xFF, len(frame_format[1]), master_format[0] & 0xFF])
    height, width = int(rng.integers(20, 70)), int(rng.integers(20, 70))
    frame = _planes_of(_random_plane_file(tmp_path / "f.fit", rng, height, width, bitpix=frame_format[0],
                                          roworder=frame_format[1], nans=True))
    master = _planes_of(_random_plane_file(tmp_path / "m.fit", rng, height, width, bitpix=master_format[0],
                                           roworder=master_format[1]))
    bottom_edge = 0
    for trial in range(300):
        y0 = int(rng.integers(0, height))
        y1 = height if trial % 4 == 0 else int(rng.integers(y0 + 1, height + 1))
        x0 = int(rng.integers(0, width))
        x1 = int(rng.integers(x0 + 1, width + 1))
        step = int(rng.integers(1, 9))
        bottom_edge += y1 == height and step > 1
        for planes in (frame, master):
            for plane in range(3):
                expected = tt._rows(*planes.planes[plane], y0, y1)[:, x0:x1][::step, ::step]
                actual = tt._sampled(*planes.planes[plane], x0, x1, y0, y1, step)
                h.assert_same(np.ascontiguousarray(expected), actual, f"plane {plane} box {(x0, x1, y0, y1, step)}")
    assert bottom_edge > 20


@pytest.mark.parametrize("file_format", FORMATS, ids=lambda f: f"{f[0]}{f[1][0]}")
def test_sampled_reads_probe_every_32_rows_and_stay_bit_identical(tmp_path, file_format):
    """T-3 (R18): a pooled crop's sampled read calls the unit's probe before every SAMPLED_PROBE_ROWS sampled rows
    (once for a short read); the values are those of the unprobed read, bit for bit (NaNs included)."""
    rng = np.random.default_rng([file_format[0] & 0xFF, len(file_format[1]), 11])
    planes = _planes_of(_random_plane_file(tmp_path / "f.fit", rng, 301, 37, bitpix=file_format[0],
                                           roworder=file_format[1], nans=True))
    for x0, x1, y0, y1, step in ((0, 37, 0, 301, 1), (3, 30, 5, 301, 3), (0, 37, 0, 20, 1), (1, 37, 0, 301, 9),
                                 (0, 37, 10, 74, 1), (0, 37, 10, 75, 2)):
        for plane in range(3):
            calls = []
            actual = tt._sampled(*planes.planes[plane], x0, x1, y0, y1, step, check=lambda: calls.append(1))
            h.assert_same(tt._sampled(*planes.planes[plane], x0, x1, y0, y1, step), actual, (x0, x1, y0, y1, step))
            rows = len(range(y0, y1, step))
            assert len(calls) == max(1, -(-rows // tt.SAMPLED_PROBE_ROWS)), (rows, len(calls))


def test_a_probe_that_raises_inside_a_crop_writes_nothing(tmp_path):
    rng = np.random.default_rng(12)
    frame = _random_plane_file(tmp_path / "f.fit", rng, 400, 60, bitpix=-32, roworder="BOTTOM-UP")
    master = _random_plane_file(tmp_path / "m.fit", rng, 400, 60, bitpix=16, roworder="TOP-DOWN")
    tall = _candidate(30.0, 0.0, 31.0, 399.0)
    calls = []
    assert tt.write_crop_png(frame, master, tall, tmp_path / "probed.png", check=lambda: calls.append(1))["written"]
    tt.write_crop_png(frame, master, tall, tmp_path / "plain.png")
    h.assert_same(tmp_path / "plain.png", tmp_path / "probed.png")
    assert len(calls) == 2 * -(-400 // tt.SAMPLED_PROBE_ROWS)       # frame and master: 400 sampled rows each

    def stop_at_third():
        calls.append(1)
        if len(calls) == 3:
            raise h.Cancelled("stop")
    calls.clear()
    with pytest.raises(h.Cancelled):
        tt.write_crop_png(frame, master, tall, tmp_path / "stopped.png", check=stop_at_third)
    assert not (tmp_path / "stopped.png").exists()


def _candidate(x0, y0, x1, y1, status="measured"):
    return {"id": "c", "measurement": {"status": status, "x0": x0, "y0": y0, "x1": x1, "y1": y1}}


@pytest.mark.parametrize("frame_format", FORMATS, ids=lambda f: f"frame{f[0]}{f[1][0]}")
@pytest.mark.parametrize("master_format", [(-32, "BOTTOM-UP"), (16, "TOP-DOWN")], ids=lambda f: f"master{f[0]}{f[1][0]}")
def test_crop_png_is_bit_identical_to_the_reference(tmp_path, frame_format, master_format):
    """Property test: write_crop_png against _write_crop_png_reference -- PNG bytes and the returned dict -- for frame
    and master in different formats (16 against -32 bit, BOTTOM-UP against TOP-DOWN), boxes inside, across and outside
    every edge (y1 == H among them), tiny and large boxes, sizes forcing every downsampling step."""
    rng = np.random.default_rng([frame_format[0] & 0xFF, len(frame_format[1]), master_format[0] & 0xFF, 5])
    height, width = int(rng.integers(40, 120)), int(rng.integers(40, 120))
    frame = _random_plane_file(tmp_path / "f.fit", rng, height, width, bitpix=frame_format[0],
                               roworder=frame_format[1], nans=True)
    master = _random_plane_file(tmp_path / "m.fit", rng, height, width, bitpix=master_format[0],
                                roworder=master_format[1])
    written = bottom = 0
    for trial in range(120):
        ax, bx = rng.uniform(-80, width + 80, 2)
        ay, by = rng.uniform(-80, height + 80, 2)
        if trial % 5 == 0:
            ay, by = height - rng.uniform(0, 20), height + rng.uniform(0, 60)       # reaching past the bottom
        size = int(rng.choice([3, 7, 16, 33, 512]))
        candidate = _candidate(float(ax), float(ay), float(bx), float(by), "measured" if trial % 17 else "too_short")
        for source in ("path", "planes"):
            f, m = (frame, master) if source == "path" else (tt.Planes(frame), tt.Planes(master))
            ref_path, new_path = tmp_path / f"ref-{trial}-{source}.png", tmp_path / f"new-{trial}-{source}.png"
            expected = tt._write_crop_png_reference(f, m, candidate, ref_path, size=size)
            actual = tt.write_crop_png(f, m, candidate, new_path, size=size)
            h.assert_same(expected, actual, f"trial {trial} info")
            assert ref_path.exists() == new_path.exists() == bool(expected.get("written"))
            if expected.get("written"):
                h.assert_same(ref_path, new_path, f"trial {trial} png")
                written += 1
                bottom += tt._crop_box(candidate["measurement"])[3] >= height and expected["downsample"] > 1
    assert written > 100 and bottom > 4


def test_frame_and_master_on_different_grids_take_the_reference(tmp_path, monkeypatch):
    rng = np.random.default_rng(3)
    frame = _random_plane_file(tmp_path / "f.fit", rng, 200, 300, bitpix=-32, roworder="BOTTOM-UP")
    master = _random_plane_file(tmp_path / "m.fit", rng, 200, 301, bitpix=-32, roworder="BOTTOM-UP")
    calls = []
    real = tt._write_crop_png_reference
    monkeypatch.setattr(tt, "_write_crop_png_reference", lambda *a, **k: calls.append(a) or real(*a, **k))
    candidate = _candidate(100.0, 100.0, 110.0, 102.0)      # the box stays inside both: the reference can subtract
    info = tt.write_crop_png(frame, master, candidate, tmp_path / "x.png")
    assert calls and info["written"]


def test_the_scan_refuses_frames_off_the_master_grid(tmp_path):
    """The crop phase's scope (prereg §3.4): detect_job crops only frames the scan measured, and the scan's detection
    refuses a frame whose shape differs from the master's -- so in a job write_crop_png never takes the reference."""
    rng = np.random.default_rng(4)
    frame = _random_plane_file(tmp_path / "f.fit", rng, 50, 60, bitpix=-32, roworder="BOTTOM-UP")
    master = _random_plane_file(tmp_path / "m.fit", rng, 50, 61, bitpix=-32, roworder="BOTTOM-UP")
    with pytest.raises(ValueError, match="same grid"):
        ts._scan_frame(str(frame), str(master), dict(tt.DEFAULT_DETECTION), [], 16)


# ================================================================== T-3: the synthetic data set

@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    """The 13-frame injection set (800 x 600, 11 frames with candidates), scanned once in this process."""
    from test_transient_trails import make_dataset
    folder = tmp_path_factory.mktemp("trail-crops")
    make_dataset(folder)
    stems = ts.frame_order(folder)
    marks = {"light_00005": [(120.0, 80.0, 420.0, 560.0)], "light_00006": [(480.0, 60.0, 600.0, 200.0)],
             "light_00002": [(100.0, 100.0, 400.0, 120.0)]}           # hand marks: on two trails, on empty sky
    grid_marks = {stem: [ts.mark_to_grid({"stem": stem, "x0": a, "y0": b, "x1": c, "y1": d, "half_width_px": 30},
                                         np.eye(3), None) for a, b, c, d in marks.get(stem, [])] for stem in stems}
    scanned = {stem: ts._scan_frame(str(folder / f"r_{stem}.fit"), str(folder / "sky_master.fit"),
                                    dict(tt.DEFAULT_DETECTION), grid_marks[stem], 16) for stem in stems}
    scanned = json.loads(json.dumps(scanned, allow_nan=False))        # as a scan child publishes it
    inputs = [{"path": str(folder / f"in_{i:05d}.fit"), "size": 1, "sha256": "0" * 64} for i in range(1, 14)]
    transforms = {stem: np.eye(3).tolist() for stem in stems}
    (folder / "dataset.json").write_text(json.dumps({"stems": stems, "scanned": scanned, "inputs": inputs,
                                                     "transforms": transforms}), encoding="utf-8")
    return str(folder)


def load(folder):
    folder = Path(folder)
    data = json.loads((folder / "dataset.json").read_text(encoding="utf-8"))
    return {**data, "work": folder, "master": tt.Planes(folder / "sky_master.fit")}


def skeleton(stems):
    """detect_job's record before the crop phase (the keys it fills)."""
    return {"version": tt.TRANSIENT_VERSION, "frames_scanned": 0, "frame_order": list(stems), "candidates": [],
            "per_frame": {}, "frame_stats": {}}


def run_side(folder, side, crops, *, workers=None, inflight=None, check=None, cap=ts.TRAIL_CROP_CAP, record=None):
    data = load(folder)
    record = skeleton(data["stems"]) if record is None else record
    kwargs = dict(work=data["work"], inputs=data["inputs"], transforms=data["transforms"], model=None,
                  own_registration=False)
    if side == "reference":
        ts._trail_crops_reference(record, data["scanned"], data["stems"], data["master"], crops, check, **kwargs)
    else:
        ts._crop_phase(record, data["scanned"], data["stems"], data["master"], crops, check, workers, cap=cap,
                       inflight=inflight, **kwargs)
    return record


def normalised(record, crops):
    """The record with each review_png reduced to its file name (after checking its folder)."""
    record = copy.deepcopy(record)
    for c in record["candidates"]:
        if c.get("review_png") is not None:
            path = Path(c["review_png"])
            assert path.parent == Path(crops), (path, crops)
            c["review_png"] = path.name
    return record


def site_target(folder, side, workers=None):
    """For the PYTHONHASHSEED / numeric reruns: (normalised record, PNG bytes) of one side in a fresh folder."""
    crops = Path(folder) / f"crops-{side}-{os.getpid()}-{time.monotonic_ns()}"
    record = run_side(folder, side, crops, workers=workers)
    return normalised(record, crops), h.tree_bytes(crops)


def reference_target(folder):
    return site_target(folder, "reference")


def pool_target(folder):
    return site_target(folder, "pool", 8)


@pytest.fixture(scope="module")
def expected(dataset):
    crops = Path(dataset) / "crops-expected"
    record = run_side(dataset, "reference", crops)
    return normalised(record, crops), h.tree_bytes(crops)


def frames_with_candidates(folder):
    data = load(folder)
    return [i for i, stem in enumerate(data["stems"]) if data["scanned"][stem]["result"]["segments"]]


# ================================================================== (a)

def test_the_data_set_exercises_the_site(dataset, expected):
    record, pngs = expected
    data = load(dataset)
    assert len(data["stems"]) == 13 and len(frames_with_candidates(dataset)) >= 8
    assert len(record["candidates"]) >= 12 and len(pngs) >= 10
    assert any(len(data["scanned"][s]["result"]["segments"]) >= 2 for s in data["stems"])
    assert all(list(c)[-1] == "review_png" for c in record["candidates"])          # review_png is the last key
    assert record["frames_scanned"] == 13


@pytest.mark.parametrize("case", h.params(h.synthetic_cases()))
def test_pool_matches_reference_over_the_matrix(dataset, expected, tmp_path, case):
    before = h.aester_threads()
    crops = tmp_path / "crops"
    with h.pool_conditions(case):
        record = run_side(dataset, "pool", crops, workers=case.workers, inflight=case.inflight)
    h.assert_same(expected[0], normalised(record, crops))
    h.assert_same(expected[1], h.tree_bytes(crops))
    h.assert_pool_quiet(before)


def test_pool_statistics_note_and_shadow_decision(dataset, tmp_path, monkeypatch):
    notes = []
    monkeypatch.setattr(performance, "note_pool", lambda site, stats: notes.append((site, dict(stats))))
    shadows = []
    from starstack import resources
    monkeypatch.setattr(resources, "shadow_pool", lambda stage, **kw: shadows.append((stage, kw)))
    operations = []

    class Timer:
        def operation(self, name):
            operations.append(name)
            return contextlib.nullcontext()

    data = load(dataset)
    record = skeleton(data["stems"])
    with h.pool_conditions():
        ts._crop_phase(record, data["scanned"], data["stems"], data["master"], tmp_path / "c", None, 4,
                       work=data["work"], inputs=data["inputs"], transforms=data["transforms"], model=None,
                       own_registration=False, timer=Timer())
    (site, stats), = notes
    assert site == "trail_crops" and stats["workers"] == 4 and stats["units"] == 13 and not stats["serial"]
    assert stats["unit_bytes"] == ts.TRAIL_CROP_UNIT_BYTES and stats["reserved_bytes"] == 4 * ts.TRAIL_CROP_UNIT_BYTES
    assert shadows == [("py.trail_crops", {"actual": 4, "items": 13})]
    assert operations == ["trail_crops"]


def test_reference_and_pool_are_stable_under_two_hash_seeds(dataset, expected):
    digest = h.assert_hashseed_stable("test_p1c_site_trail_crops:reference_target",
                                      "test_p1c_site_trail_crops:pool_target", dataset)
    assert digest == h.canonical_digest(expected)


@pytest.mark.parametrize("env", h.NUMERIC_ENVS, ids=lambda e: ",".join(f"{k}={v}" for k, v in e.items()))
def test_numeric_environments_give_the_same_crops(dataset, expected, env):
    """(g): the crops read and compute in the job process; equal under the numeric variants (T-3m's argument)."""
    want = h.canonical_digest(expected)
    assert h.subprocess_digest("test_p1c_site_trail_crops:reference_target", dataset, env=env) == want
    assert h.subprocess_digest("test_p1c_site_trail_crops:pool_target", dataset, env=env) == want


def test_detect_job_end_to_end_is_independent_of_scan_and_crop_workers(tmp_path, monkeypatch):
    """detect_job at scan processes {1, 3} and crop threads {1, 4, 16}: candidates.json byte for byte (timings
    removed) and the PNGs, in one work folder (paths equal)."""
    from test_transient_trails import make_dataset
    work = tmp_path / "w"
    work.mkdir()
    make_dataset(work)
    inputs = [{"path": str(work / f"in_{i:05d}.fit"), "size": 1, "sha256": "0" * 64} for i in range(1, 14)]
    transforms = {f"light_{i:05d}": np.eye(3).tolist() for i in range(1, 14)}
    outcomes = []
    for scan_workers, crop_workers in ((1, "1"), (3, "4"), (3, "16"), (1, "16")):
        shutil.rmtree(work / ts.TRANSIENTS_DIR, ignore_errors=True)
        monkeypatch.setenv("AESTER_FRAME_WORKERS", crop_workers)
        record = ts.detect_job(work, own_registration=False, transforms=transforms, model=None, inputs=inputs,
                               reference_stem="light_00002", log=lambda *_: None, workers=scan_workers)
        assert record["workers"] == scan_workers
        text = json.loads((work / ts.TRANSIENTS_DIR / ts.CANDIDATES_FILE).read_text(encoding="utf-8"))
        for key in ("seconds", "started_at", "workers"):
            text.pop(key)
        for frame in text["per_frame"].values():
            frame.pop("seconds")
        outcomes.append((text, h.tree_bytes(work / ts.TRANSIENTS_DIR / "crops")))
    for outcome in outcomes[1:]:
        h.assert_same(outcomes[0], outcome)


# ================================================================== (b) the F48 reference job

@pytest.fixture(scope="module")
def f48():
    return load_f48()


def load_f48():
    """The F48 job's scan results, record, inputs and registration (read only), for the crop phase."""
    job_dir = h.f48_job()
    work = job_dir / "work"
    job = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
    fixture = json.loads((work / ts.TRANSIENTS_DIR / ts.CANDIDATES_FILE).read_text(encoding="utf-8"))
    runs = sorted(p for p in (work / ts.TRANSIENTS_DIR / "scan-runs").iterdir() if p.is_dir())
    assert len(runs) == 1
    scanned = {}
    for path in sorted(runs[0].glob("result-*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        scanned[payload["stem"]] = payload["scanned"]
    from starstack import core
    calibration = json.loads((work / "distortion.json").read_text(encoding="utf-8"))
    transforms, model = core.registration_transforms(work, calibration, True, int(job["diagnostics"]["height"]))
    stems = fixture["frame_order"]
    assert stems == ts.frame_order(work) and sorted(scanned) == sorted(stems)
    return {"work": work, "job": job, "fixture": fixture, "scanned": scanned, "transforms": transforms,
            "model": model, "stems": stems, "master": tt.Planes(Path(fixture["master"]))}


def f48_record(f48):
    """The job record before its crop phase, as detect_job built it (its own values: K's frozen run)."""
    keys = ("version", "stage_version", "grid", "master", "frames_scanned", "frame_order", "reference_stem", "params",
            "policy", "marks", "candidates", "per_frame", "frame_stats", "workers", "started_at")
    record = {key: copy.deepcopy(f48["fixture"][key]) for key in keys}
    record.update(frames_scanned=0, candidates=[], per_frame={}, frame_stats={})
    return record


def f48_run(f48, crops, side="pool", workers=None, check=None, inflight=None, finish=True):
    record = f48_record(f48)
    kwargs = dict(work=f48["work"], inputs=f48["job"]["inputs"], transforms=f48["transforms"], model=f48["model"],
                  own_registration=True)
    started = time.perf_counter()
    if side == "reference":
        ts._trail_crops_reference(record, f48["scanned"], f48["stems"], f48["master"], crops, check, **kwargs)
    else:
        ts._crop_phase(record, f48["scanned"], f48["stems"], f48["master"], crops, check, workers,
                       inflight=inflight, **kwargs)
    seconds = time.perf_counter() - started
    record["seconds"] = f48["fixture"]["seconds"]
    if finish:
        ts._finish(record, f48["work"], persist=False, check=check)
    return record, seconds


def as_written(record):
    """The record as write_candidates stores it (numpy scalars as numbers), read back."""
    return json.loads(json.dumps(record, ensure_ascii=False, default=tt._json_default))


def assert_rebuilds_the_fixture(f48, record, crops):
    fixture_crops = Path(f48["fixture"]["master"]).parent / ts.TRANSIENTS_DIR / "crops"
    want = copy.deepcopy(f48["fixture"])
    for c in want["candidates"]:
        if c.get("review_png") is not None:
            assert Path(c["review_png"]).parent == fixture_crops
            c["review_png"] = Path(c["review_png"]).name
    h.assert_same(want, as_written(normalised(record, crops)))
    written = sorted(p.name for p in Path(crops).glob("*.png"))
    assert written == sorted(c["review_png"] for c in want["candidates"] if c.get("review_png"))
    for name in written:
        h.assert_same(fixture_crops / name, Path(crops) / name, name)
    return len(written)


@pytest.mark.parametrize("workers", [1, None, 16], ids=["w1", "default", "w16"])
def test_f48_crop_phase_rebuilds_the_jobs_record_and_crops(f48, tmp_path, workers):
    before = h.aester_threads()
    crops = tmp_path / "crops"
    with h.pool_conditions(h.Case(workers, None, "reuse", 0)):
        record, seconds = f48_run(f48, crops, workers=workers)
    assert assert_rebuilds_the_fixture(f48, record, crops) == 299
    h.assert_pool_quiet(before)
    print(f"F48 crop phase, workers={workers}: {seconds:.2f} s")


def test_f48_reference_rebuilds_the_jobs_record_and_crops(f48, tmp_path):
    crops = tmp_path / "crops"
    record, seconds = f48_run(f48, crops, side="reference")
    assert assert_rebuilds_the_fixture(f48, record, crops) == 299
    print(f"F48 crop loop before T-3m/T-3 (reference): {seconds:.2f} s")


# ================================================================== (c) failures

@pytest.fixture
def faults(monkeypatch):
    """Fail the review crop of the candidates whose ids are in ``faults["crops"]`` and the construction of those in
    ``faults["builds"]`` (both sets may change between runs)."""
    faults = {"crops": set(), "builds": set()}
    real_build = ts._candidate

    def failing(real):
        def crop(frame, master, candidate, path, **kw):
            if candidate["id"] in faults["crops"]:
                raise RuntimeError(f"injected crop failure {candidate['id']}")
            return real(frame, master, candidate, path, **kw)
        return crop

    def build(stem, k, seg, **kw):
        if f"{stem}-{k}" in faults["builds"]:
            raise ValueError(f"injected build failure {stem}-{k}")
        return real_build(stem, k, seg, **kw)

    monkeypatch.setattr(tt, "write_crop_png", failing(tt.write_crop_png))
    monkeypatch.setattr(tt, "_write_crop_png_reference", failing(tt._write_crop_png_reference))
    monkeypatch.setattr(ts, "_candidate", build)
    return faults


def _attempt(folder, crops, workers):
    record = skeleton(load(folder)["stems"])
    outcome = h.outcome(lambda delivered: run_side(folder, "pool", crops, workers=workers, record=record))
    outcome.delivered = [c["id"] for c in record["candidates"] if "review_png" in c]
    return outcome


@pytest.mark.parametrize("what", ["crop", "build"])
@pytest.mark.parametrize("position", [0, 1, "middle", "last"])
def test_failures_end_like_the_serial_run(dataset, expected, tmp_path, faults, what, position):
    """Failure at the k-th frame with candidates: the serial run's exception and delivered prefix at every worker
    count; the crops written are the serial run's plus, at most, crops of later frames (prereg §5 item 2), each the
    clean run's bytes; a rerun into the same folder is the clean run."""
    frames = frames_with_candidates(dataset)
    k = h.failure_positions(len(frames))[[0, 1, "middle", "last"].index(position)]
    data = load(dataset)
    target = f"{data['stems'][frames[k]]}-1"
    faults["crops" if what == "crop" else "builds"].add(target)
    serial_crops = tmp_path / "serial"
    serial = _attempt(dataset, serial_crops, 1)
    assert serial.error_type is (RuntimeError if what == "crop" else ValueError) and target in serial.message
    assert len(serial.delivered) == sum(len(data["scanned"][s]["result"]["segments"]) for s in data["stems"][:frames[k]])
    serial_files = h.tree_state(serial_crops) if serial_crops.exists() else {}
    later = {f"{s}-{n}.png" for s in data["stems"][frames[k] + 1:] for n in range(1, 30)}
    clean = expected[1]
    for workers in (2, 3, 4, 8, 16):
        before = h.aester_threads()
        crops = tmp_path / f"w{workers}"
        faults["crops" if what == "crop" else "builds"].add(target)
        with h.pool_conditions(h.Case(workers, None, "reuse", workers)):
            parallel = _attempt(dataset, crops, workers)
        h.assert_like_serial(serial, parallel)
        assert parallel.delivered == serial.delivered
        files = h.tree_state(crops) if crops.exists() else {}
        assert set(serial_files) <= set(files), (workers, set(serial_files) - set(files))
        assert set(files) - set(serial_files) <= later, (workers, sorted(set(files) - set(serial_files)))
        for name in files:                       # every crop written is the clean run's crop
            assert (crops / name).read_bytes() == clean[name], name
        h.assert_pool_quiet(before)
        faults["crops"].clear()
        faults["builds"].clear()
        rerun = run_side(dataset, "pool", crops, workers=workers)
        h.assert_same(expected[0], normalised(rerun, crops))
        h.assert_same(expected[1], h.tree_bytes(crops))


def test_within_a_frame_a_later_build_failure_is_reported_before_an_earlier_crop_failure(dataset, tmp_path, faults):
    """Prereg §5 item 4: a frame's candidates are all built before its first crop. With crop 1 and build 2 of one
    frame failing, the pool (any worker count) reports build 2; the former loop reported crop 1."""
    data = load(dataset)
    stem = next(s for s in data["stems"] if len(data["scanned"][s]["result"]["segments"]) >= 2)
    faults["crops"].add(f"{stem}-1")
    faults["builds"].add(f"{stem}-2")
    for workers in (1, 2, 4, 16):
        with h.pool_conditions(h.Case(workers, None, "reuse", 1)):
            with pytest.raises(ValueError, match=f"injected build failure {stem}-2"):
                run_side(dataset, "pool", tmp_path / f"w{workers}", workers=workers)
    with pytest.raises(RuntimeError, match=f"injected crop failure {stem}-1"):
        ts._trail_crops_reference(skeleton(data["stems"]), data["scanned"], data["stems"], data["master"],
                                  tmp_path / "ref", None, work=data["work"], inputs=data["inputs"],
                                  transforms=data["transforms"], model=None, own_registration=False)


# ================================================================== (d) cancellation

@pytest.fixture
def slow_crops(monkeypatch):
    """Each review crop takes 0.25 s longer (sleeping, GIL released): a run lasts at least 0.5 s at any worker count
    (two candidates in some frames), so a cancel at 0.1 s lands while units run."""
    real = tt.write_crop_png

    def slow(*args, **kwargs):
        time.sleep(0.25)
        return real(*args, **kwargs)
    monkeypatch.setattr(tt, "write_crop_png", slow)


@pytest.mark.parametrize("workers", [1, None, 8, 16], ids=["w1", "default", "w8", "w16"])
@pytest.mark.parametrize("when", ["waiting", "submission"])
def test_cancel_returns_within_one_second(dataset, tmp_path, slow_crops, workers, when):
    switch = h.CancelSwitch()
    if when == "waiting":
        switch.trip_after(0.1)
    else:
        switch.trip_at_call(3)
    with h.pool_conditions(h.Case(workers, None, "reuse", 2)):
        latency = h.assert_cancels_in_time(lambda check: run_side(dataset, "pool", tmp_path / "c", workers=workers,
                                                                  check=check), switch)
    print(f"cancel latency {when}, workers={workers}: {latency:.3f} s")


@pytest.fixture
def crop_checks(monkeypatch):
    """The ``check`` each review crop receives, in call order (the real crop runs)."""
    seen = []
    real = tt.write_crop_png

    def recording(frame, master, candidate, path, **kwargs):
        seen.append(kwargs.get("check", "absent"))
        return real(frame, master, candidate, path, **kwargs)
    monkeypatch.setattr(tt, "write_crop_png", recording)
    return seen


def test_each_path_checks_as_the_prereg_says(dataset, expected, tmp_path, crop_checks, monkeypatch):
    """Prereg §3.4 and R18 at the pool level: ``unit.check`` before each candidate on both paths; the parallel path
    hands each crop the unit's abort probe (for its sampled reads), the serial path -- 1 worker, and a pool nested in a
    pool unit, which runs serially on a worker thread -- hands it none. So a serial run calls the job's check once per
    frame (the pool) and once per candidate (the unit), nothing more; a parallel run calls it on the calling thread
    only (the probes never call it)."""
    notes = []
    monkeypatch.setattr(performance, "note_pool", lambda site, stats: notes.append(dict(stats)))
    data = load(dataset)
    candidates = sum(len(data["scanned"][s]["result"]["segments"]) for s in data["stems"])
    callers = []

    def job_check():
        callers.append(threading.get_ident())

    with h.pool_conditions(h.Case(4, None, "reuse", 0)):
        record = run_side(dataset, "pool", tmp_path / "parallel", workers=4, check=job_check)
    assert not notes[-1]["serial"] and notes[-1]["workers"] == 4
    assert len(crop_checks) == candidates
    assert all(callable(check) and check is not job_check for check in crop_checks)
    assert set(callers) == {threading.get_ident()}
    h.assert_same(expected[0], normalised(record, tmp_path / "parallel"))

    crop_checks.clear()
    callers.clear()
    with h.pool_conditions(h.Case(1, None, "reuse", 0)):
        record = run_side(dataset, "pool", tmp_path / "serial", workers=1, check=job_check)
    assert notes[-1]["serial"] and notes[-1]["reason"] == "workers"
    assert crop_checks == [None] * candidates
    assert len(callers) == len(data["stems"]) + candidates
    h.assert_same(expected[0], normalised(record, tmp_path / "serial"))

    crop_checks.clear()
    callers.clear()
    nested = {}

    def outer(item, unit):
        if item == 0:
            nested["record"] = run_side(dataset, "pool", tmp_path / "nested", workers=4, check=job_check)
            nested["thread"] = threading.get_ident()
        return item

    with h.pool_conditions(h.Case(2, None, "reuse", 0)):
        assert frame_pool.run_ordered(outer, [0, 1], workers=2, name="outer", blas=False) == [0, 1]
    assert notes[-1]["serial"] and notes[-1]["reason"] == "nested"
    assert crop_checks == [None] * candidates
    assert len(callers) == len(data["stems"]) + candidates and set(callers) == {nested["thread"]}
    h.assert_same(expected[0], normalised(nested["record"], tmp_path / "nested"))


def run_inflated(folder, crops, *, workers, check=None, copies=6):
    """The crop phase on the data set with each frame's segments repeated ``copies`` times: 6-12 candidates in each
    frame with candidates, so a unit lasts many crops."""
    data = load(folder)
    scanned = copy.deepcopy(data["scanned"])
    for stem in data["stems"]:
        result = scanned[stem]["result"]
        result["segments"] = [copy.deepcopy(seg) for seg in result["segments"] for _ in range(copies)]
    record = skeleton(data["stems"])
    ts._crop_phase(record, scanned, data["stems"], data["master"], crops, check, workers, work=data["work"],
                   inputs=data["inputs"], transforms=data["transforms"], model=None, own_registration=False)
    return record


@pytest.fixture
def blind_crops(monkeypatch):
    """Each review crop takes 0.25 s and never calls the check it is given: only the check before each candidate can
    stop a unit between its crops."""
    real = tt.write_crop_png

    def blind(frame, master, candidate, path, **kwargs):
        time.sleep(0.25)
        return real(frame, master, candidate, path)
    monkeypatch.setattr(tt, "write_crop_png", blind)


@pytest.mark.parametrize("workers", [1, None, 8, 16], ids=["w1", "default", "w8", "w16"])
def test_cancel_needs_the_check_before_each_candidate(dataset, tmp_path, blind_crops, workers):
    """Units of 6-12 candidates of 0.25 s (1.5-3 s): a cancel at 0.1 s returns within 1 s only through the check
    before each candidate (prereg §3.4, both paths); without it the running units end first (over 1.4 s)."""
    switch = h.CancelSwitch().trip_after(0.1)
    with h.pool_conditions(h.Case(workers, None, "reuse", 2)):
        latency = h.assert_cancels_in_time(lambda check: run_inflated(dataset, tmp_path / "c", workers=workers,
                                                                      check=check), switch)
    print(f"cancel latency, 6-12 candidates a unit, workers={workers}: {latency:.3f} s")


@pytest.fixture
def slow_reads(monkeypatch):
    """Each sampled read takes 0.15 s per SAMPLED_PROBE_ROWS sampled rows (a cold read), whether or not it probes; one
    given a probe calls it before each chunk, as tt._sampled does (pinned by
    test_sampled_reads_probe_every_32_rows_and_stay_bit_identical). Returns the chunk count of each read."""
    real = tt._sampled
    chunks = []

    def slow(view, header, roworder, x0, x1, y0, y1, step, check=None):
        count = (y1 - y0 + step - 1) // step
        chunks.append(-(-count // tt.SAMPLED_PROBE_ROWS))
        for _ in range(0, count, tt.SAMPLED_PROBE_ROWS):
            if check is not None:
                check()
            time.sleep(0.15)
        return real(view, header, roworder, x0, x1, y0, y1, step)
    monkeypatch.setattr(tt, "_sampled", slow)
    return chunks


@pytest.mark.parametrize("workers", [None, 8, 16], ids=["default", "w8", "w16"])
def test_cancel_needs_the_probe_within_a_crop(dataset, tmp_path, slow_reads, monkeypatch, workers):
    """A crop whose two sampled reads take over 1 s (4 or more chunks each): a cancel at 0.1 s returns within 1 s only
    through the unit's probe inside the reads (R18; the parallel path's, so ``default`` is the automatic decision,
    whatever AESTER_FRAME_WORKERS says)."""
    monkeypatch.delenv("AESTER_FRAME_WORKERS", raising=False)
    if workers is None:
        decision = {}
        if performance.frame_workers(len(load(dataset)["stems"]), ts.TRAIL_CROP_UNIT_BYTES, ts.TRAIL_CROP_CAP,
                                     blas=False, decision=decision) < 2:
            pytest.skip(f"the automatic crop pool is serial here ({decision}): no probe within a crop (R18)")
    notes = []
    monkeypatch.setattr(performance, "note_pool", lambda site, stats: notes.append(dict(stats)))
    switch = h.CancelSwitch().trip_after(0.1)
    with h.pool_conditions(h.Case(workers, None, "reuse", 2)):
        latency = h.assert_cancels_in_time(lambda check: run_side(dataset, "pool", tmp_path / "c", workers=workers,
                                                                  check=check), switch)
    (stats,) = notes
    assert not stats["serial"]
    assert slow_reads and min(slow_reads) >= 4                    # every crop started lasts over 1.2 s unprobed
    print(f"cancel latency, slow sampled reads, workers={workers}: {latency:.3f} s")


def test_finish_probes_the_pairwise_loops(dataset, expected):
    """_finish's add_evidence / related_candidates loops call check every EVIDENCE_CHECK_EVERY outer iterations."""
    record = copy.deepcopy(expected[0])
    record.update(params={"evidence": dict(tt.DEFAULT_EVIDENCE), "detection": dict(tt.DEFAULT_DETECTION),
                          "layer": dict(tt.DEFAULT_LAYER)}, policy=dict(tt.DEFAULT_POLICY), marks=[])
    unprobed = copy.deepcopy(record)
    ts._finish(unprobed, Path(dataset), persist=False)
    probed = copy.deepcopy(record)
    calls = []
    ts._finish(probed, Path(dataset), persist=False, check=lambda: calls.append(1))
    h.assert_same(unprobed, probed)                          # the probes change nothing
    measured = sum(1 for c in record["candidates"] if c["measurement"].get("status") == "measured")
    every = ts.EVIDENCE_CHECK_EVERY
    assert len(calls) == -(-measured // every) + -(-len(record["candidates"]) // every)
    switch = h.CancelSwitch().trip_at_call(0)
    with pytest.raises(h.Cancelled):
        ts._finish(copy.deepcopy(record), Path(dataset), persist=False, check=switch)


@pytest.mark.parametrize("workers", [1, None, 8, 16], ids=["w1", "default", "w8", "w16"])
def test_f48_cancel_with_real_size_units(f48, tmp_path, workers):
    switch = h.CancelSwitch().trip_after(0.5)
    with h.pool_conditions(h.Case(workers, None, "reuse", None)):
        latency = h.assert_cancels_in_time(lambda check: f48_run(f48, tmp_path / "c", workers=workers, check=check,
                                                                 finish=False), switch)
    print(f"F48 cancel latency, workers={workers}: {latency:.3f} s")


# ================================================================== (f) memory

def test_worker_decision_under_memory(monkeypatch):
    """performance.frame_workers for the crop pool: 32 MiB a unit, no BLAS slot, cap 8, half the CPUs."""
    monkeypatch.delenv("AESTER_FRAME_WORKERS", raising=False)
    monkeypatch.setattr(frame_pool, "usable_cpus", lambda: 32)
    # budget = min(available / 3, available - 1 GiB): 1.1 GiB leaves 102 MiB (3 units), 1.05 GiB 51 MiB (1 unit)
    for physical, commit, want in ((None, None, 1), (64 * GiB, 64 * GiB, 8), (2 * GiB, 64 * GiB, 8),
                                   (int(1.1 * GiB), 64 * GiB, 3), (64 * GiB, int(1.1 * GiB), 3),
                                   (int(1.05 * GiB), None, 1)):
        monkeypatch.setattr(performance, "_memory_status", lambda p=physical, c=commit: (p, c))
        decision = {}
        assert performance.frame_workers(48, ts.TRAIL_CROP_UNIT_BYTES, ts.TRAIL_CROP_CAP, blas=False,
                                         decision=decision) == want, (physical, commit, decision)
    monkeypatch.setattr(frame_pool, "usable_cpus", lambda: 6)
    monkeypatch.setattr(performance, "_memory_status", lambda: (64 * GiB, 64 * GiB))
    assert performance.frame_workers(48, ts.TRAIL_CROP_UNIT_BYTES, ts.TRAIL_CROP_CAP, blas=False) == 3


def test_unit_bytes_outside_the_scope_follow_the_reference_box():
    """In a job every frame with candidates is on the master's grid (32 MiB a unit); a frame of another shape (only a
    tool could pass one) is charged what the reference body holds for its tallest box."""
    seg = {"measurement": {"status": "measured", "x0": 10.0, "y0": 100.0, "x1": 20.0, "y1": 1100.0}}
    scanned = {"a": {"result": {"segments": [seg], "shape": [5520, 8280]}},
               "b": {"result": {"segments": [], "shape": [10, 10]}}}
    assert ts._crop_unit_bytes(scanned, ["a", "b"], (5520, 8280)) == ts.TRAIL_CROP_UNIT_BYTES
    scanned["a"]["result"]["shape"] = [5520, 8000]
    box = tt._crop_box(seg["measurement"])
    assert box[1] < 0                                        # the reference reads the box's rows inside the image
    rows = min(5520, box[3]) - max(0, box[1])
    assert ts._crop_unit_bytes(scanned, ["a", "b"], (5520, 8280)) == 12 * rows * 8280 + ts.TRAIL_CROP_UNIT_BYTES


def test_f48_pool_memory_stays_within_its_declared_limit(f48, tmp_path, monkeypatch):
    """(f): 16 workers, 32 in flight, the head unit finishing last; the sampled rise of private bytes stays within
    granted x unit_bytes (no BLAS slot: the crops call no BLAS)."""
    if h.private_bytes() is None:
        pytest.skip("private bytes are read on Windows only")
    notes = []
    monkeypatch.setattr(performance, "note_pool", lambda site, stats: notes.append(dict(stats)))
    with h.pool_conditions(h.Case(16, 32, "reuse", 0), head_last=True, head_seconds=1.0):
        with h.PeakSampler(interval=0.002) as peak:
            f48_run(f48, tmp_path / "c", workers=16, inflight=32, finish=False)
    stats, = notes
    limit = h.pool_memory_limit(stats, blas=False)
    print(f"F48 crop pool: rise {peak.rise / 2**20:.1f} MiB (traced {peak.traced / 2**20:.1f} MiB), limit "
          f"{limit / 2**20:.0f} MiB, {peak.samples} samples, workers {stats['workers']}")
    assert stats["workers"] == 16 and stats["inflight"] == 32
    assert peak.rise <= limit


def test_one_crop_peak_t3m_against_the_reference(f48, tmp_path):
    """(f) single unit: a full-height review crop on the real 45.7 MP frame. T-3m reads the sampled pixels only (a few
    MB); the reference holds the box's full-width rows of frame and master (about 1 GB). Reported; T-3m <= 32 MiB."""
    if h.private_bytes() is None:
        pytest.skip("private bytes are read on Windows only")
    work = f48["work"]
    frame = tt.Planes(work / f"r_{f48['stems'][0]}.fit")
    height, width = frame.shape
    candidate = {"id": "tall", "measurement": {"status": "measured", "x0": width / 2 - 100, "y0": 0.0,
                                               "x1": width / 2 + 100, "y1": height - 1.0}}
    peaks = {}
    for name, function in (("t3m", tt.write_crop_png), ("reference", tt._write_crop_png_reference)):
        with h.PeakSampler(interval=0.001) as peak:
            function(frame, f48["master"], candidate, tmp_path / f"{name}.png")
        peaks[name] = peak.rise
    h.assert_same(tmp_path / "reference.png", tmp_path / "t3m.png")
    print("one full-height crop, private-bytes rise MiB:", {k: round(v / 2**20, 1) for k, v in peaks.items()})
    assert peaks["t3m"] <= ts.TRAIL_CROP_UNIT_BYTES
    assert peaks["reference"] > 4 * peaks["t3m"]
