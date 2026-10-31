"""Efficiency P1c C-3 (docs/EFFICIENCY_P1C_PREREG_20261005.md §3.1, §6.2 A5): sky zeroing of the converted frames
on frame_pool (core._sky_zero) against the kept serial loop (core._sky_zero_reference, which the re-zeroing route of
pre-sky-validity jobs still runs).

(a) the synthetic matrix (16-bit and float32 frames, static and per-frame masks, ledger on / shadow / off / none,
    threaded digests): files, statistics (with key order) and the ledger's report equal; two PYTHONHASHSEED values;
(b) the F48 DNGs re-converted: 16 frames at workers {1, default, 16} and 48 frames through _ensure_sky_validity
    (median, write_linear, zeroing) give the F48 job's foreground master, every frame's convert-checkpoint digest and
    its sky_validity record; (c) failures at frame {0, 1, middle, last}: the serial error, the pre-registered extra
    files (frames after the stop zeroed, fully or by blocks), and a rerun ends where an uninterrupted run does;
(d) cancellation within 1 s; (e) the R6 audit; (f) memory within unit_bytes and the worker decision.
"""
import contextlib
import gc
import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

import p1c_convert_data as data
import p1c_site_harness as h
from starstack import core, fits_crop, frame_pool, hashing, imaging, performance, profiling
from starstack import sky_validity as sv
from test_foreground_median import banded_frames, float_frame, siril_frame

FRAMES, HEIGHT, WIDTH, DILATION = 6, 600, 17, 2      # 3 row blocks of 256 per plane


@pytest.fixture(autouse=True)
def clean_pool(monkeypatch):
    monkeypatch.delenv(frame_pool.ENV_WORKERS, raising=False)
    monkeypatch.delenv(hashing.ENV_LEDGER, raising=False)
    before = h.aester_threads()
    yield
    h.assert_pool_quiet(before)


def make_set(folder: Path, seed: int, depth: int) -> SimpleNamespace:
    """FRAMES converted-like frames (16-bit Siril or float32), a static mask with partial weights and per-frame
    masks for every other frame."""
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    if depth == 16:
        values = banded_frames(rng, FRAMES, HEIGHT, WIDTH)
        frames = [siril_frame(folder / f"light_{i + 1:05d}.fit", values[i], "TOP-DOWN" if i == 2 else "BOTTOM-UP")
                  for i in range(FRAMES)]
    else:
        frames = [float_frame(folder / f"light_{i + 1:05d}.fit", rng.random((HEIGHT, WIDTH, 3)).astype(np.float32),
                              "TOP-DOWN" if i == 2 else "BOTTOM-UP") for i in range(FRAMES)]
    mask = np.ones((HEIGHT, WIDTH), dtype=np.float32)
    horizon = (HEIGHT * 0.6 + 40 * np.sin(np.arange(WIDTH) / 3.0 + seed)).astype(int)
    for x in range(WIDTH):
        mask[horizon[x]:, x] = 0.0
    mask[100:104, 3:9] = 0.5
    files = {}
    for i in range(1, FRAMES, 2):
        own = np.full((HEIGHT, WIDTH), 255, dtype=np.uint8)
        own[rng.integers(0, HEIGHT - 50):][:30, rng.integers(0, WIDTH - 5):][:, :5] = 0
        path = folder / "masks" / f"light_{i + 1:05d}.png"
        path.parent.mkdir(exist_ok=True)
        Image.fromarray(own).save(path)
        files[f"light_{i + 1:05d}"] = path
    return SimpleNamespace(folder=folder, frames=frames, mask=mask, files=files, depth=depth)


def source_of(fixture, kind):
    static = sv.StaticMaskSource(fixture.mask, dilation_px=DILATION)
    return static if kind == "static" else sv.PerFrameMaskSource(static, fixture.files, WIDTH, HEIGHT)


def copies(fixture, folder: Path) -> list:
    return data.copy_frames(fixture.frames, folder)


def ledger(mode):
    return contextlib.nullcontext() if mode is None else hashing.run_scope("c3-test", mode=mode)


def zero(paths, source, side, *, case=None, mode="on", check=None):
    """(stats, the ledger's report) of one side on ``paths``."""
    check = check or (lambda: None)
    with ledger(mode):
        if side == "reference":
            stats = core._sky_zero_reference(paths, source, check)
        else:
            stats = core._sky_zero(paths, source, check, WIDTH, workers=case.workers, inflight=case.inflight)
        report = hashing.scope_report()
    if report is not None:
        report = {k: v for k, v in report.items() if k not in ("label", "ledger", "largest_misses")}
    return stats, report


def frame_bytes(paths):
    return [Path(p).read_bytes() for p in paths]


def assert_zeroed_by_blocks(path, original, validity_top_down, rows=core.SKY_ZERO_ROWS) -> int:
    """A frame after a stopped zeroing (prereg §5 item 2, C-3; §6.2 (c)/(d)): header and padding as before, and
    every ``rows``-row block of every plane either untouched or zeroed exactly where the validity is False (the
    zeroing writes whole blocks: one write per block in the stream, one masked assignment per block in the
    memory-mapped pass). Returns the number of zeroed blocks."""
    header, offset = fits_crop.read_header(original)
    width, height, planes, bitpix, roworder = fits_crop._geometry(header)
    dtype, zero = (">i2", -int(float(header.get("BZERO", 0)))) if bitpix == 16 else (">f4", 0.0)
    valid = validity_top_down[::-1, :] if roworder == "BOTTOM-UP" else validity_top_down
    now, before = Path(path).read_bytes(), Path(original).read_bytes()
    end = offset + planes * height * width * np.dtype(dtype).itemsize
    assert len(now) == len(before) and now[:offset] == before[:offset] and now[end:] == before[end:], path
    if now == before:
        return 0
    got, was = (np.frombuffer(b, dtype=dtype, count=planes * height * width, offset=offset)
                .reshape(planes, height, width) for b in (now, before))
    zeroed = 0
    for plane in range(planes):
        for y in range(0, height, rows):
            block, earlier = got[plane, y:y + rows], was[plane, y:y + rows]
            if block.tobytes() == earlier.tobytes():
                continue
            expected = earlier.copy()
            expected[~valid[y:y + rows]] = zero
            assert block.tobytes() == expected.tobytes(), (Path(path).name, plane, y)
            zeroed += 1
    return zeroed


def assert_stopped_zeroing_left(folder, paths, originals, source) -> list:
    """(c)/(d): a stopped zeroing created and deleted nothing in ``folder`` and changed only frames, each untouched or
    zeroed by whole blocks (assert_zeroed_by_blocks). Returns the zeroed block count per frame."""
    assert sorted(p.name for p in Path(folder).iterdir()) == sorted(Path(p).name for p in paths)
    return [assert_zeroed_by_blocks(path, original, source.for_frame(Path(path).stem))
            for path, original in zip(paths, originals)]


# ---- targets for the PYTHONHASHSEED reruns ----------------------------------------------------------------------------

def hashseed_target(side, seed, workers):
    with tempfile.TemporaryDirectory() as folder:
        fixture = make_set(Path(folder) / "src", seed, 16 if seed % 2 == 0 else -32)
        paths = copies(fixture, Path(folder) / side)
        stats, report = zero(paths, source_of(fixture, "per_frame"), side, case=h.Case(workers, None))
        return stats, frame_bytes(paths), report


def reference_target(seed):
    return hashseed_target("reference", seed, 1)


def pooled_target(seed):
    return hashseed_target("pool", seed, 4)


# ================================================================== (a)

@pytest.fixture(scope="module")
def sets(tmp_path_factory):
    root = tmp_path_factory.mktemp("zero-sets")
    out = {}
    for depth, kind in ((16, "static"), (-32, "per_frame"), (16, "per_frame"), (-32, "static")):
        fixture = make_set(root / f"{depth}-{kind}", 7, depth)
        expected_paths = copies(fixture, root / f"{depth}-{kind}-reference")
        stats, report = zero(expected_paths, source_of(fixture, kind), "reference")
        out[(depth, kind)] = SimpleNamespace(fixture=fixture, kind=kind, stats=stats, report=report,
                                             files=frame_bytes(expected_paths), originals=frame_bytes(fixture.frames))
    return out


@pytest.mark.parametrize("case", h.params(h.synthetic_cases()))
def test_pool_matches_reference_over_the_matrix(sets, tmp_path, case):
    entry = sets[(16, "static") if case.seed % 2 == 0 else (-32, "per_frame")]
    paths = copies(entry.fixture, tmp_path / "pool")
    with h.pool_conditions(case):
        stats, report = zero(paths, source_of(entry.fixture, entry.kind), "pool", case=case)
    h.assert_same(entry.stats, stats)
    h.assert_same(entry.files, frame_bytes(paths))
    assert report == entry.report
    assert entry.files != entry.originals                       # the masks zero something in every frame


@pytest.mark.parametrize("mode", [None, "off", "on", "shadow"])
@pytest.mark.parametrize("key", [(16, "static"), (-32, "per_frame"), (16, "per_frame"), (-32, "static")],
                         ids=lambda k: f"{k[0]}-{k[1]}")
@pytest.mark.parametrize("threaded", [False, True])
def test_every_frame_kind_mask_kind_and_ledger_mode(sets, tmp_path, monkeypatch, mode, key, threaded):
    if threaded:                                                # the digest's helper thread in every unit (R13)
        monkeypatch.setattr(hashing, "PARALLEL_MIN_BYTES", 0)
    entry = sets[key]
    reference_paths = copies(entry.fixture, tmp_path / "reference")
    expected, expected_report = zero(reference_paths, source_of(entry.fixture, entry.kind), "reference", mode=mode)
    for workers in (1, 4, 16):
        paths = copies(entry.fixture, tmp_path / f"w{workers}")
        with h.pool_conditions(h.Case(workers, None, "reuse", workers)):
            stats, report = zero(paths, source_of(entry.fixture, entry.kind), "pool", case=h.Case(workers, None),
                                 mode=mode)
        h.assert_same(expected, stats)
        h.assert_same(frame_bytes(reference_paths), frame_bytes(paths))
        assert report == expected_report
        if mode in ("on", "shadow"):                            # every zeroed frame is a write hit afterwards
            with hashing.run_scope("c3-after", mode=mode):
                for path in paths:
                    assert hashing.sha256_file(path, source="test") == hashlib.sha256(path.read_bytes()).hexdigest()


def test_the_serial_path_keeps_todays_call_pattern(sets, tmp_path):
    """workers=1: the same sequence of check / for_frame / apply calls as the loop, interleaved the same way."""
    entry = sets[(16, "per_frame")]
    logs = []
    for side in ("reference", "pool"):
        log = []
        source = source_of(entry.fixture, entry.kind)
        real_for_frame = source.for_frame
        source.for_frame = lambda stem, _log=log: (_log.append(("for_frame", stem)), real_for_frame(stem))[1]
        real_apply = sv.apply_validity_to_fits

        def apply(path, validity, *, check=None, rows=256, _log=log):
            _log.append(("apply", Path(path).name))
            return real_apply(path, validity, check=check, rows=rows)
        paths = copies(entry.fixture, tmp_path / side)
        with _patched(sv, "apply_validity_to_fits", apply):
            zero(paths, source, side, case=h.Case(1, None), check=lambda _log=log: _log.append(("check",)))
        logs.append(log)
    assert logs[0] == logs[1] and logs[0][:3] == [("check",), ("for_frame", "light_00001"), ("apply", "light_00001.fit")]


@contextlib.contextmanager
def _patched(owner, name, value):
    saved = getattr(owner, name)
    setattr(owner, name, value)
    try:
        yield
    finally:
        setattr(owner, name, saved)


def test_prepare_looks_masks_up_on_the_calling_thread_in_frame_order(sets, tmp_path):
    entry = sets[(-32, "per_frame")]
    seen = []
    source = source_of(entry.fixture, "per_frame")
    real = source.for_frame
    import threading
    source.for_frame = lambda stem: (seen.append((stem, threading.current_thread().name)), real(stem))[1]
    paths = copies(entry.fixture, tmp_path / "pool")
    with h.pool_conditions(h.Case(8, 6, "reuse", 3)):
        zero(paths, source, "pool", case=h.Case(8, FRAMES))
    assert [stem for stem, _ in seen] == [Path(p).stem for p in paths]
    assert {name for _, name in seen} == {threading.current_thread().name}


def test_reruns_under_two_hash_seeds_give_one_digest():
    digest = h.assert_hashseed_stable("test_p1c_site_sky_zero:reference_target",
                                      "test_p1c_site_sky_zero:pooled_target", 4)
    assert digest == h.canonical_digest(hashseed_target("reference", 4, 1))


@pytest.mark.parametrize("env", h.NUMERIC_ENVS, ids=lambda e: ",".join(f"{k}={v}" for k, v in e.items()))
def test_numeric_variants_give_the_same_zeroing(env):
    """(g): a unit is integer work plus one division for its statistics; no BLAS. Equal under every variant."""
    expected = h.canonical_digest(hashseed_target("reference", 4, 1))
    assert h.subprocess_digest("test_p1c_site_sky_zero:hashseed_target", "pool", 4, 4, env=env) == expected


def test_notes_shadow_and_the_timer_operation(sets, tmp_path, monkeypatch):
    entry = sets[(16, "static")]
    timer = profiling.StageTimer()
    try:
        for workers in (3, None):
            paths = copies(entry.fixture, tmp_path / f"w{workers}")
            with ledger("on"):
                stats = core._sky_zero(paths, source_of(entry.fixture, "static"), lambda: None, WIDTH,
                                       workers=workers, timer=timer)
            h.assert_same(entry.stats, stats)
        monkeypatch.setenv(frame_pool.ENV_WORKERS, "1")
        paths = copies(entry.fixture, tmp_path / "serial")
        core._sky_zero(paths, source_of(entry.fixture, "static"), lambda: None, WIDTH, timer=timer)
    finally:
        report = timer.finish()
    assert report["operation_counts"]["sky_zero"] == 3
    entries = report["notes"]["frame_pool.sky_zero"]
    assert [(e["workers"], e["serial"], e.get("reason")) for e in entries][0] == (3, False, None)
    assert entries[-1]["serial"] is True and entries[-1]["reason"] == "workers"
    for e in entries:
        assert e["unit_bytes"] == core._sky_zero_unit_bytes(WIDTH) and e["units"] == FRAMES
        assert {"cpu_user_seconds", "cpu_kernel_seconds", "max_unit_seconds"} <= set(e)


# ================================================================== the routes of _ensure_sky_validity

def _job_dir(root: Path, mask_png: Path, frames: list, frame_masks=None) -> Path:
    directory = root / "job"
    (directory / "work").mkdir(parents=True)
    shutil.copyfile(mask_png, directory / "sky_mask.png")
    job = {"state": "succeeded", "updated_at": core.now(), "checkpoints": [], "frame_masks": frame_masks}
    (directory / "job.json").write_text(json.dumps(job), encoding="utf-8")
    return directory


@pytest.mark.parametrize("rezero", [False, True])
def test_the_rezeroing_route_keeps_the_serial_loop(sets, tmp_path, monkeypatch, rezero):
    entry = sets[(16, "static")]
    png = tmp_path / "mask.png"
    Image.fromarray((entry.fixture.mask * 255).round().astype(np.uint8)).save(png)
    directory = _job_dir(tmp_path, png, entry.fixture.frames)
    paths = data.copy_frames(entry.fixture.frames, directory / "work")
    ground = directory / "work" / "foreground_master.fit"
    imaging.write_linear(ground, np.zeros((HEIGHT, WIDTH, 3), np.float32), "placeholder")
    calls = []
    monkeypatch.setattr(core, "SKY_VALIDITY_DILATION_PX", DILATION)
    real_pool, real_reference = core._sky_zero, core._sky_zero_reference
    monkeypatch.setattr(core, "_sky_zero", lambda *a, **k: calls.append("pool") or real_pool(*a, **k))
    monkeypatch.setattr(core, "_sky_zero_reference", lambda *a, **k: calls.append("serial") or real_reference(*a, **k))
    pools = []
    real_run = frame_pool.run_ordered
    monkeypatch.setattr(frame_pool, "run_ordered", lambda *a, **k: pools.append(k["name"]) or real_run(*a, **k))
    timer = profiling.StageTimer()
    try:
        with ledger("on"):
            record = core._ensure_sky_validity(directory, directory / "work", paths, (3, HEIGHT, WIDTH), lambda: None,
                                               lambda name, files: None, ["foreground"], timer=timer, rezero=rezero)
    finally:
        report = timer.finish()
    assert calls == (["serial"] if rezero else ["pool"]) and pools == ([] if rezero else ["sky-zero"])
    assert report["operation_counts"].get("sky_zero") == (None if rezero else 1)
    reference = copies(entry.fixture, tmp_path / "reference")
    stats, _ = zero(reference, source_of(entry.fixture, "static"), "reference")
    h.assert_same(frame_bytes(reference), frame_bytes(paths))
    assert record["zeroed_pixels_per_frame"] == {s["file"]: s["zeroed_pixels"] for s in stats}
    assert record["frames_masked"] == FRAMES


# ================================================================== (c) failures

@pytest.mark.parametrize("k", h.failure_positions(FRAMES))
@pytest.mark.parametrize("workers", [2, 4, 16])
@pytest.mark.parametrize("how", ["raise", "readonly"])
def test_a_failure_at_frame_k_is_the_serial_failure(sets, tmp_path, k, workers, how):
    import re
    import stat
    entry = sets[(16, "static")]
    zeroed = entry.files
    outcomes, trees = {}, {}
    for side in ("reference", "pool"):
        paths = copies(entry.fixture, tmp_path / side)
        stem = Path(paths[k]).stem
        if how == "readonly":                                 # a real error: the in-place open is refused
            os.chmod(paths[k], stat.S_IREAD)
        real = sv.apply_validity_to_fits

        def apply(path, validity, *, check=None, rows=256, _stem=stem):
            if how == "raise" and Path(path).stem == _stem:
                raise OSError(f"injected failure at {_stem}")
            return real(path, validity, check=check, rows=rows)
        delivered = []
        real_run = frame_pool.run_ordered

        def run(fn, items, *, on_result, **kwargs):
            return real_run(fn, items, on_result=lambda i, s: (delivered.append(i), on_result(i, s)), **kwargs)
        before = h.tree_state(tmp_path / side)
        with _patched(sv, "apply_validity_to_fits", apply), _patched(frame_pool, "run_ordered", run), \
                h.pool_conditions(h.Case(workers, None, "reuse", k)):
            outcome = h.outcome(lambda d: zero(paths, source_of(entry.fixture, "static"), side,
                                               case=h.Case(workers, None)))
        outcome.delivered = delivered if side == "pool" else list(range(k))
        if outcome.message:                                   # the folder differs between the two sides
            outcome.message = re.sub(r"'[^']*[\\/](light_\d+\.fit)'", r"'\1'", outcome.message)
        outcomes[side] = outcome
        os.chmod(paths[k], stat.S_IREAD | stat.S_IWRITE)
        trees[side] = (paths, h.changed_files(before, h.tree_state(tmp_path / side)))
    serial, pooled = outcomes["reference"], outcomes["pool"]
    assert serial.error_type is (OSError if how == "raise" else PermissionError)
    assert pooled.error_type is serial.error_type
    h.assert_like_serial(serial, pooled)
    serial_changed, pool_changed = (set(t[1]["modified"]) for t in (trees["reference"], trees["pool"]))
    names = [Path(p).name for p in trees["pool"][0]]
    assert serial_changed == set(names[:k])
    extra = pool_changed - serial_changed
    assert serial_changed <= pool_changed and extra <= set(names[k + 1:])     # prereg §5 item 2 (C-3)
    paths = trees["pool"][0]
    validity = source_of(entry.fixture, "static")
    for index, name in enumerate(names):                     # zeroed fully, by whole blocks, or not at all
        if index == k:
            continue
        assert_zeroed_by_blocks(paths[index], entry.fixture.frames[index], validity.for_frame(Path(name).stem))
        if index < k:
            assert Path(paths[index]).read_bytes() == zeroed[index], name
    if how == "raise":                                       # a rerun ends where an uninterrupted run does
        zero(paths, source_of(entry.fixture, "static"), "pool", case=h.Case(workers, None))
        h.assert_same(zeroed, frame_bytes(paths))


@pytest.mark.parametrize("k", h.failure_positions(7))
def test_a_job_failing_in_the_zeroing_fails_and_resumes_like_the_serial_job(tmp_path, monkeypatch, k):
    """(c) at job level, on a controlled engine: a failure at frame k of the zeroing fails the job with 4 threads
    exactly as with 1 (job.json's state, stage, error, message, checkpoints, transients); resumed, both end as the
    uninterrupted job, bit for bit (checkpoint digests, records, products: the main route has no convert checkpoint,
    so the resume converts again and zeroes every frame once)."""
    from test_hashing_job_slice import _comparable
    victim, fail = f"light_{k + 1:05d}", {"on": False}
    real = sv.apply_validity_to_fits

    def apply(path, validity, *, check=None, rows=256):
        if fail["on"] and Path(path).stem == victim:
            raise OSError(f"injected zeroing failure at {victim}")
        return real(path, validity, check=check, rows=rows)
    monkeypatch.setattr(sv, "apply_validity_to_fits", apply)
    jobs = data.failing_jobs(tmp_path, monkeypatch, fail)
    assert jobs["uninterrupted"]["state"] == "succeeded", jobs["uninterrupted"].get("error")
    assert data.pool_workers(jobs["uninterrupted"], "sky_zero") == [4]
    serial, parallel = data.job_outcome(jobs["serial"]), data.job_outcome(jobs["parallel"])
    assert serial["state"] == "failed" and serial["error"] == f"injected zeroing failure at {victim}"
    assert parallel == serial
    pools = jobs["pools"]
    assert ("sky-zero", 4, 4) in pools["uninterrupted"] and ("sky-zero", 4, 4) in pools["parallel"]
    assert ("sky-zero", 1, None) in pools["serial"]
    assert data.pool_workers(jobs["parallel"], "median") == [4] and data.pool_workers(jobs["serial"], "median") == [1]
    expected = _comparable(jobs["uninterrupted"])
    for name in ("serial-resumed", "parallel-resumed"):
        assert jobs[name]["state"] == "succeeded", (name, jobs[name].get("error"))
        assert _comparable(jobs[name]) == expected, name


# ================================================================== (d) cancellation

@pytest.mark.parametrize("workers", [1, 4, 8, 16])
@pytest.mark.parametrize("call", [0, 2, 5])
def test_a_cancel_at_submission(sets, tmp_path, workers, call):
    entry = sets[(-32, "static")]
    paths = copies(entry.fixture, tmp_path / "pool")
    switch = h.CancelSwitch().trip_at_call(call)
    h.assert_cancels_in_time(lambda check: zero(paths, source_of(entry.fixture, "static"), "pool",
                                                case=h.Case(workers, None), check=check), switch)
    blocks = assert_stopped_zeroing_left(tmp_path / "pool", paths, entry.fixture.frames,
                                         source_of(entry.fixture, "static"))
    if call == 0:                                            # cancelled before the first unit: nothing written
        assert blocks == [0] * FRAMES


# ================================================================== (f) memory and the worker decision

GIB = 1 << 30


@pytest.mark.parametrize("available,expected", [(None, 1), (1.0, 1), (1.15, 2), (1.2, 3), (1.24, 4), (8.0, 4)])
def test_the_worker_decision_follows_memory(monkeypatch, available, expected):
    """63.6 MB per frame at W = 8280 (no BLAS): four threads from about 1.24 GiB available (prereg §3.1)."""
    monkeypatch.setattr(performance, "_memory_status",
                        lambda: (None, None) if available is None else (int(available * GIB), int(available * GIB)))
    monkeypatch.setattr(frame_pool, "usable_cpus", lambda: 16)
    unit_bytes = core._sky_zero_unit_bytes(8280)
    assert unit_bytes == 256 * 8280 * 30 == 63_590_400
    assert performance.frame_workers(48, unit_bytes, core.SKY_ZERO_WORKER_CAP, blas=False) == expected


def test_the_decision_takes_the_override(monkeypatch):
    monkeypatch.setattr(performance, "_memory_status", lambda: (64 * GIB, 64 * GIB))
    monkeypatch.setattr(frame_pool, "usable_cpus", lambda: 32)
    for value, expected in (("max", 16), ("2", 2), ("1", 1)):
        monkeypatch.setenv(frame_pool.ENV_WORKERS, value)
        assert performance.frame_workers(48, core._sky_zero_unit_bytes(8280), core.SKY_ZERO_WORKER_CAP,
                                         blas=False) == expected


# ================================================================== (b) real data: the F48 DNGs re-converted

def _sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(16 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@pytest.fixture(scope="module")
def f48_source():
    job = h.f48_job()
    record = data.validity_record()
    _, height, width = core._validate_frame(sorted((job / "work").glob("light_*.fit"))[0])
    source = sv.StaticMaskSource(sv.load_mask_top_down(job / "sky_mask.png", width, height),
                                 dilation_px=record["dilation_px"])
    return SimpleNamespace(source=source, width=width, height=height, record=record, digests=data.zeroed_sha256())


def job_like_copies(tmp_path_factory, folder: Path, count: int, mode="on") -> list:
    """``count`` re-converted frames as the zeroing meets them in a job: just written (copied and not flushed, so the
    write-back of gigabytes is under way, as after Siril's conversion), then read by the median and the foreground
    master written (_ensure_sky_validity's steps before the zeroing, in the ledger ``mode``). The median is also each
    frame's first open and read after its write: with gigabytes being written back, that has stalled for seconds
    here (14 s in sixteen concurrent header reads, 2.4 s in the first blocks of four memory-mapped passes) -- in a
    job the median meets it, not the zeroing. Measured job-like (48 frames, 3 runs each, cancel 0.3 s in): stream
    0.016-0.091 s at 4, 8 and 16 threads (serial <= 0.006 s), memmap 0.035-1.403 s (serial 0.005-0.040 s, its flushes
    0.46-0.53 s)."""
    paths = data.copy_frames(data.reconverted(tmp_path_factory, count), folder)
    with ledger(mode):
        ground = imaging.median_foreground(paths, lambda: None, workers=None)
        imaging.write_linear(folder.parent / "foreground_master.fit", ground, data.FOREGROUND_NOTE)
    del ground
    return paths


@pytest.mark.parametrize("case", h.params(h.real_cases()))
def test_real_16_frames_zero_to_the_f48_frames(tmp_path_factory, tmp_path, f48_source, case):
    paths = data.copy_frames(data.reconverted(tmp_path_factory, 16), tmp_path / "work")
    with h.pool_conditions(case), ledger("on"):
        started = time.perf_counter()
        stats = core._sky_zero(paths, f48_source.source, lambda: None, f48_source.width, workers=case.workers)
        seconds = time.perf_counter() - started
    for path, entry in zip(paths, stats):
        assert data.sha256_as_f48(path) == f48_source.digests[path.name], path.name
        assert entry["zeroed_pixels"] == f48_source.record["zeroed_pixels_per_frame"][path.name]
        assert list(entry) == ["file", "zeroed_pixels", "invalid_fraction"]
    print(f"real 16-frame zeroing, workers {case.workers}: {seconds:.2f} s")


MEMMAP_HANG_GUARD = 5.0     # seconds: the memory-mapped pass is an exception route (prereg §3.1 C-3), not unbounded


@pytest.mark.parametrize("mode", ["on", None], ids=["stream", "memmap"])
@pytest.mark.parametrize("workers", [None, 8, 16])
def test_real_size_cancel_within_a_second(tmp_path_factory, tmp_path, f48_source, workers, mode):
    """(d) as a job meets it (job_like_copies: the 48 frames just written and not flushed, read by the median, the
    foreground master written), the zeroing cancelled 0.3 s in. 'stream', the job's route (inside its ledger): the
    pool returns within 1.0 s. 'memmap', the memory-mapped pass outside a ledger (AESTER_HASH_LEDGER=off, a
    diagnostic setting): its flush at each frame's end, the release of a frame's mapping and the page faults of
    numpy's masked assignment into it are not probed (the frozen sky_validity.py; the serial loop has them too) and
    can stall while gigabytes are written back -- C-3's registered exception route (measured job-like 0.035-1.40 s),
    held to a hang guard. Afterwards only frames changed, each untouched or zeroed by whole blocks (prereg §5 item 2)."""
    paths = job_like_copies(tmp_path_factory, tmp_path / "work", 48, mode)
    switch = h.CancelSwitch().trip_after(0.3)
    limit = h.CANCEL_LIMIT_SECONDS if mode else MEMMAP_HANG_GUARD
    with h.pool_conditions(h.Case(workers, None, "reuse", None)), ledger(mode):
        latency = h.assert_cancels_in_time(
            lambda check: core._sky_zero(paths, f48_source.source, check, f48_source.width, workers=workers), switch,
            limit=limit)
    switch.raised.clear()                       # their tracebacks hold the memory-mapped pass's views
    gc.collect()
    blocks = assert_stopped_zeroing_left(tmp_path / "work", paths, data.reconverted(tmp_path_factory, 48),
                                         f48_source.source)
    print(f"real-size zeroing cancel, {'stream' if mode else 'memmap'}, workers {workers}: {latency:.3f} s; frames "
          f"touched {sum(1 for b in blocks if b)}, blocks zeroed {sum(blocks)}")
    shutil.rmtree(tmp_path / "work")            # 13 GB per case; a failed case keeps its frames


@pytest.fixture(scope="module")
def own_mask_png(tmp_path_factory):
    """A real-size mask of a frame's own: the job's sky mask with a 400 x 300 px patch of ground weight in the sky."""
    with Image.open(h.f48_job() / "sky_mask.png") as img:
        mask = np.array(img.convert("L"))
    mask[200:500, 1000:1400] = 0
    path = tmp_path_factory.mktemp("own-mask") / "own.png"
    Image.fromarray(mask).save(path)
    return path


@pytest.mark.parametrize("workers", [None, 8, 16])
def test_real_size_cancel_during_a_per_frame_mask_lookup(tmp_path_factory, tmp_path, f48_source, own_mask_png,
                                                          workers):
    """(d) with every other frame's own mask (PerFrameMaskSource): its lookup in prepare, on the calling thread, loads
    and dilates a 45.7 MP mask in the frozen sky_validity.py without a probe -- C-3's one exception segment (prereg
    §3.1, §5 item 5), as long as in the serial loop. A cancel raised when the fourth frame's lookup starts: the pool
    returns within 1.0 s of that lookup's end; afterwards only frames changed, untouched or zeroed by whole blocks."""
    count = 8
    paths = job_like_copies(tmp_path_factory, tmp_path / "work", count)
    files = {path.stem: shutil.copyfile(own_mask_png, tmp_path / f"{path.stem}.png") for path in paths[1::2]}
    source = sv.PerFrameMaskSource(f48_source.source, files, f48_source.width, f48_source.height)
    switch, spans = h.CancelSwitch(), {}
    real = source.for_frame

    def for_frame(stem):
        started = time.perf_counter()
        if stem == paths[3].stem:
            switch.trip()                        # on the calling thread, as the lookup starts
        try:
            return real(stem)
        finally:
            spans[stem] = (started, time.perf_counter())
    source.for_frame = for_frame
    with h.pool_conditions(h.Case(workers, None, "reuse", None)), ledger("on"):
        latency = h.assert_cancels_in_time(
            lambda check: core._sky_zero(paths, source, check, f48_source.width, workers=workers), switch,
            limit=60.0)
    source.for_frame = real
    lookup = spans[paths[3].stem][1] - switch.tripped_at
    assert [stem for stem in spans] == [path.stem for path in paths[:4]]       # nothing looked up past the cancel
    assert latency - lookup <= h.CANCEL_LIMIT_SECONDS, (latency, lookup)
    blocks = assert_stopped_zeroing_left(tmp_path / "work", paths, data.reconverted(tmp_path_factory, count), source)
    print(f"real-size zeroing cancel in a per-frame mask lookup, workers {workers}: {latency:.3f} s, of which the "
          f"lookup {lookup:.3f} s; blocks zeroed {sum(blocks)}")
    shutil.rmtree(tmp_path / "work")            # a failed case keeps its frames


@pytest.mark.skipif(os.name != "nt", reason="private bytes are read on Windows")
@pytest.mark.parametrize("workers", [1, 4, 16])
def test_real_size_pool_memory_within_unit_bytes(tmp_path_factory, tmp_path, f48_source, workers):
    """(f): in flight 2 x workers, the first frame finishing last; the pool's rise, sampled around run_ordered after a
    collection (garbage freed inside the window would offset the pool's rise), is at least one read block (256 rows
    of 16-bit samples) per granted thread -- a run that measures nothing fails -- and at most granted x unit_bytes
    (no BLAS slot)."""
    paths = data.copy_frames(data.reconverted(tmp_path_factory, 16), tmp_path / "work")
    peaks, stats = {}, {}
    real = frame_pool.run_ordered

    def measured(*args, **kwargs):
        gc.collect()
        sampler = h.PeakSampler(trace=False)
        try:
            with sampler:
                return real(*args, **kwargs)
        finally:
            stats.update(kwargs.get("stats") or {})
            peaks["rise"] = sampler.rise
    with h.pool_conditions(h.Case(workers, 2 * workers, "reuse", 0), head_last=True, head_seconds=0.3), \
            _patched(frame_pool, "run_ordered", measured), ledger("on"):
        core._sky_zero(paths, f48_source.source, lambda: None, f48_source.width, workers=workers,
                       inflight=2 * workers)
    limit = h.pool_memory_limit(stats, blas=False)
    granted = stats["workers"]
    unit = core._sky_zero_unit_bytes(f48_source.width)
    block = granted * core.SKY_ZERO_ROWS * f48_source.width * 2
    print(f"real 16-frame zeroing pool, {granted} threads: rise {peaks['rise'] / 2**20:.1f} MiB, read blocks "
          f"{block / 2**20:.1f} MiB, limit {limit / 2**20:.1f} MiB, per thread {peaks['rise'] / granted / 2**20:.1f} "
          f"MiB, unit_bytes {unit / 2**20:.1f} MiB")
    assert block <= peaks["rise"] <= max(limit, unit), (peaks, block, limit)


def test_real_48_frames_through_ensure_sky_validity(tmp_path_factory, tmp_path, monkeypatch):
    """The convert stage's Python part on the 48 re-converted frames, as the job runs it (default threads, ledger on,
    the job's timer): the F48 job's foreground master, its 48 zeroed frames and its sky_validity record."""
    job = h.f48_job()
    directory = _job_dir(tmp_path, job / "sky_mask.png", [])
    paths = data.copy_frames(data.reconverted(tmp_path_factory, 48), directory / "work")
    checkpoints = []
    timer = profiling.StageTimer()
    try:
        with ledger("on"), h.pool_conditions(h.Case(None, None, "reuse", None)):
            started = time.perf_counter()
            record = core._ensure_sky_validity(directory, directory / "work", paths, core._validate_frame(paths[0]),
                                               lambda: None, lambda name, files: checkpoints.append(name), [],
                                               timer=timer)
            seconds = time.perf_counter() - started
    finally:
        report = timer.finish()
    assert checkpoints == ["foreground"]
    assert _sha(directory / "work" / "foreground_master.fit") == data.foreground_sha256()
    digests = data.zeroed_sha256()
    assert len(digests) == 48
    for path in paths:
        assert data.sha256_as_f48(path) == digests[path.name], path.name
    expected = {k: v for k, v in data.validity_record().items() if k != "applied_at"}
    h.assert_same(expected, {k: v for k, v in record.items() if k != "applied_at"})
    assert report["operation_counts"]["foreground_median"] == 1 and report["operation_counts"]["sky_zero"] == 1
    median_note, zero_note = report["notes"]["frame_pool.median"], report["notes"]["frame_pool.sky_zero"]
    print(f"real 48-frame convert stage (median + write + zeroing): {seconds:.2f} s; median "
          f"{median_note[0]['workers']} threads {median_note[0]['wall_seconds']} s, zeroing "
          f"{zero_note[0]['workers']} threads {zero_note[0]['wall_seconds']} s")
    assert median_note[0]["workers"] == imaging.MEDIAN_WORKER_CAP and zero_note[0]["workers"] == core.SKY_ZERO_WORKER_CAP
