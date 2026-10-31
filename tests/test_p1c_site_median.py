"""Efficiency P1c C-1 (docs/EFFICIENCY_P1C_PREREG_20261005.md §3.1, §6.2 A5): the foreground median's
integer-network blocks on frame_pool (imaging._median_explicit) against the kept serial reference
(imaging._median_reference, the pre-P1c reader verbatim).

(a) the synthetic matrix and real sizes, bit for bit; two PYTHONHASHSEED values and the numeric variants (g);
(b) the F48 DNGs re-converted (16 and 48 frames): the 48-frame median written by write_linear is the F48 job's
    foreground master byte for byte; (c) read and open failures at blocks / frames {0, 1, middle, last};
(d) cancellation within 1 s; (e) the R6 audit; (f) the pool's memory within its declared unit_bytes and the
worker decision; plus the routes that stay serial ('route', 'fp-env'), the notes and the timer operation.
"""
import contextlib
import math
import os
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pytest

import p1c_convert_data as data
import p1c_site_harness as h
from starstack import frame_pool, hashing, imaging, performance, profiling
from test_foreground_median import banded_frames, float_frame, roworder_for, siril_frame, stored_codes, write_fits

N, HEIGHT, WIDTH = 9, 200, 23                      # 4 blocks (64, 64, 64, 8)
UNITS = math.ceil(HEIGHT / imaging.MEDIAN_BLOCK_ROWS)


@pytest.fixture(autouse=True)
def clean_pool(monkeypatch):
    monkeypatch.delenv(frame_pool.ENV_WORKERS, raising=False)
    before = h.aester_threads()
    yield
    h.assert_pool_quiet(before)


def synthetic_paths(folder: Path, seed: int, n: int = N, height: int = HEIGHT, width: int = WIDTH,
                    order: str = "mixed") -> list:
    values = banded_frames(np.random.default_rng(seed), n, height, width)
    return [siril_frame(folder / f"light_{i + 1:05d}.fit", values[i], roworder_for(order, i)) for i in range(n)]


def frames_of(paths):
    frames = imaging._median_frames(paths)
    assert frames is not None
    return frames


def counting(stop=None):
    """A cancel_check that counts its calls (and raises from call ``stop`` on)."""
    calls = [0]

    def check():
        calls[0] += 1
        if stop is not None and calls[0] >= stop:
            raise h.Cancelled("用户取消")
    check.calls = calls
    return check


# ---- targets for the PYTHONHASHSEED / numeric reruns (module level: imported in a fresh interpreter) ---------------

def synthetic_target(side, seed, workers):
    with tempfile.TemporaryDirectory() as folder:
        frames = frames_of(synthetic_paths(Path(folder), seed))
        if side == "reference":
            return imaging._median_reference(frames, lambda: None)
        return imaging._median_explicit(frames, lambda: None, workers=workers)


def reference_target(seed):
    return synthetic_target("reference", seed, 1)


def pooled_target(seed):
    return synthetic_target("pool", seed, 4)


# ================================================================== (a) synthetic matrix

@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    folder = tmp_path_factory.mktemp("median-synthetic")
    paths = synthetic_paths(folder, 11)
    return paths, imaging._median_reference(frames_of(paths), lambda: None)


@pytest.mark.parametrize("case", h.params(h.synthetic_cases()))
def test_pool_matches_reference_over_the_matrix(synthetic, case):
    paths, expected = synthetic
    stats = {}
    real = frame_pool.run_ordered

    def spy(*args, **kwargs):
        result = real(*args, **kwargs)
        stats.update(kwargs.get("stats") or {})
        return result
    with h.pool_conditions(case), _patched(frame_pool, "run_ordered", spy):
        actual = imaging._median_explicit(frames_of(paths), lambda: None, workers=case.workers,
                                          inflight=case.inflight)
    h.assert_same(expected, actual)
    assert stats["workers"] == min(case.workers, UNITS) and stats["unit_bytes"] == imaging.median_unit_bytes(N, WIDTH)
    assert stats["serial"] == (min(case.workers, UNITS) < 2)


@contextlib.contextmanager
def _patched(owner, name, value):
    saved = getattr(owner, name)
    setattr(owner, name, value)
    try:
        yield
    finally:
        setattr(owner, name, saved)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 8, 47, 48])
@pytest.mark.parametrize("order", ["all-bottom", "all-top", "mixed", "none"])
def test_frame_counts_and_row_orders(tmp_path, n, order):
    paths = synthetic_paths(tmp_path, n, n=n, height=131, width=7, order=order)
    expected = imaging._median_reference(frames_of(paths), lambda: None)
    for workers in (1, 3, 16):
        h.assert_same(expected, imaging._median_explicit(frames_of(paths), lambda: None, workers=workers))
    h.assert_same(expected, imaging._median_foreground_reference(paths, lambda: None))


def test_the_serial_path_keeps_todays_call_pattern(synthetic):
    """workers=1 (and median_foreground's default): one cancel_check per block before the block, as the reference."""
    paths, expected = synthetic
    for run in (lambda c: imaging._median_reference(frames_of(paths), c),
                lambda c: imaging._median_explicit(frames_of(paths), c, workers=1),
                lambda c: imaging.median_foreground(paths, c)):
        check = counting()
        h.assert_same(expected, run(check))
        assert check.calls[0] == UNITS
    for stop in range(1, UNITS + 1):
        old, new = counting(stop), counting(stop)
        with pytest.raises(h.Cancelled):
            imaging._median_reference(frames_of(paths), old)
        with pytest.raises(h.Cancelled):
            imaging._median_explicit(frames_of(paths), new, workers=1)
        assert old.calls == new.calls == [stop]


def test_reruns_under_two_hash_seeds_give_one_digest():
    digest = h.assert_hashseed_stable("test_p1c_site_median:reference_target", "test_p1c_site_median:pooled_target", 3)
    assert digest == h.canonical_digest(synthetic_target("reference", 3, 1))


@pytest.mark.parametrize("env", h.NUMERIC_ENVS, ids=lambda e: ",".join(f"{k}={v}" for k, v in e.items()))
def test_numeric_variants_give_the_same_median(env):
    """(g): the median is integer selection plus one float32 mean; no BLAS. Equal under every variant."""
    expected = h.canonical_digest(synthetic_target("reference", 4, 1))
    assert h.subprocess_digest("test_p1c_site_median:synthetic_target", "pool", 4, 16, env=env) == expected


# ================================================================== routes that stay serial, notes, the operation

def test_float_and_former_routes_stay_serial_and_say_route(tmp_path, monkeypatch):
    notes = []
    monkeypatch.setattr(performance, "note_pool", lambda site, stats: notes.append((site, dict(stats))))
    rng = np.random.default_rng(2)
    floats = [float_frame(tmp_path / f"f{i}.fit", rng.random((HEIGHT, WIDTH, 3)).astype(np.float32)) for i in range(3)]
    pooled = []
    monkeypatch.setattr(frame_pool, "run_ordered", lambda *a, **k: pooled.append(1))
    expected = imaging._median_foreground_reference(floats, lambda: None)
    h.assert_same(expected, imaging.median_foreground(floats, lambda: None, workers=None))
    gz = tmp_path / "former"
    gz.mkdir()
    former = [Path(str(p) + ".gz") for p in synthetic_paths(gz, 3, n=3, height=70, width=5)]
    import gzip
    for path in former:
        path.write_bytes(gzip.compress(Path(str(path)[:-3]).read_bytes()))
    h.assert_same(imaging._median_foreground_reference(former, lambda: None),
                  imaging.median_foreground(former, lambda: None, workers=None))
    assert pooled == []
    assert [(site, s["reason"], s["units"], s["workers"], s["serial"]) for site, s in notes] == [
        ("median", "route", UNITS, 1, True), ("median", "route", None, 1, True)]
    for _, entry in notes:
        assert {"unit_bytes", "cpu_user_seconds", "cpu_kernel_seconds", "max_unit_seconds"} <= set(entry)


def test_a_table_near_the_subnormal_range_stays_serial(tmp_path, monkeypatch):
    """BSCALE 1e-35: every table value is subnormal or 0 -> serial, reason 'fp-env', still the reference's bits."""
    notes = []
    monkeypatch.setattr(performance, "note_pool", lambda site, stats: notes.append(dict(stats)))
    values = banded_frames(np.random.default_rng(5), 5, 70, 9)
    paths = [write_fits(tmp_path / f"light_{i:05d}.fit", stored_codes(values[i]), 16,
                        [("BZERO", "RAW:0."), ("BSCALE", "RAW:1E-35"), ("ROWORDER", "BOTTOM-UP")]) for i in range(5)]
    frames = frames_of(paths)
    assert frames[0].bscale == 1e-35
    network = imaging._median_network(frames)
    assert network is not None and imaging._table_fp_sensitive(network.table)
    expected = imaging._median_reference(frames, lambda: None)
    h.assert_same(expected, imaging._median_explicit(frames, lambda: None, workers=4))
    assert [(n["reason"], n["serial"]) for n in notes] == [("fp-env", True)]
    assert not imaging._table_fp_sensitive(imaging.median_conversion_table(1.0, 32768.0))


def test_notes_operation_and_kernel_note(synthetic, monkeypatch):
    paths, expected = synthetic
    kernel = []
    real_note = profiling.note
    monkeypatch.setattr(profiling, "note", lambda key, value: kernel.append((key, value)) or real_note(key, value))
    timer = profiling.StageTimer()
    try:
        h.assert_same(expected, imaging.median_foreground(paths, lambda: None, workers=3, timer=timer))
        monkeypatch.setenv(frame_pool.ENV_WORKERS, "1")
        h.assert_same(expected, imaging.median_foreground(paths, lambda: None, workers=None, timer=timer))
    finally:
        report = timer.finish()
    assert report["operation_counts"]["foreground_median"] == 2
    entries = report["notes"]["frame_pool.median"]
    assert [(e["workers"], e["serial"], e.get("reason"), e["bound"]) for e in entries] == [
        (3, False, None, "caller"), (1, True, "workers", "override")]
    for entry in entries:
        assert entry["unit_bytes"] == imaging.median_unit_bytes(N, WIDTH) and entry["units"] == UNITS
        assert {"cpu_user_seconds", "cpu_kernel_seconds", "max_unit_seconds", "unit_cpu_seconds"} <= set(entry)
    assert "fp_env_applied" in entries[0]
    assert [k for k, _ in kernel].count("foreground_median") == 2           # the P1a note, unchanged
    assert kernel[0][1] == {"kernel": imaging.MEDIAN_KERNEL, "frames": N, "tile_rows": 64,
                            "comparators": len(imaging.median_pairs(N))}


@pytest.mark.skipif(os.name != "nt", reason="the C runtime's _controlfp")
def test_r21_workers_take_the_coordinators_denormal_bits(synthetic):
    """With FTZ/DAZ set in the calling thread the pool's workers run with it too (fp_env_applied); the median of
    normal values is the reference's either way."""
    paths, expected = synthetic
    api = frame_pool._controlfp()
    saved = api(0, 0)
    stats = {}
    real = frame_pool.run_ordered

    def spy(*args, **kwargs):
        try:
            return real(*args, **kwargs)
        finally:
            stats.update(kwargs.get("stats") or {})
    try:
        api(0x01000000, frame_pool._MCW_DN)       # _DN_FLUSH
        with _patched(frame_pool, "run_ordered", spy):
            actual = imaging._median_explicit(frames_of(paths), lambda: None, workers=4)
    finally:
        api(saved, frame_pool.FP_ENV_MASK)
    h.assert_same(expected, actual)
    assert stats["fp_env_applied"] >= 1 and int(stats["fp_env"], 16) & frame_pool._MCW_DN == 0x01000000


# ================================================================== (c) failures

@pytest.mark.parametrize("k", h.failure_positions(UNITS))
@pytest.mark.parametrize("workers", [2, 4, 16])
def test_a_read_error_at_block_k_is_the_serial_error(tmp_path, monkeypatch, k, workers):
    paths = synthetic_paths(tmp_path, 21, order="all-bottom")
    real = imaging._read_rows

    def read(handle, frame, plane, start, stop, out):
        if start // imaging.MEDIAN_BLOCK_ROWS >= k and plane == 1 and frame.path.endswith("00004.fit"):
            raise OSError(f"FITS data ended early (the file changed while it was read): {frame.path} @{start}")
        return real(handle, frame, plane, start, stop, out)
    monkeypatch.setattr(imaging, "_read_rows", read)
    serial = h.outcome(lambda d: imaging._median_explicit(frames_of(paths), lambda: None, workers=1))
    with h.pool_conditions(h.Case(workers, None, "reuse", k)):
        parallel = h.outcome(lambda d: imaging._median_explicit(frames_of(paths), lambda: None, workers=workers))
    assert serial.error_type is OSError and serial.message.endswith(f"@{k * imaging.MEDIAN_BLOCK_ROWS}")
    h.assert_like_serial(serial, parallel)
    for path in paths:                                # finalize closed every handle (Windows refuses otherwise)
        path.unlink()


def test_a_truncated_frame_fails_at_its_lowest_block_with_the_serial_text(tmp_path):
    paths = synthetic_paths(tmp_path, 22, order="all-bottom")
    frames = frames_of(paths)                         # headers read while the frame was complete
    victim = paths[5]
    size = victim.stat().st_size
    with open(victim, "r+b") as raw:                  # cut into plane 2 (rows 100.. of the last plane gone)
        raw.truncate(frames[5].offset + (2 * HEIGHT + 100) * WIDTH * 2)
    outcomes = [h.outcome(lambda d: imaging._median_reference(frames, lambda: None))]
    for workers in (1, 3, 16):
        with h.pool_conditions(h.Case(workers, None, "reuse", 1)):
            outcomes.append(h.outcome(lambda d: imaging._median_explicit(frames, lambda: None, workers=workers)))
    assert outcomes[0].error_type is OSError and "ended early" in outcomes[0].message and size > 0
    for other in outcomes[1:]:
        h.assert_like_serial(outcomes[0], other)


@pytest.mark.parametrize("j", [0, 1, N // 2, N - 1])
@pytest.mark.parametrize("workers", [1, 2, 4])
def test_an_open_failure_is_the_first_in_frame_order(tmp_path, monkeypatch, j, workers):
    paths = synthetic_paths(tmp_path, 23)
    frames = frames_of(paths)
    real_open = open
    failing = {frames[j].path, frames[-1].path}

    def opener(path, *args, **kwargs):
        if path in failing:
            raise PermissionError(13, "locked", path)
        return real_open(path, *args, **kwargs)
    monkeypatch.setattr(imaging, "open", opener, raising=False)
    serial = h.outcome(lambda d: imaging._median_reference(frames, lambda: None))
    with h.pool_conditions(h.Case(workers, None, "fresh" if workers == 4 else "reuse", j)):
        pooled = h.outcome(lambda d: imaging._median_explicit(frames, lambda: None, workers=workers))
    monkeypatch.undo()
    assert serial.error_type is PermissionError and Path(frames[j].path).name in serial.message
    h.assert_like_serial(serial, pooled)
    for path in paths:
        path.unlink()


@pytest.mark.parametrize("k", h.failure_positions(4))
def test_a_job_failing_in_the_median_fails_and_resumes_like_the_serial_job(tmp_path, monkeypatch, k):
    """(c) at job level, on a controlled engine (16-bit frames: the integer route, 4 blocks): a read error from
    block k on fails the job with 4 threads exactly as with 1 (job.json's state, stage, error, message,
    checkpoints, transients); resumed, both end as the uninterrupted job, bit for bit."""
    from test_hashing_job_slice import _comparable
    fail = {"on": False}
    real = imaging._read_rows

    def read(handle, frame, plane, start, stop, out):
        if (fail["on"] and Path(frame.path).name == "light_00004.fit" and plane == 1
                and start // imaging.MEDIAN_BLOCK_ROWS >= k):
            raise OSError(f"injected read failure at block {start // imaging.MEDIAN_BLOCK_ROWS}")
        return real(handle, frame, plane, start, stop, out)
    monkeypatch.setattr(imaging, "_read_rows", read)
    jobs = data.failing_jobs(tmp_path, monkeypatch, fail)
    assert jobs["uninterrupted"]["state"] == "succeeded", jobs["uninterrupted"].get("error")
    assert data.pool_workers(jobs["uninterrupted"], "median") == [4]
    serial, parallel = data.job_outcome(jobs["serial"]), data.job_outcome(jobs["parallel"])
    assert serial["state"] == "failed" and serial["error"] == f"injected read failure at block {k}"
    assert parallel == serial
    pools = jobs["pools"]
    assert ("median", 4, 4) in pools["uninterrupted"] and ("median", 4, 4) in pools["parallel"]
    assert ("median", 1, None) in pools["serial"]
    expected = _comparable(jobs["uninterrupted"])
    for name in ("serial-resumed", "parallel-resumed"):
        assert jobs[name]["state"] == "succeeded", (name, jobs[name].get("error"))
        assert _comparable(jobs[name]) == expected, name


# ================================================================== (d) cancellation

@pytest.mark.parametrize("workers", [1, 4, 8, 16])
def test_a_submission_time_cancel(synthetic, workers):
    paths, _ = synthetic
    switch = h.CancelSwitch().trip_at_call(2)
    h.assert_cancels_in_time(lambda check: imaging._median_explicit(frames_of(paths), check, workers=workers), switch)


# ================================================================== (f) memory and the worker decision

GIB = 1 << 30


@pytest.mark.parametrize("available,expected", [(None, 1), (1.0, 1), (1.2, 2), (1.25, 3), (1.32, 4), (4.0, 4), (64.0, 4)])
def test_the_worker_decision_follows_memory(monkeypatch, available, expected):
    """72.6 MB per thread at n = 48, W = 8280 (no BLAS slot): four threads from about 1.3 GiB (1.4 GB) available
    (prereg §3.1: "about 1.3")."""
    monkeypatch.setattr(performance, "_memory_status",
                        lambda: (None, None) if available is None else (int(available * GIB), int(available * GIB)))
    monkeypatch.setattr(frame_pool, "usable_cpus", lambda: 16)
    # measured per granted thread (whole call, state 0's network included; docstring of median_unit_bytes) <= declared
    assert 62.4 * 2**20 <= imaging.median_unit_bytes(48, 8280) <= 75e6
    assert 36.3 * 2**20 <= imaging.median_unit_bytes(16, 8280)   # integrated tree, 1 thread
    workers, bound = imaging._median_workers(48, 8280, 87)
    assert workers == expected, bound
    assert bound == ("memory-unknown" if available is None else "memory" if expected < 4 else "cap")


def test_the_worker_decision_limits_open_files_and_takes_the_override(monkeypatch):
    monkeypatch.setattr(performance, "_memory_status", lambda: (64 * GIB, 64 * GIB))
    monkeypatch.setattr(frame_pool, "usable_cpus", lambda: 32)
    assert imaging._median_workers(300, 8280, 87) == (3, "handles")
    assert imaging._median_workers(600, 8280, 87) == (1, "handles")
    assert imaging._median_workers(2000, 8280, 87) == (1, "handles")
    monkeypatch.setenv(frame_pool.ENV_WORKERS, "max")
    assert imaging._median_workers(48, 8280, 87) == (16, "override")
    assert imaging._median_workers(100, 8280, 87) == (10, "handles")
    monkeypatch.setenv(frame_pool.ENV_WORKERS, "2")
    assert imaging._median_workers(48, 8280, 87) == (2, "override")
    monkeypatch.setenv(frame_pool.ENV_WORKERS, "1")
    assert imaging._median_workers(48, 8280, 87) == (1, "override")


def test_explicit_workers_are_limited_by_open_files(tmp_path):
    paths = synthetic_paths(tmp_path, 31, n=9)
    seen = {}
    real = frame_pool.run_ordered

    def spy(*args, **kwargs):
        seen.update(workers=kwargs["workers"], stats=kwargs["stats"])
        return real(*args, **kwargs)
    with _patched(imaging, "MEDIAN_OPEN_FILES", 30), _patched(frame_pool, "run_ordered", spy):
        expected = imaging._median_reference(frames_of(paths), lambda: None)
        h.assert_same(expected, imaging._median_explicit(frames_of(paths), lambda: None, workers=16))
    assert seen["workers"] == 3 and seen["stats"]["bound"] == "handles"


# ================================================================== (b) real data: the F48 DNGs re-converted

@pytest.fixture(scope="module")
def real16(tmp_path_factory):
    frames = data.reconverted(tmp_path_factory, 16)
    return frames, imaging._median_reference(frames_of(frames), lambda: None)


@pytest.mark.parametrize("case", h.params(h.real_cases()))
def test_real_16_frames_match_the_reference(real16, case):
    paths, expected = real16
    stats = {}
    real = frame_pool.run_ordered

    def spy(*args, **kwargs):
        try:
            return real(*args, **kwargs)
        finally:
            stats.update(kwargs.get("stats") or {})
    with h.pool_conditions(case), _patched(frame_pool, "run_ordered", spy):
        started = time.perf_counter()
        actual = imaging.median_foreground(paths, lambda: None, workers=case.workers)
        seconds = time.perf_counter() - started
    h.assert_same(expected, actual)
    print(f"real 16 frames, workers {case.workers}: {stats.get('workers')} threads, {seconds:.2f} s")


def test_real_48_frames_give_the_f48_foreground_master(tmp_path_factory, tmp_path):
    """The 48-frame median with the site's own decision, written by write_linear in a ledger scope: the F48 job's
    foreground master byte for byte (its foreground checkpoint digest)."""
    import hashlib
    paths = data.reconverted(tmp_path_factory, 48)
    with h.pool_conditions(h.Case(None, None, "reuse", None), audit=True):
        ground = imaging.median_foreground(paths, lambda: None, workers=None)
    target = tmp_path / "foreground_master.fit"
    with hashing.run_scope("c1-real", mode="on"):
        imaging.write_linear(target, ground, data.FOREGROUND_NOTE)
    assert hashlib.sha256(target.read_bytes()).hexdigest() == data.foreground_sha256()


@pytest.mark.parametrize("workers", [None, 8, 16])
def test_real_size_cancel_within_a_second(real16, workers):
    paths, _ = real16
    switch = h.CancelSwitch().trip_after(0.4)
    with h.pool_conditions(h.Case(workers, None, "reuse", None)):
        latency = h.assert_cancels_in_time(
            lambda check: imaging.median_foreground(paths, check, workers=workers), switch)
    print(f"real-size cancel, workers {workers}: {latency:.3f} s")


@pytest.mark.skipif(os.name != "nt", reason="private bytes are read on Windows")
@pytest.mark.parametrize("workers", [1, 4, 16])
def test_real_size_pool_memory_within_unit_bytes(real16, workers):
    """(f): in flight 2 x workers, the first block finishing last. Sampled around the whole _median_explicit call,
    so that state 0's network (built before the pool, then the first thread's state) counts, less the output array
    the caller receives, after a collection (garbage freed inside the window would offset the pool's rise). The
    pool's memory is at least its threads' raw blocks (a run that measures nothing fails) and at most granted x
    unit_bytes (no BLAS slot)."""
    import gc
    paths, expected = real16
    frames = frames_of(paths)
    n, width = len(frames), frames[0].shape[2]
    stats = {}
    real = frame_pool.run_ordered

    def spy(*args, **kwargs):
        try:
            return real(*args, **kwargs)
        finally:
            stats.update(kwargs.get("stats") or {})
    case = h.Case(workers, 2 * workers, "reuse", 0)
    with h.pool_conditions(case, head_last=True, head_seconds=0.3), _patched(frame_pool, "run_ordered", spy):
        gc.collect()
        with h.PeakSampler(trace=False) as peak:
            actual = imaging._median_explicit(frames, lambda: None, workers=workers, inflight=2 * workers)
    h.assert_same(expected, actual)
    rise = peak.rise - actual.nbytes
    granted = stats["workers"]
    limit = h.pool_memory_limit(stats, blas=False)
    raw = granted * n * imaging.MEDIAN_BLOCK_ROWS * width * 2
    print(f"real 16-frame median pool, {granted} threads: rise {rise / 2**20:.1f} MiB (output excluded), raw blocks "
          f"{raw / 2**20:.1f} MiB, limit {limit / 2**20:.1f} MiB, per thread {rise / granted / 2**20:.1f} MiB, "
          f"unit_bytes {imaging.median_unit_bytes(n, width) / 2**20:.1f} MiB")
    assert raw <= rise <= limit, (rise, raw, limit)
