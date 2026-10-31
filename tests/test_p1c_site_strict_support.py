"""Efficiency P1c F-2 (docs/EFFICIENCY_P1C_PREREG_20261005.md §3.2, §6.2 A5): strict-support exports on the frame pool.

registered_support.prepare_exports against its kept serial body registered_support._prepare_exports_reference:
(a) the synthetic matrix (workers x in flight x state policy x jitter seed) and R11's in flight = every frame, in four
    scenarios (F-like: SIP, static mask, ledger on, retained maps; strict L-like: no SIP, per-frame masks, outside a
    hash scope, no retained maps; ledger shadow; the reference kernel route), files by bytes, the record with its key
    order, the progress calls; PYTHONHASHSEED reruns of both sides;
(b)/(d)/(f) real size on a copy of the F48 reference job's exports (tests needing it skip without the fixture):
    workers {1, default, 16}, cancellation within 1 s (also during a per-frame mask decode in prepare), the pool's
    memory against its declared bound with the in-flight window full;
(c) injected failures at frames {0, 1, middle, last} in prepare, the kernel, the zeroing and the .npy write: the serial
    error, a delivered prefix no longer than the serial one, the changed files the serial ones plus only the pre-
    registered extras (later exports zeroed, later maps written), no record, then a clean retry equal to an
    uninterrupted run;
(e) every pool run under the context audit; (g) the numeric variants; plus the R16 checkpoint list, the SIP failure
    memo, the volume cap and the worker decision of §6.2 (f); at job level (a strict L job on a controlled engine, the
    CLI/MCP strict L path) the same products at 1 / 3 / 16 workers, a failing export unit failing the job as the serial
    run does, and a resume that drops F-2's sidecar temporaries and ends like an uninterrupted job;
(b) through scripts/replay_support.py --workers, read-only on the whole F48 job.
"""
from __future__ import annotations

import contextlib
import gc
import json
import os
import random
import shutil
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

import p1c_site_harness as h
from starstack import (core, frame_pool, hashing, performance, registered_support as rs, resources,
                       siril_support_map as sm, sky_validity as sv, support_kernel as sk)
from starstack.registration_distortion import RadialDistortion
from test_registered_support_independent import _seq, _write_rgb
from test_strict_support_kernel import _sip_header

HEIGHT, WIDTH, FRAMES = 43, 67, 9
GB = 10 ** 9
# name -> (SIP export, per-frame masks, hash scope mode or None, retained maps, AESTER_STRICT_KERNEL)
SCENARIOS = {
    "f": (True, False, "on", True, "buffered"),
    "sl": (False, True, None, False, "buffered"),
    "shadow": (True, True, "shadow", True, "buffered"),
    "reference_kernel": (True, False, "on", True, "reference"),
}


# ------------------------------------------------------------------------------------------------ worlds

def build_world(root, name) -> Path:
    """The scenario's work folder template (exports, export sequence, distortion.wcs): deterministic per name."""
    root = Path(root)
    if root.exists():
        return root
    sip = SCENARIOS[name][0]
    rng = np.random.default_rng(sum(map(ord, name)))
    root.mkdir(parents=True)
    matrices = [np.array([[1 + rng.normal(0, .01), rng.normal(0, .01), rng.normal(0, 2)],
                          [rng.normal(0, .01), 1 + rng.normal(0, .01), rng.normal(0, 2)],
                          [rng.normal(0, 1e-5), rng.normal(0, 1e-5), 1.]]) for _ in range(FRAMES)]
    _seq(root / rs.EXPORT_SEQUENCE, matrices)
    if sip:
        with (root / rs.EXPORT_SEQUENCE).open("a", encoding="utf-8") as stream:
            stream.write("D1 2 distortion.wcs\n")
        fits.PrimaryHDU(header=fits.Header(_sip_header(rng, WIDTH, HEIGHT, 3, 1e-3))).writeto(root / "distortion.wcs")
    for number in range(1, FRAMES + 1):
        _write_rgb(root / f"r_light_{number:05d}.fit", rng.random((HEIGHT, WIDTH)) + .1)
    return root


class _Static:
    def __init__(self, v):
        self.v = v

    def for_frame(self, stem):
        return self.v


class _PerFrame:
    def __init__(self, table):
        self.table = table

    def for_frame(self, stem):
        return self.table[stem]


def source_for(name):
    """A fresh validity source (fresh arrays, same values): frames 1, 4, 7 share the static array, the rest have
    their own, so both the anchor cache and the digest rule see repeats and changes."""
    v = np.ones((HEIGHT, WIDTH), bool)
    v[int(HEIGHT * .7):, :] = False
    if not SCENARIOS[name][1]:
        return _Static(v)
    rng = np.random.default_rng(5)
    table = {}
    for number in range(1, FRAMES + 1):
        own = v & (rng.random((HEIGHT, WIDTH)) > .03)
        table[f"light_{number:05d}"] = v if number % 3 == 1 else own
    return _PerFrame(table)


def model_for(name):
    return RadialDistortion(.01, -.002, WIDTH / 2, HEIGHT / 2, max(WIDTH, HEIGHT) / 2, .99, "u_to_p") \
        if SCENARIOS[name][0] else None


@contextlib.contextmanager
def scenario_environment(name):
    _, _, scope, _, kernel = SCENARIOS[name]
    saved = os.environ.get("AESTER_STRICT_KERNEL")
    if kernel == "reference":
        os.environ["AESTER_STRICT_KERNEL"] = "reference"
    else:
        os.environ.pop("AESTER_STRICT_KERNEL", None)
    try:
        with (hashing.run_scope("p1c-strict", mode=scope) if scope else contextlib.nullcontext()):
            yield
    finally:
        if saved is None:
            os.environ.pop("AESTER_STRICT_KERNEL", None)
        else:
            os.environ["AESTER_STRICT_KERNEL"] = saved


def strip(record):
    return {key: value for key, value in record.items() if key != "filter_seconds"}


def files(work: Path) -> dict:
    return {name: data for name, data in h.tree_bytes(work).items() if name != rs.RECORD}


def run_site(name, template, work, *, workers=1, reference=False, check=None, pool_stats=None, unit_bytes=None,
             source=None):
    """Copy the template to ``work`` and run one side; (record without its timing, files, progress calls)."""
    shutil.copytree(template, work)
    progress = []
    source = source_for(name) if source is None else source
    retain = SCENARIOS[name][3]
    with scenario_environment(name):
        if reference:
            record = rs._prepare_exports_reference(work, source, model_for(name), check=check,
                                                   progress=lambda *a: progress.append(a),
                                                   retain_frame_support=retain)
        else:
            record = rs.prepare_exports(work, source, model_for(name), check=check,
                                        progress=lambda *a: progress.append(a), retain_frame_support=retain,
                                        workers=workers, pool_stats=pool_stats, unit_bytes=unit_bytes)
    assert json.loads((work / rs.RECORD).read_text(encoding="utf-8"))["frames"] == record["frames"]
    return strip(record), files(work), progress


@contextlib.contextmanager
def forced_inflight(inflight):
    """Run the site's pool with this many units in flight (the site itself asks for one per worker)."""
    real = frame_pool.run_ordered

    def run(*args, **kwargs):
        if inflight is not None:
            kwargs["inflight"] = inflight
        return real(*args, **kwargs)
    frame_pool.run_ordered = run
    try:
        yield
    finally:
        frame_pool.run_ordered = real


def hashseed_reference(name, root):
    """For the PYTHONHASHSEED and numeric reruns: the serial body in a fresh interpreter."""
    template = build_world(Path(root) / f"template-{name}", name)
    return run_site(name, template, Path(root) / f"reference-{os.getpid()}-{time.time_ns()}", reference=True)


def hashseed_pool(name, root):
    template = build_world(Path(root) / f"template-{name}", name)
    with h.pool_conditions(h.Case(8, None, "reuse", 3)):
        return run_site(name, template, Path(root) / f"pool-{os.getpid()}-{time.time_ns()}", workers=8)


@pytest.fixture(scope="module")
def worlds(tmp_path_factory):
    """name -> (template, the reference side's (record, files, progress))."""
    root = tmp_path_factory.mktemp("strict-worlds")
    sm.release_distortion_grid()
    out = {}
    for name in SCENARIOS:
        template = build_world(root / f"template-{name}", name)
        out[name] = (template, run_site(name, template, root / f"reference-{name}", reference=True))
    return out


@pytest.fixture(autouse=True)
def clean_knobs(monkeypatch):
    for knob in ("AESTER_FRAME_WORKERS", "AESTER_STRICT_KERNEL"):
        monkeypatch.delenv(knob, raising=False)
    yield
    assert sm._distortion_grid.cache_info().currsize == 0 and not sm._GRID_FAILED


def _matrix():
    full = h.synthetic_cases()
    small = h.synthetic_cases(workers=(1, 3, 16), seeds=(0,))
    return ([pytest.param("f", case, id=f"f-{case.id}") for case in full]
            + [pytest.param(name, case, id=f"{name}-{case.id}") for name in ("sl", "shadow", "reference_kernel")
               for case in small])


# ================================================================== (a) synthetic

@pytest.mark.parametrize("name,case", _matrix())
def test_pool_matches_the_serial_body(worlds, tmp_path, name, case):
    template, expected = worlds[name]
    before = h.aester_threads()
    pool_stats = {"decision": {"bound": "test"}}
    with h.pool_conditions(case), forced_inflight(case.inflight):
        actual = run_site(name, template, tmp_path / "work", workers=case.workers, pool_stats=pool_stats,
                          unit_bytes=1 << 20)
    h.assert_same(expected, actual)
    assert pool_stats["decision"] == {"bound": "test"} and list(pool_stats)[0] == "decision"   # notes only add keys
    assert pool_stats["workers"] == min(case.workers, FRAMES) and pool_stats["unit_bytes"] == 1 << 20
    assert pool_stats["serial"] is (case.workers == 1)
    expected_implementation = rs.REFERENCE_IMPLEMENTATION if name == "reference_kernel" else rs.SUPPORT_IMPLEMENTATION
    assert actual[0]["implementation"] == expected_implementation
    h.assert_pool_quiet(before)


@pytest.mark.parametrize("name", list(SCENARIOS))
@pytest.mark.parametrize("policy", ["reuse", "fresh"])
def test_prepare_depends_only_on_earlier_frames(worlds, tmp_path, name, policy):
    """R11: every frame in flight at once (prepare runs as far ahead as it can)."""
    template, expected = worlds[name]
    with h.pool_conditions(h.Case(4, FRAMES, policy, 1)), forced_inflight(FRAMES):
        actual = run_site(name, template, tmp_path / "work", workers=4)
    h.assert_same(expected, actual)


def test_the_default_serial_path_keeps_the_former_calls(worlds, tmp_path):
    """workers=1 calls check() exactly as the serial body does (R18: the extra probes are the parallel path's only),
    and reports progress 1..n with the same stems."""
    template, _ = worlds["f"]
    calls = {"reference": 0, "pool": 0}

    def counter(side):
        def check():
            calls[side] += 1
        return check
    reference = run_site("f", template, tmp_path / "ref", reference=True, check=counter("reference"))
    pooled = run_site("f", template, tmp_path / "pool", workers=1, check=counter("pool"))
    h.assert_same(reference, pooled)
    assert calls["pool"] == calls["reference"] > FRAMES
    assert [p[0] for p in pooled[2]] == list(range(1, FRAMES + 1))


def test_the_parallel_path_probes_the_digests_and_the_grid_wait(worlds, tmp_path, monkeypatch):
    """On the parallel path the export, map and read-back digests and the SIP grid wait get the unit's probe."""
    template, expected = worlds["sl"]
    seen = {"sha": [], "grid": []}
    real_sha, real_grid = rs.sha256, sm.distortion_grid

    def sha(path, check=None):
        seen["sha"].append(check is not None)
        return real_sha(path, check)

    def grid(width, height, spec, *, check=None):
        seen["grid"].append(check is not None)
        return real_grid(width, height, spec, check=check)
    monkeypatch.setattr(rs, "sha256", sha)
    monkeypatch.setattr(sk, "distortion_grid", grid)
    actual = run_site("sl", template, tmp_path / "pool", workers=4)
    h.assert_same(expected, actual)
    # outside a hash scope: the digest before and after each export; then the record's sequence and counts digests
    assert seen["sha"].count(True) == 2 * FRAMES and seen["sha"].count(False) == 2
    seen["sha"].clear()
    template_f, expected_f = worlds["f"]
    actual = run_site("f", template_f, tmp_path / "pool-f", workers=4)
    h.assert_same(expected_f, actual)
    # ledger on: the export after (a hit), the map's read-back (a hit); then the record's sequence, wcs and counts
    assert seen["sha"].count(True) == 2 * FRAMES and seen["sha"].count(False) == 3
    assert seen["grid"] and all(seen["grid"])
    seen["grid"].clear()
    run_site("f", template_f, tmp_path / "serial-f", workers=1)
    assert seen["grid"] and not any(seen["grid"])


def test_reruns_under_two_hash_seeds_give_one_digest(tmp_path):
    for name in ("f", "sl"):
        h.assert_hashseed_stable("test_p1c_site_strict_support:hashseed_reference",
                                 "test_p1c_site_strict_support:hashseed_pool", name, str(tmp_path))


def test_numeric_variants_give_the_same_result(worlds, tmp_path):
    """(g): the pooled site under each numeric environment equals the in-process serial body."""
    _, expected = worlds["f"]
    wanted = h.canonical_digest(expected)
    for env in h.NUMERIC_ENVS:
        assert h.subprocess_digest("test_p1c_site_strict_support:hashseed_pool", "f", str(tmp_path), env=env) \
            == wanted, env


# ================================================================== (c) failure

STAGES = ("prepare", "kernel", "zero", "save")


def _inject(monkeypatch, stage, stem):
    """Make ``stage`` of frame ``stem`` raise ValueError (thread-safe)."""
    error = lambda key: ValueError(f"injected {stage} failure at {key}")
    if stage == "prepare":
        original = source_for

        def failing(name):
            source = original(name)
            for_frame = source.for_frame
            source.for_frame = h.FailOn(for_frame, at={stem}, error=error, key=lambda s: s)
            return source
        return failing
    if stage == "kernel":
        monkeypatch.setattr(sk, "registered_mask_kernel", h.FailOn(
            sk.registered_mask_kernel, at={stem}, error=error,
            key=lambda valid, h_stored, **kw: _stem_of_matrix(h_stored)))
    elif stage == "zero":
        monkeypatch.setattr(sv, "apply_validity_with_digests", h.FailOn(
            sv.apply_validity_with_digests, at={stem}, error=error, key=lambda path, *a, **kw: Path(path).stem[2:]))
    else:
        monkeypatch.setattr(rs, "save_array", h.FailOn(
            rs.save_array, at={stem}, error=error,
            key=lambda path, *a, **kw: Path(path).stem.removeprefix("frame_support_")))
    return source_for


_MATRIX_STEMS = {}


def _stem_of_matrix(matrix):
    return _MATRIX_STEMS[np.asarray(matrix, dtype=np.float64).tobytes()]


def _remember_matrices(template):
    for stem, matrix in rs.export_transforms(template / rs.EXPORT_SEQUENCE).items():
        _MATRIX_STEMS[matrix.tobytes()] = stem


def _failed_run(name, template, work, workers, make_source, reference=False):
    shutil.copytree(template, work)
    before = h.tree_state(work)
    progress = []
    source = make_source(name)
    with scenario_environment(name):
        if reference:
            run = lambda delivered: rs._prepare_exports_reference(
                work, source, model_for(name), progress=lambda *a: delivered.append(a[0]),
                retain_frame_support=SCENARIOS[name][3])
        else:
            run = lambda delivered: rs.prepare_exports(
                work, source, model_for(name), progress=lambda *a: delivered.append(a[0]),
                retain_frame_support=SCENARIOS[name][3], workers=workers)
        result = h.outcome(run)
    return result, h.changed_files(before, h.tree_state(work))


@pytest.mark.parametrize("stage", STAGES)
@pytest.mark.parametrize("k", h.failure_positions(FRAMES))
def test_a_failure_ends_like_the_serial_body_and_a_retry_recovers(worlds, tmp_path, monkeypatch, stage, k):
    template, expected = worlds["f"]
    _remember_matrices(template)
    stem = f"light_{k + 1:05d}"
    make_source = _inject(monkeypatch, stage, stem)
    serial, serial_files = _failed_run("f", template, tmp_path / "serial", 1, make_source, reference=True)
    assert serial.error_type is ValueError and stem in serial.message and serial.delivered == list(range(1, k + 2))
    assert not (tmp_path / "serial" / rs.RECORD).exists()
    later = {f"r_light_{n:05d}.fit" for n in range(k + 2, FRAMES + 1)}
    later_maps = {f"frame_support_light_{n:05d}.npy" for n in range(k + 2, FRAMES + 1)}
    for workers in (1, 2, 4, 16):
        before = h.aester_threads()
        work = tmp_path / f"w{workers}"
        with h.pool_conditions(h.Case(workers, None, "reuse", k)):
            parallel, changed = _failed_run("f", template, work, workers, make_source)
        h.assert_like_serial(serial, parallel)
        assert serial.delivered == parallel.delivered               # frames before k are always delivered
        assert not changed["deleted"] and not (work / rs.RECORD).exists() and not (work / rs.COUNTS).exists()
        assert set(serial_files["created"]) <= set(changed["created"]) <= set(serial_files["created"]) | later_maps
        assert set(serial_files["modified"]) <= set(changed["modified"]) <= set(serial_files["modified"]) | later
        if workers == 1:
            assert changed == serial_files
        h.assert_pool_quiet(before)
        assert sm._distortion_grid.cache_info().currsize == 0       # released after the pool finalized its kernels
        # the job's retry: Siril re-exports, the attempt deletes the maps and temporaries; then an uninterrupted run
        for stale in sorted(work.glob("frame_support_light_*.npy*")):
            stale.unlink()
        for export in sorted(template.glob("r_light_*.fit")):
            shutil.copyfile(export, work / export.name)
        monkeypatch.undo()
        with h.pool_conditions(h.Case(workers, None, "reuse", k)):
            retried = _rerun("f", work, workers)
        h.assert_same(expected, retried)
        make_source = _inject(monkeypatch, stage, stem)


def _rerun(name, work, workers):
    progress = []
    with scenario_environment(name):
        record = rs.prepare_exports(work, source_for(name), model_for(name), progress=lambda *a: progress.append(a),
                                    retain_frame_support=SCENARIOS[name][3], workers=workers)
    return strip(record), files(work), progress


def test_a_failing_sip_grid_is_built_once_and_fails_every_frame_alike(tmp_path, monkeypatch):
    """A non-finite SIP grid: the serial error; with 8 workers the grid is built once (the memo), not per worker."""
    root = build_world(tmp_path / "template", "f")
    header = {"CTYPE1": "RA---TAN-SIP", "CTYPE2": "DEC--TAN-SIP", "CRPIX1": 10., "CRPIX2": 10.,
              "AP_ORDER": 5, "BP_ORDER": 0, "AP_5_0": 1e300}
    (root / "distortion.wcs").unlink()
    fits.PrimaryHDU(header=fits.Header(header)).writeto(root / "distortion.wcs")
    builds = []
    real = sm._distortion_grid

    def counted(width, height, spec):
        builds.append(1)
        return real(width, height, spec)              # lru_cache keeps no failure: every call here is a build
    counted.cache_clear, counted.cache_info = real.cache_clear, real.cache_info
    monkeypatch.setattr(sm, "_distortion_grid", counted)
    serial, _ = _failed_run("f", root, tmp_path / "serial", 1, source_for, reference=True)
    assert serial.error_type is ValueError and serial.message == "SIP source grid is not finite in float32"
    serial_builds = len(builds)
    builds.clear()
    with h.pool_conditions(h.Case(8, None, "reuse", 0)):
        parallel, _ = _failed_run("f", root, tmp_path / "pool", 8, source_for)
    h.assert_like_serial(serial, parallel)
    assert serial_builds == 1 and len(builds) == 1
    assert not sm._GRID_FAILED                                          # released with the grid


def test_the_grid_memo_lasts_until_release_and_the_probe_runs_before_the_lock():
    spec = sm._sip_spec({"CTYPE1": "RA---TAN-SIP", "CTYPE2": "DEC--TAN-SIP", "CRPIX1": 10., "CRPIX2": 10.,
                         "AP_ORDER": 5, "BP_ORDER": 0, "AP_5_0": 1e300})
    for _ in range(2):
        with pytest.raises(ValueError, match="not finite"):
            sm.distortion_grid(30, 24, spec)
    assert (30, 24, spec) in sm._GRID_FAILED
    order = []

    class Stop(Exception):
        pass

    def probe():
        order.append(sm._GRID_LOCK.locked())
        raise Stop()
    with sm._GRID_LOCK:                                   # another worker is building: the probe still runs
        with pytest.raises(Stop):
            sm.distortion_grid(30, 24, spec, check=probe)
    assert order == [True]
    sm.release_distortion_grid()
    assert not sm._GRID_FAILED


# ================================================================== (d) cancellation (synthetic part)

@pytest.mark.parametrize("call", [0, 1, 4])           # check() runs before each of the 9 submissions
@pytest.mark.parametrize("workers", [1, 4, 16])
def test_a_cancel_at_a_submission_publishes_nothing(worlds, tmp_path, call, workers):
    template, _ = worlds["f"]
    shutil.copytree(template, tmp_path / "work")
    switch = h.CancelSwitch().trip_at_call(call)
    with h.pool_conditions(h.Case(workers, None, "reuse", call)), scenario_environment("f"):
        h.assert_cancels_in_time(lambda check: rs.prepare_exports(
            tmp_path / "work", source_for("f"), model_for("f"), check=check, retain_frame_support=True,
            workers=workers), switch)
    assert not (tmp_path / "work" / rs.RECORD).exists() and not (tmp_path / "work" / rs.COUNTS).exists()


# ================================================================== core: R16, retry clean-up, decision, volume cap

def test_r16_the_f_checkpoint_lists_the_maps_by_name_whatever_the_listing_order(tmp_path, monkeypatch):
    work = tmp_path / "job" / "work"
    work.mkdir(parents=True)
    for number in (3, 1, 2, 10):
        (work / f"frame_support_light_{number:05d}.npy").write_bytes(bytes([number]))
    from starstack import finite_stack as fs
    for name in (fs.NVALID, fs.NUSED):
        (work / name).write_bytes(name.encode())
    expected = core._checkpoint_records(tmp_path / "job", core._finite_stack_files(work))
    real_glob = Path.glob
    for seed in range(5):
        def shuffled(self, pattern, **kwargs):
            found = list(real_glob(self, pattern, **kwargs))
            random.Random(seed).shuffle(found)
            return iter(found)
        monkeypatch.setattr(Path, "glob", shuffled)
        assert core._checkpoint_records(tmp_path / "job", core._finite_stack_files(work)) == expected
    names = [record["path"] for record in expected]
    assert names[2:] == sorted(names[2:]) and len(names) == 6


@pytest.mark.parametrize("available_gb,pixels,expected", [
    (10, 5520 * 8280, 8), (6, 5520 * 8280, 6), (2.3, 5520 * 8280, 2), (2.1, 5520 * 8280, 2), (1.7, 5520 * 8280, 1),
    (10, 102_000_000, 5), (6, 102_000_000, 3), (2.3, 102_000_000, 1), (2.1, 102_000_000, 1)])
def test_the_worker_decision_follows_memory_and_frame_size(monkeypatch, tmp_path, available_gb, pixels, expected):
    """§6.2 (f) with the measured unit figure (256 MiB + the BLAS slot per worker at 45.7 MP; the prereg's 8 / 5 / 2 / 1
    at 10 / 6 / 2.3 / 2.1 GB assumed the provisional 320 MiB): memory bounds the pool as min(avail / 3, avail - 1 GiB)
    / per worker, scaled by the frame's pixels. The cap is held at 8 here to show the memory curve."""
    available = int(available_gb * GB)
    monkeypatch.setattr(performance, "_memory_status", lambda: (available, available * 2))
    monkeypatch.setattr(frame_pool, "usable_cpus", lambda: 64)
    monkeypatch.setattr(resources, "drive_type", lambda path: "fixed")
    monkeypatch.setattr(rs, "STRICT_WORKER_CAP", 8)
    width = 8280 if pixels == 5520 * 8280 else 12000
    height = pixels // width
    decision = {}
    workers, unit_bytes = core._strict_support_workers(tmp_path, {}, height, width, 48, decision)
    assert workers == expected, decision
    assert unit_bytes == max(rs.STRICT_UNIT_BYTES, -(-rs.STRICT_UNIT_BYTES * height * width // rs.STRICT_UNIT_PIXELS))
    assert decision["workers"] == expected and decision["blas"] is True and decision["volume"] == "fixed"
    assert decision["bound"] == ("cap" if expected == 8 else "memory")


def test_the_measured_cap_bounds_the_pool_when_memory_is_ample(monkeypatch, tmp_path):
    monkeypatch.setattr(performance, "_memory_status", lambda: (100 * GB, 100 * GB))
    monkeypatch.setattr(frame_pool, "usable_cpus", lambda: 64)
    monkeypatch.setattr(resources, "drive_type", lambda path: "fixed")
    decision = {}
    assert core._strict_support_workers(tmp_path, {}, 5520, 8280, 48, decision)[0] == rs.STRICT_WORKER_CAP
    assert decision["bound"] == "cap"
    monkeypatch.setattr(frame_pool, "usable_cpus", lambda: 6)              # half the CPUs bound it below the cap
    decision = {}
    assert core._strict_support_workers(tmp_path, {}, 5520, 8280, 48, decision)[0] == min(3, rs.STRICT_WORKER_CAP)
    assert decision["bound"] == "cpu"


def test_two_workers_need_about_1_8_gb_and_2_7_gb_with_per_frame_masks(monkeypatch, tmp_path):
    """The prereg's "about 2.2 GB, 2.8 GB with per-frame masks" for two workers, with the measured unit figure: 1.81 GB,
    and 2.68 GB with two own masks (each worker also holds a frame's anchor grid (R17); prepare's decode peak is 11 B/px
    (0.50 GB) and the source caches each own mask (1 B/px))."""
    monkeypatch.setattr(frame_pool, "usable_cpus", lambda: 64)
    monkeypatch.setattr(resources, "drive_type", lambda path: "fixed")
    job = {"frame_masks": {"files": {"light_00002": {}, "light_00005": {}}}}

    def workers(available, job_):
        monkeypatch.setattr(performance, "_memory_status", lambda: (int(available * GB), None))
        return core._strict_support_workers(tmp_path, job_, 5520, 8280, 48, {})[0]
    plain = min(x / 100 for x in range(100, 400) if workers(x / 100, {}) >= 2)
    masked = min(x / 100 for x in range(100, 400) if workers(x / 100, job) >= 2)
    pixels = 5520 * 8280
    per_worker = rs.STRICT_UNIT_BYTES + frame_pool.BLAS_SLOT_BYTES
    assert plain == pytest.approx(3 * 2 * per_worker / GB, abs=.011)
    extra = rs.STRICT_MASK_DECODE_BYTES_PER_PIXEL * pixels + 2 * pixels
    assert masked == pytest.approx((3 * 2 * (per_worker + pixels) + extra) / GB, abs=.011)
    assert 1.75 <= plain <= 1.85 and 2.6 <= masked <= 2.8, (plain, masked)
    decision = {}
    core._strict_support_workers(tmp_path, job, 5520, 8280, 48, decision)
    assert decision["own_masks"] == 2 and decision["extra_bytes"] == extra


@pytest.mark.parametrize("kind,cap", [("removable", 2), ("remote", 2), ("fixed", None), ("unknown", None)])
def test_a_removable_or_network_work_folder_caps_the_pool_at_two(monkeypatch, tmp_path, kind, cap):
    monkeypatch.setattr(performance, "_memory_status", lambda: (100 * GB, 100 * GB))
    monkeypatch.setattr(frame_pool, "usable_cpus", lambda: 64)
    monkeypatch.setattr(resources, "drive_type", lambda path: kind)
    decision = {}
    workers, _ = core._strict_support_workers(tmp_path, {}, 5520, 8280, 48, decision)
    assert workers == (cap or rs.STRICT_WORKER_CAP) and decision["volume"] == kind
    assert decision["bound"] == ("volume" if cap else "cap")
    monkeypatch.setenv("AESTER_FRAME_WORKERS", "6")                    # the override still wins (measurements)
    assert core._strict_support_workers(tmp_path, {}, 5520, 8280, 48, {})[0] == 6


def test_drive_type_reads_get_drive_type_only(monkeypatch):
    if os.name != "nt":
        assert resources.drive_type("/tmp/x") == "unknown"
        return
    calls = []
    monkeypatch.setattr(resources, "_drive_type", lambda root: calls.append(root) or 2)
    assert resources.drive_type(r"E:\jobs\work") == "removable" and calls == ["E:\\"]
    monkeypatch.setattr(resources, "_drive_type", lambda root: None)
    assert resources.drive_type(r"C:\x") == "unknown"
    monkeypatch.undo()
    assert resources.drive_type(r"\\server\share\work") == "remote"          # a share: never queried
    assert resources.drive_type(Path.cwd()) in ("fixed", "removable", "remote", "ramdisk", "cdrom", "unknown")


def test_supported_stack_decides_notes_and_shadows_the_pool(monkeypatch, tmp_path):
    """core._supported_stack hands the decision to prepare_exports, runs it in the 'registered_support' operation
    and notes the pool (frame_pool.strict_support) with its decision; nothing reaches a record."""
    from starstack import profiling
    work = tmp_path / "work"
    work.mkdir()
    (work / core.SEQUENCE_FILE).write_text("x", encoding="utf-8")
    (work / "light_00001.fit").write_bytes(b"")
    for number in (1, 2, 3):
        (work / f"r_light_{number:05d}.fit").write_bytes(b"")
    seen = {}

    def prepare_exports(*args, **kwargs):
        seen.update(kwargs)
        kwargs["pool_stats"].update(workers=kwargs["workers"], units=3, serial=kwargs["workers"] == 1)
    monkeypatch.setattr(rs, "prepare_exports", prepare_exports)
    monkeypatch.setattr(core, "_validate_frame", lambda *a: (3, 5520, 8280))
    monkeypatch.setattr(core, "_validity_source", lambda *a: object())
    monkeypatch.setattr(core, "_grid_model", lambda *a: None)
    monkeypatch.setattr(core, "_update", lambda *a, **k: None)
    monkeypatch.setenv("AESTER_FRAME_WORKERS", "3")
    timer = profiling.StageTimer()
    core._supported_stack("siril", tmp_path, work, {}, {}, False, {"dilation_px": 3}, lambda: None,
                          runner=lambda *a, **k: None, timer=timer)
    report = timer.finish()
    assert seen["workers"] == 3 and seen["unit_bytes"] == rs.STRICT_UNIT_BYTES and "retain_frame_support" not in seen
    note = report["notes"]["frame_pool.strict_support"]
    assert len(note) == 1 and note[0]["workers"] == 3 and note[0]["decision"]["bound"] == "override"
    assert report["operation_counts"]["registered_support"] == 1


# ================================================================== job level: a strict L job, controlled engine

JOB_FRAMES = 5


class StrictEngine:
    """Siril's file contract for a strict L job on Siril's own registration (never Siril): convert, the probe, the
    fallback registration's sequence, the strict export (r_<stem>.fit per selected frame) and the contract stack."""

    def __init__(self):
        self.names = []

    def __call__(self, executable, directory, name, commands, cancel_check, **options):
        from starstack import imaging, stack_settings
        import d4_fakes
        from test_core_two_pass_reference import _seq_text
        commands, work = list(commands), Path(directory) / "work"
        self.names.append(name)
        cancel_check()
        if name == "01_convert":
            for i, path in enumerate(sorted((Path(directory) / "stage").glob("*")), 1):
                shutil.copyfile(path, work / f"light_{i:05d}.fit")
        elif name == "02_probe":
            assert commands == core.PROBE_SCRIPT
        elif name == "02_register_fallback":
            (work / core.SEQUENCE_FILE).write_text(_seq_text(JOB_FRAMES, 1), encoding="utf-8")
        elif name == "02_export":
            assert commands == ["cd work", rs.EXPORT_COMMAND]
            selected = [image["stem"] for image in core.read_seq(work / core.SEQUENCE_FILE)["images"]
                        if image["selected"]]
            for stem in selected:
                shutil.copyfile(work / f"{stem}.fit", work / f"r_{stem}.fit")
            (work / "r_light_.seq").write_text(d4_fakes.sequence_text(len(selected), name="r_light_"),
                                               encoding="utf-8")
        elif name == "02_stack":
            plain, _ = stack_settings.without_settings(commands)
            assert plain == ["cd work", "seqclean r_light -stat", core.STACK_COMMAND], commands
            frames = sorted(work.glob("r_light_*.fit"))
            data = np.mean([imaging.read_rgb(p) for p in frames], axis=0).astype(np.float32)
            imaging.write_linear(work / "sky_master.fit", data, "Controlled engine, not Siril",
                                 extra={"STACKCNT": len(frames)})
            d4_fakes.stack_log(directory, name, commands)
        else:
            raise AssertionError(name)


def _strict_job_runner(tmp_path, monkeypatch, *, height=30, own=False):
    """run(label, workers, fail=None, resume=None): a strict L job (CLI/MCP's strict L/R: no retained maps) through
    core.run_job, with AESTER_FRAME_WORKERS=workers; ``fail`` = (stage, stem) makes that frame's kernel or zeroing
    raise. ``own``: the job's own registration (a fitted calibration with a real SIP distortion.wcs): the exports carry
    the SIP term and the stack is warped back through finish_stack's native mask; else Siril's fallback registration."""
    from PIL import Image
    from types import SimpleNamespace
    from starstack import imaging, registration_calibration as rc
    source = tmp_path / "sources"
    source.mkdir()
    rng = np.random.default_rng(8)
    for number in range(JOB_FRAMES):
        data = (.03 + rng.random((height, 40, 3)) * .01).astype(np.float32)
        data[3:6, 5:8] += .2
        imaging.write_linear(source / f"frame {number}.fit", data, "Test fixture")
    mask = np.full((height, 40), 255, dtype=np.uint8)
    edge = round(height * 22 / 30)                          # 22 at the fallback job's 30 rows
    mask[edge, :] = 128
    mask[edge + 1:, :] = 0
    Image.fromarray(mask).save(tmp_path / "sky_mask.png")
    engine = {"available": True, "executable": str(tmp_path / "engine.exe"), "version": "1.4.4", "message": "controlled"}
    monkeypatch.setattr(core, "doctor", lambda *_: dict(engine))
    monkeypatch.setattr(core, "_spawn_worker", lambda directory: None)
    monkeypatch.setattr(core.shutil, "disk_usage", lambda _: SimpleNamespace(free=10**12))
    monkeypatch.setattr(rs, "require_engine", lambda info: None)

    def unavailable(work, **kwargs):
        summary = {"status": "unavailable", "reason": "test: Siril's own registration"}
        (Path(work) / rc.DISTORTION_JSON).write_text(json.dumps(summary), encoding="utf-8")
        return summary

    def fitted(work, **kwargs):
        from test_core_two_pass_reference import JobDiagnosticsTest
        from test_strict_support_kernel import _sip_header
        stems = {f"light_{i:05d}" for i in range(1, JOB_FRAMES + 1)}
        summary = JobDiagnosticsTest._fake_calibration(work, stems, count=JOB_FRAMES)
        summary["parameters"] = {"k1": 2e-3, "k2": 0.0, "cx": 19.5, "cy": height / 2 - .5, "scale": float(height),
                                 "magnification": .99, "direction": "u_to_p"}
        (Path(work) / rc.DISTORTION_JSON).write_text(json.dumps(summary), encoding="utf-8")
        header = _sip_header(np.random.default_rng(31), 40, height, 3, 1e-3)
        (Path(work) / rc.DISTORTION_WCS).unlink()
        fits.PrimaryHDU(header=fits.Header(header)).writeto(Path(work) / rc.DISTORTION_WCS)
        return summary
    monkeypatch.setattr(rc, "calibrate", fitted if own else unavailable)
    monkeypatch.setattr(core, "run_script", StrictEngine())
    real_kernel, real_zero = sk.registered_mask_kernel, sv.apply_validity_with_digests

    def run(label, workers, fail=None, resume=None):
        monkeypatch.setenv("AESTER_FRAME_WORKERS", str(workers))
        monkeypatch.setattr(sk, "registered_mask_kernel", real_kernel)
        monkeypatch.setattr(sv, "apply_validity_with_digests", real_zero)
        if fail is not None:
            stage, stem = fail
            error = lambda key: ValueError(f"injected {stage} failure at {key}")
            if stage == "kernel":
                monkeypatch.setattr(sk, "registered_mask_kernel", h.FailOn(real_kernel, at={stem}, error=error,
                                                                           key=_job_frame_key))
            else:
                monkeypatch.setattr(sv, "apply_validity_with_digests", h.FailOn(
                    real_zero, at={stem}, error=error, key=lambda path, *a, **kw: Path(path).stem[2:]))
        if resume is None:
            job = core.submit_job(str(source), str(tmp_path / label), sky_mask_path=str(tmp_path / "sky_mask.png"),
                                  support_mode=rs.MODE, transient_mode="off")
            directory = job["job_dir"]
        else:
            directory = resume
            core.resume_job(directory)
        core.run_job(directory)
        return core.load_job(directory)
    return SimpleNamespace(run=run)


@pytest.fixture
def strict_job(tmp_path, monkeypatch):
    """A strict L job on Siril's fallback registration (30 x 40 frames): F-2 only."""
    return _strict_job_runner(tmp_path, monkeypatch)


@pytest.fixture
def own_strict_job(tmp_path, monkeypatch):
    """A strict L job on its own registration (140 x 40 frames: three 64-row native-mask blocks): F-2 and G-4t."""
    return _strict_job_runner(tmp_path, monkeypatch, height=140, own=True)


def _job_frame_key(valid, h_stored, **kwargs):
    """The fallback registration's matrices shift frame n by 1.5 (n - 2) px (test_core_two_pass_reference._seq_text)."""
    shift = float(np.asarray(h_stored)[0, 2])
    return f"light_{int(round(shift / 1.5)) + 2:05d}"


def _job_view(job):
    """What must not depend on the worker count: state, error, checkpoints, the support record without its timing,
    the support sidecars, the filtered exports, the stack checkpoint's digests (the record's aside: it carries the
    timing) and the products' pixels."""
    directory = Path(job["job_dir"])
    work = directory / "work"
    record = json.loads((work / rs.RECORD).read_text(encoding="utf-8")) if (work / rs.RECORD).exists() else None
    stack = [(r["path"], r["sha256"]) for r in (job.get("checkpoint_files") or {}).get("stack", [])
             if not r["path"].endswith(rs.RECORD)]
    data = {name: h.tree_bytes(work)[name] for name in sorted(h.tree_bytes(work))
            if name.startswith(("r_light_0", "sky_support_"))}
    return {"state": job["state"], "error": job.get("error"), "checkpoints": job["checkpoints"], "stack": stack,
            "record": None if record is None else {k: v for k, v in record.items() if k != "filter_seconds"},
            "data": data}


def test_a_strict_l_job_is_the_same_at_every_worker_count(strict_job):
    """W19-W21 in small: the strict L job (prepare_exports with retain_frame_support=False) at 1, 3 and 16 workers."""
    views, notes = {}, {}
    for workers in (1, 3, 16):
        job = strict_job.run(f"w{workers}", workers)
        assert job["state"] == "succeeded", job.get("error")
        views[workers] = _job_view(job)
        notes[workers] = job["diagnostics"]["timings"]["notes"]["frame_pool.strict_support"]
        assert "frame_pool.native_mask" not in job["diagnostics"]["timings"]["notes"]     # no warp-back: no G-4t
        assert not list((Path(job["job_dir"]) / "work").glob("frame_support_light_*"))
    assert views[3] == views[1] and views[16] == views[1]
    assert [n[0]["workers"] for n in (notes[1], notes[3], notes[16])] == [1, 3, JOB_FRAMES]
    assert notes[1][0]["serial"] is True and notes[16][0]["decision"]["bound"] == "override"
    record = views[1]["record"]
    assert record["implementation"] == rs.SUPPORT_IMPLEMENTATION and len(record["frames"]) == JOB_FRAMES
    assert all("explicit_validity" not in frame for frame in record["frames"])


def test_an_own_registration_strict_job_is_the_same_at_every_worker_count(own_strict_job):
    """The strict L job on its own registration: F-2's exports and G-4t's native mask (three 64-row blocks, run inside
    the "native_mask" operation) at 1, 3 and 16 workers give the same job; both pools are noted under timings only."""
    views, notes, operations = {}, {}, {}
    for workers in (1, 3, 16):
        job = own_strict_job.run(f"own-w{workers}", workers)
        assert job["state"] == "succeeded", job.get("error")
        assert job["diagnostics"]["distortion"]["applied"] is True
        views[workers] = _job_view(job)
        work = Path(job["job_dir"]) / "work"
        views[workers]["native"] = (work / rs.NATIVE_VALID).read_bytes()
        views[workers]["master"] = (work / "sky_master.fit").read_bytes()
        timings = job["diagnostics"]["timings"]
        notes[workers] = {site: timings["notes"][f"frame_pool.{site}"] for site in ("strict_support", "native_mask")}
        operations[workers] = timings["operation_counts"]
    assert views[3] == views[1] and views[16] == views[1]
    record = views[1]["record"]
    assert record["implementation"] == rs.SUPPORT_IMPLEMENTATION and record["model"] is not None
    assert [notes[w]["native_mask"][0]["workers"] for w in (1, 3, 16)] == [1, 3, 3]           # three blocks
    assert notes[1]["native_mask"][0]["serial"] is True and notes[16]["native_mask"][0]["units"] == 3
    assert notes[3]["native_mask"][0]["decision"]["bound"] == "override"
    assert notes[3]["native_mask"][0]["unit_bytes"] == rs.NATIVE_UNIT_BYTES
    assert [notes[w]["strict_support"][0]["workers"] for w in (1, 3, 16)] == [1, 3, JOB_FRAMES]
    assert all(operations[w]["native_mask"] == 1 and operations[w]["registered_support"] == 1 for w in operations)
    for workers in (1, 3, 16):        # R19: the pools' figures stay in the timings
        job = json.loads((Path(own_strict_job.run(f"own-check-w{workers}", workers)["job_dir"]) / "job.json")
                         .read_text(encoding="utf-8"))
        outside = json.dumps({k: v for k, v in job.items() if k != "diagnostics"}) + json.dumps(
            {k: v for k, v in job["diagnostics"].items() if k != "timings"})
        assert "frame_pool" not in outside and "pool_stats" not in outside


@pytest.mark.parametrize("stage", ["kernel", "zero"])
@pytest.mark.parametrize("k", [0, 2, JOB_FRAMES - 1])
def test_a_failing_export_fails_the_job_like_the_serial_run_and_a_resume_recovers(strict_job, stage, k):
    """(c) at job level: the same job state and error at 1 and 4 workers; the resumed attempt (Siril re-exports, the
    attempt drops F-2's sidecar temporaries) ends like an uninterrupted job."""
    stem = f"light_{k + 1:05d}"
    clean = _job_view(strict_job.run("clean", 4))
    failed = {}
    for workers in (1, 4):
        job = strict_job.run(f"{stage}-{k}-w{workers}", workers, fail=(stage, stem))
        failed[workers] = job
        assert job["state"] == "failed" and f"injected {stage} failure at {stem}" in job["error"], job.get("error")
        assert not (Path(job["job_dir"]) / "work" / rs.RECORD).exists()
    assert failed[4]["error"] == failed[1]["error"] and failed[4]["checkpoints"] == failed[1]["checkpoints"]
    work = Path(failed[4]["job_dir"]) / "work"
    for name in ("frame_support_light_00002.npy.tmp", "sky_support_eligible.npy.tmp"):
        (work / name).write_bytes(b"left by a cancelled pool")
    resumed = strict_job.run(None, 4, resume=failed[4]["job_dir"])
    assert resumed["state"] == "succeeded", resumed.get("error")
    assert not list(work.glob("*.npy.tmp"))
    view = _job_view(resumed)
    assert view == {**clean, "error": view["error"]} and view["error"] is None


# ================================================================== real size: the F48 reference job's exports

@pytest.mark.parametrize("workers", [None, 16])
def test_replay_with_workers_reproduces_every_f48_map_and_the_native_map(tmp_path, workers):
    """(b) through scripts/replay_support.py --workers, read-only on the F48 job: every frame's map by the buffered
    kernel on the frame pool equals the .npy the job retained (by digest), the count its record; native_mask on the
    pool equals the job's native map (G-4t)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("support_replay_tool_p1c", h.REPO_ROOT / "scripts" / "replay_support.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    job = h.f48_job()
    if workers is None:
        record = json.loads((job / "work" / rs.RECORD).read_text(encoding="utf-8"))
        height, width = record["shape"]
        workers = core._strict_support_workers(job / "work", {}, height, width, len(record["frames"]), {})[0]
    report = tool.replay(job, output_root=tmp_path / "replays", workers=workers)
    assert report["state"] == "succeeded", (report.get("error"), report.get("integrity_error"))
    assert len(report["frames"]) == 48 and all(f["matches_recorded_count"] for f in report["frames"])
    pool = report["pool"]
    assert pool["retained_maps_compared"] == 48 and pool["native_matches_record"] is True
    assert pool["runs"][0]["pool"]["workers"] == workers and pool["native_pool"]["workers"] == min(workers, 87)
    print(f"replay --workers {workers}: frames {pool['runs'][0]['frames_seconds']:.1f} s, "
          f"native {pool['native_seconds']:.1f} s")


REAL_FRAMES = 6


@pytest.fixture(scope="module")
def f48(tmp_path_factory):
    """A copy of the first REAL_FRAMES exports of the F48 job (already filtered, so filtering them again rewrites the
    same bytes) with a sequence selecting just them; the job's mask source and model; its per-frame record."""
    job = h.f48_job()
    work = job / "work"
    record = json.loads((work / rs.RECORD).read_text(encoding="utf-8"))
    stems = [entry["stem"] for entry in record["frames"]][:REAL_FRAMES]
    root = tmp_path_factory.mktemp("f48")
    dest = root / "work"
    dest.mkdir()
    text = (work / rs.EXPORT_SEQUENCE).read_text(encoding="utf-8")
    lines = []
    for line in text.splitlines():
        parts = line.split()
        if parts[:1] == ["I"] and len(parts) == 3:
            parts[2] = "1" if f"light_{int(parts[1]):05d}" in stems else "0"
            line = " ".join(parts)
        elif parts[:1] == ["S"]:
            parts[4] = str(len(stems))
            line = " ".join(parts)
        lines.append(line)
    (dest / rs.EXPORT_SEQUENCE).write_text("\n".join(lines) + "\n", encoding="utf-8")
    h.copy_subset(work, dest, ["distortion.wcs"] + [f"r_{stem}.fit" for stem in stems])
    job_record = json.loads((job / "job.json").read_text(encoding="utf-8"))
    height, width = record["shape"]
    dilation = json.loads((work / "sky_validity.json").read_text(encoding="utf-8"))["dilation_px"]
    source = core._validity_source(job, job_record, width, height, dilation)
    return {"work": dest, "source": source, "model": RadialDistortion(**record["model"]),
            "frames": {entry["stem"]: entry for entry in record["frames"]}, "stems": stems, "shape": (height, width)}


def _real(f48, *, workers=None, reference=False, check=None, pool_stats=None, unit_bytes=None, source=None,
          around=None):
    """One side on the copied exports; ``source`` replaces the job's static mask source, ``around`` (a context
    manager, e.g. a PeakSampler) is entered around the prepare_exports call alone."""
    source = f48["source"] if source is None else source
    with hashing.run_scope("p1c-strict-real", mode="on"), (around or contextlib.nullcontext()):
        if reference:
            record = rs._prepare_exports_reference(f48["work"], source, f48["model"], check=check,
                                                   retain_frame_support=True)
        else:
            record = rs.prepare_exports(f48["work"], source, f48["model"], check=check,
                                        retain_frame_support=True, workers=workers, pool_stats=pool_stats,
                                        unit_bytes=unit_bytes)
    maps = {stem: (f48["work"] / f"frame_support_{stem}.npy").read_bytes() for stem in f48["stems"]}
    return strip(record), maps, (f48["work"] / rs.COUNTS).read_bytes()


@pytest.fixture(scope="module")
def f48_reference(f48):
    expected = _real(f48, reference=True)
    for entry in expected[0]["frames"]:                     # the job's own record: same maps, same filtered exports
        old = f48["frames"][entry["stem"]]
        assert entry["supported_pixels"] == old["supported_pixels"]
        assert entry["export_after_sha256"] == old["export_after_sha256"] == entry["export_before_sha256"]
        assert entry["explicit_validity"]["sha256"] == old["explicit_validity"]["sha256"]
    return expected


def _default_workers(f48):
    height, width = f48["shape"]
    return core._strict_support_workers(f48["work"], {}, height, width, REAL_FRAMES, {})


@pytest.mark.parametrize("case", h.params(h.real_cases()))
def test_real_size_pool_matches_the_serial_body(f48, f48_reference, case):
    workers, unit_bytes = _default_workers(f48)
    workers = workers if case.workers is None else case.workers
    before = h.aester_threads()
    pool_stats = {}
    with h.pool_conditions(case):
        actual = _real(f48, workers=workers, pool_stats=pool_stats, unit_bytes=unit_bytes)
    h.assert_same(f48_reference, actual)
    assert pool_stats["workers"] == min(workers, REAL_FRAMES)
    h.assert_pool_quiet(before)


@pytest.mark.parametrize("workers", [None, 8, 16])
def test_real_size_cancel_returns_within_a_second(f48, f48_reference, workers):
    """(d): tripped after the SIP grid exists (its one-off build is the pre-registered exception)."""
    workers = _default_workers(f48)[0] if workers is None else workers
    started = threading.Event()
    real = sk.ExportMaskKernel.block

    def block(self, *args, **kwargs):
        real(self, *args, **kwargs)
        if self.grid_flat is not None:
            started.set()
    switch = h.CancelSwitch()

    def trip_when_running():
        if started.wait(30):
            time.sleep(0.3)
            switch.trip()
    sk.ExportMaskKernel.block = block
    try:
        threading.Thread(target=trip_when_running, daemon=True).start()
        with h.pool_conditions(h.Case(workers, None, "reuse", None)):
            latency = h.assert_cancels_in_time(lambda check: _real(f48, workers=workers, check=check), switch)
    finally:
        sk.ExportMaskKernel.block = real
    print(f"cancel latency {latency:.3f} s at {workers} workers")
    assert not (f48["work"] / rs.RECORD).exists() or json.loads(
        (f48["work"] / rs.RECORD).read_text(encoding="utf-8"))["frames"] == f48_reference[0]["frames"]
    h.assert_same(f48_reference, _real(f48, workers=workers))           # a rerun is whole again


@pytest.fixture(scope="module")
def f48_own_masks(f48, tmp_path_factory):
    """An own mask PNG for every copied frame, each the job's sky mask again: the per-frame route (a decode in prepare
    for every frame, an anchor grid per own array) on the static route's values, so the exports and maps stay the same."""
    root = tmp_path_factory.mktemp("f48-own-masks")
    return {stem: Path(shutil.copyfile(h.f48_job() / "sky_mask.png", root / f"{stem}.png")) for stem in f48["stems"]}


class _TrippingMasks(sv.PerFrameMaskSource):
    """A per-frame mask source that trips ``switch`` as the decode of ``stem``'s own mask starts, and times it."""

    def __init__(self, f48, files, switch=None, stem=None):
        height, width = f48["shape"]
        super().__init__(f48["source"], files, width, height)
        self.switch, self.stem, self.decode_seconds = switch, stem, None

    def for_frame(self, stem):
        if stem != self.stem or stem in self._cache:
            return super().for_frame(stem)
        self.switch.trip()
        started = time.perf_counter()
        try:
            return super().for_frame(stem)
        finally:
            self.decode_seconds = time.perf_counter() - started


@pytest.mark.parametrize("workers", [None, 8, 16])
def test_real_size_cancel_during_a_per_frame_mask_decode_returns_within_a_second(f48, f48_reference, f48_own_masks,
                                                                                 workers):
    """(d) with per-frame masks (strict L/R jobs with frame masks): prepare decodes each own mask in the coordinator
    (PerFrameMaskSource.for_frame: the PNG, the minimum with the static mask, the dilation), one call that cannot be
    probed because sky_validity is frozen (§6.7 A16), so it joins the pool's unprobeable segments (§5 item 5, P1c
    validation report). Tripped as the decode of the last frame of the first window starts, while the earlier units
    run (the SIP grid built before), the pool still returns within 1 s; then a whole per-frame run equals the static
    route's reference (same values, so the same record, exports and maps)."""
    height, width = f48["shape"]
    workers = _default_workers(f48)[0] if workers is None else workers
    granted = min(workers, REAL_FRAMES)
    switch = h.CancelSwitch()
    source = _TrippingMasks(f48, f48_own_masks, switch, f48["stems"][granted - 1])
    header = fits.getheader(f48["work"] / "distortion.wcs")
    sm.distortion_grid(width, height, sm._sip_spec(header))
    try:
        with h.pool_conditions(h.Case(workers, None, "reuse", None)):
            latency = h.assert_cancels_in_time(lambda check: _real(f48, workers=workers, check=check, source=source),
                                               switch)
    finally:
        sm.release_distortion_grid()
    print(f"per-frame mask cancel latency {latency:.3f} s at {workers} workers "
          f"(the decode in prepare {source.decode_seconds:.3f} s)")
    assert source.decode_seconds is not None and len(source._cache) == granted
    h.assert_same(f48_reference, _real(f48, workers=workers, source=_TrippingMasks(f48, f48_own_masks)))


@pytest.mark.parametrize("workers,inflight", [(2, 4), (4, 8)])
def test_real_size_pool_memory_stays_within_its_declared_bound(f48, f48_reference, workers, inflight):
    """(f): in flight 2 x workers, the first unit finishing last, so the window fills (2 workers: 3 results held while
    unit 0 runs; 4 workers: all 6 frames in flight, 5 held). Only the prepare_exports call is sampled. Its rise must stay
    within granted x (unit_bytes + BLAS slot), unit_bytes already counting a worker's held result (R17), plus the
    coordinator's own arrays: the uint16 counts (2 B/px), the static validity's anchor grid built in prepare(0) (1 B/px)
    and the one digest copy of that validity in on_result (1 B/px), and 32 MiB for the record, the ledger and Python
    objects. The SIP grid is built before (a shared 366 MB the unit figure does not charge)."""
    height, width = f48["shape"]
    _, unit_bytes = _default_workers(f48)
    header = fits.getheader(f48["work"] / "distortion.wcs")
    sm.distortion_grid(width, height, sm._sip_spec(header))
    pool_stats = {}
    peak = h.PeakSampler(0.002, trace=False)
    gc.collect()                    # earlier tests' arrays freed inside the window would hide the pool's rise
    try:
        with h.pool_conditions(h.Case(workers, inflight, "reuse", 0), head_last=True, head_seconds=8.0), \
                forced_inflight(inflight):
            actual = _real(f48, workers=workers, pool_stats=pool_stats, unit_bytes=unit_bytes, around=peak)
    finally:
        sm.release_distortion_grid()
    h.assert_same(f48_reference, actual)
    assert pool_stats["workers"] == workers and pool_stats["inflight"] == inflight
    assert pool_stats["max_held"] == min(inflight, REAL_FRAMES), pool_stats      # the window was full (head included)
    coordinator = 4 * height * width + (32 << 20)
    limit = pool_stats["workers"] * (pool_stats["unit_bytes"] + frame_pool.BLAS_SLOT_BYTES) + coordinator
    print(f"strict_support pool rise {peak.rise / 2**20:.0f} MiB, limit {limit / 2**20:.0f} MiB "
          f"(workers {workers}, in flight {inflight}, held {pool_stats['max_held']}; coordinator "
          f"{coordinator / 2**20:.0f} MiB)")
    if os.name == "nt":
        assert peak.rise <= limit, (peak.rise, limit)
        assert peak.rise >= min(inflight, REAL_FRAMES) * height * width, peak.rise      # it saw the held maps
