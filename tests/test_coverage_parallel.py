"""Efficiency P1b-1p (G-6 pilot): ``sky_validity.coverage_at`` on the frame pool equals the pre-P1b loop
(``_coverage_at_reference``) term by term for any worker count, keeps the validity lookups in the coordinator in
frame order, cancels like the pool, and ``core._write_coverage`` / the job start use it as pre-registered.

Efficiency P1c O-1 (docs/EFFICIENCY_P1C_PREREG_20261005.md §3.6, §2.1 R19/R20): ``core._write_coverage`` decides
through ``performance.frame_workers`` (the smaller of available physical memory and commit, net of in-process
reservations), takes O-1's ``worker_cap``, reserves its workers' bytes while the pool runs (sky_validity, in the frozen
holdout closure, declares none) and notes the pool through ``performance.note_pool`` (frame_pool.coverage, a list, one
entry per call, with the decision).

The real-fixture test reads the kept P1a F48 calibration read-only and must run whenever the fixture exists (the
acceptance records that it ran, P1b cross-check M6).
"""
import hashlib
import json
import os
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits
from PIL import Image

from starstack import core, frame_pool, imaging
from starstack import sky_validity as sv
from starstack.registration_distortion import RadialDistortion
from test_legacy_stack_split import world   # noqa: F401  (a finished L job on the controlled engine, no Siril)

# Controlled engines here model Siril before the EFF-D4 stack-settings contract (efficiency P1c §4.2): jobs are
# admitted without it (tests/conftest.py pre_contract_jobs); contract jobs: tests/test_stack_settings.py.
pytestmark = pytest.mark.usefixtures("pre_contract_jobs")

REPO = Path(__file__).resolve().parents[1]
FIXTURE_REL = Path("outputs") / "perf-baseline-20260928" / "runs" / "F48"
H, W = 48, 64
LEGACY_JOB = {"support_mode": "legacy"}          # _write_coverage's frame-count branch (not the strict support map)


class Cancelled(RuntimeError):
    pass


@pytest.fixture(autouse=True)
def clean_pool_environment(monkeypatch):
    monkeypatch.delenv(frame_pool.ENV_WORKERS, raising=False)
    yield
    assert frame_pool._budget_used == 0
    assert not [t for t in threading.enumerate() if t.name.startswith("aester-coverage")]


def _transforms(rng, frames):
    out = {}
    for k in range(frames):
        angle = rng.normal(0, 0.01)
        shift = rng.normal(0, 6, size=2)
        scale = 1 + rng.normal(0, 0.004)
        h = np.array([[scale * np.cos(angle), -scale * np.sin(angle), shift[0]],
                      [scale * np.sin(angle), scale * np.cos(angle), shift[1]],
                      [rng.normal(0, 1e-5), rng.normal(0, 1e-5), 1.0]])
        out[f"light_{k + 1:05d}"] = np.eye(3) if k == 0 else h
    return out


def _static_mask(rng):
    mask = np.ones((H, W), dtype=np.float32)
    horizon = (H * 0.7 + 4 * np.sin(np.arange(W) / 7.0 + rng.random() * 6)).astype(int)
    for x in range(W):
        mask[horizon[x]:, x] = 0.0
    mask[horizon.min() - 2:horizon.min(), : W // 3] = 0.5
    return mask


def _sources(rng, transforms, tmp_path):
    """The four validity forms coverage_at accepts: array, dict, StaticMaskSource, PerFrameMaskSource."""
    mask = _static_mask(rng)
    static = sv.StaticMaskSource(mask, dilation_px=2)
    files = {}
    folder = tmp_path / f"masks-{rng.integers(1 << 30)}"
    folder.mkdir()
    for k, stem in enumerate(transforms):
        if k % 2:
            own = np.full((H, W), 255, dtype=np.uint8)
            y, x = rng.integers(2, H // 2), rng.integers(2, W - 12)
            own[y:y + 5, x:x + 10] = 0
            Image.fromarray(own).save(folder / f"{stem}.png")
            files[stem] = folder / f"{stem}.png"
    per_frame = lambda: sv.PerFrameMaskSource(static, files, W, H)       # noqa: E731 (fresh cache per run)
    by_frame = per_frame().by_frame(transforms)
    return {"array": lambda: static.validity, "dict": lambda: by_frame, "static": lambda: static,
            "per_frame": per_frame}


def _model(kind):
    if kind is None:
        return None
    return RadialDistortion(-0.03, 0.004, W / 2, H / 2, max(W, H) / 2, magnification=0.97, direction=kind)


def _points(step):
    gy, gx = np.mgrid[0:H:step, 0:W:step]
    return np.column_stack([gx.ravel(), gy.ravel()]).astype(np.float64)


@pytest.fixture
def spy(monkeypatch):
    """Wrap frame_pool.run_ordered as coverage_at calls it: capture the per-frame terms and stats; optionally force
    the maximum prepare lead (inflight = frames, rule R11) and the context audit (rule R6)."""
    real = frame_pool.run_ordered
    box = SimpleNamespace(terms=None, max_lead=False, audit=False, calls=0)

    def run_ordered(fn, items, *, on_result, inflight=None, **kw):
        box.calls += 1
        box.terms = []

        def capture(i, value):
            box.terms.append((i, value.copy()))
            on_result(i, value)
        monkeypatch.setattr(frame_pool, "_AUDIT_CONTEXT", box.audit)
        try:
            return real(fn, items, on_result=capture, inflight=len(items) if box.max_lead else inflight, **kw)
        finally:
            monkeypatch.setattr(frame_pool, "_AUDIT_CONTEXT", False)
    monkeypatch.setattr(frame_pool, "run_ordered", run_ordered)
    return box


@pytest.mark.parametrize("model_kind", [None, "p_to_u", "u_to_p"])
@pytest.mark.parametrize("seed", range(20))
def test_coverage_terms_and_counts_equal_the_reference_for_any_workers(tmp_path, spy, seed, model_kind):
    rng = np.random.default_rng(seed)
    transforms = _transforms(rng, 9 + seed % 5)
    model = _model(model_kind)
    points = _points(1 + seed % 3)
    for name, make in _sources(rng, transforms, tmp_path).items():
        reference = sv._coverage_at_reference(points, make(), transforms, model)
        assert reference.dtype == np.int32
        # the reference's own per-frame terms: one frame at a time
        reference_terms = [sv._coverage_at_reference(points, make(), {stem: h}, model) for stem, h in transforms.items()]
        assert np.array_equal(sum(reference_terms), reference)
        for workers in (1, 2, 4, 16):
            for max_lead in (False, True):
                spy.max_lead = spy.audit = max_lead
                stats = {}
                got = sv.coverage_at(points, make(), transforms, model, workers=workers, stats=stats)
                assert got.dtype == np.int32 and np.array_equal(got, reference), (name, workers, max_lead)
                assert [i for i, _ in spy.terms] == list(range(len(transforms)))
                for (i, term), expected in zip(spy.terms, reference_terms):
                    assert term.dtype == np.int32 and np.array_equal(term, expected), (name, workers, i)
                assert stats["serial"] is (workers == 1) and stats["units"] == len(transforms)
                if workers > 1:
                    granted = min(workers, len(transforms))
                    assert stats["workers"] == granted and stats["max_held"] <= stats["inflight"]
                    assert stats["inflight"] == (max(granted, len(transforms)) if max_lead else 2 * granted)


def test_coverage_map_passes_workers_and_stats_through(tmp_path):
    rng = np.random.default_rng(7)
    transforms = _transforms(rng, 6)
    static = sv.StaticMaskSource(_static_mask(rng), dilation_px=3)
    serial = sv.coverage_map(static, transforms, _model("p_to_u"), step=2)
    stats = {}
    parallel = sv.coverage_map(static, transforms, _model("p_to_u"), step=2, workers=3, stats=stats)
    assert parallel.shape == (H // 2, W // 2) and np.array_equal(parallel, serial) and parallel.dtype == serial.dtype
    assert stats["workers"] == 3 and not stats["serial"]


def test_effective_sky_weight_keeps_the_serial_default_and_the_reference_weight(monkeypatch):
    """The effective sky weight (band points only) stays on the pool's serial path (design pool §8) and equals the
    weight computed through the pre-P1b loop (review pool-4)."""
    rng = np.random.default_rng(7)
    transforms = _transforms(rng, 6)
    static = sv.StaticMaskSource(_static_mask(rng), dilation_px=3)
    model = _model("p_to_u")
    real, calls = frame_pool.run_ordered, []

    def run_ordered(fn, items, *, workers, name, stats=None, **kw):
        seen = {} if stats is None else stats
        calls.append({"workers": workers, "name": name, "stats": seen})
        return real(fn, items, workers=workers, name=name, stats=seen, **kw)
    monkeypatch.setattr(frame_pool, "run_ordered", run_ordered)
    weight, info = sv.effective_sky_weight(static.mask, static.validity, transforms, model)
    assert [(c["workers"], c["name"]) for c in calls] == [(1, "coverage")]
    assert calls[0]["stats"]["serial"] is True and calls[0]["stats"]["reason"] == "workers"
    assert calls[0]["stats"]["workers"] == 1 and calls[0]["stats"]["units"] == len(transforms)
    # the same weight through _coverage_at_reference (no pool at all)
    band = (~static.validity) & (static.mask > 0.0)
    ys, xs = np.nonzero(band)
    reference_cov = sv._coverage_at_reference(np.column_stack([xs, ys]).astype(np.float64), static.validity,
                                              transforms, model)
    expected = static.mask.astype(np.float32).copy()
    expected[ys[reference_cov == 0], xs[reference_cov == 0]] = 0.0
    zero = int((reference_cov == 0).sum())
    assert 0 < zero < len(ys) == info["band_pixels"]                    # the comparison has both outcomes
    assert info["no_coverage_pixels"] == zero
    assert weight.dtype == np.float32 and np.array_equal(weight, expected)


def test_serial_path_checks_once_per_frame_in_order(tmp_path):
    rng = np.random.default_rng(3)
    transforms = _transforms(rng, 7)
    order = []

    class Source:
        def for_frame(self, stem):
            order.append(("lookup", stem))
            return np.ones((H, W), dtype=bool)

    def check():
        order.append(("check", len(order)))

    stats = {}
    sv.coverage_at(_points(4), Source(), transforms, None, check=check, stats=stats)
    assert [kind for kind, _ in order] == ["check", "lookup"] * len(transforms)
    assert [stem for kind, stem in order if kind == "lookup"] == list(transforms)
    assert stats["serial"] and stats["reason"] == "workers"


def test_per_frame_mask_loads_stay_in_the_coordinator_in_frame_order(tmp_path, monkeypatch):
    rng = np.random.default_rng(11)
    transforms = _transforms(rng, 10)
    static = sv.StaticMaskSource(_static_mask(rng), dilation_px=2)
    files = {}
    for stem in transforms:
        Image.fromarray(np.full((H, W), 255, dtype=np.uint8)).save(tmp_path / f"{stem}.png")
        files[stem] = tmp_path / f"{stem}.png"
    real = sv.load_mask_top_down
    loads = []

    def spy_load(path, width, height):
        loads.append((Path(path).stem, threading.current_thread() is threading.main_thread()))
        return real(path, width, height)
    monkeypatch.setattr(sv, "load_mask_top_down", spy_load)
    source = sv.PerFrameMaskSource(static, files, W, H)
    sv.coverage_at(_points(2), source, transforms, None, workers=4)
    assert loads == [(stem, True) for stem in transforms]


def test_parallel_cancel_raises_the_checks_object_and_cleans_up(tmp_path, monkeypatch):
    rng = np.random.default_rng(5)
    transforms = _transforms(rng, 40)
    flag = threading.Event()
    raised = []
    calls = []

    class SlowSource:
        def for_frame(self, stem):
            return np.ones((H, W), dtype=bool)

    def check():
        calls.append(flag.is_set())
        if flag.is_set():
            exc = Cancelled("用户取消")
            raised.append(exc)
            raise exc

    original = sv.frame_to_reference_native

    def slow(*args, **kwargs):
        time.sleep(0.05)
        return original(*args, **kwargs)

    monkeypatch.setattr(sv, "frame_to_reference_native", slow)
    threading.Timer(0.2, flag.set).start()
    t0 = time.perf_counter()
    with pytest.raises(Cancelled) as info:
        sv.coverage_at(_points(2), SlowSource(), transforms, None, check=check, workers=4)
    assert time.perf_counter() - t0 < 0.2 + 0.05 + frame_pool.POLL_SECONDS + 0.5
    assert len(raised) == 1 and info.value is raised[0] and calls[-1] is True and calls.count(True) == 1


# --------------------------------------------------------------------------- core._write_coverage and the job start

def _coverage_job(tmp_path, monkeypatch, *, frames=6, model_kind="p_to_u"):
    rng = np.random.default_rng(21)
    work = tmp_path / "work"
    work.mkdir()
    fits.PrimaryHDU(np.zeros((3, H, W), dtype=np.float32)).writeto(work / "sky_master.fit")
    static = sv.StaticMaskSource(_static_mask(rng), dilation_px=3)
    transforms, model = _transforms(rng, frames), _model(model_kind)
    monkeypatch.setattr(core, "_validity_source", lambda *a: static)
    monkeypatch.setattr(core, "registration_transforms", lambda *a: (transforms, model))
    notes = {}
    from starstack import performance

    def note_pool(site, stats):                      # performance.note_pool: frame_pool.<site>, a list (R19)
        notes.setdefault(f"frame_pool.{site}", []).append(json.loads(json.dumps(stats)))
    monkeypatch.setattr(performance, "note_pool", note_pool)
    return work, static, transforms, model, notes


def _memory(monkeypatch, physical, commit):
    """performance's readings (R20: both pass through performance._memory_status, net of reservations)."""
    from starstack import performance
    monkeypatch.setattr(performance, "_memory_status", lambda: (physical, commit))


@pytest.mark.parametrize("setting, serial, reason", [("1", True, "workers"), (None, False, None), ("4", False, None),
                                                     ("max", False, None)])
def test_write_coverage_uses_the_pool_and_notes_the_decision(tmp_path, monkeypatch, setting, serial, reason):
    work, static, transforms, model, notes = _coverage_job(tmp_path, monkeypatch)
    if setting is not None:
        monkeypatch.setenv(frame_pool.ENV_WORKERS, setting)
    monkeypatch.setattr(core.frame_pool, "usable_cpus", lambda: 192)
    _memory(monkeypatch, 64 << 30, 128 << 30)
    returned = core._write_coverage(work, tmp_path, None, True, {"dilation_px": 3}, lambda: None, job=LEGACY_JOB)
    assert returned == work / sv.COVERAGE_PNG
    expected = sv._coverage_at_reference(_points(core.SKY_COVERAGE_STEP), static, transforms, model)
    expected = expected.reshape(len(range(0, H, core.SKY_COVERAGE_STEP)), len(range(0, W, core.SKY_COVERAGE_STEP)))
    np.testing.assert_array_equal(sv.read_coverage_png(returned), expected)
    note, = notes["frame_pool.coverage"]                            # one entry for one call (R19)
    want = {"1": 1, None: 6, "4": 4, "max": 6}[setting]       # auto: min(cap 8, 96, memory, 6 frames)
    assert note["serial"] is serial and note["workers"] == want and note["requested"] == want
    assert note["units"] == len(transforms) and note.get("reason") == reason
    decision = note["decision"]
    assert decision["unit_bytes"] == sv.COVERAGE_UNIT_BYTES and decision["blas"] is True
    assert decision["bound"] == ("override" if setting is not None else "units")
    assert decision["memory_source"] == "physical" and "o1_cap" not in decision
    assert decision["reserved_bytes"] == (0 if want == 1 else want * (sv.COVERAGE_UNIT_BYTES + frame_pool.BLAS_SLOT_BYTES))
    summary = json.loads((work / "sky_coverage.json").read_text(encoding="utf-8"))
    assert summary["frames"] == len(transforms) and "frame_pool" not in json.dumps(summary)


def test_write_coverage_notes_one_entry_per_call(tmp_path, monkeypatch):
    """R19: two calls (a job's stack stage and a later rebuild through _compose_sky) give two list entries in the job
    timer's note frame_pool.coverage, through performance.note_pool."""
    from starstack import profiling
    work = tmp_path / "work"
    work.mkdir()
    fits.PrimaryHDU(np.zeros((3, H, W), dtype=np.float32)).writeto(work / "sky_master.fit")
    static = sv.StaticMaskSource(_static_mask(np.random.default_rng(21)), dilation_px=3)
    transforms = _transforms(np.random.default_rng(21), 6)
    monkeypatch.setattr(core, "_validity_source", lambda *a: static)
    monkeypatch.setattr(core, "registration_transforms", lambda *a: (transforms, None))
    timer = profiling.StageTimer()
    try:
        for _ in range(2):
            core._write_coverage(work, tmp_path, None, True, {"dilation_px": 3}, lambda: None, job=LEGACY_JOB)
    finally:
        notes = timer.finish()["notes"]
    entries = notes["frame_pool.coverage"]
    assert isinstance(entries, list) and len(entries) == 2 and all(e["units"] == 6 for e in entries)


@pytest.mark.parametrize("cap, setting, want", [(3, None, 3), (1, None, 1), (8, None, 6), (20, None, 6),
                                                (3, "5", 5)])
def test_write_coverage_takes_o1s_worker_cap(tmp_path, monkeypatch, cap, setting, want):
    """O-1 (prereg §3.6): while the trail scan overlaps, the coverage pool's workers are capped by the plan's share
    (``worker_cap``); AESTER_FRAME_WORKERS still overrides (benches). The cap is noted in the decision."""
    work, static, transforms, model, notes = _coverage_job(tmp_path, monkeypatch)
    if setting is None:
        monkeypatch.delenv(frame_pool.ENV_WORKERS, raising=False)
    else:
        monkeypatch.setenv(frame_pool.ENV_WORKERS, setting)
    monkeypatch.setattr(core.frame_pool, "usable_cpus", lambda: 192)
    _memory(monkeypatch, 64 << 30, 128 << 30)
    core._write_coverage(work, tmp_path, None, True, {"dilation_px": 3}, lambda: None, job=LEGACY_JOB,
                         worker_cap=cap)
    note, = notes["frame_pool.coverage"]
    assert note["workers"] == want and note["decision"]["o1_cap"] == cap
    assert note["decision"]["cap"] == min(sv.COVERAGE_WORKER_CAP, cap)


def test_write_coverage_reserves_its_workers_while_the_pool_runs(tmp_path, monkeypatch):
    """sky_validity declares no unit_bytes, so _write_coverage holds workers x (unit + BLAS slot) through
    frame_pool.reserved while coverage_map runs (a decision on O-1's helper thread sees it), and nothing after."""
    from starstack import performance
    work, static, transforms, model, notes = _coverage_job(tmp_path, monkeypatch)
    monkeypatch.delenv(frame_pool.ENV_WORKERS, raising=False)
    monkeypatch.setattr(core.frame_pool, "usable_cpus", lambda: 192)
    _memory(monkeypatch, 64 << 30, 128 << 30)
    seen = {}
    real = sv.coverage_map

    def coverage_map(*args, workers, **kw):
        seen.update(workers=workers, reserved=frame_pool.reserved_bytes(), available=performance.available_memory())
        return real(*args, workers=workers, **kw)
    monkeypatch.setattr(sv, "coverage_map", coverage_map)
    before = frame_pool.reserved_bytes()
    core._write_coverage(work, tmp_path, None, True, {"dilation_px": 3}, lambda: None, job=LEGACY_JOB)
    held = 6 * (sv.COVERAGE_UNIT_BYTES + frame_pool.BLAS_SLOT_BYTES)
    assert seen["workers"] == 6 and seen["reserved"] == before + held
    assert seen["available"] == (64 << 30) - before - held                 # R20: deducted from the readings
    assert frame_pool.reserved_bytes() == before


def test_write_coverage_notes_and_releases_on_a_failure(tmp_path, monkeypatch):
    work, static, transforms, model, notes = _coverage_job(tmp_path, monkeypatch)
    monkeypatch.setenv(frame_pool.ENV_WORKERS, "4")
    before = frame_pool.reserved_bytes()

    def check():
        raise Cancelled("用户取消")
    with pytest.raises(Cancelled):
        core._write_coverage(work, tmp_path, None, True, {"dilation_px": 3}, check, job=LEGACY_JOB)
    assert frame_pool.reserved_bytes() == before and len(notes["frame_pool.coverage"]) == 1


def test_write_coverage_files_are_byte_identical_serial_and_parallel(tmp_path, monkeypatch):
    outputs = {}
    for setting in ("1", "max"):
        folder = tmp_path / setting
        folder.mkdir()
        work, *_ = _coverage_job(folder, monkeypatch, frames=12)
        monkeypatch.setenv(frame_pool.ENV_WORKERS, setting)
        core._write_coverage(work, folder, None, True, {"dilation_px": 3}, lambda: None, job=LEGACY_JOB)
        outputs[setting] = ((work / sv.COVERAGE_PNG).read_bytes(), (work / "sky_coverage.json").read_bytes())
    assert outputs["1"] == outputs["max"]


class _Decided(Exception):
    """Stops _write_coverage right after its worker decision (no coverage of a large frame is computed)."""


@pytest.mark.parametrize("height, width, unit_mib, workers", [
    (4000, 6000, 230, 8),           # 24 MP: below the measured frame, the 230 MiB floor (cap 8)
    (5520, 8280, 230, 8),           # 45.7 MP: the frame COVERAGE_UNIT_BYTES was measured on (cap 8)
    (8736, 11648, 512.06, 5),       # 102 MP: the unit grows with the pixels, so fewer workers
])
def test_write_coverage_charges_each_worker_by_frame_pixels(tmp_path, monkeypatch, height, width, unit_mib, workers):
    """Review pool-1: at 8 GiB free the budget is min(avail/3, avail-1 GiB) = 2730.7 MiB.  A constant 230 MiB unit
    (+32 MiB BLAS slot) granted 8 workers at any size; scaled by the frame's pixels a 102 MP frame gets 5."""
    from starstack import performance, resources
    assert sv.COVERAGE_UNIT_PIXELS == resources.REFERENCE_PIXELS == 5520 * 8280
    work, static, transforms, model, notes = _coverage_job(tmp_path, monkeypatch, frames=48)
    monkeypatch.setattr(fits, "getheader", lambda *a, **k: {"NAXIS1": width, "NAXIS2": height})
    monkeypatch.setattr(core.frame_pool, "usable_cpus", lambda: 192)
    _memory(monkeypatch, 8 << 30, 16 << 30)                 # frame_workers takes the smaller one (R20)
    monkeypatch.delenv(frame_pool.ENV_WORKERS, raising=False)
    real_default, charged, chosen = frame_pool.default_workers, [], []

    def default_workers(**kw):
        charged.append(kw["unit_bytes"])
        return real_default(**kw)

    def coverage_map(*args, workers, **kw):
        chosen.append(workers)
        raise _Decided()
    monkeypatch.setattr(frame_pool, "default_workers", default_workers)
    monkeypatch.setattr(sv, "coverage_map", coverage_map)
    with pytest.raises(_Decided):
        core._write_coverage(work, tmp_path, None, True, {"dilation_px": 3}, lambda: None, job=LEGACY_JOB)
    expected_unit = max(sv.COVERAGE_UNIT_BYTES, -(-sv.COVERAGE_UNIT_BYTES * height * width // (5520 * 8280)))
    assert charged == [expected_unit] and round(expected_unit / 2 ** 20, 2) == unit_mib
    assert chosen == [workers]
    budget = min((8 << 30) // 3, (8 << 30) - (1 << 30))
    assert workers == min(sv.COVERAGE_WORKER_CAP, budget // (expected_unit + frame_pool.BLAS_SLOT_BYTES))
    assert workers * (expected_unit + frame_pool.BLAS_SLOT_BYTES) <= budget        # the rule keeps its own budget


@pytest.fixture
def job_environment(tmp_path, monkeypatch):
    """tests/test_jobs.py's controlled harness (no Siril)."""
    source = tmp_path / "source"
    source.mkdir()
    for number in range(3):
        data = np.full((18, 24, 3), .03 + number * .001, dtype=np.float32)
        imaging.write_linear(source / f"frame {number}.fit", data, "Test fixture")
    engine = {"available": True, "executable": str(tmp_path / "fake engine.exe"), "version": "1.4.4",
              "message": "Test engine"}
    monkeypatch.setattr(core, "doctor", lambda *_: dict(engine))
    monkeypatch.setattr(core.shutil, "disk_usage", lambda _: SimpleNamespace(free=10**12))
    monkeypatch.setattr(core, "_spawn_worker", lambda directory: None)
    calls = []
    monkeypatch.setattr(core, "run_script", lambda *a, **k: calls.append(a[2]))
    return SimpleNamespace(source=source, output=tmp_path / "results", calls=calls)


@pytest.mark.parametrize("bad", ["eight", "0", "17"])
def test_malformed_knob_fails_the_job_at_its_start(job_environment, monkeypatch, bad):
    job = core.submit_job(str(job_environment.source), str(job_environment.output))
    monkeypatch.setenv(frame_pool.ENV_WORKERS, bad)
    core.run_job(job["job_dir"])
    result = core.load_job(job["job_dir"])
    assert result["state"] == "failed" and "AESTER_FRAME_WORKERS" in result["error"]
    assert job_environment.calls == []                              # no engine stage ran
    assert not (Path(job["job_dir"]) / "work" / "light_00001.fit").exists()


def test_recompose_refuses_a_malformed_knob_before_any_lock_or_export(world, monkeypatch):
    """Review pool-2: recompose_transients can rebuild a missing coverage map on the frame pool (_compose_sky ->
    _write_coverage), so it validates AESTER_FRAME_WORKERS first, with the job's message, before taking the job's
    lock or making an export directory."""
    finished = world.run("3a")
    assert finished["state"] == "succeeded", finished.get("error")
    folder = Path(finished["job_dir"])
    listing = lambda: sorted((str(p.relative_to(folder)), p.is_dir()) for p in folder.rglob("*"))   # noqa: E731
    before = listing()
    entered = []
    monkeypatch.setattr(core, "_recompose_locked", lambda directory, job, *a, **k: entered.append(job["state"]) or job)
    for bad in ("eight", "0", "17", "²", "８"):
        monkeypatch.setenv(frame_pool.ENV_WORKERS, bad)
        with pytest.raises(ValueError) as info:
            core.recompose_transients(str(folder), overrides={"any": "omit"})
        with pytest.raises(ValueError) as job_start:
            frame_pool.validate_environment()                     # what the job worker raises at its start
        assert str(info.value) == str(job_start.value) == "AESTER_FRAME_WORKERS must be auto, max or 1-16"
    assert entered == [] and listing() == before
    monkeypatch.setenv(frame_pool.ENV_WORKERS, "4")
    core.recompose_transients(str(folder), overrides={"any": "omit"})
    assert entered == ["succeeded"]


# --------------------------------------------------------------------------- the real F48 calibration (read-only)

def _fixture_work():
    roots = []
    if os.environ.get("AESTER_P1A_F48_FIXTURE"):
        roots.append(Path(os.environ["AESTER_P1A_F48_FIXTURE"]))
    roots.append(REPO / FIXTURE_REL)
    try:                                   # a git worktree: the fixture lives in the main checkout
        common = subprocess.run(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"], cwd=REPO,
                                capture_output=True, text=True, timeout=30).stdout.strip()
        if common:
            roots.append(Path(common).parent / FIXTURE_REL)
    except (OSError, subprocess.SubprocessError):
        pass
    for root in roots:
        found = sorted(root.glob("c/workflow-*/stack/job-*/work/distortion.json")) if root.is_dir() else []
        if found:
            return found[0].parent
    return None


def test_real_f48_calibration_identical_at_1_8_16_and_cancels_promptly(monkeypatch):
    work = _fixture_work()
    if work is None:
        pytest.skip(f"kept P1a fixture {FIXTURE_REL} is absent")
    from starstack.registration_calibration import load_calibration
    calibration = load_calibration(work)
    with Image.open(work.parent / "sky_mask.png") as image:
        width, height = image.size
    transforms, model = core.registration_transforms(work, calibration, True, height)
    assert len(transforms) == 48 and model is not None
    static = sv.StaticMaskSource(sv.load_mask_top_down(work.parent / "sky_mask.png", width, height))
    gy, gx = np.mgrid[0:height:core.SKY_COVERAGE_STEP, 0:width:core.SKY_COVERAGE_STEP]
    points = np.column_stack([gx.ravel(), gy.ravel()]).astype(np.float64)

    t0 = time.perf_counter()
    reference = sv._coverage_at_reference(points, static, transforms, model)
    timings = {"reference": round(time.perf_counter() - t0, 2)}

    real_run, real_native = frame_pool.run_ordered, sv.frame_to_reference_native
    captured = {}

    def p_digest(*args, **kwargs):           # per-frame projected positions p (float64), digested as the units run
        p = real_native(*args, **kwargs)
        captured["p"][args[1].tobytes()] = hashlib.sha256(np.ascontiguousarray(p).tobytes()).hexdigest()
        return p

    def run_ordered(fn, items, *, on_result, **kw):
        def capture(i, value):
            captured["terms"].append(hashlib.sha256(value.tobytes()).hexdigest())
            on_result(i, value)
        return real_run(fn, items, on_result=capture, **kw)

    monkeypatch.setattr(frame_pool, "run_ordered", run_ordered)
    monkeypatch.setattr(sv, "frame_to_reference_native", p_digest)
    runs = {}
    for workers in (1, 8, 16):
        captured.update(p={}, terms=[])
        stats = {}
        t0 = time.perf_counter()
        got = sv.coverage_at(points, static, transforms, model, workers=workers, stats=stats)
        timings[f"w{workers}"] = round(time.perf_counter() - t0, 2)
        assert got.dtype == np.int32 and np.array_equal(got, reference), workers
        assert len(captured["terms"]) == 48 and len(captured["p"]) == 48
        assert stats["workers"] == workers and stats["serial"] is (workers == 1)
        runs[workers] = (list(captured["terms"]), dict(captured["p"]))
    for workers in (8, 16):
        assert runs[workers][0] == runs[1][0], "per-frame terms differ"
        assert runs[workers][1] == runs[1][1], "per-frame positions p differ"
    monkeypatch.setattr(frame_pool, "run_ordered", real_run)
    monkeypatch.setattr(sv, "frame_to_reference_native", real_native)

    latencies = []
    for trial in range(5):
        flag = threading.Event()
        raised = []

        def check():
            if flag.is_set():
                exc = Cancelled("用户取消")
                raised.append(exc)
                raise exc
        at = 1.0 + 0.23 * trial
        threading.Timer(at, flag.set).start()
        t0 = time.perf_counter()
        with pytest.raises(Cancelled) as info:
            sv.coverage_at(points, static, transforms, model, workers=8, check=check)
        latencies.append(round(time.perf_counter() - t0 - at, 3))
        assert len(raised) == 1 and info.value is raised[0]
    print("real F48 coverage:", json.dumps({"seconds": timings, "cancel_latency_s": latencies,
                                            "points": len(points), "frames": len(transforms)}))
    assert max(latencies) <= 2.0
