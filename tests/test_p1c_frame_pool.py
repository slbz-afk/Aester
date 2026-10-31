"""Efficiency P1c-0 (docs/EFFICIENCY_P1C_PREREG_20261005.md §2.2): the frame pool's additions.

In-process memory reservations, the floating-point control word given to workers (R21), exclusive sections, the
``Beside`` helper thread and the ``AESTER_OVERLAP`` knob, and the new per-run diagnostics. The P1b suite
(tests/test_frame_pool.py) keeps covering the pool's serial-loop semantics.
"""
import contextvars
import gc
import os
import threading
import time
import weakref

import numpy as np
import pytest

from starstack import frame_pool as fp
from starstack import hashing, performance
from test_coverage_parallel import job_environment  # noqa: F401  (fixture: tests/test_jobs.py's controlled harness)

VAR = contextvars.ContextVar("p1c_frame_pool_test_var", default="unset")
windows = pytest.mark.skipif(os.name != "nt", reason="the C runtime's _controlfp (R21 is skipped off Windows)")
MiB, GiB = 1 << 20, 1 << 30


class Cancelled(RuntimeError):
    pass


def aester_threads():
    return {t for t in threading.enumerate() if t.name.startswith("aester-")}


@pytest.fixture(autouse=True)
def quiet_pool(monkeypatch):
    for name in (fp.ENV_WORKERS, fp.ENV_OVERLAP):
        monkeypatch.delenv(name, raising=False)
    before = aester_threads()
    yield
    assert fp._budget_used == 0 and fp.reserved_bytes() == 0 and fp.exclusive_sections() == ()
    deadline = time.monotonic() + 5
    while aester_threads() - before and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not (aester_threads() - before)


# ================================================================== reservations

def test_reserved_nests_counts_and_releases_on_exceptions():
    assert fp.reserved_bytes() == 0
    with fp.reserved(5 * MiB) as held:
        assert held == 5 * MiB and fp.reserved_bytes() == 5 * MiB
        with fp.reserved(np.int64(3 * MiB)):                      # an integer-like count
            assert fp.reserved_bytes() == 8 * MiB
        with pytest.raises(KeyError):
            with fp.reserved(MiB):
                assert fp.reserved_bytes() == 6 * MiB
                raise KeyError("inside")
        assert fp.reserved_bytes() == 5 * MiB
    assert fp.reserved_bytes() == 0
    for bad in (-1, 1.5, "8", None):
        with pytest.raises(ValueError, match="nbytes"):
            with fp.reserved(bad):
                pass
    assert fp.reserved_bytes() == 0


@pytest.mark.parametrize("blas", [True, False])
def test_a_parallel_pool_reserves_its_workers_while_they_run(blas):
    seen = []

    def fn(x, unit):
        seen.append(fp.reserved_bytes())
        time.sleep(0.002)
        return x

    stats = {}
    fp.run_ordered(fn, range(24), workers=4, name="t-res", unit_bytes=10 * MiB, blas=blas, stats=stats)
    expected = 4 * (10 * MiB + (fp.BLAS_SLOT_BYTES if blas else 0))
    assert set(seen) == {expected} and fp.reserved_bytes() == 0
    assert stats["unit_bytes"] == 10 * MiB and stats["reserved_bytes"] == expected and stats["workers"] == 4


def test_the_serial_path_reserves_nothing_and_bad_unit_bytes_fail_first():
    seen = []
    stats = {}
    fp.run_ordered(lambda x, u: seen.append(fp.reserved_bytes()), range(4), workers=1, name="t-res1",
                   unit_bytes=10 * MiB, stats=stats)
    assert seen == [0] * 4 and stats["reserved_bytes"] == 0 and stats["unit_bytes"] == 10 * MiB
    ran = []
    for bad in (-1, 2.5):
        with pytest.raises(ValueError, match="unit_bytes"):
            fp.run_ordered(lambda x, u: ran.append(x), range(4), workers=2, name="t-res2", unit_bytes=bad)
    assert ran == []


@pytest.mark.parametrize("ending", ["failure", "cancel", "finalize"])
def test_the_reservation_is_released_on_every_exit(ending):
    flag = threading.Event()

    def check():
        if flag.is_set():
            raise Cancelled("stop")

    def fn(x, unit):
        if ending == "failure" and x == 3:
            raise ValueError("unit 3")
        if ending == "cancel" and x == 2:
            flag.set()
        unit.check()
        time.sleep(0.005)
        return x

    def finalize(state):
        if ending == "finalize":
            raise OSError("close")

    with pytest.raises((ValueError, Cancelled, OSError)):
        fp.run_ordered(fn, range(16), workers=4, name="t-res-exit", check=check, unit_bytes=MiB,
                       init=lambda: object(), finalize=finalize)
    assert fp.reserved_bytes() == 0 and fp._budget_used == 0


def test_available_memory_and_commit_deduct_reservations_once(monkeypatch):
    monkeypatch.setattr(performance, "_memory_status", lambda: (20 * GiB, 12 * GiB))
    assert (performance.available_memory(), performance.available_commit()) == (20 * GiB, 12 * GiB)
    with fp.reserved(3 * GiB):
        assert (performance.available_memory(), performance.available_commit()) == (17 * GiB, 9 * GiB)
        with fp.reserved(30 * GiB):
            assert performance.available_memory() == 0 and performance.available_commit() == 0
        decision = {}
        performance.frame_workers(48, 1 * GiB, 16, blas=False, decision=decision)
        assert decision["available_bytes"] == 9 * GiB               # min(17, 9): commit, net of the 3 GiB, once
        assert decision["memory_source"] == "commit"
    monkeypatch.setattr(performance, "_memory_status", lambda: (None, 12 * GiB))
    assert performance.available_memory() is None and performance.frame_workers(48, MiB, 8) == 1


def test_real_memory_readings_are_plausible():
    physical, commit = performance.available_memory(), performance.available_commit()
    assert physical is not None and physical > 0
    if os.name == "nt":
        assert commit is not None and commit > 0
    with fp.reserved(1 << 40):                                       # 1 TiB: more than this machine has free
        assert performance.available_memory() == 0


def test_frame_workers_follows_default_workers_and_the_override(monkeypatch):
    monkeypatch.setattr(fp, "usable_cpus", lambda: 32)
    monkeypatch.setattr(performance, "_memory_status", lambda: (8 * GiB, 64 * GiB))
    decision = {}
    # 8 GiB: budget min(8/3, 7) GiB; per worker 230 MiB + 32 MiB -> 10; cap 8
    assert performance.frame_workers(48, 230 * MiB, 8, decision=decision) == 8
    assert decision["bound"] == "cap" and decision["memory"] == 10 and decision["cpu"] == 16
    assert performance.frame_workers(48, 230 * MiB, 16, decision=decision) == 10 and decision["bound"] == "memory"
    assert performance.frame_workers(3, 230 * MiB, 16, decision=decision) == 3 and decision["bound"] == "units"
    assert performance.frame_workers(48, 230 * MiB, 16, extra_bytes=4 * GiB, decision=decision) == 5
    assert decision["extra_bytes"] == 4 * GiB and decision["available_bytes"] == 4 * GiB
    assert performance.frame_workers(48, 230 * MiB, 16) == fp.default_workers(
        cap=16, unit_bytes=230 * MiB, available_bytes=8 * GiB, units=48)
    monkeypatch.setenv(fp.ENV_WORKERS, "3")
    assert performance.frame_workers(48, 230 * MiB, 16, decision=decision) == 3 and decision["bound"] == "override"
    monkeypatch.setenv(fp.ENV_WORKERS, "max")
    assert performance.frame_workers(48, 230 * MiB, 8) == 16


def test_default_workers_reports_why_without_changing_its_answer():
    cpus = fp.usable_cpus()
    for available in (None, 512 << 20, 2 * GiB, 8 * GiB, 1 << 40):
        for units in (1, 3, 48):
            why = {}
            got = fp.default_workers(cap=8, unit_bytes=230 * MiB, available_bytes=available, units=units, why=why)
            if available is None:
                assert got == 1 and why["bound"] == "memory-unknown"
                continue
            budget = max(0, min(available // 3, available - GiB))
            assert got == max(1, min(8, 16, max(1, cpus // 2), budget // (230 * MiB + fp.BLAS_SLOT_BYTES), units))
            assert why[why["bound"]] == min(why[k] for k in ("cap", "cpu", "memory", "units"))


def test_concurrent_pools_see_each_others_reservations(monkeypatch):
    """O-1's case: a pool deciding while another pool runs sees the other's workers as reserved."""
    monkeypatch.setattr(fp, "usable_cpus", lambda: 64)
    monkeypatch.setattr(performance, "_memory_status", lambda: (12 * GiB, None))
    alone = performance.frame_workers(48, 400 * MiB, 16)
    running, decided = threading.Event(), []
    release = threading.Event()

    def outer(x, unit):
        running.set()
        release.wait(5)
        return x

    thread = threading.Thread(target=lambda: fp.run_ordered(outer, range(4), workers=4, name="t-res-outer",
                                                           unit_bytes=1 * GiB), name="t-res-driver")
    thread.start()
    try:
        assert running.wait(5)
        decided.append(performance.frame_workers(48, 400 * MiB, 16))
    finally:
        release.set()
        thread.join(10)
    assert alone == 9 and decided == [max(1, ((12 * GiB - 4 * (GiB + fp.BLAS_SLOT_BYTES)) // 3)
                                          // (400 * MiB + fp.BLAS_SLOT_BYTES))]


# ================================================================== R21: floating-point environment

def _subnormal_products():
    tiny = np.full(256, np.finfo(np.float32).tiny, dtype=np.float32)
    scale = np.linspace(0.01, 0.99, 256, dtype=np.float32)
    product = tiny * scale                                          # subnormal unless flushed to zero
    return product.tobytes() + (np.float64(2.5e-308) * np.float64(1e-3)).tobytes()


@pytest.fixture
def coordinator_flushes_subnormals():
    """The coordinator thread with FTZ/DAZ (``_DN_FLUSH``), as a library might leave it; restored afterwards."""
    api = fp._controlfp()
    old = int(api(0, 0)) & fp.FP_ENV_MASK
    api(0x01000000, fp._MCW_DN)                                     # _DN_FLUSH
    try:
        yield
    finally:
        api(old, fp.FP_ENV_MASK)
    assert int(api(0, 0)) & fp.FP_ENV_MASK == old


@windows
def test_r21_workers_compute_subnormal_products_like_the_coordinator(coordinator_flushes_subnormals):
    coordinator = _subnormal_products()
    assert coordinator != _plain_thread(_subnormal_products)        # a new thread would not flush: the test bites
    stats = {}
    results = fp.run_ordered(lambda x, u: (time.sleep(0.002), _subnormal_products())[1], range(32), workers=8,
                             name="t-fp", stats=stats)
    assert all(r == coordinator for r in results)
    assert stats["fp_env"] == hex(0x01000000) and 1 <= stats["fp_env_applied"] <= 32
    helper = fp.Beside.start(lambda probe: _subnormal_products(), "t-fp")
    assert helper.result() == coordinator and helper.fp_env_applied is True


@windows
def test_r21_default_environment_needs_no_change():
    reference = _subnormal_products()
    stats = {}
    results = fp.run_ordered(lambda x, u: _subnormal_products(), range(16), workers=4, name="t-fp0", stats=stats)
    assert all(r == reference for r in results) and stats["fp_env_applied"] == 0
    assert stats["fp_env"] == hex(fp.fp_environment())
    assert reference == _plain_thread(_subnormal_products)          # no flush by default: subnormals kept


def test_fp_environment_is_skipped_off_windows(monkeypatch):
    monkeypatch.setattr(fp, "_fp_api", False)
    assert fp.fp_environment() is None and fp._apply_fp_environment(None) is False
    stats = {}
    fp.run_ordered(lambda x, u: x, range(8), workers=2, name="t-fp-off", stats=stats)
    assert stats["fp_env"] is None and stats["fp_env_applied"] == 0


def _plain_thread(function):
    box = []
    thread = threading.Thread(target=lambda: box.append(function()))
    thread.start()
    thread.join()
    return box[0]


# ================================================================== exclusive sections

def test_pools_inside_an_exclusive_section_run_serially_and_say_so():
    main = threading.current_thread()
    names = []
    with fp.exclusive("siril-stack"):
        assert fp.exclusive_sections() == ("siril-stack",)
        stats = {}
        out = fp.run_ordered(lambda x, u: (names.append(threading.current_thread()), x * 2)[1], range(10),
                             workers=8, name="t-excl", stats=stats)
        assert out == [x * 2 for x in range(10)] and set(names) == {main}
        assert stats["serial"] and stats["reason"] == "exclusive" and stats["exclusive"] == "siril-stack"
        with fp.exclusive("nox"), fp.exclusive("nox"):
            assert fp.exclusive_sections() == ("nox", "siril-stack")
            stats = {}
            fp.run_ordered(lambda x, u: x, range(4), workers=2, name="t-excl2", stats=stats)
            assert stats["exclusive"] == "nox+siril-stack"
        assert fp.exclusive_sections() == ("siril-stack",)
        stats = {}
        fp.run_ordered(lambda x, u: x, range(4), workers=1, name="t-excl3", stats=stats)
        assert stats["reason"] == "workers"                         # a 1-worker pool is serial for its own reason
    assert fp.exclusive_sections() == ()
    stats = {}
    fp.run_ordered(lambda x, u: x, range(4), workers=2, name="t-excl4", stats=stats)
    assert not stats["serial"]


def test_an_exclusive_section_is_process_wide_and_released_on_errors():
    entered, release = threading.Event(), threading.Event()

    def nox_like():
        with fp.exclusive("nox"):
            entered.set()
            release.wait(5)

    thread = threading.Thread(target=nox_like)
    thread.start()
    try:
        assert entered.wait(5)
        stats = {}
        fp.run_ordered(lambda x, u: x, range(6), workers=3, name="t-excl-pw", stats=stats)
        assert stats["reason"] == "exclusive"                       # opened by another thread
    finally:
        release.set()
        thread.join(5)
    with pytest.raises(RuntimeError):
        with fp.exclusive("siril-stack"):
            raise RuntimeError("Siril failed")
    assert fp.exclusive_sections() == ()


def test_the_exclusive_seam_raises_for_a_pool_that_would_fan_out(monkeypatch):
    monkeypatch.setattr(fp, "_EXCLUSIVE_RAISES", True)
    ran = []
    with fp.exclusive("siril-stack"):
        with pytest.raises(fp.ExclusiveViolation, match="t-seam.*siril-stack"):
            fp.run_ordered(lambda x, u: ran.append(x), range(8), workers=4, name="t-seam")
        assert fp.run_ordered(lambda x, u: x, range(3), workers=1, name="t-seam1") == [0, 1, 2]
    assert ran == [] and fp._budget_used == 0


# ================================================================== AESTER_OVERLAP

@pytest.mark.parametrize("value, enabled", [(None, True), ("", True), ("auto", True), (" AUTO ", True), ("off", False),
                                            ("Off", False)])
def test_overlap_knob_values(monkeypatch, value, enabled):
    if value is not None:
        monkeypatch.setenv(fp.ENV_OVERLAP, value)
    fp.validate_environment()
    assert fp.overlap_enabled() is enabled


@pytest.mark.parametrize("bad", ["on", "1", "0", "false", "ｏｆｆ", "auto,off", "of"])
def test_a_malformed_overlap_knob_fails_at_job_start(monkeypatch, bad):
    monkeypatch.setenv(fp.ENV_OVERLAP, bad)
    with pytest.raises(ValueError, match=f"^{fp.ENV_OVERLAP} must be auto or off$"):
        fp.validate_environment()
    with pytest.raises(ValueError, match=fp.ENV_OVERLAP):
        fp.overlap_enabled()


@pytest.mark.parametrize("workers, enabled", [("1", False), ("01", False), ("2", True), ("16", True), ("max", True),
                                              ("auto", True)])
def test_overlap_needs_more_than_one_frame_worker(monkeypatch, workers, enabled):
    monkeypatch.setenv(fp.ENV_WORKERS, workers)
    assert fp.overlap_enabled() is enabled


def test_overlap_is_off_inside_a_pool_unit_and_the_frame_knob_is_still_checked(monkeypatch):
    inside = fp.run_ordered(lambda x, u: fp.overlap_enabled(), range(4), workers=2, name="t-ovl")
    assert inside == [False] * 4 and fp.overlap_enabled()
    monkeypatch.setenv(fp.ENV_WORKERS, "17")
    with pytest.raises(ValueError, match=fp.ENV_WORKERS):
        fp.validate_environment()


@pytest.mark.parametrize("bad", ["on", "ｏｆｆ"])
def test_a_malformed_overlap_knob_fails_the_job_at_its_start(job_environment, monkeypatch, bad):
    """core runs frame_pool.validate_environment() first thing at job start (P1b X6): AESTER_OVERLAP fails the job
    there like AESTER_FRAME_WORKERS (U-P2 precedent), before any engine stage."""
    from starstack import core
    job = core.submit_job(str(job_environment.source), str(job_environment.output))
    monkeypatch.setenv(fp.ENV_OVERLAP, bad)
    core.run_job(job["job_dir"])
    result = core.load_job(job["job_dir"])
    assert result["state"] == "failed" and "AESTER_OVERLAP must be auto or off" in result["error"]
    assert job_environment.calls == []                              # no engine stage ran


# ================================================================== Beside

def test_beside_returns_the_value_and_runs_beside_the_caller():
    main = threading.current_thread()
    seen = {}

    def work(probe):
        seen["thread"] = threading.current_thread()
        probe()
        return 42

    helper = fp.Beside.start(work, "t-value")
    assert helper.result() == 42 and helper.done()
    assert seen["thread"] is not main and seen["thread"].name == "aester-beside-t-value"
    with pytest.raises(RuntimeError, match="already taken"):
        helper.result()


def test_beside_reraises_the_helpers_exception_object():
    error = LookupError("scan failed")

    def work(probe):
        raise error

    helper = fp.Beside.start(work, "t-error")
    with pytest.raises(LookupError) as info:
        helper.result()
    assert info.value is error


def test_beside_probe_calls_the_job_check_and_abandon_stops_the_helper():
    calls, swallowed = [], []
    first = threading.Event()

    def check():
        calls.append(threading.current_thread().name)
        first.set()

    def work(probe):
        while True:
            try:
                probe()
            except Exception:                                       # _Abandoned is not an Exception
                swallowed.append(1)
            time.sleep(0.005)

    helper = fp.Beside.start(work, "t-abandon", check=check)
    assert first.wait(5)
    t0 = time.perf_counter()
    helper.abandon()
    assert time.perf_counter() - t0 < 1.0 and helper.done() and helper.abandoned
    assert calls and set(calls) == {"aester-beside-t-abandon"} and swallowed == []
    helper.abandon()                                                # repeatable
    helper.abandon(wait=False)
    with pytest.raises(RuntimeError, match="after abandon"):
        helper.result()
    assert helper.errors == []


def test_beside_cancel_through_the_job_check_reaches_result():
    flag = threading.Event()

    def check():
        if flag.is_set():
            raise Cancelled("用户取消")

    def work(probe):
        while True:
            probe()
            time.sleep(0.005)

    helper = fp.Beside.start(work, "t-cancel", check=check)
    flag.set()
    with pytest.raises(Cancelled, match="用户取消"):
        helper.result()


@pytest.mark.parametrize("exc_type", [Cancelled, KeyboardInterrupt])
def test_a_failing_wait_check_abandons_the_helper_and_is_reraised(exc_type):
    stopped = threading.Event()

    def work(probe):
        try:
            while True:
                probe()
                time.sleep(0.005)
        finally:
            stopped.set()

    raised = exc_type("main thread stops")
    calls = []

    def wait_check():
        calls.append(time.perf_counter())
        if len(calls) == 3:
            raise raised

    helper = fp.Beside.start(work, "t-wait")
    t0 = time.perf_counter()
    with pytest.raises(exc_type) as info:
        helper.result(wait_check)
    assert info.value is raised and stopped.is_set() and helper.done() and helper.abandoned
    assert time.perf_counter() - t0 < 3 * fp.POLL_SECONDS + 1.0
    assert all(b - a < fp.POLL_SECONDS + 0.15 for a, b in zip(calls, calls[1:]))


def test_beside_runs_in_a_copy_of_the_callers_context_outside_any_pool():
    seen = {}

    def work(probe):
        seen.update(var=VAR.get(), mode=hashing.active_mode(), over=np.geterr()["over"], in_pool=fp.in_pool())
        VAR.set("helper-wrote")
        return None

    token = VAR.set("caller")
    try:
        with hashing.run_scope("beside-test", mode="on"), np.errstate(over="raise"):
            def from_unit(x, unit):
                helper = fp.Beside.start(work, "t-ctx")
                helper.result()
                return fp.in_pool()
            assert fp.run_ordered(from_unit, range(2), workers=2, name="t-ctx-pool") == [True, True]
    finally:
        VAR.reset(token)
    assert seen == {"var": "caller", "mode": "on", "over": "raise", "in_pool": False}
    assert VAR.get() == "unset"


def test_pools_inside_beside_fan_out_and_share_the_process_budget():
    peak, lock = {"n": 0}, threading.Lock()

    def count():
        n = sum(1 for t in threading.enumerate() if t.name.startswith("aester-t-share"))
        with lock:
            peak["n"] = max(peak["n"], n)

    def unit(x, u):
        count()
        time.sleep(0.02)
        count()
        return x

    inner_stats = {}

    def work(probe):
        return fp.run_ordered(unit, range(40), workers=12, name="t-share-b", check=probe, stats=inner_stats)

    outer_started = threading.Event()

    def outer_unit(x, u):
        outer_started.set()
        return unit(x, u)

    outer_stats = {}
    driver = threading.Thread(target=lambda: fp.run_ordered(outer_unit, range(40), workers=12, name="t-share-a",
                                                            stats=outer_stats))
    driver.start()
    assert outer_started.wait(5)
    helper = fp.Beside.start(work, "t-share")
    assert helper.result() == list(range(40))
    driver.join(10)
    assert peak["n"] <= fp.MAX_FRAME_WORKERS
    assert not inner_stats["serial"] or inner_stats["reason"] == "budget"
    assert outer_stats["workers"] + (0 if inner_stats["serial"] else inner_stats["workers"]) <= fp.MAX_FRAME_WORKERS


def test_a_pool_inside_beside_is_abandoned_midway():
    started = threading.Event()
    seen = {}

    def unit(x, u):
        started.set()
        for _ in range(200):
            u.check()
            time.sleep(0.005)
        return x

    def work(probe):
        try:
            return fp.run_ordered(unit, range(32), workers=4, name="t-midway", check=probe)
        except BaseException as exc:
            seen["exc"] = type(exc).__name__
            raise

    helper = fp.Beside.start(work, "t-midway")
    assert started.wait(5)
    t0 = time.perf_counter()
    helper.abandon()
    assert time.perf_counter() - t0 < 1.0 and seen["exc"] == "_Abandoned"
    assert fp._budget_used == 0


def test_beside_hands_a_failed_result_over_once_and_keeps_nothing():
    class Big:
        def __init__(self):
            self.buf = bytearray(16 << 20)

    refs = []

    def work(probe):
        big = Big()
        refs.append(weakref.ref(big))
        raise ValueError("fails holding a large intermediate")

    helper = fp.Beside.start(work, "t-weak")
    gc.collect()
    gc.disable()
    try:
        try:
            helper.result()
        except ValueError as exc:
            logged = repr(exc)                                      # the caller logs it, then drops the reference
        assert "fails holding" in logged
        assert refs[0]() is None
    finally:
        gc.enable()
    helper_ref = weakref.ref(helper)
    del helper
    gc.collect()
    assert helper_ref() is None


@pytest.mark.parametrize("ending", ["raises", "returns"])
def test_an_outcome_arriving_after_abandon_without_waiting_is_dropped(ending):
    class Big:
        def __init__(self):
            self.buf = bytearray(16 << 20)

    refs, release = [], threading.Event()

    def work(probe):
        release.wait(5)                                             # still running when abandon(wait=False) returns
        big = Big()
        refs.append(weakref.ref(big))
        if ending == "raises":
            raise ValueError("fails after it was abandoned", big)
        return big

    helper = fp.Beside.start(work, "t-late")
    helper.abandon(wait=False)
    assert not helper.done() and helper.abandoned
    release.set()
    helper._thread.join(5)
    assert helper.done() and helper._outcome is None
    gc.collect()
    assert refs and refs[0]() is None
    with pytest.raises(RuntimeError, match="after abandon"):
        helper.result()
    assert helper.errors == []


def test_beside_abandon_records_and_does_not_raise_its_own_errors(monkeypatch):
    release = threading.Event()
    helper = fp.Beside.start(lambda probe: release.wait(5), "t-abandon-err")

    def broken_join(timeout=None):
        raise RuntimeError("join failed")

    monkeypatch.setattr(helper._thread, "join", broken_join)
    helper.abandon()
    assert helper.errors and "join failed" in helper.errors[0]
    monkeypatch.undo()
    release.set()
    helper.abandon()
    assert helper.done()


def test_beside_takes_no_frame_pool_slot():
    release = threading.Event()
    helper = fp.Beside.start(lambda probe: release.wait(5), "t-slot")
    try:
        assert fp._budget_used == 0
        stats = {}
        fp.run_ordered(lambda x, u: x, range(32), workers=16, name="t-slot-pool", stats=stats)
        assert stats["workers"] == 16
    finally:
        release.set()
        helper.result()


# ================================================================== per-run diagnostics

def test_stats_carry_cpu_longest_unit_and_memory_for_both_paths():
    def burn(x, unit):
        end = time.thread_time() + 0.02
        while time.thread_time() < end:
            pass
        if x == 3:
            time.sleep(0.1)
        return x

    for workers in (1, 4):
        stats = {}
        fp.run_ordered(burn, range(8), workers=workers, name="t-diag", stats=stats, unit_bytes=MiB)
        assert stats["unit_cpu_seconds"] >= 8 * 0.02 * 0.9, stats
        assert 0.1 <= stats["max_unit_seconds"] < 1.0
        assert stats["cpu_user_seconds"] + stats["cpu_kernel_seconds"] >= stats["unit_cpu_seconds"] * 0.5
        for key in ("unit_bytes", "reserved_bytes", "fp_env", "fp_env_applied", "wall_seconds", "units"):
            assert key in stats
        assert stats["unit_bytes"] == MiB


def test_render_workers_is_one_inside_a_pool_unit():
    big = 1 << 40
    outside = performance.render_workers(6000, 8000, cpu_count=192, memory_bytes=big)
    inside = fp.run_ordered(lambda x, u: performance.render_workers(6000, 8000, cpu_count=192, memory_bytes=big),
                            range(4), workers=2, name="t-render")
    assert outside > 1 and inside == [1] * 4
