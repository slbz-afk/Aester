"""Efficiency P1b-1: the frame pool (``starstack.frame_pool``) against its serial reference.

The first block is the design's prototype suite (p1b design pool ``final/test_frame_pool_final.py``); the second turns
every review probe into a regression test; the third holds the cross-check additions (R15 prepare check, X2 usable
CPUs on this machine, X11 frozen closure).  Timing limits are the design's (§9.1); the overhead figures take the best
of three repetitions because other work may share the machine.
"""
import contextvars
import gc
import importlib.util
import json
import os
import random
import statistics
import subprocess
import sys
import threading
import time
import weakref
from pathlib import Path

import numpy as np
import pytest

from starstack import frame_pool as fp
from starstack import hashing, profiling, siril

SRC = Path(fp.__file__).resolve().parents[1]
REPO = SRC.parent
VAR = contextvars.ContextVar("frame_pool_test_var", default="unset")
windows = pytest.mark.skipif(os.name != "nt", reason="Windows processor-group APIs")


class Cancelled(RuntimeError):
    pass


class WorkflowCancelled(KeyboardInterrupt):
    pass


def pool_threads(name):
    return [t for t in threading.enumerate() if t.name.startswith(f"aester-{name}")]


@pytest.fixture(autouse=True)
def budget_is_released(monkeypatch):
    monkeypatch.delenv(fp.ENV_WORKERS, raising=False)
    yield
    assert fp._budget_used == 0


# ================================================================== prototype suite

def test_serial_path_is_the_plain_loop():
    trace = []
    main = threading.current_thread()

    def check():
        trace.append("check")

    def prepare(i, x, own_check):
        assert own_check is check                     # serial: prepare's check is the caller's check itself
        trace.append(("submit", i))
        return x

    def fn(item, unit):
        assert threading.current_thread() is main and unit.check is check
        trace.append(("fn", item))
        VAR.set("leaks-like-the-old-loop")
        return item * 2

    before = set(threading.enumerate())
    stats = {}
    out = fp.run_ordered(fn, [1, 2, 3], workers=1, name="t-serial", check=check, stats=stats, prepare=prepare)
    assert out == [2, 4, 6] and not (set(threading.enumerate()) - before)
    assert trace == ["check", ("submit", 0), ("fn", 1), "check", ("submit", 1), ("fn", 2),
                     "check", ("submit", 2), ("fn", 3)]
    assert VAR.get() == "leaks-like-the-old-loop"    # literally the old loop: no context copy
    assert stats["serial"] and stats["reason"] == "workers" and stats["workers"] == 1
    VAR.set("unset")


def test_serial_reasons_units_and_check_absent():
    stats = {}
    assert fp.run_ordered(lambda x, u: x + 1, [5], workers=8, name="t-units", stats=stats,
                          prepare=lambda i, x, c: (c(), x)[1]) == [6]          # no check: prepare gets a no-op
    assert stats["serial"] and stats["reason"] == "units" and stats["requested"] == 8
    stats = {}
    assert fp.run_ordered(lambda x, u: x, [], workers=4, name="t-empty", stats=stats) == []
    assert stats["reason"] == "units" and stats["units"] == 0


@pytest.mark.parametrize("workers", [2, 4, 8, 16])
@pytest.mark.parametrize("seed", range(5))
def test_ordered_under_jitter(workers, seed):
    rng = random.Random(seed)
    delays = [rng.random() * 0.004 for _ in range(150)]
    order = []

    def fn(i, unit):
        time.sleep(delays[i])
        return i * i

    stats = {}
    fp.run_ordered(fn, range(150), workers=workers, name="t-order", on_result=lambda i, r: order.append((i, r)),
                   stats=stats)
    assert order == [(i, i * i) for i in range(150)]
    assert stats["max_running"] <= workers and stats["max_held"] <= stats["inflight"]
    assert stats["workers"] == workers and not stats["serial"]
    assert not pool_threads("t-order")


def test_float_fold_in_unit_order_is_bit_identical():
    rng = np.random.default_rng(3)
    blocks = [rng.standard_normal(1000) * 10.0 ** rng.integers(-8, 8) for _ in range(64)]
    serial = 0.0
    for b in blocks:
        serial += float(b.sum())
    acc = {"s": 0.0}

    def fold(i, r):
        acc["s"] += r
    fp.run_ordered(lambda b, u: (time.sleep(random.random() * 0.002), float(b.sum()))[1], blocks, workers=8,
                   name="t-fold", on_result=fold)
    assert acc["s"] == serial
    coalesced = 0.0
    for k in range(0, 64, 8):
        part = 0.0
        for b in blocks[k:k + 8]:
            part += float(b.sum())
        coalesced += part
    print("coalesced differs:", coalesced != serial)        # rule R2: never fold float partials of coalesced ranges


@pytest.mark.parametrize("seed", range(10))
def test_lowest_index_failure_wins(seed):
    rng = random.Random(seed)
    k1, k2 = sorted(rng.sample(range(5, 60), 2))
    delivered = []
    errors = {k1: ValueError("k1"), k2: ValueError("k2")}

    def fn(i, unit):
        time.sleep(0.02 if i == k1 else 0.001)       # k2 fails first in wall time
        if i in errors:
            raise errors[i]
        return i

    with pytest.raises(ValueError) as info:
        fp.run_ordered(fn, range(80), workers=6, name="t-fail", on_result=lambda i, r: delivered.append(i))
    assert info.value is errors[k1]
    assert delivered == list(range(k1))
    assert not pool_threads("t-fail")


def test_on_result_and_prepare_failures_are_positions():
    def boom_result(i, r):
        if i == 7:
            raise KeyError("fold7")
    with pytest.raises(KeyError, match="fold7"):
        fp.run_ordered(lambda x, u: x, range(20), workers=4, name="t-pos", on_result=boom_result)

    def boom_submit(i, x, check):
        if i == 9:
            raise LookupError("submit9")
        return x
    seen = []
    with pytest.raises(LookupError, match="submit9"):
        fp.run_ordered(lambda x, u: x, range(20), workers=4, name="t-pos", prepare=boom_submit,
                       on_result=lambda i, r: seen.append(i))
    assert seen == list(range(9))


def test_base_exception_from_unit_propagates_after_join():
    def fn(i, unit):
        if i == 3:
            raise SystemExit("unit-exit")
        time.sleep(0.01)
        return i
    with pytest.raises(SystemExit):
        fp.run_ordered(fn, range(30), workers=4, name="t-base")
    assert not pool_threads("t-base")


@pytest.mark.parametrize("exc_type", [Cancelled, WorkflowCancelled])
def test_cancel_latency_and_identity(exc_type):
    latencies = []
    for trial in range(10):
        flag = threading.Event()
        raised = []

        def check():
            if flag.is_set():
                exc = exc_type("用户取消")               # a fresh object per call, as core.check does
                raised.append(exc)
                raise exc

        def fn(i, unit):
            end = time.perf_counter() + 2.0
            while time.perf_counter() < end:
                unit.check()
                time.sleep(0.01)
            return i

        timer = threading.Timer(0.3 + 0.05 * trial, flag.set)
        timer.start()
        t0 = time.perf_counter()
        with pytest.raises(exc_type) as info:
            fp.run_ordered(fn, range(64), workers=8, name="t-cancel", check=check)
        latencies.append(time.perf_counter() - t0 - (0.3 + 0.05 * trial))
        assert len(raised) == 1 and info.value is raised[0]
        assert not pool_threads("t-cancel")
    print("cancel latency s: max %.3f median %.3f" % (max(latencies), statistics.median(latencies)))
    assert max(latencies) <= 0.35


def test_cancel_with_non_probing_units_waits_one_unit():
    flag = threading.Event()

    def check():
        if flag.is_set():
            raise Cancelled("x")

    def fn(i, unit):
        time.sleep(0.5)
        return i
    threading.Timer(0.2, flag.set).start()
    t0 = time.perf_counter()
    with pytest.raises(Cancelled):
        fp.run_ordered(fn, range(16), workers=4, name="t-noprobe", check=check)
    assert time.perf_counter() - t0 <= 0.5 + fp.POLL_SECONDS + 0.2


def test_no_leaked_threads_over_many_exits():
    # Compare thread sets, not counts: a thread left by an earlier test may end meanwhile (full-suite flake, P1b T2).
    base = set(threading.enumerate())
    for n in range(100):
        mode = n % 4
        try:
            if mode == 0:
                fp.run_ordered(lambda x, u: x, range(10), workers=4, name="t-leak")
            elif mode == 1:
                fp.run_ordered(lambda x, u: 1 / (x - 5), range(10), workers=4, name="t-leak")
            elif mode == 2:
                def check(c=iter(range(3))):
                    if next(c, None) is None:
                        raise Cancelled("stop")
                fp.run_ordered(lambda x, u: time.sleep(0.01), range(10), workers=4, name="t-leak", check=check)
            else:
                fp.run_ordered(lambda x, u: x, range(10), workers=4, name="t-leak",
                               on_result=lambda i, r: (_ for _ in ()).throw(WorkflowCancelled("k")) if i == 4 else None)
        except (ZeroDivisionError, Cancelled, WorkflowCancelled):
            pass
    assert not (set(threading.enumerate()) - base) and not pool_threads("t-leak")


def test_context_is_copied_per_task():
    seen = []

    def fn(i, unit):
        seen.append((hashing.active_mode(), hashing.digest_writes(), siril._JOB_STAGE.get(), VAR.get(),
                     np.geterr()["over"], len(profiling.active()), fp.in_pool(),
                     threading.current_thread().name.startswith("aester-t-ctx")))
        VAR.set("worker-wrote")
        try:
            np.array([1e308]) * 10.0
        except FloatingPointError:
            return "fpe"
        return "no-fpe"

    token = VAR.set("caller")
    try:
        with hashing.run_scope("frame-pool-test", mode="on"), siril.job_stage_launches(), np.errstate(over="raise"):
            out = fp.run_ordered(fn, range(12), workers=4, name="t-ctx")
    finally:
        VAR.reset(token)
    assert out == ["fpe"] * 12
    assert set(seen) == {("on", True, True, "caller", "raise", 0, True, True)}
    assert VAR.get() == "unset" and not fp.in_pool()


def test_nested_pool_in_unit_runs_serially():
    names, reasons = [], []

    def inner(x, u):
        names.append(threading.current_thread().name)
        return x

    def outer(x, u):
        st = {}
        r = fp.run_ordered(inner, range(3), workers=4, name="t-inner", stats=st)
        reasons.append(st["reason"])
        return r
    fp.run_ordered(outer, range(4), workers=2, name="t-outer")
    assert all(n.startswith("aester-t-outer") for n in names) and set(reasons) == {"nested"}


def test_per_worker_state_is_exclusive_and_finalized_once():
    made, closed = [], []

    def init():
        s = {"busy": False, "id": len(made), "units": 0}
        made.append(s)
        return s

    def fn(i, unit):
        assert not unit.state["busy"]
        unit.state["busy"] = True
        time.sleep(random.random() * 0.002)
        unit.state["units"] += 1
        unit.state["busy"] = False
        return unit.state["id"]
    order = []
    fp.run_ordered(fn, range(40), workers=4, name="t-state", init=init, finalize=closed.append,
                   check=lambda: order.append("check") if len(made) == 4 else pytest.fail("init after check"))
    assert len(made) == 4 and closed == made and sum(s["units"] for s in made) == 40


def test_worker_resolution(monkeypatch):
    monkeypatch.setenv(fp.ENV_WORKERS, "3")
    assert fp.resolve_workers(8, units=10) == 3
    monkeypatch.setenv(fp.ENV_WORKERS, "max")
    assert fp.resolve_workers(2, units=100) == min(16, fp.usable_cpus())
    assert fp.resolve_workers(2, units=5) == min(5, fp.usable_cpus())
    monkeypatch.setenv(fp.ENV_WORKERS, "auto")
    assert fp.resolve_workers(5, units=3) == 3
    assert fp.resolve_workers(None, units=100, cap=6) == min(6, fp.usable_cpus())
    monkeypatch.setenv(fp.ENV_WORKERS, " MAX ")
    assert fp.resolve_workers(1, units=100) == min(16, fp.usable_cpus())
    monkeypatch.delenv(fp.ENV_WORKERS)
    assert fp.resolve_workers(7, units=100) == 7 and fp.resolve_workers(40, units=100) == 16
    for bad in ("0", "17", "32", "eight", "-1", "4.0"):
        monkeypatch.setenv(fp.ENV_WORKERS, bad)
        with pytest.raises(ValueError, match="AESTER_FRAME_WORKERS"):
            fp.resolve_workers(1, units=3)
        with pytest.raises(ValueError, match="AESTER_FRAME_WORKERS"):
            fp.validate_environment()
    monkeypatch.delenv(fp.ENV_WORKERS)
    fp.validate_environment()
    with pytest.raises(ValueError):
        fp.run_ordered(lambda x, u: x, range(3), workers=0, name="t-bad")


@pytest.mark.parametrize("bad", ["²", "¹⁶", "８", "１６", "٣", "1²", "+8", "1_6",
                                 "Ⅸ"])
def test_non_ascii_or_signed_digits_fail_with_the_knobs_own_message(monkeypatch, bad):
    """Review pool-3: str.isdigit() accepts superscript ('²', which int() rejects with its own message), full-width
    ('８', which int() reads) and other Unicode digits; every malformed value must raise the knob's message."""
    monkeypatch.setenv(fp.ENV_WORKERS, bad)
    with pytest.raises(ValueError, match=f"^{fp.ENV_WORKERS} must be auto, max or 1-{fp.MAX_FRAME_WORKERS}$"):
        fp.validate_environment()
    with pytest.raises(ValueError, match=f"^{fp.ENV_WORKERS} must be"):
        fp.resolve_workers(1, units=3)
    for good in ("1", "08", " 16 ", "MAX", "Auto", ""):                # ASCII forms stay accepted
        monkeypatch.setenv(fp.ENV_WORKERS, good)
        fp.validate_environment()


def test_default_workers_charges_blas_slot():
    gib = 1 << 30
    cpus = fp.usable_cpus()
    # 8 GiB free: budget min(8/3, 7) GiB = 2.67 GiB; per worker 230 MiB + 32 MiB -> 10
    assert fp.default_workers(cap=8, unit_bytes=230 << 20, available_bytes=8 * gib, units=48) == min(8, max(1, cpus // 2), 10)
    # 2 GiB free: budget min(0.67, 1) GiB = 682 MiB -> 2 workers (3 without the BLAS slot)
    assert fp.default_workers(cap=8, unit_bytes=230 << 20, available_bytes=2 * gib, units=48) == min(2, max(1, cpus // 2))
    assert fp.default_workers(cap=8, unit_bytes=230 << 20, available_bytes=2 * gib, units=48, blas=False) == min(2, max(1, cpus // 2), 3)
    assert fp.default_workers(cap=8, unit_bytes=230 << 20, available_bytes=None, units=48) == 1
    assert fp.default_workers(cap=8, unit_bytes=230 << 20, available_bytes=512 << 20, units=48) == 1
    assert fp.default_workers(cap=8, unit_bytes=1, available_bytes=1 << 40, units=3) == min(3, max(1, cpus // 2))


def test_overhead_per_unit():
    pers, spins = [], []
    for _ in range(3):
        t0 = time.perf_counter()
        fp.run_ordered(lambda x, u: x, range(20000), workers=4, name="t-ovh")
        pers.append((time.perf_counter() - t0) / 20000)
        t0 = time.perf_counter()
        for _ in range(50):
            fp.run_ordered(lambda x, u: x, range(16), workers=16, name="t-spin")
        spins.append((time.perf_counter() - t0) / 50)
    per, spin = min(pers), min(spins)
    print("overhead per unit %.1f us; 16-thread pool create+16 units+join %.2f ms" % (per * 1e6, spin * 1e3))
    assert per < 50e-6 and spin < 10e-3


@pytest.mark.parametrize("seed", range(3))
def test_jitter_seam_keeps_order_and_bits(monkeypatch, seed):
    rng = random.Random(seed)
    monkeypatch.setattr(fp, "_before_unit", lambda i: time.sleep(rng.random() * 0.003))
    data = np.random.default_rng(seed).standard_normal((64, 4096)).astype(np.float32)
    serial = fp.run_ordered(lambda row, u: np.sqrt(np.abs(row)).sum(dtype=np.float64), data, workers=1, name="t-j")
    for workers in (2, 8, 16):
        assert fp.run_ordered(lambda row, u: np.sqrt(np.abs(row)).sum(dtype=np.float64), data, workers=workers,
                              name="t-j") == serial


def test_prepare_runs_in_coordinator_in_order():
    main = threading.current_thread()
    seen = []

    def prepare(i, x, check):
        assert threading.current_thread() is main
        seen.append(i)
        return x + 100
    out = fp.run_ordered(lambda x, u: x, range(50), workers=8, name="t-prep", prepare=prepare)
    assert out == list(range(100, 150)) and seen == list(range(50))


# ================================================================== review regressions

def test_M1_stop_index_never_moves_up_and_submission_stops():
    """crit1 probe A: unit 2 fails early; prepare(4) must not run and unit 3 must abort at its probe."""
    events, prepared = [], []
    err2 = ValueError("unit2")

    def prepare(i, x, check):
        prepared.append(i)
        if i == 4:
            raise LookupError("prepare4")
        return x

    def fn(x, unit):
        i = unit.index
        if i == 0:
            time.sleep(0.5); return i
        if i == 1:
            time.sleep(1.2); return i
        if i == 2:
            time.sleep(0.05); raise err2
        if i == 3:
            time.sleep(0.8)
            unit.check()
            events.append("unit3 side effect")
            return i
        return i

    with pytest.raises(ValueError) as info:
        fp.run_ordered(fn, range(8), workers=3, inflight=4, name="t-m1", prepare=prepare)
    assert info.value is err2 and prepared == [0, 1, 2, 3] and events == []


@pytest.mark.parametrize("probing", [False, True])
def test_m1_check_exception_is_a_cancel_and_check_is_not_called_again(probing):
    """crit1 probe C: a fresh exception per raising call; check() must raise once and that object propagates."""
    calls, raises, updates = [], [], []
    flag = {"on": False}

    def check():
        calls.append(time.perf_counter())
        if flag["on"]:
            updates.append("job.json cancel_requested")      # core.check's side effect
            exc = Cancelled("用户取消")
            raises.append(exc)
            raise exc

    def prepare(i, x, own_check):
        if i == 3:
            flag["on"] = True
        return x

    def fn(x, unit):
        end = time.perf_counter() + 0.5
        while time.perf_counter() < end:
            if probing:
                unit.check()
            time.sleep(0.01)
        return x

    delivered = []
    with pytest.raises(Cancelled) as info:
        fp.run_ordered(fn, range(10), workers=2, inflight=8, name="t-m1c", check=check, prepare=prepare,
                       on_result=lambda i, r: delivered.append(i))
    assert len(raises) == 1 and info.value is raises[0] and len(updates) == 1
    assert delivered == list(range(len(delivered))) and len(delivered) <= 4
    assert not pool_threads("t-m1c")


def test_m1_finished_head_is_delivered_not_cancelled():
    """A cancel that arrives after the last unit finished must not turn the run into Cancelled (serial completes)."""
    for trial in range(10):
        flag = threading.Event()

        def check():
            if flag.is_set():
                raise Cancelled("late")

        def fn(i, unit):
            if i == 5:
                time.sleep(0.25 + 0.013 * trial)
                flag.set()                   # the cancel appears exactly when the last unit is done
            else:
                time.sleep(0.01)
            return i
        assert fp.run_ordered(fn, range(6), workers=3, name="t-late", check=check) == list(range(6))


def test_m2_init_failure_finalizes_created_states():
    made, finalized = [], []

    def init():
        if len(made) == 2:
            raise MemoryError("init #3")
        s = {"id": len(made)}
        made.append(s)
        return s

    with pytest.raises(MemoryError):
        fp.run_ordered(lambda x, u: x, range(10), workers=4, name="t-m2d", init=init, finalize=finalized.append)
    assert finalized == made and len(made) == 2 and not pool_threads("t-m2d")


@pytest.mark.parametrize("workers", [1, 4])
def test_m2_every_state_finalized_and_serial_precedence(workers):
    """finalize() raising: every state is still finalized; like the serial loop's `finally:`, the finalize error is
    raised and carries the unit's failure as __context__."""
    finalized = []
    unit_error = ValueError("unit 5")

    def finalize(state):
        finalized.append(state)
        if len(finalized) == 1:
            raise OSError("close failed")

    def fn(x, u):
        if u.index == 5:
            raise unit_error
        return x

    with pytest.raises(OSError) as info:
        fp.run_ordered(fn, range(10), workers=workers, name="t-m2e", init=lambda: object(), finalize=finalize)
    assert len(finalized) == workers and info.value.__context__ is unit_error


def test_M4_budget_caps_pool_threads_process_wide():
    """crit1 probe F: a pool started from on_result gets only the free slots (none here: it runs serially)."""
    peak = {"n": 0}
    lock = threading.Lock()
    inner_stats = []

    def count():
        n = sum(1 for t in threading.enumerate() if t.name.startswith("aester-t-m4"))
        with lock:
            peak["n"] = max(peak["n"], n)

    def inner(x, u):
        count(); time.sleep(0.02); count(); return x

    def outer(x, u):
        count(); time.sleep(0.1); count(); return x

    def on_result(i, r):
        if i in (0, 3):
            st = {}
            fp.run_ordered(inner, range(16), workers=16, name="t-m4-inner", stats=st)
            inner_stats.append(st)

    fp.run_ordered(outer, range(32), workers=16, name="t-m4-outer", on_result=on_result)
    assert peak["n"] <= fp.MAX_FRAME_WORKERS
    assert all(s["serial"] and s["reason"] == "budget" for s in inner_stats)
    st = {}
    fp.run_ordered(outer, range(20), workers=10, name="t-m4-outer",
                   on_result=lambda i, r: fp.run_ordered(inner, range(16), workers=16, name="t-m4-inner", stats=st)
                   if i == 0 else None)
    assert st["workers"] == 6 and st["requested"] == 16      # 16 - 10 slots were free


def test_M4_no_openblas_warning_with_pool_started_from_on_result():
    """crit5 as a test: outer 16 + inner 16 requested from on_result -> no 'NUM_THREADS exceeded'."""
    child = r'''
import sys
sys.path.insert(0, sys.argv[1])
import numpy as np
from starstack import frame_pool as fp
a = np.random.default_rng(0).random((1_000_000, 3)); h = np.random.default_rng(1).random((3, 3))
ref = a @ h.T
bad = []
def blas(x, u):
    for _ in range(6):
        if not np.array_equal(a @ h.T, ref): bad.append(x)
    return x
def on_result(i, r):
    if i == 0:
        fp.run_ordered(blas, range(16), workers=16, name="inner")
fp.run_ordered(blas, range(64), workers=16, name="outer", on_result=on_result)
print("RESULT", "MISMATCH" if bad else "EQUAL", flush=True)
'''
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith(("OPENBLAS_", "OMP_"))}
    r = subprocess.run([sys.executable, "-X", "utf8", "-c", child, str(SRC)], capture_output=True, text=True,
                       timeout=170, env=env)
    text = r.stdout + r.stderr
    assert "RESULT EQUAL" in text and "NUM_THREADS exceeded" not in text, text[-600:]


def test_m3_aborted_is_not_swallowed_by_except_exception():
    side = []
    flag = threading.Event()

    def check():
        if flag.is_set():
            raise Cancelled("x")

    def fn(x, unit):
        try:
            for _ in range(100):
                unit.check(); time.sleep(0.01)
        except Exception:
            side.append(unit.index)
            return None
        return x

    threading.Timer(0.15, flag.set).start()
    with pytest.raises(Cancelled):
        fp.run_ordered(fn, range(16), workers=8, name="t-m3", check=check)
    assert side == [] and issubclass(fp._Aborted, BaseException) and not issubclass(fp._Aborted, Exception)


@pytest.mark.parametrize("workers", [1, 2])
def test_m4_audit_detects_unbalanced_contextvar_write(monkeypatch, workers):
    monkeypatch.setattr(fp, "_AUDIT_CONTEXT", True)

    def fn(x, unit):
        if unit.index == 0:
            np.seterr(over="raise")
        return x
    ctx = contextvars.copy_context()
    with pytest.raises(fp.ContextLeak):
        ctx.run(lambda: fp.run_ordered(fn, range(4), workers=workers, name="t-audit"))


@pytest.mark.parametrize("workers", [1, 4])
def test_m4_audit_accepts_balanced_writes(monkeypatch, workers):
    monkeypatch.setattr(fp, "_AUDIT_CONTEXT", True)

    def fn(x, unit):
        with np.errstate(over="ignore"), hashing.run_scope("t", mode="on"):
            token = VAR.set("inside")
            VAR.reset(token)
        return x
    assert fp.run_ordered(fn, range(8), workers=workers, name="t-audit-ok", check=lambda: None,
                          prepare=lambda i, x, c: x, on_result=lambda i, r: None) is None


@pytest.mark.parametrize("kind", ["failure", "cancel", "prepare", "on_result", "prepare_check"])
def test_m5_failure_does_not_retain_states_or_results(kind):
    class Big:
        def __init__(self):
            self.buf = bytearray(8 << 20)
    refs = {"states": [], "results": []}

    def init():
        s = Big(); refs["states"].append(weakref.ref(s)); return s

    def fn(x, unit):
        if x == 0:
            time.sleep(0.3)
        if kind == "failure" and x == 1:
            raise ValueError("unit 1 fails")
        r = Big(); refs["results"].append(weakref.ref(r))
        return r

    flag = threading.Event()

    def check():
        if flag.is_set():
            raise Cancelled("stop")

    def prepare(i, x, own_check):
        if kind == "prepare" and i == 6:
            raise LookupError("prepare 6")
        if kind == "prepare_check" and i == 6:
            flag.set()
            own_check()
        return x

    def on_result(i, r):
        if kind == "on_result" and i == 0:
            raise KeyError("fold 0")

    if kind == "cancel":
        threading.Timer(0.15, flag.set).start()
    gc.collect()
    gc.disable()
    try:
        try:
            fp.run_ordered(fn, range(8), workers=4, name="t-m5", init=init, check=check, prepare=prepare,
                           on_result=on_result)
        except (ValueError, Cancelled, LookupError, KeyError):
            pass
        time.sleep(0.05)
        alive = {k: sum(1 for r in v if r() is not None) for k, v in refs.items()}
    finally:
        gc.enable()
    assert alive == {"states": 0, "results": 0}, alive


def test_M2_prepare_may_lead_the_fold_by_inflight():
    """Documents rule R11: prepare(k) can run before on_result(j<k); max lead = inflight - 1."""
    trace, fold = [], {"n": 0}

    def prepare(i, x, check):
        trace.append(("P", i, fold["n"]))
        return x
    fp.run_ordered(lambda x, u: (time.sleep(0.01), x)[1], range(12), workers=2, inflight=12, name="t-lead",
                   prepare=prepare, on_result=lambda i, r: fold.__setitem__("n", fold["n"] + 1))
    lead = max(i - n for _, i, n in trace)
    assert lead >= 2          # the parallel path does lead; serial would give 0
    assert lead <= 12 - 1


def test_M3_fresh_state_seam_exposes_history_dependent_units(monkeypatch):
    """Rule R12 seam: with _STATE_POLICY='fresh' a history-free unit gives the serial result; a unit that reads
    state written by earlier units does not."""
    def init():
        return {"seen": 0}

    def history_free(x, unit):
        unit.state["scratch"] = x * 3            # written before read in the same unit
        return unit.state["scratch"]

    def history_dependent(x, unit):
        unit.state["seen"] += 1
        return unit.state["seen"]

    ref_free = fp.run_ordered(history_free, range(30), workers=1, name="t-fresh", init=init)
    ref_dep = fp.run_ordered(history_dependent, range(30), workers=1, name="t-fresh", init=init)
    monkeypatch.setattr(fp, "_STATE_POLICY", "fresh")
    made = []
    assert fp.run_ordered(history_free, range(30), workers=4, name="t-fresh",
                          init=lambda: made.append(1) or {"seen": 0}) == ref_free
    assert len(made) == 30
    assert fp.run_ordered(history_dependent, range(30), workers=4, name="t-fresh", init=init) != ref_dep


def test_budget_released_on_every_exit():
    for mode in range(5):
        try:
            if mode == 0:
                fp.run_ordered(lambda x, u: x, range(10), workers=8, name="t-bud")
            elif mode == 1:
                fp.run_ordered(lambda x, u: 1 / (x - 3), range(10), workers=8, name="t-bud")
            elif mode == 2:
                fp.run_ordered(lambda x, u: x, range(10), workers=8, name="t-bud",
                               init=lambda: (_ for _ in ()).throw(MemoryError("init")))
            elif mode == 3:
                fp.run_ordered(lambda x, u: x, range(10), workers=8, name="t-bud",
                               check=lambda: (_ for _ in ()).throw(Cancelled("c")))
            else:
                fp.run_ordered(lambda x, u: x, range(10), workers=8, name="t-bud",
                               prepare=lambda i, x, c: (_ for _ in ()).throw(Cancelled("p")) if i == 4 else x)
        except (ZeroDivisionError, MemoryError, Cancelled):
            pass
        assert fp._budget_used == 0


# ================================================================== cross-check additions

@pytest.mark.parametrize("swallow", [False, True])
def test_R15_blocking_prepare_check_is_a_cancel_with_one_side_effect(swallow):
    """X18: a prepare that blocks (a procpool ring) and waits through its own check: the job's check() raises once,
    its side effect happens once, the first object is raised, and check() is never called again.  ``swallow``: the
    prepare catches the cancel and asks again; it gets the kept object, the job's check() is not called again."""
    calls, raises, updates = [], [], []
    flag = threading.Event()
    asked_again = []

    def check():
        calls.append("raise" if flag.is_set() else "ok")
        if flag.is_set():
            updates.append("job.json cancel_requested")
            exc = Cancelled("用户取消")
            raises.append(exc)
            raise exc

    def prepare(i, x, own_check):
        if i < 4:
            return x
        if i == 4:
            threading.Timer(0.15, flag.set).start()
        while True:                                 # blocks until a cancel, polling its check like a ring would
            try:
                own_check()
            except Cancelled as exc:
                if not swallow:
                    raise
                before = len(calls)
                with pytest.raises(Cancelled) as again:
                    own_check()
                asked_again.append((again.value is exc, len(calls) == before))
                raise RuntimeError("ring gave up") from exc
            time.sleep(0.01)

    delivered = []
    with pytest.raises(Cancelled) as info:
        fp.run_ordered(lambda x, u: (time.sleep(0.02), x)[1], range(10), workers=3, name="t-r15", check=check,
                       prepare=prepare, on_result=lambda i, r: delivered.append(i))
    assert len(raises) == 1 and len(updates) == 1 and info.value is raises[0]
    assert calls.count("raise") == 1 and calls[-1] == "raise"      # never called again after it raised
    assert delivered == list(range(len(delivered))) and len(delivered) <= 4
    if swallow:
        assert asked_again == [(True, True)]
    assert not pool_threads("t-r15")


def test_R15_prepare_that_swallows_the_cancel_submits_nothing_more():
    seen, raises = [], []
    flag = threading.Event()

    def check():
        if flag.is_set():
            exc = Cancelled("stop")
            raises.append(exc)
            raise exc

    def prepare(i, x, own_check):
        seen.append(i)
        if i == 3:
            flag.set()
            try:
                own_check()
            except Cancelled:
                pass                                    # rule R15 forbids this; the pool still cancels
        return x

    ran = []
    with pytest.raises(Cancelled) as info:
        fp.run_ordered(lambda x, u: ran.append(x) or x, range(10), workers=2, name="t-r15s", check=check,
                       prepare=prepare, check_each=False)
    assert info.value is raises[0] and len(raises) == 1
    assert seen == [0, 1, 2, 3] and 3 not in ran


def test_R15_serial_prepare_check_is_the_callers_check():
    flag = {"on": False}

    def check():
        if flag["on"]:
            raise Cancelled("serial")

    def prepare(i, x, own_check):
        if i == 2:
            flag["on"] = True
            own_check()
        return x
    with pytest.raises(Cancelled, match="serial"):
        fp.run_ordered(lambda x, u: x, range(5), workers=1, name="t-r15-serial", check=check, prepare=prepare,
                       check_each=False)


CPU_PROBE = r'''
import ctypes, json, os, sys
from ctypes import wintypes
sys.path.insert(0, sys.argv[1])
from starstack import frame_pool as fp
k32 = ctypes.WinDLL("kernel32", use_last_error=True)
k32.GetCurrentProcess.restype = wintypes.HANDLE
k32.GetProcessGroupAffinity.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.USHORT), ctypes.POINTER(wintypes.USHORT)]
k32.GetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
k32.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
k32.GetActiveProcessorCount.argtypes = [wintypes.WORD]; k32.GetActiveProcessorCount.restype = wintypes.DWORD
k32.GetActiveProcessorGroupCount.restype = wintypes.WORD
def facts():
    me = k32.GetCurrentProcess()
    n = wintypes.USHORT(64); groups = (wintypes.USHORT * 64)()
    ok = bool(k32.GetProcessGroupAffinity(me, ctypes.byref(n), groups))
    p, s = ctypes.c_size_t(), ctypes.c_size_t()
    k32.GetProcessAffinityMask(me, ctypes.byref(p), ctypes.byref(s))
    return {"groups_ok": ok, "process_groups": list(groups[:n.value]), "mask_bits": bin(p.value).count("1"),
            "system_mask_bits": bin(s.value).count("1"), "os_cpu_count": os.cpu_count(),
            "active_groups": k32.GetActiveProcessorGroupCount(), "active_all": k32.GetActiveProcessorCount(0xFFFF),
            "usable_cpus": fp.usable_cpus()}
out = {"default": facts()}
if sys.argv[2] == "16":
    if not k32.SetProcessAffinityMask(k32.GetCurrentProcess(), (1 << 16) - 1):   # as bench_workflow --affinity 16
        raise SystemExit("SetProcessAffinityMask failed %d" % ctypes.get_last_error())
    out["affinity16"] = facts()
print(json.dumps(out))
'''


def _cpu_probe(mode):
    r = subprocess.run([sys.executable, "-X", "utf8", "-c", CPU_PROBE, str(SRC), mode], capture_output=True, text=True,
                       timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


@windows
def test_X2_usable_cpus_default_process_and_16_processor_affinity(record_property):
    """X2: record what a default process sees on this machine (groups, mask, result) and check the bench's 16-LP
    affinity (the first 16 processors of the process's group) gives 16, also in a child it starts (job workers)."""
    default = _cpu_probe("default")["default"]
    print("default process:", json.dumps(default))
    record_property("usable_cpus_default", json.dumps(default))
    if default["groups_ok"] and len(default["process_groups"]) == 1:
        assert default["usable_cpus"] == default["mask_bits"]
    else:
        assert default["usable_cpus"] == default["os_cpu_count"]
    assert 1 <= default["usable_cpus"] <= max(default["active_all"], default["os_cpu_count"])
    if default["system_mask_bits"] < 16:
        pytest.skip("fewer than 16 logical processors in this process's group")
    pinned = _cpu_probe("16")["affinity16"]
    print("affinity 16:", json.dumps(pinned))
    record_property("usable_cpus_affinity16", json.dumps(pinned))
    assert pinned["mask_bits"] == 16 and len(pinned["process_groups"]) == 1 and pinned["usable_cpus"] == 16
    # a job worker inherits the bench's affinity: a child of the pinned process sees 16 as well
    child_code = ("import ctypes, sys, subprocess\nfrom ctypes import wintypes\n"
                  "k32 = ctypes.WinDLL('kernel32', use_last_error=True)\n"
                  "k32.GetCurrentProcess.restype = wintypes.HANDLE\n"
                  "k32.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]\n"
                  "assert k32.SetProcessAffinityMask(k32.GetCurrentProcess(), (1 << 16) - 1)\n"
                  "r = subprocess.run([sys.executable, '-X', 'utf8', '-c', "
                  "'import sys; sys.path.insert(0, sys.argv[1]); from starstack import frame_pool as fp; "
                  "print(fp.usable_cpus())', sys.argv[1]], capture_output=True, text=True, timeout=60)\n"
                  "print(r.stdout.strip() or r.stderr)\n")
    r = subprocess.run([sys.executable, "-X", "utf8", "-c", child_code, str(SRC)], capture_output=True, text=True,
                       timeout=120)
    assert r.returncode == 0 and r.stdout.strip().splitlines()[-1] == "16", (r.stdout, r.stderr)


def test_X11_frozen_holdout_closure_includes_frame_pool():
    """sky_validity imports frame_pool: the image-holdout runner freezes it (an unlisted import fails closed)."""
    spec = importlib.util.spec_from_file_location("image_holdout_runner_frame_pool", REPO / "scripts" / "verify_image_holdout.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    assert "frame_pool" in runner.SOURCE_MODULES
    manifest = runner.source_manifest()
    assert str(Path("src/starstack") / "frame_pool.py") in manifest
    assert runner._local_module_names((SRC / "starstack" / "frame_pool.py").read_text(encoding="utf-8")) == set()
