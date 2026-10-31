"""StageTimer instrumentation for the governor (efficiency P1b-2 §3.11, A5): local peaks, sampler, windows."""
import os
import threading
import time

import numpy as np
import pytest

from starstack import profiling

WINDOWS = os.name == "nt"
MB = 10 ** 6


@pytest.mark.skipif(not WINDOWS, reason="process memory counters")
def test_local_peaks_are_per_phase_while_the_lifetime_peak_is_not():
    timer = profiling.StageTimer(peak_sampler_hz=10)
    timer.record_system_memory = True          # what resources.shadow_start sets at shadow-full (U-G4)
    start = profiling.peak_reading()[0]
    timer.phase("A")
    a = np.ones(400 * MB // 8)
    time.sleep(0.25)
    del a
    timer.phase("B")
    b = np.ones(200 * MB // 8)
    time.sleep(0.25)
    timer.phase("C")
    del b
    report = timer.finish()
    phases = report["resources"]["phases"]
    peak_a = phases["A"]["local_peak_private_bytes"]
    peak_b = phases["B"]["local_peak_private_bytes"]
    lifetime = report["resources"]["total"]["peak_private_bytes"]
    assert peak_a >= start + 350 * MB
    assert start + 150 * MB <= peak_b < lifetime
    assert peak_b < peak_a - 100 * MB
    assert phases["C"]["local_peak_private_bytes"] < peak_a
    assert all(view["min_system_available_bytes"] for view in (phases["A"], phases["B"]))
    sampler = report["resources"]["peak_sampler"]
    assert sampler["hz"] == 10 and sampler["ticks"] >= 3 and sampler["thread"] == "aester-peak-sampler"
    assert report["resources"]["local_peak_semantics"].startswith("sampled lower bound")
    assert report["resources"]["version"] == "process-resources-2"


def test_the_sampler_thread_is_joined_by_finish_and_survives_errors():
    calls = []

    def reader():
        calls.append(1)
        if len(calls) == 2:
            raise OSError("counter unavailable")
        return 100, 50, 7

    timer = profiling.StageTimer(peak_sampler_hz=50, peak_reader=reader)
    timer.record_system_memory = True
    thread = timer._peak_thread
    assert thread.name == "aester-peak-sampler" and thread.daemon and thread.is_alive()
    with timer.operation("op"):
        time.sleep(0.1)
    report = timer.finish()
    assert not thread.is_alive() and timer._peak_thread is None
    assert len(calls) >= 3
    view = report["resources"]["operations"]["op"]
    assert view["min_system_available_bytes"] == 7
    assert report["resources"]["phases"]["setup"]["local_peak_working_set_bytes"] >= 50
    assert report["resources"]["peak_sampler"]["ticks"] >= 3
    assert timer.finish() == report


def test_effective_rate_under_gil_load():
    timer = profiling.StageTimer(peak_sampler_hz=10, peak_reader=lambda: (1, 1, 1))
    deadline = time.perf_counter() + 1.2
    x = 0
    while time.perf_counter() < deadline:      # a pure-Python busy loop holding the GIL
        x += 1
    stats = timer.finish()["resources"]["peak_sampler"]
    assert stats["effective_hz"] >= 5 and stats["seconds"] >= 1.0
    assert stats["cpu_share"] is not None and stats["cpu_share"] < 0.05


def test_disabled_by_default_and_sampler_keeps_its_meaning():
    names_before = {t.name for t in threading.enumerate()}
    readings = []

    def fake():
        readings.append(1)
        return {"cpu_user_seconds": 0.0, "cpu_kernel_seconds": 0.0, "private_bytes": 1, "working_set_bytes": 1}

    timer = profiling.StageTimer(sampler=fake)
    assert timer._peak_thread is None and {t.name for t in threading.enumerate()} == names_before
    with timer.operation("x"):
        pass
    report = timer.finish()
    assert readings and "local_peak_semantics" not in report["resources"]
    assert "operation_windows" not in report
    assert "local_peak_private_bytes" not in report["resources"]["phases"]["setup"]


def test_an_unfinished_timer_does_not_keep_its_sampler_alive():
    timer = profiling.StageTimer(peak_sampler_hz=50, peak_reader=lambda: (1, 1, 1))
    thread = timer._peak_thread
    del timer
    import gc
    gc.collect()
    thread.join(2.0)
    assert not thread.is_alive()


def test_operation_windows_are_capped_and_counted():
    now = [0.0]
    timer = profiling.StageTimer(clock=lambda: now[0], peak_sampler_hz=10, peak_reader=lambda: None)
    for index in range(70):
        with timer.operation("script:x"):
            now[0] += 1.0
        now[0] += 0.5
    with timer.operation("governor"):
        now[0] += 0.25
    report = timer.finish()
    windows = report["operation_windows"]
    assert len(windows["script:x"]) == profiling.WINDOW_LIMIT == 64
    assert windows["script:x"][0] == [0.0, 1.0] and windows["script:x"][1] == [1.5, 2.5]
    assert report["operation_windows_dropped"] == {"script:x": 6}
    assert windows["governor"] == [[105.0, 105.25]]


def test_child_peaks_are_aggregated_as_maxima():
    timer = profiling.StageTimer()
    with timer.operation("script:02_export"):
        profiling.note_child_accounting({"peak_job_memory_bytes": 9, "child_peak_working_set_bytes": 5,
                                         "child_peak_commit_bytes": 6})
        profiling.note_child_accounting({"peak_job_memory_bytes": 3, "child_peak_working_set_bytes": 8,
                                         "child_peak_commit_bytes": None})
    children = timer.finish()["resources"]["operations"]["script:02_export"]["children"]
    assert children["child_peak_working_set_bytes"] == 8 and children["child_peak_commit_bytes"] == 6
    assert children["peak_job_memory_bytes"] == 9 and children["processes"] == 2


def test_system_available_memory_is_written_only_when_the_timer_is_told_so():
    """U-G4 (prereg section 8): min_system_available_bytes is a raw system RAM value; a timer omits it unless
    record_system_memory is set (resources.shadow_start does so at shadow-full only). Local peaks stay."""
    for flag in (None, False, True):
        timer = profiling.StageTimer(peak_sampler_hz=50, peak_reader=lambda: (100, 50, 123456789))
        if flag is not None:
            timer.record_system_memory = flag
        with timer.operation("op"):
            time.sleep(0.05)
        report = timer.finish()
        views = list(report["resources"]["phases"].values()) + list(report["resources"]["operations"].values())
        assert views and all(view["local_peak_private_bytes"] >= 100 for view in views)
        if flag:
            assert all(view["min_system_available_bytes"] == 123456789 for view in views)
        else:
            assert not any("min_system_available_bytes" in view for view in views)
            assert "123456789" not in str(report)


def test_a_sampler_thread_that_cannot_start_never_fails_finish(monkeypatch):
    """GOV-3: Thread.start() raising leaves no thread behind; finish() returns a receipt that says why."""
    real = threading.Thread.start

    def failing(self):
        if self.name == profiling.PEAK_SAMPLER_THREAD:
            raise RuntimeError("can't start new thread")
        return real(self)

    monkeypatch.setattr(threading.Thread, "start", failing)
    timer = profiling.StageTimer(peak_sampler_hz=10, peak_reader=lambda: (1, 1, 1))
    monkeypatch.setattr(threading.Thread, "start", real)
    assert timer._peak_thread is None and timer._peak_stop is None and timer._peak_hz is None
    timer.phase("stack")
    report = timer.finish()
    assert report["resources"]["peak_sampler"]["error"].startswith("RuntimeError")
    assert "local_peak_private_bytes" not in report["resources"]["phases"]["stack"]
    assert timer.finish() == report


def test_stopping_a_never_started_sampler_thread_does_not_raise():
    timer = profiling.StageTimer(peak_sampler_hz=10, peak_reader=lambda: (1, 1, 1))
    timer._stop_peak_sampler()                                  # the real one, joined
    timer._peak_thread = threading.Thread(target=lambda: None, name=profiling.PEAK_SAMPLER_THREAD)   # never started
    timer._peak_stop = None
    report = timer.finish()
    assert report["resources"]["version"] == "process-resources-2" and timer._peak_thread is None
