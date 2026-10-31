"""Bounded result-file sharing faults, independent of candidate classification."""
from contextlib import contextmanager
import ctypes
import errno
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from starstack import transient_scan as scan


class Clock:
    now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


@pytest.fixture
def world(tmp_path, monkeypatch):
    clock = Clock()
    processes, logs = [], []
    state = SimpleNamespace(clock=clock, processes=processes, logs=logs,
                            status=lambda index: 0, omit=set(), content=None)

    class Process:
        def __init__(self, index):
            self.index, self.killed = index, False

        def poll(self):
            return state.status(self.index)

        def kill(self):
            self.killed = True

    @contextmanager
    def guard(command, **kwargs):
        request = json.loads(Path(command[-1]).read_text(encoding='utf-8'))
        process = Process(len(processes))
        processes.append(process)
        for job in request['jobs']:
            if job['stem'] not in state.omit:
                value = {'version': 1, 'stem': job['stem'], 'scanned': {
                    'result': {'segments': [{'source': job['stem'], 'x0': 4}]},
                    'stats': {'zero': 0.0}, 'seconds': .01}}
                Path(job['result']).write_text(state.content if state.content is not None else json.dumps(value), encoding='utf-8')
        yield process

    monkeypatch.setattr(scan, 'guarded_process', guard)
    monkeypatch.setattr(scan, 'time', clock)
    state.run = lambda **kwargs: scan.scan_frames(
        {'z': {}, 'a': {}}, workers=2, output_dir=tmp_path, log=logs.append, **kwargs)
    return state


def denial(winerror=None, code=errno.EACCES):
    error = PermissionError(code, 'injected result-file denial')
    if winerror is not None:
        error.winerror = winerror
    return error


def deny_reads(monkeypatch, fail, *, index=None, error=None):
    real = Path.read_text
    calls = {}

    def read(path, *args, **kwargs):
        if path.name.startswith('result-') and (index is None or path.name == f'result-{index:06d}.json'):
            calls[path.name] = calls.get(path.name, 0) + 1
            if fail(calls[path.name]):
                raise error or denial()
        return real(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'read_text', read)
    return calls


@pytest.mark.parametrize('winerror', [None, 5, 32, 33])
@pytest.mark.parametrize('already_exited', [False, True])
def test_transient_denial_recovers_without_result_change(world, monkeypatch, winerror, already_exited):
    world.status = lambda index: 0 if already_exited or world.clock.now >= .3 else None
    calls = deny_reads(monkeypatch, lambda count: count <= 2, error=denial(winerror))
    result = world.run()
    assert list(result) == ['z', 'a']
    assert result == {stem: {'result': {'segments': [{'source': stem, 'x0': 4}]},
                            'stats': {'zero': 0.0}, 'seconds': .01} for stem in ('z', 'a')}
    assert set(calls.values()) == {3}
    assert len(world.logs) == 2
    assert not any(process.killed for process in world.processes)


def test_persistent_denial_has_own_deadline_without_global_timeout(world, monkeypatch):
    calls = deny_reads(monkeypatch, lambda _: True)
    with pytest.raises(scan.ScanWorkerError, match='持续无法读取') as error:
        world.run(total_timeout=None)
    assert isinstance(error.value.__cause__, PermissionError)
    assert 2 <= world.clock.now <= 2.11
    assert max(calls.values()) <= 22
    assert 'result-000000.json' in str(error.value)


def test_cancel_interrupts_retries_and_stops_every_live_worker(world, monkeypatch):
    world.status = lambda _: None
    deny_reads(monkeypatch, lambda _: True)

    def check():
        if world.clock.now >= .1:
            raise RuntimeError('caller cancelled')

    with pytest.raises(RuntimeError, match='caller cancelled'):
        world.run(check=check)
    assert world.clock.now == .1
    assert all(process.killed for process in world.processes)


def test_global_timeout_still_bounds_read_retries(world, monkeypatch):
    world.status = lambda _: None
    deny_reads(monkeypatch, lambda _: True)
    with pytest.raises(TimeoutError, match='总时限'):
        world.run(total_timeout=.15)
    assert world.clock.now == .2
    assert all(process.killed for process in world.processes)


def test_other_worker_failure_waits_for_the_earlier_frame_whose_read_is_denied(world, monkeypatch):
    """P1c T-2: frame order decides. Child 1 published 'a' and exits 9 (a failure after every frame); frame 0's denied
    read is still pending, so that failure waits; when the denial lifts at 0.3 s, frame 0 is read and the late exit is
    reported."""
    world.status = lambda index: 9 if index == 1 and world.clock.now >= .1 else (0 if world.clock.now >= .5 else None)
    deny_reads(monkeypatch, lambda count: world.clock.now < .3, index=0)
    with pytest.raises(scan.ScanWorkerError, match='异常退出（9）') as error:
        world.run()
    assert '失败' in str(error.value) and 'worker' not in str(error.value) and 'scan-runs' not in str(error.value)
    assert world.clock.now == pytest.approx(.5)
    assert not world.processes[0].killed


def test_persistent_denial_of_an_earlier_frame_outranks_another_workers_exit(world, monkeypatch):
    world.status = lambda index: 9 if index == 1 and world.clock.now >= .1 else None
    deny_reads(monkeypatch, lambda _: True, index=0)
    with pytest.raises(scan.ScanWorkerError, match='第 1 帧（z）的结果持续无法读取') as error:
        world.run()
    assert isinstance(error.value.__cause__, PermissionError)
    assert 2 <= world.clock.now <= 2.11
    assert world.processes[0].killed


def test_clean_exit_missing_result_waits_for_an_earlier_denied_frame(world, monkeypatch):
    world.omit = {'a'}
    deny_reads(monkeypatch, lambda count: world.clock.now < .3, index=0)
    with pytest.raises(scan.ScanWorkerError, match='第 2 帧（a）：扫描进程退出但缺少结果'):
        world.run()
    assert world.clock.now == pytest.approx(.3)


def test_clean_exit_missing_result_is_not_hidden_by_a_later_denial(world, monkeypatch):
    world.omit = {'z'}
    deny_reads(monkeypatch, lambda _: True, index=1)
    with pytest.raises(scan.ScanWorkerError, match='第 1 帧（z）：扫描进程退出但缺少结果'):
        world.run()
    assert world.clock.now == 0


@pytest.mark.parametrize('error', [denial(code=errno.EPERM), denial(87), OSError(errno.EIO, 'I/O fault')])
def test_unrelated_io_failure_is_not_retried(world, monkeypatch, error):
    deny_reads(monkeypatch, lambda _: True, error=error)
    with pytest.raises(type(error)) as caught:
        world.run()
    assert caught.value is error
    assert world.clock.now == 0


@pytest.mark.parametrize('content,exception', [
    ('{', json.JSONDecodeError),
    ('{"version": 1, "stem": "wrong"}', scan.ScanWorkerError),
    ('{"version": 2, "stem": "z"}', scan.ScanWorkerError),
    ('{"version": 1, "stem": "z"}', KeyError),
])
def test_corrupt_or_mismatched_published_result_is_not_retried(world, content, exception):
    world.content = content
    with pytest.raises(exception):
        world.run()
    assert world.clock.now == 0


def test_decode_failure_is_not_retried(world, monkeypatch):
    error = UnicodeDecodeError('utf-8', b'\xff', 0, 1, 'invalid byte')
    deny_reads(monkeypatch, lambda _: True, error=error)
    with pytest.raises(UnicodeDecodeError):
        world.run()
    assert world.clock.now == 0


def test_stat_permission_denial_uses_same_bounded_read_path(world, monkeypatch):
    real, calls = Path.is_file, []

    def is_file(path):
        if path.name.startswith('result-') and not calls:
            calls.append(path)
            raise denial(32)
        return real(path)

    monkeypatch.setattr(Path, 'is_file', is_file)
    assert list(world.run()) == ['z', 'a']
    assert world.clock.now == .1


@pytest.mark.skipif(os.name != 'nt', reason='actual Windows file sharing semantics')
def test_actual_exclusive_windows_reader_releases_and_scan_recovers(world, monkeypatch):
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                  wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    real_read = Path.read_text
    handle = None
    denied = []

    def read(path, *args, **kwargs):
        nonlocal handle
        if path.name == 'result-000000.json' and not denied:
            handle = kernel.CreateFileW(str(path), 0x80000000, 0, None, 3, 0x80, None)
            assert handle != ctypes.c_void_p(-1).value, ctypes.get_last_error()
            try:
                return real_read(path, *args, **kwargs)
            except PermissionError as error:
                denied.append((error.errno, getattr(error, 'winerror', None)))
                raise
        return real_read(path, *args, **kwargs)

    real_sleep = world.clock.sleep

    def sleep(seconds):
        nonlocal handle
        if handle is not None:
            assert kernel.CloseHandle(handle)
            handle = None
        real_sleep(seconds)

    monkeypatch.setattr(Path, 'read_text', read)
    monkeypatch.setattr(world.clock, 'sleep', sleep)
    try:
        assert list(world.run()) == ['z', 'a']
        assert denied == [(errno.EACCES, None)]
        assert world.clock.now == .1
    finally:
        if handle is not None:
            kernel.CloseHandle(handle)
