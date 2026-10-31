"""Installer overall progress, time left, safe cancellation and Setup status files.

2026-09-27 user feedback: the progress bar sat empty or jumped to full, nobody
could tell how long installation takes, and Setup's window could not be moved
or closed while components were prepared. See docs/INSTALLER_UX_FIX_PREREG_20260927.md.
"""
import importlib.util
import json
import os
from pathlib import Path

import pytest

from starstack import components
from test_finite_installer_distribution import builder_case, worker_case  # noqa: F401 (fixtures)

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('installer_progress_test', ROOT / 'packaging/aester_p2/installer_progress.py')
progress = importlib.util.module_from_spec(spec)
spec.loader.exec_module(progress)
MANIFEST = json.loads((ROOT / 'packaging/aester_p2/download-manifest.json').read_text(encoding='utf-8'))


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def read(path):
    percent, headline, detail = Path(path).read_text(encoding='utf-8').split('\n')[:3]
    return int(percent), headline, detail


def downloads(steps):
    return [s for s in steps if s['kind'] == 'download']


def test_plan_lists_every_selected_step_and_marks_cached_downloads(tmp_path):
    base = progress.plan_steps(MANIFEST)
    runtime = 2 + len(MANIFEST['wheels'])
    assert len(downloads(base)) == runtime and base[-1]['key'] == 'validate'
    full = progress.plan_steps(MANIFEST, nox=True, finite_bytes=600_000_000, retouch_skill=True, gimp='install')
    keys = [s['key'] for s in full]
    assert len(downloads(full)) == runtime + len(MANIFEST.get('optional_wheels', [])) + 1
    # 2026-10-09: the download sources are measured before the first wheel and before the GIMP download.
    assert keys[-6:] == ['finite', 'skill', 'gimp-check', 'probe:gimp', 'download:gimp-3.2.6-setup.exe', 'gimp-install']
    assert keys.index('probe:pypi') == keys.index('download:' + MANIFEST['wheels'][0]['filename']) - 1
    reuse = progress.plan_steps(MANIFEST, gimp='reuse')
    assert [s['key'] for s in reuse][-1] == 'gimp-check'
    python = MANIFEST['python']
    (tmp_path / python['filename']).write_bytes(b'x' * python['size'])
    cached = progress.plan_steps(MANIFEST, cache=tmp_path)
    assert cached[0]['cached'] is True and not any(s.get('cached') for s in downloads(cached)[1:])


def simulate(tracker, clock, item_sizes, speed):
    """Feed the worker's real event shapes: prepare, download chunks, validate, complete."""
    shown = []
    for index, (name, size) in enumerate(item_sizes):
        tracker.handle({'phase': 'prepare', 'filename': name, 'index': index + 1, 'total': len(item_sizes)})
        done = 0
        while done < size:
            clock.now += 1.0
            done = min(size, done + speed)
            tracker.handle({'phase': 'download', 'filename': name, 'downloaded': done, 'total': size})
            shown.append(tracker.snapshot())
        clock.now += size / progress.EXTRACT_RATE
    tracker.handle({'phase': 'validate'})
    for _ in range(30):
        clock.now += 1.0
        shown.append(tracker.snapshot())
    tracker.handle({'phase': 'complete'})
    shown.append(tracker.snapshot())
    return shown


def test_percent_is_one_overall_monotonic_value_and_time_left_is_given():
    items = [(MANIFEST['python']['filename'], MANIFEST['python']['size'])] + \
            [(w['filename'], w['size']) for w in MANIFEST['wheels']] + [(MANIFEST['siril']['filename'], MANIFEST['siril']['size'])]
    clock = Clock()
    tracker = progress.InstallProgress(progress.plan_steps(MANIFEST), clock=clock)
    shown = simulate(tracker, clock, items, speed=8_000_000)
    percents = [s[0] for s in shown]
    assert percents == sorted(percents) and 0 <= percents[0] and percents[-1] <= 99
    assert percents[-1] >= 95                       # not stuck near empty
    assert len(set(percents)) > 20                  # moves in many small steps, not 0 -> 100
    assert any('预计还需约' in s[1] for s in shown)
    assert all(s[1].startswith('总进度 ') for s in shown)
    assert any('正在下载' in s[2] and 'MB' in s[2] for s in shown)
    assert any(s[2] == '正在离线验证运行环境与 Siril' for s in shown)


def test_time_left_follows_the_measured_download_speed():
    size = 400 * progress.MB
    steps = [{'key': 'download:big.whl', 'kind': 'download', 'filename': 'big.whl', 'bytes': size,
              'cached': False, 'text': 'download'}]
    clock = Clock()
    slow = progress.InstallProgress(steps, clock=clock)
    slow.handle({'phase': 'download', 'filename': 'big.whl', 'downloaded': 0, 'total': size})
    for second in range(1, 11):
        clock.now += 1
        slow.handle({'phase': 'download', 'filename': 'big.whl', 'downloaded': second * progress.MB, 'total': size})
    assert abs(slow.speed - progress.MB) < 0.05 * progress.MB
    for _ in range(30):
        _, headline, _ = slow.snapshot()
    assert '预计还需约 7 分钟' in headline         # 390 MB left at 1 MB/s = 6.5 min
    assert slow._eta_text(45) == '预计不到 1 分钟' and slow._eta_text(5) == '即将完成'
    assert slow._eta_text(80) == '预计还需约 1 分钟'


def test_estimate_waits_for_a_measurement_and_english_text():
    clock = Clock()
    tracker = progress.InstallProgress(progress.plan_steps(MANIFEST), clock=clock, language='en')
    assert tracker.snapshot()[1] == 'Overall 0% · estimating time left'
    assert tracker.snapshot()[2] == 'Preparing'


def test_cached_downloads_count_as_done_immediately(tmp_path):
    clock = Clock()
    steps = progress.plan_steps(MANIFEST, cache=None)
    tracker = progress.InstallProgress(steps, clock=clock)
    python = MANIFEST['python']
    tracker.handle({'phase': 'cached', 'filename': python['filename'], 'downloaded': python['size'], 'total': python['size']})
    assert tracker.steps[0]['finished'] and tracker._current == 1   # unpacking follows at once
    assert '正在解压' in tracker.snapshot()[2]


def test_cancel_stops_at_safe_points_but_never_interrupts_the_gimp_installer(tmp_path):
    cancel = tmp_path / 'cancel.request'
    steps = progress.plan_steps(MANIFEST, gimp='install')
    tracker = progress.InstallProgress(steps, cancel_path=cancel, clock=Clock())
    tracker.handle({'phase': 'download', 'filename': MANIFEST['python']['filename'], 'downloaded': 1, 'total': 10})
    cancel.write_text('cancel')
    tracker.handle({'phase': 'install-gimp', 'message': 'Official GIMP installer is running'})  # no raise
    assert tracker.snapshot()[2] == progress.TEXT['zh_CN']['cancel_after_gimp']
    with pytest.raises(progress.InstallCancelled):
        tracker.handle({'phase': 'download', 'filename': 'gimp-3.2.6-setup.exe', 'downloaded': 5, 'total': 9})
    assert not issubclass(progress.InstallCancelled, Exception)   # optional-step fault tolerance never swallows it


SMALL = {'python': {'filename': 'python.zip', 'size': 10}, 'optional_wheels': [],
         'wheels': [{'filename': 'a-1-py3-none-any.whl', 'size': 5}, {'filename': 'b-1-py3-none-any.whl', 'size': 7}],
         'siril': {'filename': 'siril.zip', 'size': 9}}


def test_source_measurement_is_planned_only_where_something_is_fetched(tmp_path):
    """2026-10-09: before the first wheel and before the GIMP download; not with --download-source official,
    not when every wheel is already cached; never for the Python seed or single-source Siril."""
    keys = [s['key'] for s in progress.plan_steps(SMALL, gimp='install')]
    assert keys.index('probe:pypi') + 1 == keys.index('download:a-1-py3-none-any.whl')
    assert keys.index('probe:gimp') + 1 == keys.index('download:gimp-3.2.6-setup.exe')
    assert [k for k in keys if k.startswith('probe:')] == ['probe:pypi', 'probe:gimp']
    official = [s['key'] for s in progress.plan_steps(SMALL, gimp='install', probe=False)]
    assert not [k for k in official if k.startswith('probe:')]
    for wheel in SMALL['wheels']:
        (tmp_path / wheel['filename']).write_bytes(b'x' * wheel['size'])
    cached = [s['key'] for s in progress.plan_steps(SMALL, cache=tmp_path)]
    assert 'probe:pypi' not in cached
    assert set(progress.TEXT['zh_CN']) == set(progress.TEXT['en'])


@pytest.mark.parametrize('language,probing,chosen,none', [
    ('zh_CN', '正在测速，选择Python 运行库的下载源（官方或镜像）', 'Python 运行库：从 mirrors.cloud.tencent.com 下载',
     'Python 运行库：下载源测速均无应答，按默认顺序尝试'),
    ('en', 'Measuring download sources for Python packages (official and mirrors)',
     'Python packages: downloading from mirrors.cloud.tencent.com',
     'Python packages: no download source answered the speed test; trying them in the default order')])
def test_source_measurement_is_shown_and_is_a_cancel_point(tmp_path, language, probing, chosen, none):
    cancel = tmp_path / 'cancel.request'
    for reachable, expected in ((True, chosen), (False, none)):
        tracker = progress.InstallProgress(progress.plan_steps(SMALL), language=language, cancel_path=cancel,
                                           clock=Clock())
        tracker.handle({'phase': 'prepare', 'filename': 'a-1-py3-none-any.whl'})   # the probe runs inside it
        tracker.handle({'phase': 'probe', 'family': 'pypi', 'candidates': 8})
        assert tracker.snapshot()[2] == probing
        tracker.handle({'phase': 'source', 'family': 'pypi', 'host': 'mirrors.cloud.tencent.com',
                        'reachable': reachable})
        assert tracker.snapshot()[2] == expected
        tracker.handle({'phase': 'download', 'filename': 'a-1-py3-none-any.whl', 'downloaded': 2, 'total': 5})
        assert tracker.steps[tracker._current]['key'] == 'download:a-1-py3-none-any.whl'
    tracker.handle({'phase': 'probe', 'family': 'python'})   # no planned step (the seed): ignored
    cancel.write_text('cancel')
    with pytest.raises(progress.InstallCancelled):
        tracker.handle({'phase': 'probe', 'family': 'gimp'})


def test_progress_file_has_three_utf8_lines_and_final_hundred(tmp_path):
    target = tmp_path / 'progress.txt'
    tracker = progress.InstallProgress(progress.plan_steps(MANIFEST), path=target, clock=Clock())
    tracker.start(interval=0.05)
    percent, headline, detail = read(target)
    assert 0 <= percent < 100 and headline.startswith('总进度') and detail
    tracker.stop(100)
    assert read(target)[0] == 100 and read(target)[1] == '总进度 100%'
    assert not list(tmp_path.glob('progress.txt.*'))   # temporary files were replaced atomically


# ------------------------------------------------------------------- worker

def setup_files(tmp_path):
    return {name: tmp_path / f'setup-{name}.txt' for name in ('progress', 'status', 'pid', 'cancel')}


def worker_args(files):
    return ['--progress-file', str(files['progress']), '--status-file', str(files['status']),
            '--pid-file', str(files['pid']), '--cancel-file', str(files['cancel']), '--language', 'en']


def test_worker_reports_pid_progress_and_success_status_for_setup(worker_case, tmp_path):
    files = setup_files(tmp_path)
    assert worker_case.run(*worker_args(files)) == 0
    assert files['pid'].read_text() == str(os.getpid())
    assert files['status'].read_text() == 'succeeded\n0\n'
    assert read(files['progress'])[:2] == (100, 'Overall 100%')
    assert not files['cancel'].exists()


def test_worker_cancelled_at_a_safe_point_reports_code_three(worker_case, tmp_path, monkeypatch):
    files = setup_files(tmp_path)
    python = MANIFEST['python']

    def prepare(root, **kwargs):
        kwargs['progress']({'phase': 'download', 'filename': python['filename'], 'downloaded': 1, 'total': python['size']})
        files['cancel'].write_text('cancel')
        kwargs['progress']({'phase': 'download', 'filename': python['filename'], 'downloaded': 2, 'total': python['size']})
        pytest.fail('the worker must stop at the second download event')
    monkeypatch.setattr(components, 'install_components', prepare)
    before = worker_case.state.read_bytes(), worker_case.pointer.read_bytes()
    assert worker_case.run('--preflight') == 0
    assert worker_case.run(*worker_args(files)) == 3
    assert files['status'].read_text() == 'cancelled\n3\n'
    assert not worker_case.prepared.exists()
    assert (worker_case.state.read_bytes(), worker_case.pointer.read_bytes()) == before
    assert worker_case.run('--rollback') == 0


def test_worker_failure_is_reported_as_failed_status(worker_case, tmp_path, monkeypatch):
    files = setup_files(tmp_path)
    monkeypatch.setattr(components, 'install_components', lambda *a, **k: (_ for _ in ()).throw(ValueError('broken mirror')))
    assert worker_case.run(*worker_args(files)) == 1
    assert files['status'].read_text() == 'failed\n1\n'
    assert 'broken mirror' in (worker_case.area / 'install-failure.json').read_text(encoding='utf-8')


def test_activation_also_reports_status_so_setup_never_blocks_on_it(worker_case, tmp_path):
    files = setup_files(tmp_path)
    assert worker_case.run() == 0
    assert worker_case.run('--activate', str(worker_case.state), *worker_args(files)) == 0
    assert files['status'].read_text() == 'succeeded\n0\n'
    assert read(files['progress'])[0] == 100


def test_builder_ships_the_progress_module_in_the_bootstrap(builder_case):
    receipt = builder_case.builder.build(builder_case.output, builder_case.compiler, builder_case.model)
    stage = Path(receipt['stage'])
    assert (stage / 'bootstrap/installer_progress.py').read_bytes() == \
        (ROOT / 'packaging/aester_p2/installer_progress.py').read_bytes()


def test_setup_script_runs_the_worker_without_blocking_its_window():
    script = (ROOT / 'packaging/aester_p2/Aester.iss').read_text(encoding='utf-8')
    runner = (ROOT / 'packaging/aester_p2/worker-runner.iss').read_text(encoding='utf-8')
    assert '#include "worker-runner.iss"' in script
    prepare = script[script.index('function PrepareToInstall'):script.index('procedure CurUninstallStepChanged')]
    code = prepare + script[script.index('procedure CurStepChanged'):]
    # Exec*/ewWaitUntilTerminated disables every Setup window for the whole run;
    # only the seconds-long preflight still waits that way.
    long_runs = [line for line in code.splitlines() if 'Exec' in line and '--preflight' not in line]
    assert not long_runs
    assert "Args, True);" in prepare and "' --activate '" in code
    assert 'ewNoWait' in runner and 'ewWaitUntilTerminated' not in runner
    assert 'procedure CancelButtonClick' in runner and "CustomMessage('CancelPrepareConfirm')" in runner
    messages = (ROOT / 'packaging/aester_p2/installer-messages.iss').read_text(encoding='utf-8')
    for key in ('CancelPrepareConfirm', 'PrepareCancelled', 'ExtractFinite', 'ExtractFiniteDetail', 'Activating',
                'GimpFoundSame', 'GimpFoundOther', 'GimpFoundUnknown'):
        assert f'en.{key}=' in messages and f'zh_CN.{key}=' in messages
    assert '兼容性不能保证' in messages and 'Aester 使用 GIMP 3.2.6' in messages
