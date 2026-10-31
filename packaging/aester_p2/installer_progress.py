"""Overall installer progress, a rough remaining time and safe cancellation.

The worker lists every step it will run, weighted by an estimated duration:
downloads by bytes at the measured speed, extraction by bytes, the rest by a
fixed estimate. Worker events move through the list in order. A small text file
(percent, headline, detail; UTF-8) is rewritten twice a second for Setup to
show; nothing here decides what gets installed. Standard library only: this
module runs in the installer bootstrap.
"""
from __future__ import annotations

import os
from pathlib import Path
import threading
import time
import uuid

# Calibrated from a real installation on 2026-09-27 (install-events.jsonl, all
# downloads cached): verifying and unpacking ran at about 12 MB/s, the offline
# check with nox took 88 s, the improved Siril build 8.6 s, the GIMP check 5.5 s.
# The first estimate (40 MB/s, 25 s) showed "under a minute" for 129 s.
INITIAL_SPEED = 4.0e6        # bytes/s until a download has been measured
EXTRACT_RATE = 12.0e6        # bytes/s to verify and unpack wheels and archives
VALIDATE_SECONDS = 35.0      # offline runtime + Siril check
VALIDATE_NOX_SECONDS = 50.0  # extra when the nox model is loaded and checked
PROBE_SECONDS = 5.0          # measuring one family's download sources (components.PROBE_SECONDS + margin)
MB = 1024 * 1024
UNINTERRUPTIBLE = {'install-gimp'}


class InstallCancelled(BaseException):
    """Raised at a safe point after the user confirmed cancelling Setup.

    A BaseException, so optional-component fault tolerance (``except
    Exception``) never swallows the user's decision.
    """


TEXT = {
    'zh_CN': {
        'overall': '总进度 {percent}%', 'estimating': '正在估算剩余时间', 'eta_minutes': '预计还需约 {n} 分钟',
        'eta_minute': '预计还需约 1 分钟', 'eta_soon': '预计不到 1 分钟', 'finishing': '即将完成',
        'download': '正在下载 {name}（{done} / {total} MB）', 'cached': '已校验下载缓存：{name}',
        'extract': '正在解压 {name}', 'validate': '正在离线验证运行环境与 Siril',
        'finite': '正在准备 Siril · Aester 改进构建', 'skill': '正在准备修图技能包',
        'gimp_check': '正在检测本机 GIMP', 'gimp_download': '正在下载 GIMP 3.2.6（{done} / {total} MB）',
        'gimp_install': '正在运行 GIMP 官方安装程序，请稍候',
        'activate': '正在检查并启用 Aester', 'cancelling': '正在取消，请稍候…',
        'cancel_after_gimp': 'GIMP 官方安装正在进行，完成后再取消…', 'start': '正在准备',
        'probe': '正在测速，选择{name}的下载源（官方或镜像）', 'source': '{name}：从 {host} 下载',
        'source_none': '{name}：下载源测速均无应答，按默认顺序尝试',
        'family_pypi': 'Python 运行库', 'family_python': 'Python', 'family_gimp': 'GIMP 3.2.6', 'family_siril': 'Siril',
    },
    'en': {
        'overall': 'Overall {percent}%', 'estimating': 'estimating time left', 'eta_minutes': 'about {n} min left',
        'eta_minute': 'about 1 min left', 'eta_soon': 'less than a minute left', 'finishing': 'almost done',
        'download': 'Downloading {name} ({done} / {total} MB)', 'cached': 'Verified cached download: {name}',
        'extract': 'Unpacking {name}', 'validate': 'Checking the offline runtime and Siril',
        'finite': 'Preparing the Siril - Aester improved build', 'skill': 'Preparing the retouch skill package',
        'gimp_check': 'Looking for an existing GIMP', 'gimp_download': 'Downloading GIMP 3.2.6 ({done} / {total} MB)',
        'gimp_install': 'Running the official GIMP installer, please wait',
        'activate': 'Checking and enabling Aester', 'cancelling': 'Cancelling, please wait...',
        'cancel_after_gimp': 'The official GIMP installer is running; cancelling when it finishes...', 'start': 'Preparing',
        'probe': 'Measuring download sources for {name} (official and mirrors)', 'source': '{name}: downloading from {host}',
        'source_none': '{name}: no download source answered the speed test; trying them in the default order',
        'family_pypi': 'Python packages', 'family_python': 'Python', 'family_gimp': 'GIMP 3.2.6', 'family_siril': 'Siril',
    },
}


def _language(value):
    return 'en' if str(value or '').lower().startswith('en') else 'zh_CN'


def _display_name(filename):
    """A short readable package name: 'PySide6' from 'PySide6-6.9.0-cp39-abi3-win_amd64.whl'."""
    stem = str(filename).split('-')[0] if filename.endswith('.whl') else Path(filename).stem
    return stem.replace('_', '-')


def _probe_step(family):
    return {'key': 'probe:' + family, 'kind': 'timed', 'family': family, 'seconds': PROBE_SECONDS, 'text': 'probe'}


def plan_steps(manifest, *, nox=False, finite_bytes=0, retouch_skill=False, gimp=None, cache=None, probe=True):
    """The ordered step list for one preparation run.

    ``gimp`` is None when GIMP is not selected, 'reuse' when a GIMP is already
    present (only a quick check runs) and 'install' otherwise. ``probe`` adds the
    download-source measurement (automatic source choice) before the first wheel
    and before the GIMP download, when something there still has to be fetched;
    the Python seed comes with Setup and Siril has a single source.
    """
    cache = Path(cache) if cache else None
    steps = []

    def cached(item):
        target = cache / item['filename'] if cache else None
        return bool(target and target.is_file() and target.stat().st_size == item['size'])

    wheels = [*manifest['wheels'], *(manifest.get('optional_wheels', []) if nox else [])]
    runtime = [manifest['python'], *wheels, manifest['siril']]
    probe_wheels = probe and any(not cached(item) for item in wheels)
    for item in runtime:
        if probe_wheels and wheels and item is wheels[0]:
            steps.append(_probe_step('pypi'))
        steps.append({'key': 'download:' + item['filename'], 'kind': 'download', 'filename': item['filename'],
                      'bytes': item['size'], 'cached': cached(item), 'text': 'download'})
        steps.append({'key': 'extract:' + item['filename'], 'kind': 'timed', 'filename': item['filename'],
                      'seconds': item['size'] / EXTRACT_RATE, 'text': 'extract'})
    steps.append({'key': 'validate', 'kind': 'timed', 'text': 'validate',
                  'seconds': VALIDATE_SECONDS + (VALIDATE_NOX_SECONDS if nox else 0.0)})
    if finite_bytes:
        steps.append({'key': 'finite', 'kind': 'timed', 'seconds': 3.0 + finite_bytes / 150e6, 'text': 'finite'})
    if retouch_skill:
        steps.append({'key': 'skill', 'kind': 'timed', 'seconds': 1.0, 'text': 'skill'})
    if gimp is not None:
        steps.append({'key': 'gimp-check', 'kind': 'timed', 'seconds': 6.0, 'text': 'gimp_check'})
        if gimp == 'install':
            from_cache = cache is not None and cached({'filename': 'gimp-3.2.6-setup.exe', 'size': 190358080})
            if probe and not from_cache:
                steps.append(_probe_step('gimp'))
            steps.append({'key': 'download:gimp-3.2.6-setup.exe', 'kind': 'download', 'filename': 'gimp-3.2.6-setup.exe',
                          'bytes': 190358080, 'cached': from_cache, 'text': 'gimp_download'})
            steps.append({'key': 'gimp-install', 'kind': 'timed', 'seconds': 30.0, 'text': 'gimp_install'})
    return steps


class InstallProgress:
    """Thread-safe progress state; ``handle`` takes the worker's event dicts."""

    def __init__(self, steps, *, language='zh_CN', path=None, cancel_path=None, clock=time.monotonic):
        self.steps = [dict(step, done=0.0, started=None, finished=False) for step in steps]
        self.text = TEXT[_language(language)]
        self.path = Path(path) if path else None
        self.cancel_path = Path(cancel_path) if cancel_path else None
        self.clock = clock
        self.began = clock()
        self.speed = INITIAL_SPEED
        self._measured = False
        self._sample = None          # (time, bytes) of the last download event
        self._current = None
        self._shown = 0
        self._eta = None
        self._phase = None
        self._cancel_seen = False
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread = None

    # ------------------------------------------------------------------ events
    def _index(self, key):
        return next((i for i, step in enumerate(self.steps) if step['key'] == key), None)

    def _advance(self, index, now):
        """Every step before ``index`` is finished; ``index`` becomes current."""
        for step in self.steps[:index]:
            step['finished'] = True
        step = self.steps[index]
        if step['started'] is None:
            step['started'] = now
        self._current = index

    def _finish(self, index, now):
        self._advance(index, now)
        self.steps[index]['finished'] = True
        following = index + 1
        # The unpacking that follows a download starts as soon as it completes.
        if following < len(self.steps) and self.steps[following]['key'].startswith('extract:'):
            self._advance(following, now)

    def handle(self, event):
        now = self.clock()
        with self._lock:
            phase = event.get('phase')
            self._phase = phase
            filename = event.get('filename') or ''
            if phase in ('download', 'cached') and filename:
                index = self._index('download:' + filename)
                if index is not None:
                    step = self.steps[index]
                    downloaded = min(float(event.get('downloaded', 0)), float(step['bytes']))
                    if phase == 'download':
                        self._measure(now, downloaded, index)
                    self._advance(index, now)
                    step['done'] = downloaded
                    if phase == 'cached' or downloaded >= step['bytes']:
                        step['cached'] = phase == 'cached' or step.get('cached', False)
                        self._finish(index, now)
            elif phase == 'prepare' and filename:
                index = self._index('download:' + filename)
                if index is not None:
                    self._advance(index, now)
            elif phase == 'validate':
                self._goto('validate', now)
            elif phase == 'complete':
                index = self._index('validate')
                if index is not None:
                    self._finish(index, now)
            elif phase == 'prepare-finite':
                self._goto('finite', now)
            elif phase == 'prepare-retouch-skill':
                self._goto('skill', now)
            elif phase == 'check-gimp':
                self._goto('gimp-check', now)
            elif phase == 'install-gimp':
                self._goto('gimp-install', now)
            elif phase == 'probe':
                # Emitted repeatedly while the sources of one family are measured; each one is a cancel point.
                self._goto('probe:' + str(event.get('family', '')), now)
            elif phase == 'source':
                index = self._index('probe:' + str(event.get('family', '')))
                if index is not None:
                    self.steps[index].update(host=event.get('host'), reachable=bool(event.get('reachable')))
                    self._finish(index, now)
            self._check_cancel(phase)

    def begin(self, key):
        """Mark a step as started when the worker runs it without events."""
        with self._lock:
            self._goto(key, self.clock())

    def _goto(self, key, now):
        index = self._index(key)
        if index is not None:
            self._advance(index, now)

    def _measure(self, now, downloaded, index):
        if self._sample is not None and self._sample[2] == index:
            elapsed = now - self._sample[0]
            delta = downloaded - self._sample[1]
            if elapsed >= 0.5 and delta > 0:
                rate = delta / elapsed
                self.speed = rate if not self._measured else 0.7 * self.speed + 0.3 * rate
                self._measured = True
                self._sample = (now, downloaded, index)
            return
        self._sample = (now, downloaded, index)

    # ------------------------------------------------------------ cancellation
    def cancel_requested(self):
        return bool(self.cancel_path and self.cancel_path.exists())

    def _check_cancel(self, phase):
        if not self.cancel_requested():
            return
        self._cancel_seen = True
        if phase in UNINTERRUPTIBLE:
            return
        raise InstallCancelled()

    # --------------------------------------------------------------- estimates
    def _weights(self, now):
        """(done_seconds, total_seconds) under the current speed estimate."""
        done = total = 0.0
        for index, step in enumerate(self.steps):
            if step['kind'] == 'download':
                size = 0.0 if step.get('cached') else float(step['bytes'])
                weight = size / self.speed
                if step['finished']:
                    part = weight
                else:
                    part = min(weight, step['done'] / self.speed) if size else 0.0
            else:
                weight = float(step['seconds'])
                if step['finished']:
                    part = weight
                elif step['started'] is not None and index == self._current:
                    part = min(0.95 * weight, now - step['started'])
                else:
                    part = 0.0
            done += part
            total += weight
        return done, max(total, 1e-6)

    def snapshot(self):
        """(percent, headline, detail) for display; percent never goes back."""
        now = self.clock()
        with self._lock:
            done, total = self._weights(now)
            percent = int(min(99.0, 100.0 * done / total))
            self._shown = max(self._shown, percent)
            remaining = max(0.0, total - done)
            self._eta = remaining if self._eta is None else 0.7 * self._eta + 0.3 * remaining
            headline = self.text['overall'].format(percent=self._shown)
            downloads_left = any(step['kind'] == 'download' and not step['finished'] and not step.get('cached')
                                 for step in self.steps)
            if now - self.began < 3.0 or (downloads_left and not self._measured):
                headline += ' · ' + self.text['estimating']
            else:
                headline += ' · ' + self._eta_text(self._eta)
            return self._shown, headline, self._detail()

    def _eta_text(self, seconds):
        if seconds < 10:
            return self.text['finishing']
        if seconds < 60:
            return self.text['eta_soon']
        minutes = int(seconds / 60 + 0.5)
        return self.text['eta_minute'] if minutes <= 1 else self.text['eta_minutes'].format(n=minutes)

    def _detail(self):
        if self._cancel_seen or self.cancel_requested():
            return self.text['cancel_after_gimp'] if self._phase in UNINTERRUPTIBLE else self.text['cancelling']
        if self._current is None:
            return self.text['start']
        step = self.steps[self._current]
        if step['key'].startswith('probe:'):
            family = self.text.get('family_' + step.get('family', ''), step.get('family', ''))
            if not step.get('host'):
                return self.text['probe'].format(name=family)
            if not step.get('reachable'):
                return self.text['source_none'].format(name=family)
            return self.text['source'].format(name=family, host=step['host'])
        name = _display_name(step.get('filename', ''))
        if step['kind'] == 'download' and step.get('cached'):
            return self.text['cached'].format(name=name)
        if step['kind'] == 'download':
            return self.text[step['text']].format(name=name, done=int(step['done'] / MB), total=int(step['bytes'] / MB + 0.5))
        return self.text[step['text']].format(name=name)

    # ------------------------------------------------------------------ output
    def write(self, percent=None):
        if self.path is None:
            return
        shown, headline, detail = self.snapshot()
        if percent is not None:
            shown = percent
            headline = self.text['overall'].format(percent=percent)
        body = f'{shown}\n{headline}\n{detail}\n'.encode('utf-8')
        temporary = self.path.with_name(self.path.name + '.' + uuid.uuid4().hex[:8])
        try:
            temporary.write_bytes(body)
            os.replace(temporary, self.path)
        except OSError:
            # Setup may be reading the file right now; the next tick retries.
            try:
                temporary.unlink()
            except OSError:
                pass

    def start(self, interval=0.5):
        """Rewrite the file on a timer so long steps keep moving on screen."""
        if self.path is None or self._thread is not None:
            return self

        def tick():
            while not self._stop.wait(interval):
                self.write()
        self.write()
        self._thread = threading.Thread(target=tick, name='installer-progress', daemon=True)
        self._thread.start()
        return self

    def stop(self, final_percent=None):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        if final_percent is not None:
            self.write(final_percent)


def write_text_atomic(path, text):
    """Small UTF-8 files for Setup (pid, status, notices)."""
    path = Path(path)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex[:8])
    temporary.write_bytes(text.encode('utf-8'))
    os.replace(temporary, path)
