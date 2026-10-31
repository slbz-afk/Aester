"""Download source choice (2026-10-09): one installer that works in mainland China and abroad.

Pinned files come from the official host or a listed mirror, whichever measures faster; only the pinned size and
SHA-256 decide whether bytes are accepted. All traffic here is synthetic: the opener is replaced, and where noted the
real urllib redirect machinery runs against a fake HTTPS handler. Nothing is downloaded.
"""
from __future__ import annotations

import email.message
import hashlib
import inspect
import io
import json
import os
from pathlib import Path
import ssl
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import urllib.response

import pytest

from starstack import components as c
from starstack import retouch_distribution as d
from test_finite_installer_distribution import builder_case, worker_case  # noqa: F401 (fixtures)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / 'packaging/aester_p2/download-manifest.json').read_text(encoding='utf-8'))
REAL_BUILD_OPENER = urllib.request.build_opener

# The decided candidate lists (task of 2026-10-09), written out independently of components.DOWNLOAD_SOURCES.
PYPI = ['https://files.pythonhosted.org/packages/{p}',
        'https://mirrors.cloud.tencent.com/pypi/packages/{p}',
        'https://repo.huaweicloud.com/repository/pypi/packages/{p}',
        'https://pypi.tuna.tsinghua.edu.cn/packages/{p}',
        'https://mirrors.ustc.edu.cn/pypi/packages/{p}',
        'https://mirrors.bfsu.edu.cn/pypi/web/packages/{p}',
        'https://mirrors.aliyun.com/pypi/packages/{p}',
        'https://mirror.nju.edu.cn/pypi/web/packages/{p}']
PYTHON = ['https://www.python.org/ftp/python/{v}/{f}',
          'https://cdn.npmmirror.com/binaries/python/{v}/{f}',
          'https://repo.huaweicloud.com/python/{v}/{f}',
          'https://mirrors.tuna.tsinghua.edu.cn/python/{v}/{f}',
          'https://mirrors.ustc.edu.cn/python/{v}/{f}',
          'https://mirror.nju.edu.cn/python/{v}/{f}']
GIMP = ['https://download.gimp.org/gimp/v3.2/windows/gimp-3.2.6-setup.exe',
        'https://opencolo.mm.fcix.net/gimp/gimp/v3.2/windows/gimp-3.2.6-setup.exe',
        'https://mirror.nju.edu.cn/gimp/gimp/v3.2/windows/gimp-3.2.6-setup.exe',
        'https://mirrors.ustc.edu.cn/gimp/v3.2/windows/gimp-3.2.6-setup.exe',
        'https://mirrors.aliyun.com/gimp/gimp/v3.2/windows/gimp-3.2.6-setup.exe']
NOT_USED = {'mirrors.huaweicloud.com', 'mirrors.zju.edu.cn', 'mirrors.cernet.edu.cn', 'mirrors.hust.edu.cn',
            'pypi.doubanio.com', 'mirrors.jlu.edu.cn', 'mirror.lzu.edu.cn', 'mirrors.sjtug.sjtu.edu.cn',
            'www.free-astro.org', 'ziply.mm.fcix.net', 'mirror.umd.edu'}
OFFICIAL_PYPI = 'https://files.pythonhosted.org/packages/'
TENCENT = 'https://mirrors.cloud.tencent.com/pypi/packages/'


def host(url):
    return urllib.parse.urlsplit(url).hostname


def pinned(body, *, url, filename):
    return {'filename': filename, 'url': url, 'size': len(body), 'sha256': hashlib.sha256(body).hexdigest()}


def wheel(size=300 * 1024):
    body = bytes(i % 251 for i in range(size))
    name = 'fake-1.0-py3-none-any.whl'
    return body, pinned(body, url=OFFICIAL_PYPI + 'aa/bb/' + 'c' * 60 + '/' + name, filename=name)


class Reply(io.BytesIO):
    def __init__(self, body, status=200, headers=None, url=''):
        super().__init__(body)
        self.status, self.headers, self.url = status, dict(headers or {}), url


def ranged(body, request, *, total=None):
    """What a mirror answers: 206 for a Range request, 200 for the whole file."""
    spec = request.get_header('Range')
    if not spec:
        return Reply(body, 200, url=request.full_url)
    start, _, end = spec.removeprefix('bytes=').partition('-')
    start, end = int(start), (int(end) if end else len(body) - 1)
    return Reply(body[start:end + 1], 206, {'Content-Range': f'bytes {start}-{end}/{total or len(body)}'},
                 request.full_url)


class Network:
    """Thread-safe stand-in for urllib's opener. ``answer(request, redirect_handler)`` serves each request."""

    def __init__(self, monkeypatch, answer):
        self.requests, self.lock, self.answer = [], threading.Lock(), answer
        network = self

        class Opener:
            def __init__(self, handlers):
                self.redirect = handlers[0] if handlers else None

            def open(self, request, timeout=None):
                with network.lock:
                    network.requests.append((host(request.full_url), request.get_header('Range'), request))
                return network.answer(request, self.redirect)
        monkeypatch.setattr(c.urllib.request, 'build_opener', lambda *handlers: Opener(handlers))

    def hosts(self):
        return [entry[0] for entry in self.requests]


def ranking(*bases):
    return [{'source': base, 'host': host(base), 'ok': True} for base in bases]


# ----------------------------------------------------------------------------------------------- the source table

def test_every_pinned_item_maps_to_its_family_sources():
    wheels = [*MANIFEST['wheels'], *MANIFEST['optional_wheels']]
    assert len(wheels) == 44
    for item in wheels:
        rest = item['url'].removeprefix(OFFICIAL_PYPI)
        assert rest != item['url']
        assert c.candidate_urls(item) == [template.format(p=rest) for template in PYPI]
        assert c.candidate_urls(item, 'official') == [item['url']]
        assert all(url.endswith('/' + item['filename']) for url in c.candidate_urls(item))
    python = MANIFEST['python']
    expected = [template.format(v='3.13.15', f=python['filename']) for template in PYTHON]
    assert expected[0] == python['url'] and c.candidate_urls(python) == expected
    siril = MANIFEST['siril']
    assert c.candidate_urls(siril) == c.candidate_urls(siril, 'official') == [siril['url']]
    assert c.candidate_urls(d.GIMP_INSTALLER) == GIMP
    assert c.candidate_urls(d.GIMP_INSTALLER, 'official') == GIMP[:2]   # GIMP's own mirror, the old fallback


def test_excluded_hosts_are_in_no_family_and_families_keep_their_own_hosts():
    every = set().union(*(c.family_hosts(family) for family in c.DOWNLOAD_SOURCES))
    assert not every & NOT_USED
    assert 'cdn.npmmirror.com' in c.family_hosts('python') and 'cdn.npmmirror.com' not in c.family_hosts('pypi')
    assert c.family_hosts('siril') == {'free-astro.org'}
    assert 'opencolo.mm.fcix.net' in c.family_hosts('gimp') and 'opencolo.mm.fcix.net' not in c.family_hosts('pypi')


@pytest.mark.parametrize('family,url,allowed', [
    ('gimp', 'https://mirror.nju.edu.cn/gimp/gimp/v3.2/windows/x.exe', True),
    ('gimp', 'https://ziply.mm.fcix.net/gimp/v3.2/windows/x.exe', False),     # GIMP's location-based pick
    ('gimp', 'https://mirror.umd.edu/gimp/v3.2/windows/x.exe', False),
    ('python', 'https://opencolo.mm.fcix.net/gimp/gimp/x.zip', False),        # a mirror of another family
    ('pypi', 'https://cdn.npmmirror.com/binaries/python/x.whl', False),
    ('siril', 'https://mirrors.cloud.tencent.com/pypi/packages/x.zip', False),
    ('siril', 'https://www.free-astro.org/download/x.zip', False),
    ('pypi', 'https://mirrors.huaweicloud.com/repository/pypi/packages/x.whl', False),
    ('pypi', 'http://mirrors.cloud.tencent.com/pypi/packages/x.whl', False),
    ('pypi', 'https://mirrors.cloud.tencent.com:8443/pypi/packages/x.whl', False),
    ('pypi', 'https://u:p@mirrors.cloud.tencent.com/pypi/packages/x.whl', False),
    ('pypi', 'https://mirrors.cloud.tencent.com/pypi/packages/x.whl', True),
])
def test_each_family_follows_redirects_only_to_its_own_hosts(family, url, allowed):
    handler = c._HTTPSRedirect(c.family_hosts(family))
    request = urllib.request.Request('https://example.invalid/start')
    if allowed:
        assert handler.redirect_request(request, None, 302, 'Found', {}, url).full_url == url
    else:
        with pytest.raises(ValueError, match='approved HTTPS origins'):
            handler.redirect_request(request, None, 302, 'Found', {}, url)


@pytest.mark.parametrize('url', ['https://mirrors.cloud.tencent.com/pypi/packages/aa/x.whl',
                                 'https://opencolo.mm.fcix.net/gimp/gimp/v3.2/windows/x.exe',
                                 'https://mirrors.huaweicloud.com/repository/pypi/packages/aa/x.whl'])
def test_an_item_must_name_its_official_url_before_the_cache_exists(tmp_path, url):
    with pytest.raises(ValueError, match='Unsupported component origin'):
        c._download(pinned(b'x', url=url, filename='x.whl'), tmp_path / 'cache')
    assert not (tmp_path / 'cache').exists()


@pytest.mark.parametrize('url', ['https://files.pythonhosted.org/component.zip',
                                 'https://files.pythonhosted.org/packages/../x.whl',
                                 'https://files.pythonhosted.org/packages/aa/x.whl?mirror=1'])
def test_an_unmapped_official_path_has_no_mirrors(url):
    assert c.candidate_urls({'url': url}) == [url]


def test_source_choice_is_validated():
    for mode in ('fastest', 'mirror', None, ''):
        with pytest.raises(ValueError, match='download_source'):
            c.DownloadSources(mode)
    assert c.DOWNLOAD_SOURCE_MODES == ('auto', 'official')


# ----------------------------------------------------------------------------------------------- measuring

def test_probe_counts_only_exact_partial_answers_and_ranks_by_speed(monkeypatch):
    body, item = wheel()
    release = threading.Event()
    monkeypatch.setattr(c, 'PROBE_SECONDS', 1.0)
    delays = {'mirrors.cloud.tencent.com': 0.0, 'mirror.nju.edu.cn': 0.12, 'files.pythonhosted.org': 0.3}

    def answer(request, redirect):
        name = host(request.full_url)
        if name in delays:
            release.wait(delays[name])
            return ranged(body, request)
        if name == 'repo.huaweicloud.com':          # ignores Range: the whole file with 200
            return Reply(body, 200, url=request.full_url)
        if name == 'pypi.tuna.tsinghua.edu.cn':     # a 206 for a file of another size
            return ranged(body, request, total=len(body) + 1)
        if name == 'mirrors.ustc.edu.cn':           # a soft-404 page served as a partial answer
            page = b'<html>not here</html>'.ljust(2913)
            return Reply(page, 206, {'Content-Range': 'bytes 0-2912/2913', 'Content-Type': 'text/html'},
                         request.full_url)
        if name == 'mirrors.bfsu.edu.cn':           # still connecting when the measurement ends
            release.wait(10)
            raise urllib.error.URLError(TimeoutError('timed out'))
        assert name == 'mirrors.aliyun.com'
        raise urllib.error.HTTPError(request.full_url, 403, 'Forbidden', {}, None)
    network = Network(monkeypatch, answer)
    events = []
    started = time.monotonic()
    try:
        ranked = c._probe('pypi', item, events.append)
        elapsed = time.monotonic() - started
    finally:
        release.set()
    assert [entry['host'] for entry in ranked] == [
        'mirrors.cloud.tencent.com', 'mirror.nju.edu.cn', 'files.pythonhosted.org',     # measured, fastest first
        'repo.huaweicloud.com', 'pypi.tuna.tsinghua.edu.cn', 'mirrors.ustc.edu.cn',    # failures, table order
        'mirrors.bfsu.edu.cn', 'mirrors.aliyun.com']
    assert [entry['ok'] for entry in ranked] == [True] * 3 + [False] * 5
    rates = [entry['bytes_per_second'] for entry in ranked[:3]]
    assert rates == sorted(rates, reverse=True)
    errors = {entry['host']: entry['error'] for entry in ranked if not entry['ok']}
    assert 'HTTP 200' in errors['repo.huaweicloud.com']
    assert 'Content-Range' in errors['pypi.tuna.tsinghua.edu.cn'] and 'Content-Range' in errors['mirrors.ustc.edu.cn']
    assert 'deadline' in errors['mirrors.bfsu.edu.cn']
    assert 'HTTP 403' in errors['mirrors.aliyun.com']
    assert elapsed < 3.0                                     # about PROBE_SECONDS + 0.5, not the stuck host's 10 s
    assert sorted(network.hosts()) == sorted(host(url) for url in PYPI)  # one request per candidate
    for _, header, request in network.requests:
        assert header == f'bytes=0-{c.PROBE_BYTES - 1}'
        assert request.get_header('User-agent') == 'Aester-Components/1'
        assert request.get_header('Accept-encoding') == 'identity'
    assert events[0]['phase'] == 'probe' and events[-1]['phase'] == 'source'
    assert {event['phase'] for event in events[:-1]} == {'probe'}
    assert events[-1]['host'] == 'mirrors.cloud.tencent.com' and events[-1]['reachable'] is True
    assert events[-1]['ranking'] == ranked and events[-1]['family'] == 'pypi'


def test_slow_delivery_past_the_deadline_does_not_count(monkeypatch):
    body, item = wheel()
    monkeypatch.setattr(c, 'PROBE_SECONDS', 0.3)

    class Slow(Reply):
        def read1(self, size=-1):
            threading.Event().wait(0.2)
            return super().read1(size)
        read = read1
    Network(monkeypatch, lambda request, redirect: Slow(body[:c.PROBE_BYTES], 206,
            {'Content-Range': f'bytes 0-{c.PROBE_BYTES - 1}/{len(body)}'}, request.full_url))
    result = c._probe_one(OFFICIAL_PYPI, item['url'], c.family_hosts('pypi'), item['size'])
    assert result['ok'] is False and 'deadline' in result['error']
    # The right file was arriving: a partial answer with its rate so far, one receive (16 KiB) at a time.
    assert result['partial'] is True and 0 < result['received'] < c.PROBE_BYTES
    assert result['received'] % c._PROBE_CHUNK == 0 and result['bytes_per_second'] > 0


def test_nothing_reachable_keeps_the_table_order(monkeypatch):
    body, item = wheel()
    Network(monkeypatch, lambda request, redirect: (_ for _ in ()).throw(urllib.error.URLError(OSError('offline'))))
    events = []
    ranked = c._probe('pypi', item, events.append)
    assert [entry['source'] for entry in ranked] == [template.format(p='') for template in PYPI]
    assert not any(entry['ok'] for entry in ranked)
    assert events[-1]['reachable'] is False and events[-1]['message'] == '下载源测速均无应答，按默认顺序尝试'


class Stop(BaseException):
    """Stands in for the installer's InstallCancelled (a BaseException)."""


def test_cancel_during_the_measurement_stops_at_once(monkeypatch):
    body, item = wheel()
    release = threading.Event()
    monkeypatch.setattr(c, 'PROBE_SECONDS', 30.0)

    def answer(request, redirect):
        release.wait(30)
        raise urllib.error.URLError(TimeoutError('released'))
    network = Network(monkeypatch, answer)
    seen = []

    def progress(event):
        seen.append(event['phase'])
        if len(seen) == 2:
            raise Stop()
    started = time.monotonic()
    try:
        with pytest.raises(Stop):
            c.DownloadSources('auto', [item]).urls(item, progress)
        assert time.monotonic() - started < 3.0
    finally:
        release.set()
    assert seen == ['probe', 'probe']
    assert all(header.startswith('bytes=0-') for _, header, _ in network.requests)   # no download began


def probe_threads():
    return [thread for thread in threading.enumerate() if thread.name.startswith('aester-source-probe-')]


def test_measuring_threads_are_daemons_and_stop_receiving_after_a_cancel(monkeypatch):
    body, item = wheel()
    monkeypatch.setattr(c, 'PROBE_SECONDS', 30.0)
    release = threading.Event()

    class Trickle(Reply):                    # the right file, 1 KiB per receive
        def read1(self, size=-1):
            threading.Event().wait(0.01)
            return super().read1(min(size, 1024))

    def answer(request, redirect):
        if host(request.full_url) == 'files.pythonhosted.org':
            return Trickle(body[:c.PROBE_BYTES], 206, {'Content-Range': f'bytes 0-{c.PROBE_BYTES - 1}/{len(body)}'},
                           request.full_url)
        release.wait(30)                     # still connecting
        raise urllib.error.URLError(TimeoutError('released'))
    Network(monkeypatch, answer)
    seen = []

    def progress(event):
        seen.append(event['phase'])
        if len(seen) == 3:
            raise Stop()
    try:
        with pytest.raises(Stop):
            c._probe('pypi', item, progress)
        threads = probe_threads()
        assert len(threads) >= len(PYPI) - 1 and all(thread.daemon for thread in threads)
        # The candidate that was receiving sees the stop request at its next receive and ends.
        [official] = [thread for thread in threads if thread.name == 'aester-source-probe-0'] or [None]
        if official is not None:
            official.join(2.0)
            assert not official.is_alive()
    finally:
        release.set()


def test_a_cancel_during_the_measurement_does_not_keep_the_process_alive(tmp_path):
    """Review 2026-10-09: pool threads were joined at interpreter exit, so python.exe lived on until every hung
    request ended. Run as Setup's bootstrap does: components as a loose module beside install_ownership."""
    script = tmp_path / 'cancel_probe.py'
    script.write_text(
        'import threading, urllib.error\n'
        'import components as c\n'
        'class Opener:\n'
        '    def open(self, request, timeout=None):\n'
        '        threading.Event().wait(30)   # a connection that hangs far beyond the measurement\n'
        '        raise urllib.error.URLError(TimeoutError("late"))\n'
        'c.urllib.request.build_opener = lambda *handlers: Opener()\n'
        'item = {"filename": "x-1-py3-none-any.whl", "size": 1 << 20, "sha256": "0" * 64,\n'
        '        "url": "https://files.pythonhosted.org/packages/aa/bb/x-1-py3-none-any.whl"}\n'
        'class Stop(BaseException):\n'
        '    pass\n'
        'seen = []\n'
        'def progress(event):\n'
        '    seen.append(event)\n'
        '    if len(seen) == 2:\n'
        '        raise Stop()\n'
        'try:\n'
        '    c.DownloadSources("auto", [item]).urls(item, progress)\n'
        'except Stop:\n'
        '    print("cancelled", flush=True)\n', encoding='utf-8')
    environment = {**os.environ, 'PYTHONPATH': str(ROOT / 'src/starstack')}
    started = time.monotonic()
    result = subprocess.run([sys.executable, '-B', '-X', 'utf8', str(script)], capture_output=True, text=True,
                            timeout=60, env=environment, cwd=tmp_path)
    elapsed = time.monotonic() - started
    assert result.returncode == 0 and 'cancelled' in result.stdout, result.stderr
    assert elapsed < 15          # not the 30 s the hung connections would hold a non-daemon pool


@pytest.mark.parametrize('full', [True, False])
def test_sources_still_delivering_at_the_deadline_rank_after_full_answers_by_their_rate(monkeypatch, full):
    """Review 2026-10-09: on a slow or distant link a source still receiving the right file at the deadline is a
    real speed signal; it ranks after full answers and before candidates that never answered."""
    body, item = wheel()
    monkeypatch.setattr(c, 'PROBE_SECONDS', 0.6)
    release = threading.Event()
    delays = {'mirrors.cloud.tencent.com': 0.01, 'mirrors.ustc.edu.cn': 0.06}   # too slow for 256 KiB in 0.6 s

    def answer(request, redirect):
        name = host(request.full_url)
        if name == 'mirror.nju.edu.cn' and full:
            return ranged(body, request)
        if name in delays:
            delay = delays[name]

            class Trickle(Reply):
                def read1(self, size=-1):
                    threading.Event().wait(delay)
                    return super().read1(min(size, 1024))
            return Trickle(body[:c.PROBE_BYTES], 206, {'Content-Range': f'bytes 0-{c.PROBE_BYTES - 1}/{len(body)}'},
                           request.full_url)
        if name == 'mirrors.aliyun.com':
            raise urllib.error.HTTPError(request.full_url, 403, 'Forbidden', {}, None)
        release.wait(5)
        raise urllib.error.URLError(TimeoutError('timed out'))
    Network(monkeypatch, answer)
    events = []
    try:
        ranked = c._probe('pypi', item, events.append)
    finally:
        release.set()
    names = [entry['host'] for entry in ranked]
    leaders = (['mirror.nju.edu.cn'] if full else []) + ['mirrors.cloud.tencent.com', 'mirrors.ustc.edu.cn']
    assert names[:len(leaders)] == leaders
    assert names[len(leaders):] == [host(t) for t in PYPI if host(t) not in leaders]     # table order
    partial = ranked[len(leaders) - 2:len(leaders)]
    assert all(entry['partial'] and not entry['ok'] and 'deadline' in entry['error'] for entry in partial)
    assert partial[0]['received'] > partial[1]['received'] > 0
    assert not any(entry.get('partial') for entry in ranked[len(leaders):])
    assert events[-1]['host'] == leaders[0] and events[-1]['reachable'] is True
    assert events[-1]['message'] == f'已选择下载源：{leaders[0]}'
    # Two attempts for a source that answered (fully or partly), one for a source that did not.
    monkeypatch.setattr(c, '_probe', lambda family, representative, progress=None: ranked)
    attempts = {host(url): count for _, url, count in c.DownloadSources('auto', [item]).candidates(item)}
    assert all(attempts[name] == 2 for name in leaders)
    assert all(attempts[name] == 1 for name in names[len(leaders):])


def test_each_family_is_measured_once_on_its_largest_file_and_only_when_needed(tmp_path, monkeypatch):
    calls = []

    def probe(family, item, progress=None):
        calls.append((family, item['filename']))
        return ranking(*reversed(c._family_bases(family)))
    monkeypatch.setattr(c, '_probe', probe)
    wheels = MANIFEST['wheels']
    sources = c.DownloadSources('auto', [MANIFEST['python'], *wheels, MANIFEST['siril']])
    first, second = sources.urls(wheels[0]), sources.urls(wheels[1])
    assert calls == [('pypi', max(wheels, key=lambda item: item['size'])['filename'])]
    rest = wheels[0]['url'].removeprefix(OFFICIAL_PYPI)
    assert first == [template.format(p=rest) for template in reversed(PYPI)]
    assert second[0].startswith('https://mirror.nju.edu.cn/')
    assert sources.urls(MANIFEST['siril']) == [MANIFEST['siril']['url']]       # one source: never measured
    assert calls == [('pypi', max(wheels, key=lambda item: item['size'])['filename'])]
    assert c.DownloadSources('official', wheels).urls(wheels[0]) == [wheels[0]['url']]
    assert len(calls) == 1
    # A verified cached file needs no measurement and no network.
    body = b'cached wheel'
    item = pinned(body, url=OFFICIAL_PYPI + 'aa/bb/cc/x-1-py3-none-any.whl', filename='x-1-py3-none-any.whl')
    (tmp_path / item['filename']).write_bytes(body)
    monkeypatch.setattr(c, '_probe', lambda *args, **kwargs: pytest.fail('a verified cache needs no measurement'))
    Network(monkeypatch, lambda *args: pytest.fail('a verified cache needs no network'))
    assert c._download(item, tmp_path, None, c.DownloadSources('auto', [item])).read_bytes() == body


# ----------------------------------------------------------------------------------------------- downloading

def test_downloads_follow_the_measured_order_and_fall_back_in_order(tmp_path, monkeypatch):
    body = b'GIMP installer fixture ' * 64
    item = {**d.GIMP_INSTALLER, 'size': len(body), 'sha256': hashlib.sha256(body).hexdigest()}
    order = ('https://mirror.nju.edu.cn/gimp/gimp/', 'https://opencolo.mm.fcix.net/gimp/gimp/',
             'https://download.gimp.org/gimp/')
    monkeypatch.setattr(c, '_probe', lambda family, representative, progress=None: ranking(*order))

    def answer(request, redirect):
        if host(request.full_url) == 'mirror.nju.edu.cn':
            raise urllib.error.HTTPError(request.full_url, 503, 'Busy', {}, None)
        return ranged(body, request)
    network = Network(monkeypatch, answer)
    events = []
    path = c._download(item, tmp_path, events.append, c.DownloadSources('auto', [item]))
    assert path.read_bytes() == body
    assert network.hosts() == ['mirror.nju.edu.cn', 'opencolo.mm.fcix.net']
    failed = [event for event in events if event['phase'] == 'source-failed']
    assert [event['host'] for event in failed] == ['mirror.nju.edu.cn'] and 'HTTP 503' in failed[0]['error']
    assert failed[0]['message'] == '下载源 mirror.nju.edu.cn 未能完成，改用下一个'


@pytest.mark.parametrize('failure,tries', [
    ('short', 1), ('wrong-hash', 1), ('http-404', 1), ('http-503', 1), ('certificate', 1), ('tls', 1),
    ('refused-redirect', 1), ('reset', 1), ('timeout', 2)])
def test_a_failed_source_hands_over_and_the_next_one_resumes(tmp_path, monkeypatch, failure, tries):
    body = bytes(range(200)) * 5
    item = pinned(body, url=OFFICIAL_PYPI + 'aa/bb/cc/x-1-py3-none-any.whl', filename='x-1-py3-none-any.whl')
    monkeypatch.setattr(c, '_probe', lambda family, representative, progress=None: ranking(TENCENT, OFFICIAL_PYPI))
    monkeypatch.setattr(c.time, 'sleep', lambda *args: None)

    def answer(request, redirect):
        if host(request.full_url) == 'files.pythonhosted.org':
            return ranged(body, request)
        if failure == 'short':          # a silent cut-off, as seen on USTC and ZJU
            return Reply(body[:300], 200, url=request.full_url)
        if failure == 'wrong-hash':
            return Reply(bytes(len(body)), 200, url=request.full_url)
        if failure.startswith('http-'):
            raise urllib.error.HTTPError(request.full_url, int(failure[5:]), 'Error', {}, None)
        if failure == 'certificate':    # TUNA's certificate expired on 2026-10-10
            raise urllib.error.URLError(ssl.SSLCertVerificationError(1, 'certificate has expired'))
        if failure == 'tls':
            raise urllib.error.URLError(ssl.SSLError(1, 'unexpected eof while reading'))
        if failure == 'reset':
            raise ConnectionResetError('connection reset by peer')
        if failure == 'timeout':
            raise urllib.error.URLError(TimeoutError('timed out'))
        redirect.redirect_request(request, None, 302, 'Found', {}, 'https://mirrors.example.invalid/x.whl')
    network = Network(monkeypatch, answer)
    path = c._download(item, tmp_path, None, c.DownloadSources('auto', [item]))
    assert path.read_bytes() == body and not (tmp_path / (item['filename'] + '.part')).exists()
    assert network.hosts() == ['mirrors.cloud.tencent.com'] * tries + ['files.pythonhosted.org']
    assert network.requests[-1][1] == ('bytes=300-' if failure == 'short' else None)


def test_the_last_source_uses_its_second_attempt_and_a_lone_source_keeps_three(tmp_path, monkeypatch):
    body = bytes(range(250)) * 4
    item = pinned(body, url=OFFICIAL_PYPI + 'aa/bb/cc/x-1-py3-none-any.whl', filename='x-1-py3-none-any.whl')
    monkeypatch.setattr(c, '_probe', lambda family, representative, progress=None: ranking(TENCENT, OFFICIAL_PYPI))
    monkeypatch.setattr(c.time, 'sleep', lambda *args: None)
    served = []

    def answer(request, redirect):
        served.append(request.get_header('Range'))
        if len(served) < 3:  # both sources cut off once; the last one resumes on its second attempt
            start = int(request.get_header('Range').removeprefix('bytes=').rstrip('-')) if request.get_header('Range') else 0
            return Reply(body[start:start + 100], 200 if start == 0 else 206,
                         {'Content-Range': f'bytes {start}-{len(body) - 1}/{len(body)}'} if start else {},
                         request.full_url)
        return ranged(body, request)
    network = Network(monkeypatch, answer)
    assert c._download(item, tmp_path / 'a', None, c.DownloadSources('auto', [item])).read_bytes() == body
    assert network.hosts() == ['mirrors.cloud.tencent.com', 'files.pythonhosted.org', 'files.pythonhosted.org']
    assert served == [None, 'bytes=100-', 'bytes=200-']
    # A family with a single source (Siril) keeps the three attempts of the code before 2026-10-09.
    siril = pinned(b'siril', url='https://free-astro.org/download/siril-x.zip', filename='siril-x.zip')
    timeouts = Network(monkeypatch, lambda request, redirect: (_ for _ in ()).throw(
        urllib.error.URLError(TimeoutError('timed out'))))
    with pytest.raises(urllib.error.URLError):
        c._download(siril, tmp_path / 'b', None, c.DownloadSources('auto', [siril]))
    assert timeouts.hosts() == ['free-astro.org'] * 3


@pytest.mark.parametrize('page', ['html-200', 'html-unknown-length', 'wrong-length-200', 'wrong-length-206'])
def test_an_error_page_from_the_next_source_keeps_the_bytes_already_received(tmp_path, monkeypatch, page):
    """Review 2026-10-09: the official host cuts off at 70 %, the next mirror answers with something that cannot be
    the missing bytes. Its headers are refused before the partial file is opened; the third source resumes."""
    body = bytes(i % 251 for i in range(100_000))
    item = pinned(body, url=OFFICIAL_PYPI + 'aa/bb/cc/x-1-py3-none-any.whl', filename='x-1-py3-none-any.whl')
    aliyun = 'https://mirrors.aliyun.com/pypi/packages/'
    monkeypatch.setattr(c, '_probe', lambda family, representative, progress=None:
                        ranking(OFFICIAL_PYPI, TENCENT, aliyun))
    monkeypatch.setattr(c.time, 'sleep', lambda *args: None)
    partial = tmp_path / (item['filename'] + '.part')
    seen = []

    def answer(request, redirect):
        name = host(request.full_url)
        seen.append((name, request.get_header('Range'), partial.stat().st_size if partial.exists() else 0))
        if name == 'files.pythonhosted.org':     # announces the whole file, delivers 70 %
            return Reply(body[:70_000], 200, {'Content-Length': str(len(body))}, request.full_url)
        if name == 'mirrors.cloud.tencent.com':
            text = b'<html><body>404 Not Found</body></html>'
            headers = {'html-200': {'Content-Type': 'text/html; charset=utf-8', 'Content-Length': str(len(text))},
                       'html-unknown-length': {'Content-Type': 'Text/HTML'},
                       'wrong-length-200': {'Content-Type': 'application/octet-stream',
                                            'Content-Length': str(len(text))},
                       'wrong-length-206': {'Content-Range': f'bytes 70000-99999/{len(body)}',
                                            'Content-Length': str(len(text))}}[page]
            return Reply(text, 206 if page.endswith('206') else 200, headers, request.full_url)
        return ranged(body, request)
    Network(monkeypatch, answer)
    events = []
    assert c._download(item, tmp_path, events.append, c.DownloadSources('auto', [item])).read_bytes() == body
    assert seen == [('files.pythonhosted.org', None, 0), ('mirrors.cloud.tencent.com', 'bytes=70000-', 70_000),
                    ('mirrors.aliyun.com', 'bytes=70000-', 70_000)]
    failed = [event for event in events if event['phase'] == 'source-failed']
    assert [event['host'] for event in failed] == ['files.pythonhosted.org', 'mirrors.cloud.tencent.com']
    assert 'Not the pinned file' in failed[1]['error']
    downloaded = [event['downloaded'] for event in events if event['phase'] == 'download']
    assert downloaded[-1] == len(body) and min(downloaded[downloaded.index(70_000) + 1:]) > 70_000


def test_a_source_that_fails_a_whole_file_is_tried_last_for_the_rest_of_the_run(tmp_path, monkeypatch):
    """Review 2026-10-09: a source that wins the short measurement but stalls on real transfers costs its two
    timeouts once, not once per file."""
    monkeypatch.setattr(c, '_probe', lambda family, representative, progress=None: ranking(TENCENT, OFFICIAL_PYPI))
    monkeypatch.setattr(c.time, 'sleep', lambda *args: None)
    files = {}
    for number in range(5):
        name = f'x{number}-1-py3-none-any.whl'
        files[name] = (bytes([number]) * 1000, pinned(bytes([number]) * 1000,
                                                       url=OFFICIAL_PYPI + 'aa/bb/cc/' + name, filename=name))

    def answer(request, redirect):
        if host(request.full_url) == 'mirrors.cloud.tencent.com':
            raise urllib.error.URLError(TimeoutError('timed out'))
        return ranged(files[request.full_url.rsplit('/', 1)[1]][0], request)
    network = Network(monkeypatch, answer)
    sources = c.DownloadSources('auto', [item for _, item in files.values()])
    for body, item in files.values():
        assert c._download(item, tmp_path, None, sources).read_bytes() == body
    assert network.hosts().count('mirrors.cloud.tencent.com') == 2       # both attempts on the first file only
    assert network.hosts().count('files.pythonhosted.org') == 5
    assert sources.given_up == {'pypi': [TENCENT]}
    assert [host(url) for url in sources.urls(next(iter(files.values()))[1])] == ['files.pythonhosted.org',
                                                                               'mirrors.cloud.tencent.com']


class Clock:
    """Stands in for components.time: 45 s pass on every attempt that times out."""

    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


@pytest.mark.parametrize('answered,expected', [
    (False, ['files.pythonhosted.org', 'mirrors.cloud.tencent.com', 'repo.huaweicloud.com', 'pypi.tuna.tsinghua.edu.cn']),
    (True, ['files.pythonhosted.org'] * 2 + ['mirrors.cloud.tencent.com'] * 2)])
def test_a_network_that_drops_every_connection_fails_within_the_time_budget(tmp_path, monkeypatch, answered, expected):
    """Review 2026-10-09: eight sources x two 45 s timeouts were 12 minutes per file before failing. A source that
    did not answer its measurement gets one attempt, and failed attempts that received nothing stop at 180 s."""
    body = b'x' * 1000
    item = pinned(body, url=OFFICIAL_PYPI + 'aa/bb/cc/x-1-py3-none-any.whl', filename='x-1-py3-none-any.whl')
    measured = [{'source': base, 'host': host(base), 'ok': answered, 'error': 'timed out'}
                for base in c._family_bases('pypi')]
    monkeypatch.setattr(c, '_probe', lambda family, representative, progress=None: measured)
    clock = Clock()
    monkeypatch.setattr(c, 'time', clock)

    def answer(request, redirect):
        clock.now += 45.0
        raise urllib.error.URLError(TimeoutError('timed out'))
    network = Network(monkeypatch, answer)
    events = []
    with pytest.raises(urllib.error.URLError):
        c._download(item, tmp_path, events.append, c.DownloadSources('auto', [item]))
    assert network.hosts() == expected
    assert sum(45.0 for _ in network.hosts()) == c._NO_PROGRESS_SECONDS == 180.0
    assert [event['phase'] for event in events].count('source-retry') == (2 if answered else 0)


def test_each_retry_is_announced_and_is_a_cancel_point(tmp_path, monkeypatch):
    body = b'x' * 1000
    item = pinned(body, url=OFFICIAL_PYPI + 'aa/bb/cc/x-1-py3-none-any.whl', filename='x-1-py3-none-any.whl')
    monkeypatch.setattr(c, '_probe', lambda family, representative, progress=None: ranking(TENCENT, OFFICIAL_PYPI))
    monkeypatch.setattr(c.time, 'sleep', lambda *args: pytest.fail('cancelled before the pause'))
    network = Network(monkeypatch, lambda request, redirect: (_ for _ in ()).throw(
        urllib.error.URLError(TimeoutError('timed out'))))
    events = []

    def progress(event):
        events.append(event)
        if event['phase'] == 'source-retry':
            raise Stop()
    with pytest.raises(Stop):
        c._download(item, tmp_path, progress, c.DownloadSources('auto', [item]))
    assert network.hosts() == ['mirrors.cloud.tencent.com']
    assert events[-1]['message'] == '下载源 mirrors.cloud.tencent.com 再试一次' and 'timed out' in events[-1]['error']


def test_official_mode_follows_redirects_only_to_the_official_hosts(tmp_path, monkeypatch):
    """Review 2026-10-09: 'official' means the official URLs (and, for GIMP, its upstream mirror opencolo), also for
    redirects; download.gimp.org's location-based pick is refused even when it is a listed mirror."""
    assert c.family_hosts('pypi', 'official') == {'files.pythonhosted.org'}
    assert c.family_hosts('python', 'official') == {'www.python.org'}
    assert c.family_hosts('siril', 'official') == {'free-astro.org'}
    assert c.family_hosts('gimp', 'official') == {'download.gimp.org', 'opencolo.mm.fcix.net'}
    body = b'GIMP installer fixture ' * 64
    item = {**d.GIMP_INSTALLER, 'size': len(body), 'sha256': hashlib.sha256(body).hexdigest()}
    monkeypatch.setattr(c.time, 'sleep', lambda *args: None)
    monkeypatch.setattr(c, '_probe', lambda family, representative, progress=None:
                        ranking(*c._family_bases('gimp')))
    nju = 'https://mirror.nju.edu.cn/gimp/gimp/v3.2/windows/gimp-3.2.6-setup.exe'

    def answer(request, redirect):
        if host(request.full_url) == 'download.gimp.org':
            request = redirect.redirect_request(request, None, 302, 'Found', {}, nju)
        return ranged(body, request)
    for mode, expected in (('official', ['download.gimp.org', 'opencolo.mm.fcix.net']), ('auto', ['download.gimp.org'])):
        network = Network(monkeypatch, answer)
        path = c._download(item, tmp_path / mode, None, c.DownloadSources(mode, [item]))
        assert path.read_bytes() == body and network.hosts() == expected
    with pytest.raises(ValueError, match='approved HTTPS origins'):
        c._HTTPSRedirect(c.family_hosts('pypi', 'official')).redirect_request(
            urllib.request.Request(OFFICIAL_PYPI + 'x.whl'), None, 302, 'Found', {}, TENCENT + 'x.whl')


def _addinfo(body, url, code, headers=()):
    message = email.message.Message()
    for key, value in headers:
        message[key] = value
    reply = urllib.response.addinfourl(io.BytesIO(body), message, url, code)
    reply.msg = 'Found' if code == 302 else 'OK'
    return reply


class FakeHTTPS(urllib.request.HTTPSHandler):
    """Replaces urllib's HTTPS handler; the redirect and error handling around it stay urllib's own."""

    def __init__(self, answer):
        super().__init__()
        self.answer, self.seen, self.lock = answer, [], threading.Lock()

    def https_open(self, request):
        with self.lock:
            self.seen.append((request.full_url, request.get_header('Range')))
        return self.answer(request)


def real_urllib(monkeypatch, answer):
    https = FakeHTTPS(answer)
    monkeypatch.setattr(c.urllib.request, 'build_opener',
                        lambda *handlers: REAL_BUILD_OPENER(*handlers, urllib.request.ProxyHandler({}), https))
    return https


def gimp_answer(body, *, slow=()):
    def answer(request):
        if request.full_url.startswith('https://download.gimp.org/'):
            return _addinfo(b'', request.full_url, 302,
                            [('Location', 'https://ziply.mm.fcix.net/gimp/v3.2/windows/gimp-3.2.6-setup.exe')])
        if host(request.full_url) in slow:
            threading.Event().wait(0.3)
        spec = request.get_header('Range')
        if spec:
            start, _, end = spec.removeprefix('bytes=').partition('-')
            start, end = int(start), (int(end) if end else len(body) - 1)
            return _addinfo(body[start:end + 1], request.full_url, 206,
                            [('Content-Range', f'bytes {start}-{end}/{len(body)}')])
        return _addinfo(body, request.full_url, 200, [('Content-Length', str(len(body)))])
    return answer


def test_gimp_official_redirect_to_an_unlisted_mirror_is_refused_once_through_real_urllib(tmp_path, monkeypatch):
    body = b'official GIMP installer fixture ' * 32
    item = {**d.GIMP_INSTALLER, 'size': len(body), 'sha256': hashlib.sha256(body).hexdigest()}
    https = real_urllib(monkeypatch, gimp_answer(body))
    monkeypatch.setattr(c.time, 'sleep', lambda *args: pytest.fail('a refused redirect is not retried'))
    events = []
    path = c._download(item, tmp_path, events.append, c.DownloadSources('official', [item]))
    assert path.read_bytes() == body
    assert [url for url, _ in https.seen] == GIMP[:2]
    assert [event['host'] for event in events if event['phase'] == 'source-failed'] == ['download.gimp.org']
    assert 'approved HTTPS origins' in events[0]['error']


def test_automatic_gimp_choice_measures_through_real_urllib_and_skips_the_refused_host(tmp_path, monkeypatch):
    body = bytes(range(256)) * 1200      # larger than one probe
    item = {**d.GIMP_INSTALLER, 'size': len(body), 'sha256': hashlib.sha256(body).hexdigest()}
    monkeypatch.setattr(c, 'PROBE_SECONDS', 2.0)
    https = real_urllib(monkeypatch, gimp_answer(body, slow={'opencolo.mm.fcix.net', 'mirrors.aliyun.com'}))
    events = []
    path = c._download(item, tmp_path, events.append, c.DownloadSources('auto', [item]))
    assert path.read_bytes() == body
    [source] = [event for event in events if event['phase'] == 'source']
    ranked = source['ranking']
    assert ranked[-1]['host'] == 'download.gimp.org' and not ranked[-1]['ok']
    assert 'approved HTTPS origins' in ranked[-1]['error']
    assert {entry['host'] for entry in ranked[:2]} == {'mirror.nju.edu.cn', 'mirrors.ustc.edu.cn'}
    downloads = [url for url, header in https.seen if header is None]
    assert len(downloads) == 1 and host(downloads[0]) == source['host']
    assert [url for url, _ in https.seen].count(GIMP[0]) == 1      # measured once, never downloaded from


# ----------------------------------------------------------------------------------------------- plumbing

def test_install_components_hands_one_choice_to_every_download(tmp_path, monkeypatch):
    import zipfile
    python, siril = tmp_path / 'python.zip', tmp_path / 'siril.zip'
    with zipfile.ZipFile(python, 'w') as archive:
        for name in ('python.exe', 'pythonw.exe', 'python313._pth'):
            archive.writestr(name, b'test')
    with zipfile.ZipFile(siril, 'w') as archive:
        archive.writestr('Siril/bin/siril-cli.exe', b'test')
    data = {'python': {'kind': 'python'}, 'wheels': [], 'optional_wheels': [], 'siril': {'kind': 'siril'},
            'nox': {'filename': 'nox.onnx', 'sha256': '0' * 64, 'size': 5, 'adapter': {}}}
    seen = []

    def download(item, cache, progress=None, sources=None):
        seen.append(sources)
        return python if item['kind'] == 'python' else siril
    monkeypatch.setattr(c, '_download', download)
    monkeypatch.setattr(c, '_validate', lambda *args: None)
    c.install_components(tmp_path / 'components', app_root=tmp_path / 'app', manifest=data, activate=False)
    assert len(seen) == 2 and seen[0] is seen[1] and seen[0].mode == 'auto'
    seen.clear()
    c.install_components(tmp_path / 'other', app_root=tmp_path / 'app', manifest=data, activate=False,
                         download_source='official')
    assert {sources.mode for sources in seen} == {'official'}
    with pytest.raises(ValueError, match='download_source'):
        c.install_components(tmp_path / 'third', app_root=tmp_path / 'app', manifest=data, download_source='cn')
    assert not (tmp_path / 'third').exists()


def test_the_build_fetches_the_python_seed_with_a_source_choice(builder_case, monkeypatch):
    """Review 2026-10-09: the python family's mirrors serve the build too, not only the in-app repair."""
    fetch, seen = c._download, []

    def download(item, cache, progress=None, sources=None):
        seen.append(sources)
        return fetch(item, cache)
    monkeypatch.setattr(c, '_download', download)
    builder = builder_case.builder
    builder.build(builder_case.output / 'auto', builder_case.compiler, builder_case.model)
    builder.build(builder_case.output / 'official', builder_case.compiler, builder_case.model,
                  download_source='official')
    assert [sources.mode for sources in seen] == ['auto', 'official']
    assert inspect.signature(builder.build).parameters['download_source'].default == 'auto'
    with pytest.raises(ValueError, match='download_source'):
        builder.build(builder_case.output / 'bad', builder_case.compiler, builder_case.model, download_source='cn')


def test_gui_repair_and_installer_default_to_the_automatic_choice():
    for function in (c.install_components, c.install_retouch_components, d.install, d.install_gimp):
        assert inspect.signature(function).parameters['download_source'].default == 'auto'
    # The GUI component manager calls both without choosing, so it measures like Setup does.
    assert 'download_source' not in (ROOT / 'src/aester_desktop/experience.py').read_text(encoding='utf-8')


def test_worker_accepts_the_choice_in_every_mode_and_passes_it_on(worker_case, monkeypatch):
    case = worker_case
    prepare = c.install_components

    def emitting(root, **kwargs):
        kwargs['progress']({'phase': 'source', 'family': 'pypi', 'host': 'mirrors.cloud.tencent.com',
                            'reachable': True, 'ranking': [{'host': 'mirrors.cloud.tencent.com', 'ok': True}],
                            'message': '已选择下载源：mirrors.cloud.tencent.com'})
        return prepare(root, **kwargs)
    monkeypatch.setattr(c, 'install_components', emitting)
    assert case.run('--preflight', '--download-source', 'official') == 0
    assert case.run('--download-source', 'official') == 0
    assert [call[2]['download_source'] for call in case.calls if call[0] == 'base'] == ['official']
    events = [json.loads(line) for line in (case.area / 'install-events.jsonl').read_text(encoding='utf-8').splitlines()]
    assert any(event['phase'] == 'source' and event['ranking'][0]['host'] == 'mirrors.cloud.tencent.com'
               for event in events)
    assert case.run('--activate', str(case.state), '--download-source', 'official') == 0
    assert case.run('--rollback', '--download-source', 'auto') == 0
    assert case.run() == 0
    assert [call[2]['download_source'] for call in case.calls if call[0] == 'base'] == ['official', 'auto']
    with pytest.raises(SystemExit):
        case.run('--download-source', 'mirror')


def test_worker_passes_the_choice_to_the_gimp_download(worker_case, monkeypatch):
    calls = []

    def prepare(root, **kwargs):
        calls.append(kwargs)
        return {'ready': True, 'gimp': None, 'skill': None, 'notices': []}
    monkeypatch.setattr(c, 'install_retouch_components', prepare)
    monkeypatch.setattr(c, 'verify_prepared_retouch_components', lambda *args, **kwargs: kwargs['expected'])
    monkeypatch.setattr(d, '_discover', lambda *args, **kwargs: {'status': 'not_found'})
    assert worker_case.run('--gimp', '--download-source', 'official') == 0
    assert calls[0]['include_gimp'] is True and calls[0]['download_source'] == 'official'


def test_setup_passes_the_choice_to_every_worker_mode():
    script = (ROOT / 'packaging/aester_p2/Aester.iss').read_text(encoding='utf-8')
    assert "{param:DOWNLOADSOURCE|auto}" in script
    params = script[script.index('WorkerParams :='):script.index("WorkerParams + ' --preflight'")]
    assert "' --download-source ' + DownloadSource" in params
    # The value reaching the command line is one of two fixed words, never the raw parameter.
    assert "DownloadSource := 'official'" in script and "DownloadSource := 'auto'" in script
    assert "WorkerParams + ' --rollback'" in script and "WorkerParams + ' --activate '" in script
    messages = (ROOT / 'packaging/aester_p2/installer-messages.iss').read_text(encoding='utf-8')
    for prefix in ('en.', 'zh_CN.'):
        [line] = [text for text in messages.splitlines() if text.startswith(prefix + 'PrepareDescription=')]
        assert 'SHA-256' in line
