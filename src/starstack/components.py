"""Versioned, per-user component preparation; no pip, elevation, or global edits.

The module deliberately uses only the standard library so the installer can run
it in the official embedded Python before the scientific runtime exists.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from pathlib import Path, PurePosixPath
from typing import Callable
import hashlib
import http.client
import json
import os
import queue
import re
import shutil
import ssl
import stat
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zipfile

try:
    from . import install_ownership as ownership
except ImportError:  # Install/uninstall bootstrap uses loose stdlib modules.
    import install_ownership as ownership

Progress = Callable[[dict], None]
_REPARSE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
USER_AGENT = "Aester-Components/1"   # several mirrors refuse requests without a User-Agent

# Where pinned downloads may come from (2026-10-09: one installer for mainland China and abroad). An item's
# official URL picks its family by the official host; below the official prefix the pinned path stays the item's
# own, so does its size and SHA-256, and the final check accepts only those bytes from any source. When checked on
# 2026-10-09, every listed mirror had every pinned file at the official size and every sampled full download
# matched its SHA-256. The hosts below are the only ones the first URL or a redirect may reach for that family:
# HTTPS, no user information, port 443. Keep this table here, not in download-manifest.json, whose hash is part of
# the component version identity.
# Not used on purpose: mirrors.huaweicloud.com for PyPI (ignores Range), mirrors.zju, cernet, hust (stale or cut
# transfers short), doubanio, jlu, lzu (redirect elsewhere), sjtug, www.free-astro.org (certificate mismatch).
DOWNLOAD_SOURCES = {
    "pypi": {
        "official": "https://files.pythonhosted.org/packages/",
        "mirrors": (
            "https://mirrors.cloud.tencent.com/pypi/packages/",
            "https://repo.huaweicloud.com/repository/pypi/packages/",
            "https://pypi.tuna.tsinghua.edu.cn/packages/",
            "https://mirrors.ustc.edu.cn/pypi/packages/",
            "https://mirrors.bfsu.edu.cn/pypi/web/packages/",
            "https://mirrors.aliyun.com/pypi/packages/",
            "https://mirror.nju.edu.cn/pypi/web/packages/",
        ),
    },
    # Only the build and an in-app repair download Python; Setup carries its own copy.
    "python": {
        "official": "https://www.python.org/ftp/python/",
        "mirrors": (
            "https://cdn.npmmirror.com/binaries/python/",
            "https://repo.huaweicloud.com/python/",
            "https://mirrors.tuna.tsinghua.edu.cn/python/",
            "https://mirrors.ustc.edu.cn/python/",
            "https://mirror.nju.edu.cn/python/",
        ),
    },
    "gimp": {
        # download.gimp.org redirects to a mirror chosen by the client's location; a host not listed here is
        # refused and the next source is tried at once.
        "official": "https://download.gimp.org/gimp/",
        # Part of GIMP's own mirror network; the only fallback before 2026-10-09, so "official" mode keeps it.
        "upstream_mirrors": ("https://opencolo.mm.fcix.net/gimp/gimp/",),
        "mirrors": (
            "https://mirror.nju.edu.cn/gimp/gimp/",
            "https://mirrors.ustc.edu.cn/gimp/",
            "https://mirrors.aliyun.com/gimp/gimp/",
        ),
    },
    # No mirror of the Siril portable build exists; more sources can be added here.
    "siril": {"official": "https://free-astro.org/download/", "mirrors": ()},
}
DOWNLOAD_SOURCE_MODES = ("auto", "official")
PROBE_BYTES = 256 * 1024       # one ranged request per candidate
PROBE_SECONDS = 4.5            # connect + PROBE_BYTES, per candidate; slower candidates rank last
_PROBE_TICK = 0.25             # progress events while measuring keep Setup's cancel button responsive
_PROBE_CHUNK = 16 * 1024       # at most one receive per read, so the deadline or a stop request is seen promptly
_NO_PROGRESS_SECONDS = 180.0   # per file: once failed attempts that received nothing add up to this, give up
_PINNED_PATH = re.compile(r"[A-Za-z0-9._+\-]+(?:/[A-Za-z0-9._+\-]+)*")


class ComponentDamaged(RuntimeError):
    """A bound component is missing/corrupt; a new private version can repair it."""


def default_component_root() -> Path:
    override = os.environ.get("AESTER_COMPONENT_ROOT")
    if override:
        return Path(override).expanduser().absolute()
    local = os.environ.get("LOCALAPPDATA")
    return (Path(local) if local else Path.home() / ".local/share") / "Aester/components"


def _app_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _manifest(app_root=None) -> dict:
    root = Path(app_root) if app_root else _app_root()
    return json.loads((root / "packaging/aester_p2/download-manifest.json").read_text(encoding="utf-8"))


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _safe_root(path) -> Path:
    root = Path(path).expanduser().absolute()
    if root == Path(root.anchor):
        raise ValueError("A component directory, not a filesystem root, is required")
    for item in (root, *root.parents):
        if item.exists() and (item.is_symlink() or (item.lstat().st_file_attributes & _REPARSE if os.name == "nt" else False)):
            raise ValueError(f"Linked component path is unsupported: {item}")
    return root


def _inside(root: Path, path: Path) -> Path:
    _safe_root(path)
    if not path.resolve().is_relative_to(root.resolve()) or path.resolve() == root.resolve():
        raise ValueError("Component path escaped its owner directory")
    return path


def _atomic(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


@contextmanager
def _lock(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    with _inside(root, root / "components.lock").open("a+b") as stream:
        stream.seek(0)
        stream.write(b"0")
        stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError("另一个窗口正在安装组件，请稍后重试") from error
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _notify(progress, **event):
    if progress:
        progress(event)


class _OriginRefused(ValueError):
    """A redirect or response left the approved HTTPS origins of the file's source family."""


def _origin_ok(url, hosts) -> bool:
    try:
        parsed = urllib.parse.urlsplit(url)
        return (parsed.scheme == "https" and parsed.hostname in hosts and parsed.username is None
                and parsed.password is None and parsed.port in (None, 443))
    except (ValueError, TypeError, AttributeError):  # an unparsable port or a missing URL
        return False


class _HTTPSRedirect(urllib.request.HTTPRedirectHandler):
    """Follow a redirect only to an approved HTTPS origin of the file's source family (one host set per family
    since 2026-10-09, where one global set served every file before)."""

    def __init__(self, hosts=frozenset()):
        super().__init__()
        self.hosts = frozenset(hosts)

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not _origin_ok(newurl, self.hosts):
            if fp is not None:
                fp.close()
            raise _OriginRefused("Download redirect left approved HTTPS origins")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _opener(hosts):
    return urllib.request.build_opener(_HTTPSRedirect(hosts))


def _source_family(url):
    """(family, pinned path below its official prefix) for a URL on a family's official host.

    The path is None when the URL is outside the mapped prefix or is not a plain path; such an item is fetched
    from its own URL only. Any other origin is refused before the network or the cache is touched."""
    for family, spec in DOWNLOAD_SOURCES.items():
        if _origin_ok(url, {urllib.parse.urlsplit(spec["official"]).hostname}):
            path = url[len(spec["official"]):] if url.startswith(spec["official"]) else None
            if path is not None and (not _PINNED_PATH.fullmatch(path)
                                     or any(part in (".", "..") for part in path.split("/"))):
                path = None
            return family, path
    raise ValueError("Unsupported component origin")


def _family_bases(family, mode="auto"):
    """Candidate URL prefixes in the default order: official, upstream mirror, then the listed mirrors."""
    spec = DOWNLOAD_SOURCES[family]
    bases = (spec["official"], *spec.get("upstream_mirrors", ()))
    return bases if mode == "official" else bases + tuple(spec.get("mirrors", ()))


def family_hosts(family, mode="auto") -> frozenset:
    """Every host the family's files or redirects may come from in this mode: in 'official' mode only the
    official host (and, for GIMP, its upstream mirror), so a redirect cannot reach a listed mirror either."""
    return frozenset(urllib.parse.urlsplit(base).hostname for base in _family_bases(family, mode))


def candidate_urls(item, mode="auto"):
    """The item's own URL first, then the same pinned path on every other source, in the default order."""
    family, path = _source_family(item["url"])
    if path is None:
        return [item["url"]]
    return [base + path for base in _family_bases(family, mode)]


def _short_error(error) -> str:
    """One line for install-events.jsonl: what failed on one source."""
    if isinstance(error, urllib.error.HTTPError):
        text = f"HTTP {error.code}: {error.reason}"
    elif isinstance(error, urllib.error.URLError) and isinstance(error.reason, BaseException):
        text = f"{type(error.reason).__name__}: {error.reason}"
    else:
        text = f"{type(error).__name__}: {error}"
    return text[:300]


def _body_problem(response, length):
    """Why a response's headers show that its body is not the expected ``length`` bytes of the pinned file, or
    None. Checked before the body is read, so an error page never touches a partial file."""
    kind = (response.headers.get("Content-Type") or "").split(";")[0].strip().lower()
    if kind == "text/html":
        return "an HTML page instead of the pinned file"
    declared = response.headers.get("Content-Length")
    if declared is not None:
        try:
            if int(declared) != length:
                return f"Content-Length {int(declared)} instead of {length}"
        except ValueError:
            return "unreadable Content-Length"
    return None


def _probe_one(base, url, hosts, size, stop=None):
    """Time the first PROBE_BYTES of one candidate. It counts only with 206, the exact Content-Range of the
    pinned size, and every requested byte within PROBE_SECONDS; a 200, an HTML page or a stale file does not.

    A candidate that answered correctly but was still delivering at the deadline is marked ``partial`` with its
    rate so far: it ranks after the full answers and before the candidates that did not answer. ``stop`` (an
    Event) ends the measurement at the next receive."""
    result = {"host": urllib.parse.urlsplit(base).hostname, "source": base, "ok": False}
    want = min(PROBE_BYTES, size)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept-Encoding": "identity",
                                                   "Range": f"bytes=0-{want - 1}"})
    started = time.monotonic()
    answered, received = False, 0
    try:
        with _opener(hosts).open(request, timeout=PROBE_SECONDS) as response:
            if not _origin_ok(response.url, hosts):
                raise _OriginRefused("Probe response left approved HTTPS origins")
            if response.status != 206:
                raise ValueError(f"HTTP {response.status} instead of a partial response")
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
            if not match or int(match[1]) != 0 or int(match[2]) != want - 1 or int(match[3]) != size:
                raise ValueError("Content-Range does not match the pinned file")
            problem = _body_problem(response, want)
            if problem:
                raise ValueError(problem)
            answered = True
            read = getattr(response, "read1", response.read)   # what one receive brings, not a full buffer
            while received < want:
                if stop is not None and stop.is_set():
                    raise TimeoutError("probe stopped")
                if time.monotonic() - started > PROBE_SECONDS:
                    raise TimeoutError("probe deadline passed")
                chunk = read(min(_PROBE_CHUNK, want - received))
                if not chunk:
                    break
                received += len(chunk)
            if received != want:
                raise ValueError("Probe answer ended early")
        seconds = time.monotonic() - started
        if seconds > PROBE_SECONDS:
            raise TimeoutError("probe deadline passed")
        result.update(ok=True, seconds=round(seconds, 3), bytes_per_second=round(want / max(seconds, 1e-6)))
    except Exception as error:
        if isinstance(error, urllib.error.HTTPError):
            error.close()
        seconds = time.monotonic() - started
        result.update(seconds=round(seconds, 3), error=_short_error(error))
        if answered and received and isinstance(error, TimeoutError):
            # The right file, still arriving when time ran out (a slow or distant link): a real speed signal.
            result.update(partial=True, received=received, bytes_per_second=round(received / max(seconds, 1e-6)))
    return result


def _answered(entry) -> bool:
    """A measured source that served the pinned file: in full within the deadline, or still delivering it."""
    return bool(entry.get("ok") or entry.get("partial"))


def _probe(family, item, progress=None):
    """Measure every candidate of a family on one pinned file, all at once.

    Ranked: full answers by speed, then candidates still delivering at the deadline by their rate so far, then the
    rest in table order (all are still tried). Waits at most PROBE_SECONDS (plus a margin) and emits ``probe``
    events meanwhile, so a cancel request is seen within a fraction of a second. The measuring threads are daemon
    threads told to stop when this returns or raises: a request still in flight neither delays this process's
    exit nor keeps using the network beyond its next receive."""
    _, path = _source_family(item["url"])
    bases = _family_bases(family, "auto")
    hosts = family_hosts(family, "auto")
    results = [None] * len(bases)
    answers, stop = queue.SimpleQueue(), threading.Event()

    def measure(index, base):
        try:
            answers.put((index, _probe_one(base, base + path, hosts, item["size"], stop)))
        except BaseException as error:  # _probe_one reports its own failures; this is a safety net
            answers.put((index, {"error": _short_error(error)}))

    _notify(progress, phase="probe", family=family, candidates=len(bases), message="正在测速选择下载源")
    try:
        for index, base in enumerate(bases):
            threading.Thread(target=measure, args=(index, base), name=f"aester-source-probe-{index}",
                             daemon=True).start()
        pending = len(bases)
        deadline = time.monotonic() + PROBE_SECONDS + 0.5
        tick = time.monotonic() + _PROBE_TICK
        while pending:
            now = time.monotonic()
            if now >= deadline:
                break
            try:
                index, result = answers.get(timeout=max(0.001, min(tick, deadline) - now))
            except queue.Empty:
                pass
            else:
                results[index] = result
                pending -= 1
            if pending and time.monotonic() >= tick:
                _notify(progress, phase="probe", family=family, candidates=len(bases), message="正在测速选择下载源")
                tick = time.monotonic() + _PROBE_TICK
    finally:
        stop.set()
    for index, base in enumerate(bases):
        entry = results[index] or {"error": "no answer within the deadline"}
        results[index] = {"host": urllib.parse.urlsplit(base).hostname, "source": base, "ok": False, **entry}
    # Stable sort: equal speeds and all failures keep the table order (official first).
    ranked = sorted(results, key=lambda entry: (0 if entry["ok"] else 1 if entry.get("partial") else 2,
                                                -entry.get("bytes_per_second", 0) if _answered(entry) else 0))
    chosen = ranked[0]
    reachable = _answered(chosen)
    _notify(progress, phase="source", family=family, host=chosen["host"], reachable=reachable,
            representative=item["filename"], ranking=ranked,
            message=f"已选择下载源：{chosen['host']}" if reachable else "下载源测速均无应答，按默认顺序尝试")
    return ranked


class DownloadSources:
    """Source choice for one preparation run.

    ``auto`` (the default) measures all candidates of a family once, the first time a file of that family must
    really be downloaded, and tries them fastest first; ``official`` keeps the behaviour before 2026-10-09: the
    official URL, and for GIMP its upstream mirror, without measuring, and redirects only to those hosts.
    Either way a file is accepted only with its pinned size and SHA-256.

    A source that fails a whole file moves to the end of its family's order for the rest of the run, so a source
    that won the short measurement but stalls on real transfers costs its timeouts once, not once per file.
    """

    def __init__(self, mode="auto", items=()):
        if mode not in DOWNLOAD_SOURCE_MODES:
            raise ValueError("download_source must be 'auto' or 'official'")
        self.mode = mode
        self._items = [item for item in items if isinstance(item, dict)]
        self.rankings = {}          # family -> probe results, best first
        self.given_up = {}          # family -> sources that failed a whole file in this run, in that order

    def _representative(self, family, item):
        """The largest pinned file of the family, so the ranged read is never the whole file."""
        best = item
        for candidate in self._items:
            try:
                same, path = _source_family(candidate["url"])
                larger = type(candidate["size"]) is int and candidate["size"] > best["size"]
            except (KeyError, TypeError, ValueError):
                continue
            if same == family and path is not None and larger:
                best = candidate
        return best

    def candidates(self, item, progress=None):
        """[(source prefix or None, URL, attempts)] in the order to try.

        A lone source keeps three attempts; otherwise each gets two, except one that did not answer its
        measurement (no connection, no name, a wrong answer), which gets one."""
        family, path = _source_family(item["url"])
        if path is None:
            return [(None, item["url"], 3)]
        bases = _family_bases(family, self.mode)
        if self.mode == "auto" and len(bases) > 1:
            if family not in self.rankings:
                self.rankings[family] = _probe(family, self._representative(family, item), progress)
            bases = tuple(entry["source"] for entry in self.rankings[family])
        failed = self.given_up.get(family, ())
        bases = (*(base for base in bases if base not in failed), *(base for base in failed if base in bases))
        if len(bases) == 1:
            return [(bases[0], bases[0] + path, 3)]
        measured = {entry["source"]: entry for entry in self.rankings.get(family, ())}
        return [(base, base + path, 1 if base in measured and not _answered(measured[base]) else 2)
                for base in bases]

    def urls(self, item, progress=None):
        return [url for _, url, _ in self.candidates(item, progress)]

    def give_up(self, item, source):
        """``source`` failed a whole file: try it last for the rest of this run."""
        if source is None:
            return
        family, _ = _source_family(item["url"])
        failed = self.given_up.setdefault(family, [])
        if source in failed:
            failed.remove(source)
        failed.append(source)


def _failure_kind(error) -> str:
    """How a failed attempt moves on.

    'final': this source cannot serve the file (redirect off the family's hosts, insecure response, certificate,
    HTTP 4xx): the next source at once, never again here. 'source': a wrong or cut-off answer (HTTP 5xx/408/429,
    other TLS failures, wrong range, size or hash, an HTML page or wrong length, short read, refused connection):
    the next source at once, the partial file resumes there; the last source uses its remaining attempts.
    'network': a timeout or name lookup failure: another attempt here if the source has one left, then the next
    source.
    """
    if isinstance(error, _OriginRefused):
        return "final"
    if isinstance(error, urllib.error.HTTPError):
        return "source" if error.code in (408, 425, 429) or error.code >= 500 else "final"
    reason = error.reason if isinstance(error, urllib.error.URLError) else error
    if isinstance(reason, ssl.SSLCertVerificationError):
        return "final"
    if isinstance(reason, TimeoutError):
        return "network"
    if isinstance(reason, (ssl.SSLError, ConnectionError, ValueError, http.client.HTTPException)):
        return "source"
    return "network"


def _valid_file(path, item) -> bool:
    return path.is_file() and path.stat().st_size == item["size"] and _digest(path) == item["sha256"]


def _fetch(opener, url, item, partial, hosts, progress, stats=None, allowed_urls=None):
    """One attempt on one source: append exactly the missing bytes, then check size and SHA-256.

    The headers are checked before the partial file is opened: an answer that cannot be the pinned bytes (an
    HTML page, a length other than the missing bytes) leaves the bytes already received for the next source.
    ``stats['received']`` counts the body bytes this attempt received."""
    filename, size = item["filename"], item["size"]
    offset = partial.stat().st_size if partial.exists() else 0
    if offset >= size:
        partial.unlink()
        offset = 0
    headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    with opener.open(request, timeout=45) as response:
        if urllib.parse.urlsplit(response.url).scheme != "https":
            raise _OriginRefused("Insecure response")
        if not _origin_ok(response.url, hosts):
            raise _OriginRefused("Download response left approved HTTPS origins")
        if allowed_urls is not None and response.url not in allowed_urls:
            raise _OriginRefused("Model response left its exact approved URLs")
        code = response.status
        if code == 206:
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
            if not match or int(match[1]) != offset or int(match[3]) != size or int(match[2]) != size - 1:
                raise ValueError("Unexpected resumed response range")
        elif code == 200:
            offset = 0  # The server ignored Range: restart, but only with what can be the whole file.
        else:
            raise ValueError(f"Unexpected download HTTP status: {code}")
        problem = _body_problem(response, size - offset)
        if problem:
            raise ValueError(f"Not the pinned file ({problem}): {filename}")
        with partial.open("ab" if offset else "wb") as stream:
            count = offset
            while chunk := response.read(1024 * 1024):
                count += len(chunk)
                if stats is not None:
                    stats["received"] += len(chunk)
                if count > size:
                    raise ValueError("Download exceeds pinned size")
                stream.write(chunk)
                _notify(progress, phase="download", filename=filename, downloaded=count, total=size, message="正在下载并校验组件")
    if partial.stat().st_size < size:
        raise ValueError(f"下载未完成就中断：{filename}")  # kept; the next attempt or source resumes it
    if not _valid_file(partial, item):
        partial.unlink()  # A wrong complete body must not be resumed forever.
        raise ValueError(f"组件大小或 SHA256 不符：{filename}")


def _publish(partial, target):
    for attempt in range(3):
        try:
            partial.replace(target)
            return target
        except OSError:  # e.g. a scanner holding the just-written file
            if attempt == 2:
                raise
            time.sleep(.25)


def _download(item: dict, cache: Path, progress=None, sources=None) -> Path:
    """Resume only exact byte ranges; a completed file always needs size+SHA256.

    ``sources`` (a DownloadSources) orders the candidate URLs and may first measure them; without it only the
    official URL (and, for GIMP, its upstream mirror) is tried, as before 2026-10-09. Each source gets two
    attempts (one if it did not answer its measurement), a lone source three; a partial file resumes on the next
    source, since only the pinned size and SHA-256 decide. Once failed attempts that received nothing add up to
    _NO_PROGRESS_SECONDS (a network that drops every connection), the last error is raised without trying the
    remaining sources.
    """
    filename = item["filename"]
    if not isinstance(filename, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.+\-]*", filename):
        raise ValueError("Invalid component filename")
    if type(item["size"]) is not int or item["size"] <= 0 or not re.fullmatch(r"[a-f0-9]{64}", item["sha256"]):
        raise ValueError("Invalid pinned component identity")
    family, _ = _source_family(item["url"])  # before the cache folder exists
    if sources is None:
        sources = DownloadSources("official")
    hosts = family_hosts(family, sources.mode)
    cache = _safe_root(cache)
    cache.mkdir(parents=True, exist_ok=True)
    target = _inside(cache, cache / filename)
    partial = _inside(cache, cache / (filename + ".part"))
    if _valid_file(target, item):
        _notify(progress, phase="cached", filename=filename, downloaded=item["size"], total=item["size"], message="已校验下载缓存")
        return target
    if _valid_file(partial, item):
        return _publish(partial, target)
    candidates = sources.candidates(item, progress)
    opener = _opener(hosts)
    failure, idle = None, 0.0
    for index, (source, url, attempts) in enumerate(candidates):
        if not _origin_ok(url, hosts):
            raise ValueError("Unsupported component origin")
        host = urllib.parse.urlsplit(url).hostname
        last = index == len(candidates) - 1
        for attempt in range(attempts):
            if attempt:
                # Each retry is also a cancel point; the reason lands in install-events.jsonl.
                _notify(progress, phase="source-retry", family=family, filename=filename, host=host,
                        error=_short_error(failure), message=f"下载源 {host} 再试一次")
                time.sleep(.25)
            stats, started = {"received": 0}, time.monotonic()
            try:
                _fetch(opener, url, item, partial, hosts, progress, stats)
            except (OSError, ValueError, urllib.error.URLError, http.client.HTTPException) as error:
                failure = error
                if isinstance(error, urllib.error.HTTPError):
                    error.close()
                if not stats["received"]:
                    idle += time.monotonic() - started
                kind = _failure_kind(error)
                if idle >= _NO_PROGRESS_SECONDS or kind == "final" or (kind == "source" and not last):
                    break
                continue
            return _publish(partial, target)
        sources.give_up(item, source)
        if idle >= _NO_PROGRESS_SECONDS:
            break
        if not last:
            _notify(progress, phase="source-failed", family=family, filename=filename, host=host,
                    error=_short_error(failure), message=f"下载源 {host} 未能完成，改用下一个")
    raise failure


class _ExactHTTPSRedirect(_HTTPSRedirect):
    """Model releases have no source-family mirrors; each redirect URL is pinned explicitly."""

    def __init__(self, urls):
        self.urls = frozenset(urls)
        super().__init__(urllib.parse.urlsplit(url).hostname for url in self.urls)

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if newurl not in self.urls:
            if fp is not None:
                fp.close()
            raise _OriginRefused("Model redirect left its exact approved URLs")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _nox_download_item(data):
    """Read only the reviewed manifest's converted ONNX release, never its upstream PB provenance URL."""
    model = data['nox']
    spec = model.get('download', {})
    if not isinstance(spec, dict):
        raise ValueError('Invalid nox download configuration')
    url = spec.get('url')
    if url in (None, ''):
        return None
    redirects = spec.get('redirect_urls', [])
    if not isinstance(redirects, list) or len(redirects) > 16:
        raise ValueError('Invalid nox redirect URL list')
    urls = [url, *redirects]
    for candidate in urls:
        try:
            parsed = urllib.parse.urlsplit(candidate)
            valid = (bool(parsed.hostname) and _origin_ok(candidate, {parsed.hostname})
                     and not parsed.fragment and bool(parsed.path) and len(candidate) <= 4096
                     and not any(character.isspace() or ord(character) < 32 for character in candidate))
        except (TypeError, ValueError, AttributeError):
            valid = False
        if not valid or candidate == model.get('upstream_url'):
            raise ValueError('nox download requires exact approved HTTPS ONNX release URLs')
    filename = model.get('filename', '')
    if (not isinstance(filename, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.+\-]*\.onnx', filename)
            or type(model.get('size')) is not int or model['size'] <= 0
            or not isinstance(model.get('sha256'), str) or not re.fullmatch(r'[a-f0-9]{64}', model['sha256'])):
        raise ValueError('Invalid pinned nox ONNX identity')
    return {'filename': filename, 'size': model['size'], 'sha256': model['sha256'],
            'url': url, 'allowed_urls': tuple(dict.fromkeys(urls))}


def nox_install_info(app_root=None) -> dict:
    """Read-only first-launch information; inspecting availability never downloads or installs anything."""
    data = _manifest(app_root)
    item = _nox_download_item(data)
    return {'available': item is not None, 'model_bytes': data['nox']['size'],
            'runtime_bytes': sum(part['size'] for part in data.get('optional_wheels', ())),
            'source': 'nox third-party MIT weights; Aester-converted ONNX',
            'training_data_reviewed': False,
            'message': ('主动同意后下载模型和运行环境；完成后配置路径，不会自动开启效果。' if item else
                        '此版本尚未提供 nox 模型下载地址；可稍后重试，或选择已有的已验转换 ONNX 模型。')}


def _download_nox_model(item, cache, progress=None):
    """Same resumable size/hash checks as other components, with an exact URL allowlist."""
    cache = _safe_root(cache)
    cache.mkdir(parents=True, exist_ok=True)
    target = _inside(cache, cache / item['filename'])
    partial = _inside(cache, cache / (item['filename'] + '.part'))
    _notify(progress, phase='model-start', filename=item['filename'], message='正在准备可选 nox 模型')
    if _valid_file(target, item):
        return target
    if _valid_file(partial, item):
        return _publish(partial, target)
    urls = item['allowed_urls']
    hosts = frozenset(urllib.parse.urlsplit(url).hostname for url in urls)
    opener = urllib.request.build_opener(_ExactHTTPSRedirect(urls))
    for attempt in range(3):
        _notify(progress, phase='model-download', filename=item['filename'], message='正在下载并校验 nox 模型')
        try:
            _fetch(opener, item['url'], item, partial, hosts, progress, allowed_urls=urls)
        except (OSError, ValueError, urllib.error.URLError, http.client.HTTPException) as error:
            if isinstance(error, urllib.error.HTTPError):
                error.close()
            if attempt == 2 or _failure_kind(error) == 'final':
                raise
            _notify(progress, phase='model-retry', message='模型下载未完成，正在重试')
            continue
        _notify(progress, phase='model-verified', message='nox 模型大小与 SHA256 已核验')
        return _publish(partial, target)


def install_nox_support(component_root=None, *, consent=False, model_source=None, progress=None,
                        cancelled=None, cache_dir=None, app_root=None) -> dict:
    """Prepare the optional model/runtime after an explicit user action, preserving old versions on failure.

    Production sources come only from the application's reviewed manifest. No environment or UI URL override
    is accepted. A local model must match that manifest, including its complete adapter configuration.
    """
    if consent is not True:
        raise ValueError('请主动同意后再准备可选 nox 组件')
    app = _safe_root(app_root or _app_root())
    lease = nullcontext()
    if app.parent.name == 'versions':
        from .install_lifecycle import runtime_lease
        lease = runtime_lease(app.parent.parent)
    with lease:
        return install_components(component_root, include_star_support=True, model_source=model_source,
                                  progress=progress, cancelled=cancelled, cache_dir=cache_dir, app_root=app,
                                  download_model=True)


def _archive_parts(name):
    parts = PurePosixPath(name).parts
    reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
    if not parts or name.startswith("/") or "\\" in name or ":" in name or any(p in {".", ".."} or p.endswith((".", " ")) or p.split(".")[0].upper() in reserved for p in parts):
        raise ValueError(f"Unsafe archive member: {name}")
    return parts


def _extract(archive: Path, destination: Path, *, wheel=False) -> None:
    _safe_root(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            parts = _archive_parts(member.filename)
            if stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError("Archive links are unsupported")
            base = destination / "Lib/site-packages" if wheel else destination
            if wheel and parts[0].endswith(".data"):
                if member.is_dir():
                    continue
                bases = {"purelib": base, "platlib": base, "scripts": destination / "Scripts", "headers": destination / "Include"}
                if len(parts) < 3 or parts[1] not in bases:
                    raise ValueError("Unsupported wheel data scheme")
                base, parts = bases[parts[1]], parts[2:]
            target = _inside(destination, base.joinpath(*parts))
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.open(member) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst, 1024 * 1024)


def component_catalog() -> dict:
    manifest = _manifest()
    return {"schema_version": 1, "components": [
        {"id": "base", "name": "Aester 基础运行环境 · Siril 1.4.4", "required": True,
         "download_bytes": sum(x["size"] for x in manifest["wheels"]) + manifest["siril"]["size"],
         "license": "Python PSF / scientific wheels upstream licenses / Siril GPL-3.0", "availability": "upstream-download"},
        {"id": "nox", "name": "nox 去星与缩星 · CPU", "required": False,
         "download_bytes": sum(x["size"] for x in manifest["optional_wheels"]), "model_bytes": manifest["nox"]["size"],
         "license": "nox trained weights MIT; project-converted ONNX; training data not reviewed",
         "availability": "consent-download" if _nox_download_item(manifest) else "existing-model-only"}
    ]}


def _read_receipt(root: Path):
    current = _inside(root, root / "current.json")
    if not current.exists():
        return None
    pointer = json.loads(current.read_text(encoding="utf-8"))
    name = pointer.get("version")
    if not isinstance(name, str) or not re.fullmatch(r"[a-zA-Z0-9-]+", name):
        raise ValueError("Invalid component version pointer")
    receipt_path = _inside(root, root / "versions" / name / "receipt.json")
    if _digest(receipt_path) != pointer.get("receipt_sha256"):
        raise ValueError("Component receipt changed")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    for key in ("python", "pythonw", "siril", "model_path"):
        if receipt.get(key):
            path = _inside(root, Path(receipt[key]))
            if not path.is_file():
                raise ValueError(f"Component is missing: {key}")
    return receipt_path, receipt


def component_status(component_root=None) -> dict:
    root = _safe_root(component_root or default_component_root())
    result = {"schema_version": 1, "component_root": str(root), "ready": False,
              "python": None, "pythonw": None, "siril": None, "model_path": None,
              "star_backend_options": {}, "receipt_path": None,
              "components": [{"id": "base", "status": "missing", "detail": "尚未安装基础环境"},
                             {"id": "nox", "status": "missing", "detail": "可选安装"}]}
    try:
        found = _read_receipt(root)
        if found:
            receipt_path, receipt = found
            result.update({key: receipt.get(key) for key in ("python", "pythonw", "siril", "model_path", "star_backend_options")})
            result.update(ready=True, receipt_path=str(receipt_path), version=receipt["version"])
            result["components"][0].update(status="ready", detail="安装时已离线校验；Siril 1.4.4")
            if receipt.get("model_path"):
                result["components"][1].update(status="ready", detail="nox 项目转换模型 / ONNX Runtime CPU")
    except (OSError, ValueError, KeyError, TypeError) as error:
        result["components"][0].update(status="broken", detail=str(error))
    return result


def _existing_version(root, final, app, data, version, manifest_sha):
    """Never execute paths taken unchecked from a writable receipt."""
    path = _inside(root, final / "receipt.json")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    expected = {"version": version, "manifest_sha256": manifest_sha, "app_root": str(app),
                "python": str(final / "runtime/python.exe"), "pythonw": str(final / "runtime/pythonw.exe"),
                "model_sha256": data["nox"]["sha256"] if version.endswith("-nox") else None,
                "star_backend_options": data["nox"]["adapter"] if version.endswith("-nox") else {}}
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("Existing component identity differs from the requested version")
    for key in ("python", "pythonw", "siril", "model_path"):
        if receipt.get(key):
            component = _inside(final, Path(receipt[key]))
            if not component.is_file():
                raise ComponentDamaged("Existing component is missing")
    siril = Path(receipt["siril"])
    if siril.name != "siril-cli.exe" or not siril.is_relative_to(final / "siril"):
        raise ValueError("Existing Siril path differs from its owned archive root")
    if version.endswith("-nox"):
        model = final / "models/nox_color.onnx"
        if receipt.get("model_path") != str(model):
            raise ValueError("Existing nox model identity changed")
        if not _valid_file(model, data["nox"]):
            raise ComponentDamaged("Existing nox model bytes changed")
    elif receipt.get("model_path") is not None:
        raise ValueError("Unexpected model in the base component version")
    return receipt


def _siril_probe_environment(folder, base=None, *, windows: bool | None = None) -> dict:
    """The environment of the ``siril-cli -v`` probe: a stdlib twin of starstack.siril.siril_environment's
    Windows keys (efficiency P1b W4; this module must run in the installer's bare Python, so it cannot import
    siril.py). A parity test pins XDG_DATA_HOME, MSYSTEM and DBUS_SESSION_BUS_ADDRESS to siril.siril_environment.
    ``-v`` exits during option parsing, before Siril looks at a data folder or a bus; this keeps it so for any
    Siril build."""
    env = dict(os.environ if base is None else base)
    if (os.name == "nt") if windows is None else windows:
        env.update(XDG_DATA_HOME=os.path.abspath(folder), MSYSTEM="AESTER", DBUS_SESSION_BUS_ADDRESS="disabled:")
    return env


def _validate(runtime: Path, siril: Path, model: Path | None, progress=None):
    _notify(progress, phase="validate", message="离线验证运行环境、Siril 和所选去星模型")
    code = "import numpy, astropy.io.fits, PIL.Image, tifffile, PySide6.QtWidgets, rawpy; from mcp.server.fastmcp import FastMCP"
    if model:
        # Execute a real full network tile, not only an import check.
        code += ("; import onnxruntime as ort; opts=ort.SessionOptions(); opts.intra_op_num_threads=2; opts.inter_op_num_threads=1"
                 f"; s=ort.InferenceSession({str(model)!r},sess_options=opts,providers=['CPUExecutionProvider'])"
                 "; a=numpy.zeros((1,512,512,3),dtype=numpy.float32); b=s.run(None,{s.get_inputs()[0].name:a})[0]"
                 "; assert b.shape==a.shape and numpy.isfinite(b).all()")
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    subprocess.run([str(runtime / "python.exe"), "-B", "-X", "utf8", "-c", code], check=True, capture_output=True, timeout=180,
                   stdin=subprocess.DEVNULL, creationflags=flags)
    with tempfile.TemporaryDirectory(prefix="aester-siril-probe-", ignore_cleanup_errors=True) as probe:
        result = subprocess.run([str(siril), "-v"], check=True, capture_output=True, timeout=30,
                                stdin=subprocess.DEVNULL, creationflags=flags, env=_siril_probe_environment(probe))
    if b"1.4.4" not in result.stdout + result.stderr:
        raise ValueError("Unexpected Siril version")


def install_components(component_root=None, *, include_star_support=False, progress=None,
                       cache_dir=None, app_root=None, model_source=None, manifest=None, activate=True,
                       download_source="auto", download_model=False, cancelled=None) -> dict:
    """Prepare a new private version; switch only after successful offline checks.

    app_root is an immutable installed code version, or the development checkout.
    manifest override is used by the extracted installer bootstrap, not GUI input.
    download_source is 'auto' (measure the official host and the listed mirrors, fastest first), 'official'
    or a DownloadSources shared by one run; the pinned size and SHA-256 decide either way.
    """
    if type(include_star_support) is not bool:
        raise ValueError("include_star_support must be bool")
    if type(download_model) is not bool:
        raise ValueError('download_model must be bool')
    raw_progress = progress
    def check_cancelled():
        if cancelled is not None and cancelled():
            raise RuntimeError('已取消组件准备；原有组件和路径保持不变。')
    def report(event):
        check_cancelled()
        if raw_progress is not None:
            raw_progress(event)
    progress = report
    _notify(progress, phase='start', message='正在检查所选组件')
    root = _safe_root(component_root or default_component_root())
    app = _safe_root(app_root or _app_root())
    data = manifest if manifest is not None else _manifest(app)
    sources = download_source if isinstance(download_source, DownloadSources) else DownloadSources(
        download_source, [data.get("python"), *data.get("wheels", ()),
                          *(data.get("optional_wheels", ()) if include_star_support else ()), data.get("siril")])
    model_input = None
    model_download = None
    if include_star_support:
        model_input = Path(model_source) if model_source else app / "packaging/aester_p2/models/nox_color.onnx"
        if model_source is None and not _valid_file(model_input, data["nox"]):
            # Reuse a previously verified identical local model during upgrades
            # or runtime repair; never infer a new backend from an arbitrary file.
            for candidate in root.glob("versions/*/models/nox_color.onnx"):
                _inside(root, candidate)
                if _valid_file(candidate, data["nox"]):
                    model_input = candidate
                    break
        # Reject a missing/changed model before downloading any large dependency.
        if not _valid_file(model_input, data["nox"]):
            if model_source is not None or not download_model:
                raise ValueError('nox 模型缺失或哈希不符，请选择已有的已验转换 ONNX 模型')
            model_download = _nox_download_item(data)
            if model_download is None:
                raise ValueError('此版本尚未提供 nox 模型下载地址；可稍后重试，或选择已有的已验转换 ONNX 模型。')
    manifest_sha = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    app_sha = hashlib.sha256(str(app).encode()).hexdigest()[:10]
    version = f"{manifest_sha[:16]}-{app_sha}-{'nox' if include_star_support else 'base'}"
    cache = _safe_root(cache_dir or root / "downloads")
    owner = ownership.identity_for_app(app)
    if owner is not None:
        ownership.register_install(owner['install_root'], root, cache)

    def download(item):
        if owner is None:
            return _download(item, cache, progress, sources)
        with ownership.download_lock(cache):
            existed = (cache / item['filename']).exists()
            path = _download(item, cache, progress, sources)
            ownership.claim_download(cache, item, app, existed_before=existed)
            return path

    def activate_current(final, version):
        value = {"version": version, "receipt_sha256": _digest(final / "receipt.json")}
        # Last cancellable boundary, AFTER every progress callback and receipt read. Once this check passes,
        # the atomic pointer commit is non-cancellable: a later stop request cannot turn committed success
        # into "cancelled". The caller must report the returned result, not infer cancellation from its flag.
        check_cancelled()
        _atomic(root / "current.json", value)
        return component_status(root)

    with _lock(root):
        if model_download is not None:
            if owner is None:
                model_input = _download_nox_model(model_download, cache, progress)
            else:
                with ownership.download_lock(cache):
                    existed = (cache / model_download['filename']).exists()
                    model_input = _download_nox_model(model_download, cache, progress)
                    ownership.claim_download(cache, model_download, app, existed_before=existed)
        # A prior successful repair remains reusable, while an older/different
        # app or manifest can never redirect this installation's identity.
        pointer = root / "current.json"
        if pointer.exists():
            try:
                selected = json.loads(pointer.read_text(encoding="utf-8")).get("version", "")
            except (ValueError, OSError, AttributeError):
                selected = ""  # Revalidate the canonical version before repairing the marker.
            expected_pattern = re.escape(f"{manifest_sha[:16]}-{app_sha}") + r"(?:-repair[a-f0-9]{8})?-" + ("nox" if include_star_support else "base")
            if isinstance(selected, str) and re.fullmatch(expected_pattern, selected):
                version = selected
        final = _inside(root, root / "versions" / version)
        if (final / "receipt.json").exists():
            try:
                receipt = _existing_version(root, final, app, data, version, manifest_sha)
                _validate(Path(receipt["python"]).parent, Path(receipt["siril"]), Path(receipt["model_path"]) if receipt.get("model_path") else None, progress)
            except (ComponentDamaged, subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
                version = f"{manifest_sha[:16]}-{app_sha}-repair{uuid.uuid4().hex[:8]}-{'nox' if include_star_support else 'base'}"
                final = _inside(root, root / "versions" / version)
                _notify(progress, phase="repair", message="保留损坏版本，准备独立修复副本")
            else:
                _notify(progress, phase='complete', message='组件已验证，此后可离线启动')
                ownership.claim_version(root, version, app)
                if activate:
                    return activate_current(final, version)
                return {**receipt, "receipt_path": str(final / "receipt.json"), "receipt_sha256": _digest(final / "receipt.json"), "prepared": True}
        elif final.exists():
            version = f"{manifest_sha[:16]}-{app_sha}-repair{uuid.uuid4().hex[:8]}-{'nox' if include_star_support else 'base'}"
            final = _inside(root, root / "versions" / version)
        stage = _inside(root, root / "staging" / uuid.uuid4().hex)
        stage.mkdir(parents=True)
        try:
            runtime = stage / "runtime"
            _extract(download(data["python"]), runtime)
            wheels = data["wheels"] + (data["optional_wheels"] if include_star_support else [])
            for index, item in enumerate(wheels, 1):
                _notify(progress, phase="prepare", component="base", filename=item["filename"], index=index, total=len(wheels), message="安装私有运行库")
                _extract(download(item), runtime, wheel=True)
            pth_files = list(runtime.glob("python*._pth"))
            if len(pth_files) != 1:
                raise ValueError("Ambiguous embedded Python path configuration")
            pth_files[0].write_text("python313.zip\n.\nLib/site-packages\n" + str(app / "src") + "\nimport site\n", encoding="utf-8")
            _extract(download(data["siril"]), stage / "siril")
            engines = list((stage / "siril").rglob("siril-cli.exe"))
            if len(engines) != 1:
                raise ValueError("Ambiguous Siril portable executable")
            model = None
            if include_star_support:
                if not _valid_file(model_input, data["nox"]):
                    raise ValueError("nox 模型缺失或哈希不符，请重新准备已验转换的 ONNX 模型")
                model = stage / "models/nox_color.onnx"
                model.parent.mkdir()
                shutil.copyfile(model_input, model)
                licence = Path(__file__).with_name("nox-LICENSE.txt")
                if not licence.exists():
                    licence = _app_root() / "packaging/aester_p2/nox-LICENSE.txt"
                if not licence.exists():
                    raise ValueError("nox license is missing from component package")
                shutil.copyfile(licence, model.parent / "LICENSE.txt")
                for filename in ("nox-author-license-readme.md", "nox-conversion-evidence.json"):
                    evidence = Path(__file__).with_name(filename)
                    if not evidence.exists():
                        evidence = _app_root() / "packaging/aester_p2" / filename
                    if evidence.exists():
                        shutil.copyfile(evidence, model.parent / filename)
            _validate(runtime, engines[0], model, progress)
            _notify(progress, phase='validated', message='已完成离线校验，正在保存组件版本')
            final.parent.mkdir(parents=True, exist_ok=True)
            if final.exists():
                raise ValueError("Incomplete component version already exists; previous installation is preserved")
            receipt = {"schema_version": 1, "version": version, "manifest_sha256": manifest_sha,
                       "app_root": str(app), "python": str(final / "runtime/python.exe"),
                       "pythonw": str(final / "runtime/pythonw.exe"), "siril": str(final / engines[0].relative_to(stage)),
                       "model_path": str(final / "models/nox_color.onnx") if model else None,
                       "model_sha256": data["nox"]["sha256"] if model else None,
                       "star_backend_options": data["nox"]["adapter"] if model else {},
                       "network_required_at_launch": False}
            (stage / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
            ownership.seal_version(stage, version, app)
            stage.rename(final)
            ownership.claim_version(root, version, app)
            _notify(progress, phase="complete", message="安装完成，此后可离线启动")
            if activate:
                return activate_current(final, version)
            return {**receipt, "receipt_path": str(final / "receipt.json"), "receipt_sha256": _digest(final / "receipt.json"), "prepared": True}
        except Exception as error:
            # Keep the failed stage and its evidence. Never disturb a working version.
            if stage.exists():
                _atomic(stage / "failure.json", {"error": str(error), "type": type(error).__name__})
            raise


def register_model(path, component_root=None) -> dict:
    """Install the known converted model and pinned runtime into a new version."""
    return install_components(component_root, include_star_support=True, model_source=path)


def _retouch_distribution():
    try:
        from . import retouch_distribution
    except ImportError:
        import retouch_distribution
    return retouch_distribution


def install_retouch_components(component_root=None, *, include_gimp=True, include_skill=True,
                              progress=None, app_root=None, cache_dir=None, gimp_source=None,
                              selected_gimp_path=None, skill_source=None, download_source="auto") -> dict:
    """Prepare optional retouch support without reinstalling the base runtime.

    GIMP is a separate official install and is retained by Aester uninstall.
    gimp_source is an optional exact official installer EXE, never a console
    executable. Existing selections use selected_gimp_path instead. The skill
    is only copied inside Aester's component area; no global skill is installed.
    download_source chooses where the pinned GIMP installer comes from, as in install_components.
    """
    return _retouch_distribution().install(component_root, include_gimp=include_gimp,
        include_skill=include_skill, progress=progress, app_root=app_root,
        cache_dir=cache_dir, gimp_source=gimp_source, selected_gimp_path=selected_gimp_path,
        skill_source=skill_source, download_source=download_source)


def retouch_component_status(component_root=None, *, app_root=None, selected_gimp_path=None) -> dict:
    """Read-only GIMP discovery and local skill availability; never run GIMP."""
    module = _retouch_distribution()
    root = _safe_root(component_root or default_component_root())
    app = _safe_root(app_root or _app_root())
    gimp = module.discover(root, app_root=app, selected_gimp_path=selected_gimp_path, probe=False)
    skill = module.skill_status(root, app_root=app)
    return {'schema_version': 1, 'gimp': gimp, 'skill': skill,
            'components': [
                {'id': 'gimp', 'status': gimp['status'], 'required': False,
                 'detail': 'GIMP 3.2.6 · 自然语言修图所需；独立安装，卸载 Aester 时保留',
                 'download_bytes': module.GIMP_INSTALLER['size']},
                {'id': 'retouch_skill', 'status': skill['status'], 'required': False,
                 'detail': '自然语言修图技能包 · 仅应用内提供；由 Agent 加载或用户另行安装'}]}


def verify_prepared_retouch_components(component_root=None, *, app_root=None, expected) -> dict:
    return _retouch_distribution().verify_prepared(
        component_root or default_component_root(), app_root=app_root or _app_root(), expected=expected)


def install_finite_backend(component_root=None, *, source=None, app_root=None) -> dict:
    """Explicitly install a verified local stack runtime as an isolated add-on.

    Never changes the base runtime pointer, active algorithm, or GUI preference.
    This local-copy path is not a claim that a redistributable package exists.
    """
    from .finite_backend_assets import (validate_runtime, discover_finite_backend, addon_prefix, RUNTIME_VERSION,
                                        runtime_members)
    app = _safe_root(app_root or _app_root())
    root = _safe_root(component_root or default_component_root())
    if source is None:
        available = discover_finite_backend(app_root=app, component_root=root)
        if available['available'] and available['source'] == 'installed-addon':
            return available
        source = available.get('install_source')
    if not source:
        raise ValueError('Select the folder containing the verified Aester stack runtime')
    source = _safe_root(source)
    if source.name == 'siril-cli.exe' and source.is_file():
        source = source.parent
    checked = validate_runtime(source / 'siril-cli.exe')
    members = runtime_members(source)   # with runtime_aux_manifest.json and its helpers when present (P1b W7)
    owner = ownership.identity_for_app(app)
    if owner is not None:
        ownership.register_install(owner['install_root'], root, root / 'downloads')
    with _lock(root):
        prefix = addon_prefix(app)
        available = discover_finite_backend(app_root=app, environ={}, component_root=root)
        if available['available'] and available['source'] == 'installed-addon':
            return available
        # A damaged earlier version stays intact for inspection; no in-place
        # edits can disturb a currently running job or overwrite user files.
        version = prefix + '-' + uuid.uuid4().hex[:8]
        final = _inside(root, root / 'versions' / version)
        stage = _inside(root, root / 'staging' / uuid.uuid4().hex)
        runtime = stage / 'runtime'
        runtime.mkdir(parents=True)
        try:
            for name in members:
                shutil.copyfile(source / name, runtime / name)
            validate_runtime(runtime / 'siril-cli.exe')
            _atomic(stage / 'finite-receipt.json', {'schema_version': 1, 'version': RUNTIME_VERSION,
                    'app_root': str(app), 'engine_sha256': checked['engine_sha256'],
                    'runtime_manifest_sha256': checked['runtime_manifest_sha256'],
                    'source_kind': 'verified-local-copy', 'redistribution_verified': False})
            ownership.seal_version(stage, version, app)
            final.parent.mkdir(parents=True, exist_ok=True)
            stage.rename(final)
            ownership.claim_version(root, version, app)
            return discover_finite_backend(app_root=app, environ={}, component_root=root)
        except Exception as error:
            if stage.exists():
                _atomic(stage / 'failure.json', {'error': str(error), 'type': type(error).__name__})
            raise


def verify_installed_finite_distribution(component_root=None, *, app_root=None, expected) -> dict:
    """Verify the prepared, application-bound optional component before activation."""
    try:
        from . import finite_distribution as distribution
        from .finite_backend_assets import addon_prefix, RUNTIME_VERSION, ENGINE_SHA256
    except ImportError:
        import finite_distribution as distribution
        from finite_backend_assets import addon_prefix, RUNTIME_VERSION, ENGINE_SHA256
    app = _safe_root(app_root or _app_root())
    root = _safe_root(component_root or default_component_root())
    if not isinstance(expected, dict):
        raise ValueError('Invalid prepared Aester stack component')
    receipt_path = _inside(root, Path(expected.get('receipt_path', '')))
    folder = receipt_path.parent
    if (receipt_path.name != 'finite-receipt.json' or folder.parent != root / 'versions'
            or not folder.name.startswith(addon_prefix(app) + '-')
            or _digest(receipt_path) != expected.get('receipt_sha256')):
        raise ValueError('Prepared Aester stack receipt changed or belongs to another application')
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    if (receipt.get('app_root') != str(app) or receipt.get('version') != RUNTIME_VERSION
            or receipt.get('engine_sha256') != ENGINE_SHA256
            or receipt.get('distribution_sha256') != distribution.DISTRIBUTION_MANIFEST_SHA256
            or expected.get('distribution_sha256') != distribution.DISTRIBUTION_MANIFEST_SHA256
            or receipt.get('source_kind') != 'reviewed-distribution'):
        raise ValueError('The prepared Aester stack component identity is invalid')
    checked = distribution.validate_distribution(folder, installed=True)
    path = str(folder / 'runtime/siril-cli.exe')
    if expected.get('path') != path:
        raise ValueError('The prepared Aester stack executable path changed')
    return {'available': True, **checked['runtime'], 'source': 'installed-addon',
            'distribution_sha256': checked['distribution_sha256'],
            'receipt_path': str(receipt_path), 'receipt_sha256': _digest(receipt_path),
            'installable': False, 'install_source': None, 'reason': ''}


def install_finite_distribution(component_root=None, *, source, app_root=None) -> dict:
    """Install a reviewed runtime with its complete compressed source and notices.

    Preparation is immutable and does not switch the base pointer or preferences.
    A failed/corrupt older copy remains recorded for conservative uninstall.
    """
    try:
        from . import finite_distribution as distribution
        from .finite_backend_assets import addon_prefix, RUNTIME_VERSION
    except ImportError:
        import finite_distribution as distribution
        from finite_backend_assets import addon_prefix, RUNTIME_VERSION
    app = _safe_root(app_root or _app_root())
    root = _safe_root(component_root or default_component_root())
    source = _safe_root(source)
    checked = distribution.validate_distribution(source)
    owner = ownership.identity_for_app(app)
    if owner is not None:
        ownership.register_install(owner['install_root'], root, root / 'downloads')
    with _lock(root):
        prefix = addon_prefix(app)
        for path in sorted((root / 'versions').glob(prefix + '*/finite-receipt.json'), reverse=True)[:100]:
            try:
                expected = {'receipt_path': str(path), 'receipt_sha256': _digest(path),
                            'path': str(path.parent / 'runtime/siril-cli.exe'),
                            'distribution_sha256': checked['distribution_sha256']}
                result = verify_installed_finite_distribution(root, app_root=app, expected=expected)
            except (OSError, ValueError, TypeError, KeyError):
                continue
            ownership.claim_version(root, path.parent.name, app)
            return result
        version = prefix + '-' + uuid.uuid4().hex[:8]
        final = _inside(root, root / 'versions' / version)
        stage = _inside(root, root / 'staging' / uuid.uuid4().hex)
        stage.mkdir(parents=True)
        try:
            for relative in (*checked['files'], distribution.MANIFEST):
                target = _inside(stage, stage.joinpath(*PurePosixPath(relative).parts))
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source.joinpath(*PurePosixPath(relative).parts), target)
            distribution.validate_distribution(stage)
            _atomic(stage / 'finite-receipt.json', {
                'schema_version': 1, 'version': RUNTIME_VERSION, 'app_root': str(app),
                'engine_sha256': checked['runtime']['engine_sha256'],
                'runtime_manifest_sha256': checked['runtime']['runtime_manifest_sha256'],
                'source_kind': 'reviewed-distribution', 'redistribution_verified': True,
                'distribution_sha256': checked['distribution_sha256']})
            ownership.seal_version(stage, version, app)
            final.parent.mkdir(parents=True, exist_ok=True)
            stage.rename(final)
            ownership.claim_version(root, version, app)
            expected = {'receipt_path': str(final / 'finite-receipt.json'),
                        'receipt_sha256': _digest(final / 'finite-receipt.json'),
                        'path': str(final / 'runtime/siril-cli.exe'),
                        'distribution_sha256': checked['distribution_sha256']}
            return verify_installed_finite_distribution(root, app_root=app, expected=expected)
        except Exception as error:
            if stage.exists():
                _atomic(stage / 'failure.json', {'error': str(error), 'type': type(error).__name__})
            raise
