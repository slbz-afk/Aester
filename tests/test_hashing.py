"""starstack.hashing: run ledger, parallel batches, hash-while-copy, digest-while-write, handover.

docs/EFFICIENCY_P1_HASHING_PREREG_20260929.md, acceptance H1.  Files are
given modification times well in the past unless a test is about the racy
rule, so ledger entries are not racy.
"""
from pathlib import Path
import gc
import hashlib
import io
import json
import os
import random
import shutil
import threading
import time

import numpy as np
import pytest

from starstack import hashing, profiling

WINDOWS = os.name == "nt"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make(path, data, age=60.0):
    """Write ``data`` and set the modification time ``age`` seconds in the past."""
    path = Path(path)
    path.write_bytes(data)
    if age:
        stamp = time.time_ns() - int(age * 1e9)
        os.utime(path, ns=(stamp, stamp))
    return path


def age_like(path, other):
    st = os.stat(other)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))


def delta(before, after):
    out = {"real": after["real"]["bytes"] - before["real"]["bytes"],
           "real_files": after["real"]["files"] - before["real"]["files"]}
    for kind in hashing.SAVED_KINDS:
        out[kind] = after["saved"][kind]["files"] - before["saved"][kind]["files"]
    out["racy"] = after["misses"]["racy"] - before["misses"]["racy"]
    out["shadow"] = after["shadow"]["checks"] - before["shadow"]["checks"]
    return out


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv(hashing.ENV_LEDGER, raising=False)
    monkeypatch.delenv(hashing.ENV_WORKERS, raising=False)
    assert not hashing.snapshot()["active"]
    yield
    assert not hashing.snapshot()["active"]


@pytest.fixture
def ntfs(tmp_path):
    probe = make(tmp_path / "probe.bin", b"probe")
    if not hashing._volume_eligible(hashing._norm(probe), os.stat(probe)):
        pytest.skip("the ledger needs a local NTFS/ReFS volume")
    return tmp_path


@pytest.fixture
def parallel(monkeypatch):
    """Use the thread pool even for batches of small test files."""
    monkeypatch.setattr(hashing, "PARALLEL_MIN_BYTES", 0)


class Cancelled(Exception):
    pass


# --------------------------------------------------------------------------- #
# Single file and ledger                                                       #
# --------------------------------------------------------------------------- #

def test_digest_equals_file_digest_and_errors_match_open(tmp_path):
    data = os.urandom(3 * hashing.CHUNK // 2)
    path = make(tmp_path / "a.bin", data)
    assert hashing.sha256_file(path, source="t") == hashlib.sha256(data).hexdigest()
    assert hashing.sha256_file(str(path), source="t", check=lambda: None) == hashlib.sha256(data).hexdigest()
    for missing in (tmp_path / "missing.bin", str(tmp_path / "missing.bin")):
        with pytest.raises(FileNotFoundError) as expected:
            open(missing, "rb")
        for mode in (None, "on", "shadow"):
            if mode is None:
                with pytest.raises(FileNotFoundError) as got:
                    hashing.sha256_file(missing, source="t")
            else:
                with hashing.run_scope("t", mode=mode), pytest.raises(FileNotFoundError) as got:
                    hashing.sha256_file(missing, source="t")
            assert str(got.value) == str(expected.value)
    with pytest.raises(ValueError):
        hashing.sha256_file(path, source="t", trust="maybe")


def test_hits_only_inside_a_scope_and_the_outermost_scope_clears(ntfs):
    path = make(ntfs / "a.bin", os.urandom(10_000))
    before = hashing.snapshot()
    hashing.sha256_file(path, source="t")
    hashing.sha256_file(path, source="t")
    assert delta(before, hashing.snapshot())["ledger"] == 0          # no scope: always read
    with hashing.run_scope("outer", mode="on"):
        assert hashing.snapshot()["label"] == "outer"
        first = hashing.sha256_file(path, source="t")
        with hashing.run_scope("inner", mode="off"):                  # nested: shares, mode ignored
            assert hashing.snapshot()["mode"] == "on" and hashing.snapshot()["label"] == "outer"
            assert hashing.sha256_file(path, source="t") == first
        assert hashing.snapshot()["entries"] >= 1                      # inner exit keeps the ledger
        assert hashing.sha256_file(path, source="t") == first
    snap = hashing.snapshot()
    assert not snap["active"] and snap["entries"] == 0 and snap["mode"] == "off"
    counts = delta(before, snap)
    assert counts["ledger"] == 2 and counts["real_files"] == 3
    with hashing.run_scope("again", mode="on"):                        # a new scope starts empty
        before = hashing.snapshot()
        hashing.sha256_file(path, source="t")
        assert delta(before, hashing.snapshot())["real_files"] == 1


@pytest.mark.parametrize("change", ["size", "mtime", "replace", "replace_same_size_and_mtime"])
def test_ledger_misses_after_size_mtime_or_file_id_change(ntfs, change):
    path = make(ntfs / "a.bin", b"A" * 5000)
    with hashing.run_scope("t", mode="on"):
        assert hashing.sha256_file(path, source="t") == sha(path)
        old = os.stat(path)
        if change == "size":
            with open(path, "ab") as handle:
                handle.write(b"B")
        elif change == "mtime":                     # same size, new content, new time
            with open(path, "r+b") as handle:
                handle.write(b"B")
            os.utime(path, ns=(old.st_atime_ns, old.st_mtime_ns + 1_000_000))
        else:
            other = make(ntfs / "b.tmp", b"C" * 5000)
            if change == "replace_same_size_and_mtime":
                os.utime(other, ns=(old.st_atime_ns, old.st_mtime_ns))
            os.replace(other, path)
            new = os.stat(path)
            if change == "replace_same_size_and_mtime":
                assert (new.st_size, new.st_mtime_ns) == (old.st_size, old.st_mtime_ns)
                assert new.st_ino != old.st_ino
        before = hashing.snapshot()
        assert hashing.sha256_file(path, source="t") == sha(path)
        counts = delta(before, hashing.snapshot())
        assert counts["ledger"] == 0 and counts["real_files"] == 1


def test_copy_that_preserves_size_and_mtime_has_a_new_file_id(ntfs):
    """A copy (CopyFile2 keeps the source time) replacing the file: same size and mtime, other file id."""
    path = make(ntfs / "a.bin", b"A" * 4096)
    other = make(ntfs / "other.bin", b"Z" * 4096)
    age_like(other, path)
    with hashing.run_scope("t", mode="on"):
        hashing.sha256_file(path, source="t")
        copy = ntfs / "copy.tmp"
        shutil.copy2(other, copy)
        assert os.stat(copy).st_mtime_ns == os.stat(path).st_mtime_ns
        os.replace(copy, path)
        assert hashing.sha256_file(path, source="t") == sha(other)


def test_copy2_onto_an_existing_file_needs_writing(ntfs):
    """CopyFile2 onto an existing file keeps its id and copies the time: only writing() makes it safe."""
    path = make(ntfs / "a.bin", b"A" * 4096)
    other = make(ntfs / "other.bin", b"Z" * 4096)
    age_like(other, path)
    with hashing.run_scope("t", mode="shadow"):
        hashing.sha256_file(path, source="t")
        key = hashing._sig(os.stat(path))
        shutil.copy2(other, path)
        if hashing._sig(os.stat(path)) != key:
            pytest.skip("this platform gave the copy a new identity")
        with pytest.raises(hashing.HashLedgerMismatch) as caught:
            hashing.sha256_file(path, source="t.copy2")
        assert caught.value.source == "t.copy2" and caught.value.real_digest == sha(other)
        assert str(path).lower() in str(caught.value).lower()
    with hashing.run_scope("t", mode="on"):
        make(path, b"A" * 4096)
        age_like(path, other)
        hashing.sha256_file(path, source="t")
        with hashing.writing(path):
            shutil.copy2(other, path)
        assert hashing.sha256_file(path, source="t") == sha(other)


def test_memmap_write_needs_invalidate_and_shadow_catches_a_missing_one(ntfs):
    path = ntfs / "frame.npy"
    np.save(path, np.zeros((256, 256), dtype=np.float32))
    stamp = time.time_ns() - 60 * 10**9
    os.utime(path, ns=(stamp, stamp))

    def poke(value):
        array = np.load(path, mmap_mode="r+")
        array[:4, :4] = value
        array.flush()
        del array
        gc.collect()

    # Without invalidation the stat does not change: "on" answers the old digest ...
    with hashing.run_scope("t", mode="on"):
        stale = hashing.sha256_file(path, source="t")
        key = hashing._sig(os.stat(path))
        poke(1.0)
        if hashing._sig(os.stat(path)) != key:
            pytest.skip("memory-mapped writes changed the stat here")
        assert hashing.sha256_file(path, source="t") == stale != sha(path)
    # ... shadow mode reports it ...
    with hashing.run_scope("t", mode="shadow"):
        hashing.sha256_file(path, source="t")
        poke(2.0)
        before = hashing.snapshot()
        with pytest.raises(hashing.HashLedgerMismatch):
            hashing.sha256_file(path, source="t.memmap")
        assert hashing.snapshot()["shadow"]["mismatches"] == before["shadow"]["mismatches"] + 1
        assert hashing.sha256_file(path, source="t") == sha(path)          # the bad entry was dropped
    # ... and writing() / invalidate() make it right.
    with hashing.run_scope("t", mode="shadow"):
        hashing.sha256_file(path, source="t")
        with hashing.writing(path):
            poke(3.0)
        assert hashing.sha256_file(path, source="t") == sha(path)
        poke(4.0)
        hashing.invalidate(path)
        assert hashing.sha256_file(path, source="t") == sha(path)


def test_invalidate_reaches_hard_links_and_writing_blocks_records(ntfs):
    path = make(ntfs / "a.bin", b"A" * 4096)
    link = ntfs / "link.bin"
    os.link(path, link)
    with hashing.run_scope("t", mode="on"):
        hashing.sha256_file(path, source="t")
        hashing.sha256_file(link, source="t")
        assert hashing.snapshot()["entries"] == 2
        with hashing.writing(path):
            assert hashing.snapshot()["entries"] == 0
            hashing.sha256_file(link, source="t")                  # same file id: busy, not recorded
            assert hashing.snapshot()["entries"] == 0
        before = hashing.snapshot()
        hashing.sha256_file(link, source="t")
        assert delta(before, hashing.snapshot())["real_files"] == 1


def test_racy_entries_are_rehashed_and_then_upgraded(ntfs, monkeypatch):
    monkeypatch.setattr(hashing, "RACY_SECONDS", 0.3)
    path = ntfs / "fresh.bin"
    path.write_bytes(os.urandom(8192))                   # modified just now
    with hashing.run_scope("t", mode="on"):
        hashing.sha256_file(path, source="t")
        before = hashing.snapshot()
        assert hashing.sha256_file(path, source="t") == sha(path)
        counts = delta(before, hashing.snapshot())
        assert counts["racy"] == 1 and counts["real_files"] == 1 and counts["ledger"] == 0
        time.sleep(0.45)
        hashing.sha256_file(path, source="t")                # re-hash, now not racy: upgraded
        before = hashing.snapshot()
        assert hashing.sha256_file(path, source="t") == sha(path)
        counts = delta(before, hashing.snapshot())
        assert counts["ledger"] == 1 and counts["real_files"] == 0


def test_trust_real_always_reads_but_remembers(ntfs):
    path = make(ntfs / "a.bin", os.urandom(4096))
    with hashing.run_scope("t", mode="on"):
        before = hashing.snapshot()
        hashing.sha256_file(path, source="t", trust="real")
        hashing.sha256_file(path, source="t", trust="real")
        hashing.sha256_file(path, source="t")
        counts = delta(before, hashing.snapshot())
        assert counts["real_files"] == 2 and counts["ledger"] == 1


def test_check_runs_per_chunk_and_once_on_a_hit(ntfs):
    path = make(ntfs / "a.bin", os.urandom(2 * hashing.CHUNK + 5))
    calls = []
    with hashing.run_scope("t", mode="on"):
        hashing.sha256_file(path, source="t", check=lambda: calls.append(1))
        assert len(calls) == 4                             # 3 chunks + the empty read
        hashing.sha256_file(path, source="t", check=lambda: calls.append(1))
        assert len(calls) == 5

        def cancel():
            raise Cancelled()
        with pytest.raises(Cancelled):
            hashing.sha256_file(path, source="t", check=cancel)
    with pytest.raises(Cancelled):
        hashing.sha256_file(path, source="t", check=cancel)


def test_environment_modes(ntfs, monkeypatch):
    path = make(ntfs / "a.bin", os.urandom(4096))
    for value, mode in (("off", "off"), ("shadow", "shadow"), ("ON", "on"), ("", "on"), ("bogus", "off")):
        monkeypatch.setenv(hashing.ENV_LEDGER, value)
        with hashing.run_scope("t"):
            assert hashing.snapshot()["mode"] == mode
            hashing.sha256_file(path, source="t")
            before = hashing.snapshot()
            assert hashing.sha256_file(path, source="t") == sha(path)
            counts = delta(before, hashing.snapshot())
        if mode == "off":
            assert counts["ledger"] == 0 and counts["real_files"] == 1 and hashing.snapshot()["entries"] == 0
        elif mode == "shadow":
            assert counts["ledger"] == 1 and counts["shadow"] == 1 and counts["real_files"] == 1
        else:
            assert counts["ledger"] == 1 and counts["real_files"] == 0
    with pytest.raises(ValueError):
        with hashing.run_scope("t", mode="sometimes"):
            pass


def test_volume_without_ledger_support_never_records(ntfs, monkeypatch):
    path = make(ntfs / "a.bin", os.urandom(4096))
    monkeypatch.setattr(hashing, "_volume_eligible", lambda key, st: False)
    with hashing.run_scope("t", mode="on"):
        before = hashing.snapshot()
        hashing.sha256_file(path, source="t")
        hashing.sha256_file(path, source="t")
        assert delta(before, hashing.snapshot())["real_files"] == 2 and hashing.snapshot()["entries"] == 0


def test_zero_file_id_and_unknown_volumes_are_not_eligible(ntfs, monkeypatch):
    if WINDOWS:
        volumes = hashing.snapshot()["volumes"]
        assert any(value["fs"].upper() in ("NTFS", "REFS") and value["eligible"] for value in volumes.values())

    class Stat:
        st_dev, st_ino = 12345, 0
    assert hashing._volume_eligible(str(ntfs), Stat()) is False
    monkeypatch.setattr(hashing, "_volume_by_root", {})
    monkeypatch.setattr(hashing, "_volume_by_dev", {})
    monkeypatch.setattr(hashing, "_probe_root", lambda path: ("X:\\", "exFAT", 3))
    real = os.stat(ntfs / "probe.bin")
    assert hashing._volume_eligible(str(ntfs), real) is False
    monkeypatch.setattr(hashing, "_volume_by_dev", {})
    monkeypatch.setattr(hashing, "_probe_root", lambda path: ("\\\\server\\share\\", "NTFS", 4))
    assert hashing._volume_eligible(str(ntfs), real) is False


def test_carried_threads_share_one_ledger(ntfs):
    files = [make(ntfs / f"f{i}.bin", os.urandom(20_000 + i)) for i in range(8)]
    errors = []
    with hashing.run_scope("t", mode="shadow"):
        def work():
            try:
                assert hashing.snapshot()["label"] == "t"
                for _ in range(5):
                    for path in files:
                        assert hashing.sha256_file(path, source="t") == sha(path)
            except BaseException as exc:  # pragma: no cover - reported below
                errors.append(exc)
        before = hashing.snapshot()
        threads = [threading.Thread(target=hashing.carry_scope(work)) for _ in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert delta(before, hashing.snapshot())["shadow"] >= 6 * 5 * len(files) - 6 * len(files)
    assert not errors


def _in_thread(fn):
    out = {}

    def run():
        try:
            out["value"] = fn()
        except BaseException as exc:  # pragma: no cover - re-raised below
            out["error"] = exc
    thread = threading.Thread(target=run)
    thread.start()
    thread.join()
    if "error" in out:
        raise out["error"]
    return out["value"]


def test_a_scope_belongs_to_the_context_that_opened_it(ntfs):
    """Threads outside the scope read every time and record nothing; each outermost scope has its own ledger."""
    path = make(ntfs / "a.bin", os.urandom(50_000))
    with hashing.run_scope("A", mode="on"):
        hashing.sha256_file(path, source="t")
        assert hashing.snapshot()["entries"] == 1

        def outside():
            snap = hashing.snapshot()
            assert not snap["active"] and snap["mode"] == "off" and hashing.active_mode() == "off"
            before = hashing.snapshot()
            for _ in range(3):
                assert hashing.sha256_file(path, source="t") == sha(path)
            counts = delta(before, hashing.snapshot())
            assert counts["real_files"] == 3 and counts["ledger"] == 0
            return True
        assert _in_thread(outside)
        # A thread of its own opens its own scope: it starts empty, and the invalidation of
        # a writer anywhere reaches both ledgers.
        barrier = threading.Barrier(2)

        def other_scope():
            with hashing.run_scope("B", mode="on"):
                assert hashing.snapshot()["label"] == "B" and hashing.snapshot()["entries"] == 0
                before = hashing.snapshot()
                hashing.sha256_file(path, source="t")
                assert delta(before, hashing.snapshot())["real_files"] == 1
                barrier.wait()          # main thread invalidates now
                barrier.wait()
                return hashing.snapshot()["entries"]
        result = {}
        thread = threading.Thread(target=lambda: result.update(entries=other_scope()))
        thread.start()
        barrier.wait()
        assert hashing.snapshot()["open_scopes"] >= 2
        hashing.invalidate(path)
        assert hashing.snapshot()["entries"] == 0
        barrier.wait()
        thread.join()
        assert result["entries"] == 0
        # carry_scope hands this scope to another thread.
        hashing.sha256_file(path, source="t")
        before = hashing.snapshot()
        assert _in_thread(hashing.carry_scope(lambda: hashing.sha256_file(path, source="t"))) == sha(path)
        assert delta(before, hashing.snapshot())["ledger"] == 1
    # After the scope ends a carried callable runs without a ledger.
    with hashing.run_scope("C", mode="on"):
        carried = hashing.carry_scope(lambda: hashing.active_mode())
    assert _in_thread(carried) == "off"


def _poke_memmap(path, value, rows=1, tail=False):
    array = np.memmap(path, dtype=np.uint8, mode="r+")
    if tail:
        array[-rows:] = value
    else:
        array[:rows] = value
    array.flush()
    del array
    gc.collect()


def test_a_writer_inside_a_read_keeps_the_digest_out_of_the_ledger(ntfs):
    """A writing() block that begins and ends while a hash is being read: the digest is not recorded."""
    path = make(ntfs / "frame.bin", os.urandom(3 * hashing.CHUNK))
    for mode in ("on", "shadow"):
        with hashing.run_scope("t", mode=mode):
            calls = [0]

            def check():
                calls[0] += 1
                if calls[0] == 2:          # the first chunk has been read already
                    with hashing.writing(path):
                        _poke_memmap(path, calls[0] + (mode == "shadow"))
            key = hashing._sig(os.stat(path))
            torn = hashing.sha256_file(path, source="t", check=check)
            if hashing._sig(os.stat(path)) != key:
                pytest.skip("memory-mapped writes changed the stat here")
            assert torn != sha(path)
            refused = hashing.snapshot()["refused"]["touched"]
            assert hashing.snapshot()["entries"] == 0
            assert hashing.sha256_file(path, source="t") == sha(path)
            assert hashing.snapshot()["refused"]["touched"] == refused


def test_a_writer_inside_another_threads_read(ntfs):
    """The e1 race: thread A reads, thread B's whole writing() block runs meanwhile."""
    path = make(ntfs / "frame.bin", os.urandom(3 * hashing.CHUNK))
    started, written = threading.Event(), threading.Event()
    with hashing.run_scope("t", mode="on"):
        def check():
            started.set()
            assert written.wait(10)
        reader = {}
        thread = threading.Thread(target=hashing.carry_scope(
            lambda: reader.update(digest=hashing.sha256_file(path, source="t", check=check))))
        thread.start()
        assert started.wait(10)
        stat = hashing._sig(os.stat(path))
        with hashing.writing(path):
            _poke_memmap(path, 7, rows=hashing.CHUNK + 1)
        written.set()
        thread.join()
        if hashing._sig(os.stat(path)) != stat:
            pytest.skip("memory-mapped writes changed the stat here")
        before = hashing.snapshot()
        assert hashing.sha256_file(path, source="t") == sha(path)
        assert delta(before, hashing.snapshot())["real_files"] == 1


def test_shadow_does_not_blame_the_ledger_for_a_concurrent_writer(ntfs):
    path = make(ntfs / "frame.bin", os.urandom(2 * hashing.CHUNK))
    with hashing.run_scope("t", mode="shadow"):
        hashing.sha256_file(path, source="t")
        calls = [0]

        def check():
            calls[0] += 1
            if calls[0] == 2:
                with hashing.writing(path):
                    _poke_memmap(path, 9, rows=16, tail=True)      # bytes not read yet
        before = hashing.snapshot()
        digest = hashing.sha256_file(path, source="t", check=check)      # hit, verified by a real read
        after = hashing.snapshot()
        assert after["shadow"]["concurrent"] == before["shadow"]["concurrent"] + 1
        assert after["shadow"]["mismatches"] == before["shadow"]["mismatches"]
        assert digest == sha(path)
        assert hashing.sha256_file(path, source="t") == sha(path)


# --------------------------------------------------------------------------- #
# Batches                                                                      #
# --------------------------------------------------------------------------- #

def test_small_batches_stay_serial(ntfs):
    files = [make(ntfs / f"f{i}.bin", os.urandom(1000)) for i in range(6)]
    before = hashing.snapshot()["parallel_batches"]
    assert hashing.sha256_many(files, source="t", workers=4) == [sha(path) for path in files]
    assert hashing.snapshot()["parallel_batches"] == before


def test_sha256_many_order_dedupe_and_serial_equivalence(ntfs, monkeypatch):
    files = [make(ntfs / f"f{i}.bin", os.urandom(random.Random(i).randint(0, 3 * hashing.CHUNK)))
             for i in range(9)]
    spelled = [files[0], str(files[3]), files[3].parent / "." / files[3].name, str(files[3]).upper()] + files
    expected = [sha(path) for path in spelled]
    before = hashing.snapshot()
    assert hashing.sha256_many(spelled, source="t") == expected
    counts = delta(before, hashing.snapshot())
    # Repeats of the same spelling (after abspath) are hashed once; another letter case is not merged.
    assert counts["real_files"] == len(files) + 1
    assert hashing.snapshot()["parallel_batches"] > before["parallel_batches"]
    monkeypatch.setenv(hashing.ENV_WORKERS, "1")
    assert hashing.resolve_workers(8) == 1
    before = hashing.snapshot()
    assert hashing.sha256_many(spelled, source="t") == expected
    assert hashing.snapshot()["parallel_batches"] == before["parallel_batches"]
    monkeypatch.setenv(hashing.ENV_WORKERS, "zero")
    assert hashing.resolve_workers(None) == min(4, os.cpu_count() or 1) and hashing.resolve_workers(3) == 3
    monkeypatch.delenv(hashing.ENV_WORKERS)
    assert hashing.sha256_many([], source="t") == []
    with hashing.run_scope("t", mode="on"):
        assert hashing.sha256_many(spelled, source="t") == expected
        before = hashing.snapshot()
        assert hashing.sha256_many(spelled, source="t", workers=3) == expected
        counts = delta(before, hashing.snapshot())
        assert counts["real_files"] == 0 and counts["ledger"] == len(files) + 1   # the ledger key folds case
    with hashing.run_scope("t", mode="shadow"):
        hashing.sha256_many(spelled, source="t")
        before = hashing.snapshot()
        assert hashing.sha256_many(spelled, source="t") == expected
        assert delta(before, hashing.snapshot())["shadow"] == len(files) + 1


def _scenario_outcome(fn):
    try:
        return ("ok", fn())
    except BaseException as exc:
        return ("raise", type(exc).__name__, str(exc))


def test_parallel_failures_match_the_serial_loop(ntfs, parallel, monkeypatch):
    """Randomised: whatever finishes first, the parallel result equals the serial one."""
    real_read = hashing._read_digest
    plan = {}

    def fake_read(path, source, check=None, abort=None):
        kind, pause = plan.get(os.path.basename(os.fspath(path)), ("ok", 0))
        end = time.perf_counter() + pause
        while time.perf_counter() < end:
            if abort is not None:
                abort()
            time.sleep(0.002)
        if kind == "oserror":
            raise PermissionError(13, "denied", os.fspath(path))
        if kind == "boom":
            raise RuntimeError("boom " + os.path.basename(os.fspath(path)))
        return real_read(path, source, check, abort)

    monkeypatch.setattr(hashing, "_read_digest", fake_read)
    rng = random.Random(7)
    batches = hashing.snapshot()["parallel_batches"]
    for trial in range(40):
        count = rng.randint(1, 9)
        paths, expected = [], []
        plan.clear()
        for index in range(count):
            name = f"t{trial}_{index}.bin"
            kind = rng.choice(["ok", "ok", "ok", "mismatch", "missing", "oserror", "boom"])
            path = ntfs / name
            if kind != "missing":
                make(path, os.urandom(rng.randint(0, 3000)))
            plan[name] = (kind if kind in ("oserror", "boom") else "ok", rng.choice([0, 0, .01, .04]))
            paths.append(path)
            expected.append("0" * 64 if kind == "mismatch" else (sha(path) if path.exists() else "f" * 64))
        cancel_at = rng.choice([None, None, rng.randint(1, count + 1)])

        def run(workers):
            calls = [0]

            def check():
                calls[0] += 1
                if cancel_at is not None and calls[0] >= cancel_at:
                    raise Cancelled("stop")
            return _scenario_outcome(lambda: hashing.first_mismatch(
                paths, expected, source="t", workers=workers, check=check if cancel_at else None))

        serial = run(1)
        parallel = run(4)
        if cancel_at is None:
            assert parallel == serial, (trial, plan, serial, parallel)
        else:
            # Cancellation time differs between the two runs; the rest must still be serial-consistent.
            if parallel[0] == "raise" and parallel[1] == "Cancelled":
                continue
            reference = _scenario_outcome(lambda: hashing.first_mismatch(paths, expected, source="t", workers=1))
            assert parallel == reference, (trial, plan, reference, parallel)
    assert hashing.snapshot()["parallel_batches"] - batches >= 30


def test_slow_low_index_mismatch_beats_a_fast_high_index_error(ntfs, monkeypatch):
    big = make(ntfs / "big.bin", os.urandom(24 << 20))
    small = [make(ntfs / f"s{i}.bin", os.urandom(100)) for i in range(3)]
    missing = ntfs / "missing.bin"
    paths = [big, small[0], missing, small[1]]
    expected = ["0" * 64, sha(small[0]), "0" * 64, sha(small[1])]
    assert hashing.first_mismatch(paths, expected, source="t", workers=4) == 0
    with pytest.raises(FileNotFoundError):
        hashing.first_mismatch(paths, [sha(big)] + expected[1:], source="t", workers=4)


def test_cancellation_from_check(ntfs, parallel):
    files = [make(ntfs / f"f{i}.bin", os.urandom(50_000)) for i in range(12)]
    calls = []
    hashing.sha256_many(files, source="t", workers=4, check=lambda: calls.append(1))
    assert len(calls) >= len(files)                                   # at least once per file

    def cancel_after(n):
        seen = [0]

        def check():
            seen[0] += 1
            if seen[0] > n:
                raise Cancelled("cancelled")
        return check
    for n in (0, 3, 11):
        with pytest.raises(Cancelled):
            hashing.sha256_many(files, source="t", workers=4, check=cancel_after(n))
    # A failure below the cancelled position wins, as in the serial loop.
    missing = ntfs / "missing.bin"
    with pytest.raises(FileNotFoundError):
        hashing.sha256_many([files[0], missing] + files[1:], source="t", workers=4, check=cancel_after(2))


@pytest.mark.parametrize("threads", ["serial", "parallel"])
def test_verify_entries_keeps_the_callers_error_and_order(ntfs, monkeypatch, threads):
    if threads == "parallel":
        monkeypatch.setattr(hashing, "PARALLEL_MIN_BYTES", 0)
    else:
        monkeypatch.setenv(hashing.ENV_WORKERS, "1")
    files = [make(ntfs / f"f{i}.bin", os.urandom(1000 + i)) for i in range(6)]
    entries = [{"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size} for path in files]

    def error(entry):
        return ValueError("输入或已完成阶段的文件已改变／缺失：" + str(Path(entry["path"])))

    hashing.verify_entries(entries, source="t", error=error)
    bad_digest = [dict(entry) for entry in entries]
    bad_digest[2]["sha256"] = "0" * 64
    bad_digest[4]["bytes"] += 1                        # a later precheck failure does not win
    with pytest.raises(ValueError) as caught:
        hashing.verify_entries(bad_digest, source="t", error=error)
    assert str(caught.value).endswith(str(files[2]))
    size_first = [dict(entry) for entry in entries]
    size_first[1]["bytes"] += 1
    size_first[3]["sha256"] = "0" * 64
    before = hashing.snapshot()
    with pytest.raises(ValueError) as caught:
        hashing.verify_entries(size_first, source="t", error=error)
    assert str(caught.value).endswith(str(files[1]))
    assert delta(before, hashing.snapshot())["real_files"] == 1          # nothing after the failed precheck
    missing = [dict(entry) for entry in entries]
    missing[0]["path"] = str(ntfs / "gone.bin")
    with pytest.raises(ValueError) as caught:
        hashing.verify_entries(missing, source="t", error=error)
    assert "gone.bin" in str(caught.value)
    broken = [dict(entry) for entry in entries]
    del broken[3]["bytes"]
    broken[4]["sha256"] = "0" * 64
    with pytest.raises(KeyError):
        hashing.verify_entries(broken, source="t", error=error)
    broken[1]["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        hashing.verify_entries(broken, source="t", error=error)
    no_digest = [dict(entry) for entry in entries]
    del no_digest[2]["sha256"]
    before = hashing.snapshot()
    with pytest.raises(KeyError):
        hashing.verify_entries(no_digest, source="t", error=error)
    assert delta(before, hashing.snapshot())["real_files"] == 3      # the serial loop hashed it first
    # The same file listed twice is hashed once but compared per entry.
    twice = [entries[0], dict(entries[0]), entries[1]]
    twice[1]["sha256"] = "f" * 64
    with pytest.raises(ValueError) as caught:
        hashing.verify_entries(twice, source="t", error=lambda entry: ValueError(entry["sha256"]))
    assert str(caught.value) == "f" * 64
    calls = []
    hashing.verify_entries(entries, source="t", error=error, check=lambda: calls.append(1))
    assert len(calls) >= len(entries)


# --------------------------------------------------------------------------- #
# Copy                                                                         #
# --------------------------------------------------------------------------- #

def test_copy_with_digest(ntfs):
    data = os.urandom(2 * hashing.CHUNK + 17)
    src = make(ntfs / "src.bin", data)
    digest = hashlib.sha256(data).hexdigest()
    calls = []
    with hashing.run_scope("t", mode="on"):
        before = hashing.snapshot()
        result = hashing.copy_with_digest(src, ntfs / "dst.bin", source="t", expected=digest,
                                          check=lambda: calls.append(1))
        assert result == (digest, digest, len(data)) and (ntfs / "dst.bin").read_bytes() == data
        counts = delta(before, hashing.snapshot())
        assert counts["real"] == 2 * len(data)                          # copy + real read-back
        assert len(calls) >= 4
        before = hashing.snapshot()
        assert hashing.sha256_file(ntfs / "dst.bin", source="t") == digest    # read-back remembered as written
        assert hashing.sha256_file(src, source="t") == digest                 # copy digest remembered for src
        counts = delta(before, hashing.snapshot())
        assert counts["write"] == 1 and counts["ledger"] == 1 and counts["real_files"] == 0
        # The read-back is real even when the ledger knows the destination.
        before = hashing.snapshot()
        result = hashing.copy_with_digest(src, ntfs / "dst.bin", source="t", exclusive=False)
        assert delta(before, hashing.snapshot())["real"] == 2 * len(data) and result.dst_sha256 == digest
        # Mismatch: no read-back, the caller decides.
        before = hashing.snapshot()
        result = hashing.copy_with_digest(src, ntfs / "dst2.bin", source="t", expected="0" * 64)
        assert result == (digest, None, len(data))
        assert delta(before, hashing.snapshot())["real"] == len(data)
        # Exclusive create.
        with pytest.raises(FileExistsError):
            hashing.copy_with_digest(src, ntfs / "dst.bin", source="t")
        # No read-back: the copy digest is registered for the destination.
        result = hashing.copy_with_digest(src, ntfs / "dst3.bin", source="t", readback=False, fsync=False)
        assert result == (digest, digest, len(data))
        before = hashing.snapshot()
        assert hashing.sha256_file(ntfs / "dst3.bin", source="t") == digest
        assert delta(before, hashing.snapshot())["write"] == 1

        def cancel():
            raise Cancelled()
        with pytest.raises(Cancelled):
            hashing.copy_with_digest(src, ntfs / "dst4.bin", source="t", check=cancel)
    # Outside a scope: same bytes, nothing remembered.
    result = hashing.copy_with_digest(src, ntfs / "dst5.bin", source="t")
    assert result == (digest, digest, len(data))
    with pytest.raises(FileNotFoundError):
        hashing.copy_with_digest(ntfs / "nope.bin", ntfs / "dst6.bin", source="t")
    assert not (ntfs / "dst6.bin").exists()


# --------------------------------------------------------------------------- #
# Digest while writing                                                         #
# --------------------------------------------------------------------------- #

def test_hashing_writer_sequential_forward_and_backward(tmp_path):
    with open(tmp_path / "a.bin", "wb") as raw:
        writer = hashing.HashingWriter(raw)
        writer.write(b"abc")
        assert writer.tell() == 3
        writer.seek(10)                        # forward: the gap reads back as zeros
        writer.write(np.arange(4, dtype=np.uint16))
        writer.seek(0, 2)
        writer.write(memoryview(b"xyz"))
    assert writer.digest() == sha(tmp_path / "a.bin") and writer.size == (tmp_path / "a.bin").stat().st_size
    with open(tmp_path / "b.bin", "wb") as raw:
        writer = hashing.HashingWriter(raw)
        writer.write(b"abcdef")
        writer.seek(2)
        writer.write(b"Z")
    assert writer.digest() is None and writer.gave_up == "backward seek"
    with open(tmp_path / "c.bin", "wb") as raw:
        writer = hashing.HashingWriter(raw)
        writer.write(b"abcdef")
        writer.truncate(3)
    assert writer.digest() is None
    make(tmp_path / "d.bin", b"existing")
    with open(tmp_path / "d.bin", "ab") as raw:
        assert hashing.HashingWriter(raw).digest() is None     # not a fresh file
    with open(tmp_path / "e.bin", "wb") as raw:
        writer = hashing.HashingWriter(raw)
        with pytest.raises(io.UnsupportedOperation):
            writer.fileno()


def test_write_helpers_match_a_real_rehash_and_become_write_hits(ntfs):
    array = np.random.default_rng(3).random((64, 48), dtype=np.float32)
    payload = os.urandom(10_000)
    with hashing.run_scope("t", mode="shadow"):
        npy = hashing.save_npy_with_digest(ntfs / "a.npy", array, source="t")
        reference = ntfs / "reference.npy"
        np.save(reference, array)
        assert npy == sha(ntfs / "a.npy") == sha(reference)
        raw = hashing.write_bytes_with_digest(ntfs / "b.json", payload, source="t")
        assert raw == sha(ntfs / "b.json") == hashlib.sha256(payload).hexdigest()
        from astropy.io import fits
        fits_digest = hashing.write_with_digest(
            ntfs / "c.fit", lambda handle: fits.writeto(handle, array), source="t")
        fits.writeto(ntfs / "c_ref.fit", array)
        assert fits_digest == sha(ntfs / "c.fit") == sha(ntfs / "c_ref.fit")
        import tifffile
        assert hashing.write_with_digest(ntfs / "d.tif", lambda handle: tifffile.imwrite(handle, array),
                                         source="t") is None       # tifffile seeks back: read-back instead
        before = hashing.snapshot()
        for name in ("a.npy", "b.json", "c.fit"):
            hashing.sha256_file(ntfs / name, source="t")
        counts = delta(before, hashing.snapshot())
        assert counts["write"] == 3 and counts["shadow"] == 3
    with pytest.raises(FileExistsError):
        hashing.write_bytes_with_digest(ntfs / "b.json", payload, source="t", mode="xb")
    with pytest.raises(ValueError):
        hashing.register_written(ntfs / "b.json", "ABC", "t")
    assert not hashing.register_written(ntfs / "b.json", sha(ntfs / "b.json"), "t")     # no scope
    # A registered write digest is still bound to the exact stat.
    with hashing.run_scope("t", mode="on"):
        hashing.write_bytes_with_digest(ntfs / "e.bin", b"12345", source="t")
        with open(ntfs / "e.bin", "ab") as handle:
            handle.write(b"6")
        assert hashing.sha256_file(ntfs / "e.bin", source="t") == sha(ntfs / "e.bin")


# --------------------------------------------------------------------------- #
# Handover                                                                     #
# --------------------------------------------------------------------------- #

def after_stale(before):
    return hashing.snapshot()["misses"]["stale"] - before["misses"]["stale"]


def test_handover_export_import(ntfs, monkeypatch):
    root = ntfs / "job"
    root.mkdir()
    kept = make(root / "kept.bin", os.urandom(5000))
    changed = make(root / "changed.bin", os.urandom(5000))
    fresh = root / "fresh.bin"
    outside = make(ntfs / "outside.bin", os.urandom(5000))
    handover = ntfs / "handover.json"
    for bad in ("", "short", "has space in it", "../../escape", None, 12345678):
        with pytest.raises(ValueError):
            hashing.export_handover(handover, root, token=bad)
        with pytest.raises(ValueError):
            hashing.import_handover(handover, root, token=bad)
    default = hashing.handover_path("attempt-default")
    assert default.parent.name == hashing.HANDOVER_DIRECTORY and default.name == "attempt-default.json"
    with hashing.run_scope("job", mode="on"):
        hashing.sha256_file(kept, source="job")
        assert hashing.export_handover(None, root, token="attempt-default") == 1 and default.is_file()
    with hashing.run_scope("workflow", mode="on"):
        assert hashing.import_handover(None, root, token="attempt-default") == 1 and not default.exists()
    assert hashing.export_handover(handover, root, token="attempt1") is None and not handover.exists()   # no scope

    def export(token="attempt1"):
        with hashing.run_scope("job", mode="on"):
            fresh.write_bytes(os.urandom(5000))                      # racy: modified just now
            for path in (kept, changed, fresh, outside):
                hashing.sha256_file(path, source="job")
            assert hashing.export_handover(handover, root, token=token) == 3

    export()
    document = json.loads(handover.read_text(encoding="utf-8"))
    assert document["version"] == hashing.HANDOVER_VERSION and document["token"] == "attempt1"
    assert {Path(row["path"]).name for row in document["entries"]} == {"kept.bin", "changed.bin", "fresh.bin"}
    with open(changed, "ab") as handle:
        handle.write(b"x")
    assert hashing.import_handover(handover, root, token="attempt1") == 0 and not handover.exists()   # no scope
    export()
    with hashing.run_scope("workflow", mode="off"):
        assert hashing.import_handover(handover, root, token="attempt1") == 0 and not handover.exists()
    export()
    with hashing.run_scope("workflow", mode="on"):
        assert hashing.import_handover(handover, root / "elsewhere", token="attempt1") == 0
    assert not handover.exists()                                   # always consumed
    export()
    with hashing.run_scope("workflow", mode="on"):
        assert hashing.import_handover(handover, root, token="attempt0") == 0      # another attempt's file
    assert not handover.exists()
    export()
    with open(changed, "ab") as handle:              # changed after the export: the stat no longer matches
        handle.write(b"y")
    with hashing.run_scope("workflow", mode="on"):
        assert hashing.import_handover(handover, root, token="attempt1") == 3
        assert not handover.exists()
        before = hashing.snapshot()
        for path in (kept, changed, fresh, outside):
            assert hashing.sha256_file(path, source="wf") == sha(path)
        counts = delta(before, hashing.snapshot())
        assert counts["handover"] == 1 and counts["real_files"] == 3 and counts["racy"] == 1
        assert after_stale(before) == 1
    export()
    with hashing.run_scope("workflow", mode="shadow"):
        hashing.import_handover(handover, root, token="attempt1")
        before = hashing.snapshot()
        assert hashing.sha256_file(kept, source="wf") == sha(kept)
        assert delta(before, hashing.snapshot())["shadow"] == 1
    for text in ("not json", json.dumps({"version": "other", "token": "attempt1", "entries": []}),
                 json.dumps({"version": hashing.HANDOVER_VERSION, "token": "attempt1", "entries": [{"path": 1}]})):
        handover.write_text(text, encoding="utf-8")
        with hashing.run_scope("workflow", mode="on"):
            assert hashing.import_handover(handover, root, token="attempt1") == 0
    with hashing.run_scope("workflow", mode="on"):
        assert hashing.import_handover(ntfs / "absent.json", root, token="attempt1") == 0


def test_handover_is_revoked_by_a_later_in_place_write_and_by_an_empty_or_off_export(ntfs):
    root = ntfs / "work"
    root.mkdir()
    frame = make(root / "r_light_0001.fit", os.urandom(20_000))
    other = make(root / "other.bin", os.urandom(1000))
    handover = ntfs / "handover.json"
    # e4: export, then a memmap write under root (stat unchanged) in the exporting process.
    with hashing.run_scope("job", mode="on"):
        hashing.sha256_file(frame, source="job")
        hashing.sha256_file(other, source="job")
        assert hashing.export_handover(handover, root, token="token-t1") == 2
        with hashing.writing(frame):
            _poke_memmap(frame, 5)
        assert not handover.exists()
        assert hashing.snapshot()["handover"]["revoked"] >= 1
        # invalidate() revokes too, and a write outside root does not.
        hashing.sha256_file(frame, source="job")
        assert hashing.export_handover(handover, root, token="token-t1") == 2
        with hashing.writing(ntfs / "unrelated.bin"):
            (ntfs / "unrelated.bin").write_bytes(b"x")
        assert handover.exists()
        hashing.invalidate(other)
        assert not handover.exists()
    # The importer never sees the pre-write digest.
    with hashing.run_scope("workflow", mode="on"):
        assert hashing.import_handover(handover, root, token="token-t1") == 0
        assert hashing.sha256_file(frame, source="wf") == sha(frame)
    # A stale file from an earlier attempt is removed by an export that has nothing to hand over.
    with hashing.run_scope("job", mode="on"):
        hashing.sha256_file(frame, source="job")
        assert hashing.export_handover(handover, root, token="token-t1") == 1
    assert handover.exists()
    with hashing.run_scope("job", mode="off"):
        assert hashing.export_handover(handover, root, token="token-t2") is None
    assert not handover.exists()
    with hashing.run_scope("job", mode="on"):
        hashing.sha256_file(frame, source="job")
        assert hashing.export_handover(handover, root, token="token-t1") == 1
    assert handover.exists()
    with hashing.run_scope("job", mode="on"):
        assert hashing.export_handover(handover, root, token="token-t3") == 0
    assert not handover.exists()


# --------------------------------------------------------------------------- #
# Review fixes (2026-09-29)                                                    #
# --------------------------------------------------------------------------- #

def test_a_plain_rewrite_right_after_a_write_digest_is_seen(ntfs):
    """e2: a same-size "wb" rewrite within one clock tick of a registered write is not answered from the ledger."""
    stale = 0
    with hashing.run_scope("t", mode="on"):
        for index in range(60):
            path = ntfs / f"rec{index}.json"
            digest = hashing.write_bytes_with_digest(path, b'{"state": "running"}', source="t")
            path.write_bytes(b'{"state": "success"}')
            if hashing.sha256_file(path, source="t") != sha(path):
                stale += 1
            assert digest != sha(path)

        def plain_then_register(path):
            path.write_bytes(b"A" * 64)
            assert hashing.register_written(path, sha(path), "t")
        writers = (
            lambda p: hashing.write_with_digest(p, lambda h: h.write(b"A" * 64), source="t"),
            lambda p: hashing.save_npy_with_digest(p, np.zeros(8, dtype=np.uint8), source="t"),
            lambda p: hashing.copy_with_digest(make(p.with_suffix(".src"), b"A" * 64), p, source="t"),
            lambda p: hashing.copy_with_digest(make(p.with_suffix(".src"), b"A" * 64), p, source="t",
                                               readback=False),
            plain_then_register)
        for index, write in enumerate(writers):
            path = ntfs / f"other{index}.bin"
            write(path)
            before = hashing.snapshot()
            assert hashing.sha256_file(path, source="t") == sha(path)
            assert delta(before, hashing.snapshot())["write"] == 1          # the write digest is used ...
            path.write_bytes(b"B" * path.stat().st_size)
            if hashing.sha256_file(path, source="t") != sha(path):          # ... until a plain rewrite
                stale += 1
    assert stale == 0


def test_a_write_digest_is_not_recorded_when_another_writer_overlapped(ntfs):
    path = ntfs / "a.bin"
    with hashing.run_scope("t", mode="on"):
        with hashing.writing(path) as block:
            path.write_bytes(b"first")
            with hashing.writing(path):               # a nested/concurrent writer of the same file
                path.write_bytes(b"other")
            block.written(hashlib.sha256(b"first").hexdigest(), "t")
        assert not block.registered
        before = hashing.snapshot()
        assert hashing.sha256_file(path, source="t") == sha(path)
        assert delta(before, hashing.snapshot())["real_files"] == 1
        with hashing.writing(path) as block:
            path.write_bytes(b"third")
            block.written(hashlib.sha256(b"third").hexdigest(), "t")
        assert block.registered
        with pytest.raises(ValueError):
            with hashing.writing(path) as block:
                block.written("XYZ", "t")
        with pytest.raises(RuntimeError):             # a failed block records nothing
            with hashing.writing(path) as block:
                block.written(hashlib.sha256(b"third").hexdigest(), "t")
                raise RuntimeError("write failed")
        assert not block.registered
    with hashing.writing(path) as block:              # outside a scope: nothing is recorded or nudged
        block.written(hashlib.sha256(b"third").hexdigest(), "t")
    assert not block.registered


def test_non_exclusive_copy_refuses_the_same_file_like_shutil(ntfs):
    data = os.urandom(1000)
    src = make(ntfs / "src.bin", data)
    link = ntfs / "link.bin"
    os.link(src, link)
    for dst in (src, link, str(src)):
        with pytest.raises(shutil.SameFileError) as expected:
            shutil.copyfile(src, dst)
        with hashing.run_scope("t", mode="on"):
            with pytest.raises(shutil.SameFileError) as got:
                hashing.copy_with_digest(src, dst, source="t", exclusive=False)
        assert str(got.value) == str(expected.value)
        assert src.read_bytes() == data
    with pytest.raises(FileExistsError):
        hashing.copy_with_digest(src, link, source="t")
    assert src.read_bytes() == data
    with pytest.raises(FileNotFoundError):
        hashing.copy_with_digest(src, ntfs / "no-such-dir" / "x.bin", source="t", exclusive=False)


def test_copy_runs_check_after_the_copy_before_comparing(ntfs):
    """storage._copy_verified: check() after fsync; a cancel beats the copy-mismatch error."""
    data = os.urandom(3 * hashing.CHUNK // 2)
    src = make(ntfs / "src.bin", data)
    calls = []

    def cancel_after_copy():
        calls.append(1)
        if len(calls) == 4:          # 2 chunks + the empty read, then the post-copy check
            raise Cancelled()
    with pytest.raises(Cancelled):
        hashing.copy_with_digest(src, ntfs / "dst.bin", source="t", expected="0" * 64, check=cancel_after_copy)
    calls.clear()
    result = hashing.copy_with_digest(src, ntfs / "dst2.bin", source="t", check=lambda: calls.append(1))
    assert result.dst_sha256 == sha(src) and len(calls) == 4        # no check() during the read-back


def test_verify_entries_validate_runs_before_check(ntfs, monkeypatch):
    """finite_stack._verify_sources: the file-name check comes before check(), a mismatch before either."""
    monkeypatch.setattr(hashing, "PARALLEL_MIN_BYTES", 0)
    work = ntfs
    files = [make(work / f"f{i}.bin", os.urandom(1000 + i)) for i in range(3)]

    def serial(entries, check):
        for name, digest in entries:
            if not isinstance(name, str) or Path(name).name != name:
                raise ValueError("有限值堆栈收据含无效文件名")
            check()
            if not (work / name).is_file() or sha(work / name) != digest:
                raise ValueError("有限值堆栈源、序列或独立有效性已改变")

    def converted(entries, check):
        def validate(entry):
            name = entry[0]
            if not isinstance(name, str) or Path(name).name != name:
                raise ValueError("有限值堆栈收据含无效文件名")
        hashing.verify_entries(entries, source="t", check=check, validate=validate,
                               path=lambda entry: work / entry[0], expected=lambda entry: entry[1],
                               precheck=lambda entry, path: path.is_file(),
                               error=lambda entry: ValueError("有限值堆栈源、序列或独立有效性已改变"))

    good = [(path.name, sha(path)) for path in files]
    cases = [good, good[:1] + [("../bad", "0" * 64)] + good[1:], [(good[0][0], "0" * 64), ("../bad", "0")],
             [good[0], (5, "0")], good[:2] + [("missing.bin", "0" * 64)], [("../bad", "0")]]
    for entries in cases:
        for cancel_at in (None, 1, 2, 3):
            def make_check():
                calls = [0]

                def check():
                    calls[0] += 1
                    if cancel_at is not None and calls[0] >= cancel_at:
                        raise Cancelled("stop")
                return check
            for workers in ("1", "4"):
                monkeypatch.setenv(hashing.ENV_WORKERS, workers)
                assert _scenario_outcome(lambda: converted(entries, make_check())) == \
                    _scenario_outcome(lambda: serial(entries, make_check())), (entries, cancel_at, workers)


def test_off_mode_reads_every_repeat_and_repeats_are_counted(ntfs, monkeypatch):
    path = make(ntfs / "a.bin", os.urandom(4096))
    for workers in (1, 4):
        monkeypatch.setenv(hashing.ENV_LEDGER, "off")
        before = hashing.snapshot()
        assert hashing.sha256_many([path, path, path], source="t", workers=workers) == [sha(path)] * 3
        counts = delta(before, hashing.snapshot())
        assert counts["real_files"] == 3 and counts["batch"] == 0
        with hashing.run_scope("t"):
            before = hashing.snapshot()
            hashing.sha256_many([path, path], source="t", workers=workers)
            assert delta(before, hashing.snapshot())["real_files"] == 2
        monkeypatch.delenv(hashing.ENV_LEDGER)
        before = hashing.snapshot()
        assert hashing.sha256_many([path, path, path], source="t.batch", workers=workers) == [sha(path)] * 3
        counts = delta(before, hashing.snapshot())
        assert counts["real_files"] == 1 and counts["batch"] == 2
        assert hashing.snapshot()["saved"]["batch"]["bytes"] - before["saved"]["batch"]["bytes"] == 2 * 4096
    assert profiling.hash_saved_sources()["t.batch"]["hash_saved_batch_bytes"] >= 2 * 4096


def test_path_forms_fail_like_the_helpers_they_replace(tmp_path, monkeypatch):
    """sha256_file opens open(path): a Path-based helper passes Path(path) and fails as before."""
    monkeypatch.chdir(tmp_path)
    for value in ("", Path(""), tmp_path, str(tmp_path), tmp_path / "missing.bin", str(tmp_path / "missing.bin")):
        opener = (lambda v: Path(v).open("rb")) if isinstance(value, Path) else (lambda v: open(v, "rb"))
        with pytest.raises(OSError) as expected:
            with opener(value):
                pass
        for mode in (None, "on"):
            if mode is None:
                with pytest.raises(OSError) as got:
                    hashing.sha256_file(value, source="t")
            else:
                with hashing.run_scope("t", mode=mode), pytest.raises(OSError) as got:
                    hashing.sha256_file(value, source="t")
            assert type(got.value) is type(expected.value) and str(got.value) == str(expected.value), value
    with pytest.raises(TypeError):
        hashing.sha256_file(Path(3), source="t")


# --------------------------------------------------------------------------- #
# In-memory digests                                                            #
# --------------------------------------------------------------------------- #

def test_array_digest_is_array_identity():
    from starstack.photographic import array_identity
    rng = np.random.default_rng(5)
    arrays = [rng.random((300, 7), dtype=np.float32), rng.random((5, 4, 3)).astype(">f8"),
              np.arange(1000, dtype=np.uint16).reshape(40, 25)[:, ::3], np.zeros((0, 4), dtype=np.uint8),
              rng.integers(0, 255, (129, 2), dtype=np.uint8), np.asfortranarray(rng.random((260, 9)))]
    for array in arrays:
        assert hashing.array_digest(array, source="t") == array_identity(array)
    for bad in (np.float32(1.0), np.zeros((), dtype=np.uint8), [1, 2]):
        with pytest.raises(Exception) as expected:
            array_identity(bad)
        with pytest.raises(Exception) as got:
            hashing.array_digest(bad, source="t")
        assert type(got.value) is type(expected.value) and str(got.value) == str(expected.value)
    big = [rng.random((512, 1024), dtype=np.float32) for _ in range(5)]
    before = profiling.hashed_memory_sources().get("t.arrays", {}).get("hashed_memory_bytes", 0)
    assert hashing.arrays_digest_many(big + arrays, source="t.arrays", workers=4) == \
        [array_identity(array) for array in big + arrays]
    assert profiling.hashed_memory_sources()["t.arrays"]["hashed_memory_bytes"] - before == \
        sum(array.nbytes for array in big + arrays)
    assert hashing.arrays_digest_many([], source="t") == []
    calls = []
    hashing.arrays_digest_many(big, source="t", workers=4, check=lambda: calls.append(1))
    assert len(calls) == len(big)
    # The lowest failing index wins; a check() failure counts for the array it precedes.
    for workers in (1, 4):
        with pytest.raises(IndexError):
            hashing.arrays_digest_many(big[:2] + [np.zeros((), np.uint8)] + big[2:] + [[1]], source="t",
                                       workers=workers)
        seen = [0]

        def cancel():
            seen[0] += 1
            if seen[0] == 3:
                raise Cancelled()
        with pytest.raises(Cancelled):
            hashing.arrays_digest_many(big + [[1]], source="t", workers=workers, check=cancel)
        seen[0] = 0
        with pytest.raises(AttributeError):
            hashing.arrays_digest_many([big[0], [1]] + big, source="t", workers=workers, check=cancel)


def test_sha256_bytes_and_write_digests_count_memory_hashing(tmp_path):
    payload = os.urandom(12_345)
    before = hashing.snapshot()["memory"]
    assert hashing.sha256_bytes(payload, source="t.bytes") == hashlib.sha256(payload).hexdigest()
    assert hashing.sha256_bytes(memoryview(payload), source="t.bytes") == hashlib.sha256(payload).hexdigest()
    with pytest.raises(TypeError) as expected:
        hashlib.sha256("text")
    with pytest.raises(TypeError) as got:
        hashing.sha256_bytes("text", source="t.bytes")
    assert str(got.value) == str(expected.value)
    hashing.write_bytes_with_digest(tmp_path / "a.bin", payload, source="t.write")
    hashing.write_with_digest(tmp_path / "b.bin", lambda handle: (handle.write(payload), handle.seek(20_000),
                                                                  handle.write(b"x")), source="t.write")
    after = hashing.snapshot()["memory"]
    assert after["bytes"] - before["bytes"] == 2 * len(payload) + len(payload) + 20_001
    sources = profiling.hashed_memory_sources()
    assert sources["t.bytes"]["hashed_memory_calls"] >= 2 and sources["t.write"]["hashed_memory_bytes"] >= 32_346


# --------------------------------------------------------------------------- #
# Accounting                                                                   #
# --------------------------------------------------------------------------- #

def test_profiling_reports_real_and_saved_bytes_per_phase_and_source(ntfs):
    path = make(ntfs / "a.bin", os.urandom(300_000))
    timer = profiling.StageTimer(initial="first")
    with hashing.run_scope("t", mode="on"):
        hashing.sha256_file(path, source="test.helper")
        timer.phase("second")
        hashing.sha256_file(path, source="test.helper")
        hashing.sha256_file(path, source="test.other")
        hashing.write_bytes_with_digest(ntfs / "w.bin", b"w" * 1000, source="test.writer")
        hashing.sha256_file(ntfs / "w.bin", source="test.writer")
    report = timer.finish()
    phases = report["resources"]["phases"]
    assert phases["first"]["hashed_bytes"] >= 300_000 and phases["first"]["hash_saved_bytes"] == 0
    second = phases["second"]
    assert second["hash_saved_bytes"] == 601_000 and second["hash_saved_files"] == 3
    assert second["hash_saved_ledger_bytes"] == 600_000 and second["hash_saved_write_bytes"] == 1000
    assert second["hash_saved_handover_bytes"] == 0
    saved = report["resources"]["hash_saved_sources"]
    assert saved["test.helper"]["hash_saved_ledger_bytes"] == 300_000
    assert saved["test.other"]["hash_saved_bytes"] == 300_000 and saved["test.writer"]["hash_saved_write_bytes"] == 1000
    assert report["resources"]["hashed_sources"]["test.helper"]["hashed_bytes"] == 300_000
    assert report["resources"]["total"]["hash_saved_bytes"] == 601_000
    assert "hash_saved" in report["resources"]["semantics"] and "lower bound" in report["resources"]["semantics"]
    assert second["hashed_memory_bytes"] == 1000 and second["hashed_memory_calls"] == 1
    assert report["resources"]["hashed_memory_sources"] == {
        "test.writer": {"hashed_memory_bytes": 1000, "hashed_memory_calls": 1}}
    assert second["hash_saved_batch_bytes"] == 0 and "hashed_memory" in report["resources"]["semantics"]
    json.loads(json.dumps(report, allow_nan=False))
    profiling.note_hash_saved("x", source="bad")
    profiling.note_hash_saved(5, source="bad.kind", kind="other")
    assert "bad.kind" not in profiling.hash_saved_sources() and "bad" not in profiling.hash_saved_sources()


# --------------------------------------------------------------------------- #
# Review fixes (P1): serial failures under concurrent change, write failures,  #
# handover housekeeping, scope reports                                          #
# --------------------------------------------------------------------------- #

def _delete_during_first_read(monkeypatch, first, victims, replace_with_folder=False):
    """While ``first`` is being hashed, remove ``victims`` (or put a folder in their place).

    A victim's own read waits until the removal happened, so a parallel batch
    cannot hash it before: the removal always falls between the up-front
    checks of a batch and the victim's hash, the window the serial loop did
    not have.
    """
    real_read = hashing._read_digest
    removed = threading.Event()
    names = {os.path.basename(os.fspath(path)) for path in victims}

    def fake_read(path, source, check=None, abort=None):
        name = os.path.basename(os.fspath(path))
        if name == os.path.basename(os.fspath(first)) and not removed.is_set():
            for victim in victims:
                os.remove(victim)
                if replace_with_folder:
                    os.mkdir(victim)
            removed.set()
        elif name in names:
            removed.wait(5)
        return real_read(path, source, check, abort)
    monkeypatch.setattr(hashing, "_read_digest", fake_read)


@pytest.mark.parametrize("workers", ["1", "4"])
@pytest.mark.parametrize("case", ["removed", "folder", "no-digest-removed", "locked"])
def test_a_file_removed_while_a_batch_runs_fails_like_the_serial_loop(ntfs, monkeypatch, workers, case):
    """verify_entries re-runs the serial loop's checks of an entry whose hash fails (review finding)."""
    monkeypatch.setattr(hashing, "PARALLEL_MIN_BYTES", 0)
    monkeypatch.setenv(hashing.ENV_WORKERS, workers)

    def build(folder):
        folder.mkdir()
        files = [make(folder / f"f{i}.bin", bytes([i]) * (20_000 + i)) for i in range(4)]
        entries = [{"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size} for path in files]
        if case == "no-digest-removed":
            del entries[2]["sha256"]        # the serial loop hashes it, then fails on the digest lookup
        return files, entries

    def error(entry):
        return ValueError("输入或已完成阶段的文件已改变／缺失：" + str(Path(entry["path"])))

    def serial(entries, on_hash):
        for entry in entries:
            path = Path(entry["path"])
            if not path.is_file() or path.stat().st_size != entry["bytes"]:
                raise error(entry)
            on_hash(path)
            if sha(path) != entry["sha256"]:
                raise error(entry)

    files, entries = build(ntfs / "run-serial")
    done = []

    def on_hash(path):
        if path == files[0] and not done:
            done.append(1)
            if case != "locked":
                os.remove(files[2])
                if case == "folder":
                    os.mkdir(files[2])
        if case == "locked" and path == files[2]:
            raise PermissionError(13, "locked by another process", str(path))
    expected = _scenario_outcome(lambda: serial(entries, on_hash))

    files, entries = build(ntfs / "run-batch")
    if case == "locked":
        real_read = hashing._read_digest

        def locked(path, source, check=None, abort=None):
            if os.path.basename(os.fspath(path)) == files[2].name:
                raise PermissionError(13, "locked by another process", os.fspath(path))
            return real_read(path, source, check, abort)
        monkeypatch.setattr(hashing, "_read_digest", locked)
    else:
        _delete_during_first_read(monkeypatch, files[0], [files[2]], replace_with_folder=case == "folder")
    actual = _scenario_outcome(lambda: hashing.verify_entries(entries, source="t", error=error))
    same = lambda result: tuple(part.replace("run-serial", "run-X").replace("run-batch", "run-X") for part in result)
    assert same(actual) == same(expected)
    if case == "locked":
        assert actual[1] == "PermissionError"     # the file is still there: the hash's own error stays
    else:
        assert actual[:2] == ("raise", "ValueError") and actual[2].endswith(str(files[2]))


@pytest.mark.parametrize("workers", [1, 4])
def test_batch_hooks_after_and_on_error(ntfs, monkeypatch, workers):
    monkeypatch.setattr(hashing, "PARALLEL_MIN_BYTES", 0)
    files = [make(ntfs / f"h{i}.bin", os.urandom(3000 + i)) for i in range(5)]
    paths = files + [files[1]]                         # a repeat is resolved from its first occurrence
    seen = {}

    def after(index):
        seen[index] = os.stat(paths[index]).st_size
    assert hashing.sha256_many(paths, source="t", workers=workers, after=after) == [sha(p) for p in paths]
    assert sorted(seen) == list(range(len(paths)))
    with hashing.run_scope("t", mode="on"):          # ledger hits run it too
        hashing.sha256_many(files, source="t", workers=workers)
        seen.clear()
        before = hashing.snapshot()
        hashing.sha256_many(paths, source="t", workers=workers, after=after)
        assert delta(before, hashing.snapshot())["real_files"] == 0 and sorted(seen) == list(range(len(paths)))
    # after's failure is the item's failure (lowest index wins); on_error is not asked for it.
    asked = []

    def after_fails(index):
        if index in (2, 4):
            raise KeyError(f"after {index}")
    with pytest.raises(KeyError, match="after 2"):
        hashing.sha256_many(paths, source="t", workers=workers, after=after_fails,
                            on_error=lambda index, exc: asked.append(index))
    assert asked == []
    # on_error replaces a hash failure of the reported item only; None keeps the original.
    missing = [files[0], ntfs / "gone1.bin", files[2], ntfs / "gone3.bin"]
    with pytest.raises(ValueError, match="^item 1$"):
        hashing.sha256_many(missing, source="t", workers=workers,
                            on_error=lambda index, exc: asked.append(index) or ValueError(f"item {index}"))
    assert asked == [1]
    with pytest.raises(FileNotFoundError):
        hashing.sha256_many(missing, source="t", workers=workers, on_error=lambda index, exc: None)

    def check_raises(index, exc):
        raise LookupError("raised by the check")
    with pytest.raises(LookupError, match="raised by the check"):   # on_error may raise itself
        hashing.first_mismatch(missing, [sha(files[0])] * 4, source="t", workers=workers, on_error=check_raises)
    # A check() failure is not a hash failure.
    asked.clear()

    def cancel():
        raise Cancelled()
    with pytest.raises(Cancelled):
        hashing.sha256_many(missing, source="t", workers=workers, check=cancel,
                            on_error=lambda index, exc: asked.append(index))
    assert asked == []


class _NoSpace:
    """A file object whose writes fail as on a full disk and that has no descriptor (so Pillow uses write())."""

    def __init__(self, raw):
        self._raw = raw

    def write(self, data):
        raise OSError(28, "No space left on device")

    def fileno(self):
        raise io.UnsupportedOperation("fileno")

    def __getattr__(self, name):
        return getattr(self._raw, name)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._raw.close()


@pytest.fixture
def disk_full(monkeypatch):
    """Writes to the named files fail (Pillow and this module both open files through builtins.open)."""
    import builtins
    real_open = builtins.open
    names = set()

    def fake_open(file, mode="r", *args, **kwargs):
        handle = real_open(file, mode, *args, **kwargs)
        if isinstance(file, (str, os.PathLike)) and os.path.basename(os.fspath(file)) in names \
                and any(flag in mode for flag in "wxa+"):
            return _NoSpace(handle)
        return handle
    monkeypatch.setattr(builtins, "open", fake_open)
    return names


def test_a_failed_image_write_leaves_what_image_save_leaves(ntfs, disk_full):
    from PIL import Image
    from starstack import master_export as me
    image = Image.fromarray(np.arange(40 * 50 * 3, dtype=np.uint8).reshape(40, 50, 3))
    for name in ("a.jpg", "b.png"):
        for existing in (False, True):
            old, new = ntfs / f"old-{existing}-{name}", ntfs / f"new-{existing}-{name}"
            for path in (old, new):
                if existing:
                    path.write_bytes(b"an earlier file")
                disk_full.add(path.name)
            expected = _scenario_outcome(lambda: image.save(old))
            with hashing.run_scope("t", mode="on"):
                actual = _scenario_outcome(lambda: me.save_image_with_digest(image, new, source="t"))
            assert actual == expected and expected[:2] == ("raise", "OSError"), (name, existing)
            # Image.save deletes the file it created and leaves (truncated) one that was there.
            assert (new.exists(), old.exists()) == (existing, existing)
            if existing:
                assert new.read_bytes() == old.read_bytes() == b""


def test_write_digest_failures_leave_what_the_replaced_writer_leaves(ntfs, disk_full):
    def before_writing(handle):
        raise ValueError("refused before writing")

    def after_writing(handle):
        handle.write(b"half")
        raise ValueError("refused after writing")
    for writer, discard, left in ((before_writing, "unwritten", False), (after_writing, "unwritten", True),
                                  (before_writing, None, True), (after_writing, "created", False)):
        path = ntfs / f"w-{writer.__name__}-{discard}.bin"
        with hashing.run_scope("t", mode="on"), pytest.raises(ValueError):
            hashing.write_with_digest(path, writer, source="t", discard=discard)
        assert path.exists() == left, (writer.__name__, discard)
    with pytest.raises(ValueError):
        hashing.write_with_digest(ntfs / "x.bin", before_writing, source="t", discard="sometimes")
    # A disk-full write of in-memory bytes: "created" removes the file, None leaves it (as write_bytes).
    for discard, left in (("created", False), (None, True)):
        path = ntfs / f"full-{discard}.bin"
        disk_full.add(path.name)
        with pytest.raises(OSError):
            hashing.write_bytes_with_digest(path, b"data" * 100, source="t", discard=discard)
        assert path.exists() == left


def test_write_linear_failing_before_it_writes_leaves_no_file(ntfs, monkeypatch):
    """astropy validates before it creates the file: a refusal leaves no file, in and out of a run scope."""
    from astropy.io import fits
    from starstack import imaging
    data = np.ones((6, 5, 3), np.float32)
    real = fits.writeto

    def refusing(target, *args, **kwargs):
        raise ValueError("header refused by verify")
    monkeypatch.setattr(imaging.fits, "writeto", refusing)
    for mode in (None, "off", "on", "shadow"):
        path = ntfs / f"linear-{mode}.fit"
        if mode is None:
            result = _scenario_outcome(lambda: imaging.write_linear(path, data, "note"))
        else:
            with hashing.run_scope("t", mode=mode):
                result = _scenario_outcome(lambda: imaging.write_linear(path, data, "note"))
        assert result == ("raise", "ValueError", "header refused by verify") and not path.exists(), mode
    monkeypatch.setattr(imaging.fits, "writeto", real)
    with hashing.run_scope("t", mode="on"):
        imaging.write_linear(ntfs / "linear-on.fit", data, "note")        # the retry is not refused
    assert (ntfs / "linear-on.fit").is_file()


def test_prune_handovers_removes_only_old_handover_files(tmp_path, monkeypatch):
    import tempfile
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    assert hashing.prune_handovers() == 0                    # no folder yet
    folder = tmp_path / hashing.HANDOVER_DIRECTORY
    folder.mkdir()
    old = time.time() - hashing.HANDOVER_MAX_AGE_SECONDS - 60
    names = {"orphan": "abcdefgh12.json", "orphan_tmp": "abcdefgh12.json.123.456.tmp", "fresh": "ijklmnop34.json",
             "other": "notes.txt", "short": "abc.json"}
    for key, name in names.items():
        (folder / name).write_text("{}", encoding="utf-8")
        if key != "fresh":
            os.utime(folder / name, (old, old))
    (folder / "qrstuvwx56.json").mkdir()
    os.utime(folder / "qrstuvwx56.json", (old, old))
    assert hashing.handover_path("ijklmnop34").parent == folder
    before = hashing.snapshot()["handover"]["pruned"]
    assert hashing.prune_handovers() == 2
    assert hashing.snapshot()["handover"]["pruned"] - before == 2
    assert sorted(p.name for p in folder.iterdir()) == sorted(["ijklmnop34.json", "notes.txt", "abc.json",
                                                               "qrstuvwx56.json"])
    assert hashing.prune_handovers(max_age_seconds=-1) == 1     # the fresh one, once old enough


def test_scope_report_describes_the_innermost_scope_and_is_noted(ntfs):
    path = make(ntfs / "r.bin", os.urandom(40_000))
    assert hashing.scope_report() is None
    with hashing.run_scope("outer", mode="shadow"):
        hashing.sha256_file(path, source="t")                  # a real read, recorded
        with hashing.run_scope("inner"):
            hashing.sha256_file(path, source="t")              # a hit, verified by a real read
            inner = hashing.scope_report()
        outer = hashing.scope_report()
    assert (inner["label"], inner["ledger"], inner["mode"]) == ("inner", "outer", "shadow")
    assert inner["real"]["files"] == 1 and inner["shadow"]["checks"] == 1 and inner["shadow"]["mismatches"] == 0
    assert inner["saved"]["ledger"]["bytes"] == 40_000
    assert outer["label"] == "outer" and outer["real"]["files"] == 2 and outer["shadow"]["checks"] == 1
    json.loads(json.dumps(outer, allow_nan=False))
    timer = profiling.StageTimer()
    with hashing.run_scope("noted", mode="on"):
        hashing.sha256_file(path, source="t")
        hashing.note_scope_report()
    hashing.note_scope_report()                                  # outside a scope: nothing, no error
    note = timer.finish()["notes"][hashing.NOTE_KEY]
    assert note["label"] == "noted" and note["mode"] == "on" and note["real"]["bytes"] == 40_000


def test_digest_writes_follows_the_scope_mode(monkeypatch):
    assert hashing.digest_writes() is False
    for mode, expected in (("on", True), ("shadow", True), ("off", False)):
        with hashing.run_scope("t", mode=mode):
            assert hashing.digest_writes() is expected
    monkeypatch.setenv(hashing.ENV_LEDGER, "off")
    with hashing.run_scope("t"):
        assert hashing.digest_writes() is False
