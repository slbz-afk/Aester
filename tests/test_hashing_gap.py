"""starstack.hashing, efficiency P1b-5: rename re-key, guarded fresh seal, miss diagnostics.

docs/EFFICIENCY_P1B_PREREG_20261001.md (P1b-5, user decisions U-H1/U-H2).  Two
new recording rules, neither of which may change a digest or the read that
produces it:

* ``hashing.rename`` carries ledger entries to the renamed path only when the
  entry is unchanged, the file there has exactly the recorded stat, and no
  writer holds it;
* ``fresh=`` seals a would-be-racy read-back of a file the caller just wrote,
  guarded against any change between the read and the seal.

Every case checks the digest against the file's bytes; shadow mode verifies
every ledger answer by a real read.  Files are aged unless a test is about a
just-written file.
"""
from pathlib import Path
import hashlib
import os
import random
import shutil
import threading
import time

import pytest

from starstack import hashing, profiling

MB = 1 << 20


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make(path, data, age=60.0):
    """Write ``data``; with ``age`` set the modification time that many seconds in the past."""
    path = Path(path)
    path.write_bytes(data)
    if age:
        stamp = time.time_ns() - int(age * 1e9)
        os.utime(path, ns=(stamp, stamp))
    return path


def mtime(path):
    return os.stat(path).st_mtime_ns


def counters():
    snap = hashing.snapshot()
    return {"real": snap["real"]["bytes"], "real_files": snap["real"]["files"],
            "checks": snap["shadow"]["checks"], "mismatches": snap["shadow"]["mismatches"],
            "moved": snap["renamed"]["moved"], "dropped": snap["renamed"]["dropped"],
            "sealed": snap["fresh"]["sealed"], "fresh_ledger": snap["fresh"]["ledger"],
            "fresh_refused": snap["fresh"]["refused"], "write": snap["recorded"]["write"],
            "ledger": snap["recorded"]["ledger"], "parallel": snap["parallel_batches"],
            "refused_touched": snap["refused"]["touched"], "refused_seal": snap["refused"]["seal"],
            "refused_unstable": snap["refused"]["unstable"],
            **{"miss_" + key: value for key, value in snap["misses"].items()}}


def moved(before):
    after = counters()
    return {key: after[key] - before[key] for key in before}


def entry(path):
    """The current scope's ledger entry for ``path`` (None when there is none)."""
    return hashing._current().entries.get(hashing._norm(path))


def keys_under(folder):
    root = hashing._norm(folder)
    return [key for key in hashing._current().entries if key == root or hashing._under(key, root)]


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
def slow_racy(monkeypatch):
    """A just-written file stays "racy" for 5 s instead of 0.5 s, so a slow test machine cannot age it."""
    monkeypatch.setattr(hashing, "RACY_SECONDS", 5.0)


@pytest.fixture
def stat_hook(monkeypatch):
    """Run a callback around the first hashing._stat of a chosen path (the re-key's stat of the new path)."""
    real = hashing._stat
    armed = {}

    def hooked(path):
        target = armed.get("path")
        if target is not None and hashing._norm(path) == target:
            before, after = armed.pop("before", None), armed.pop("after", None)
            armed.clear()
            if before is not None:
                before()
            result = real(path)
            if after is not None:
                after()
            return result
        return real(path)

    def arm(path, *, before=None, after=None):
        armed.update(path=hashing._norm(path), before=before, after=after)
    monkeypatch.setattr(hashing, "_stat", hooked)
    return arm


def same_size_same_mtime_write(path, value):
    """A writing()-wrapped in-place write that keeps size and modification time (as a memory map does)."""
    st = os.stat(path)
    with hashing.writing(path):
        with open(path, "r+b") as handle:
            handle.write(bytes([value]) * st.st_size)
        os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))


def stage_with_three_files(root):
    """A .partial stage: an aged file, a just-written file and an aged file in a nested folder."""
    stage = root / ".partial-x"
    (stage / "sub").mkdir(parents=True)
    old = make(stage / "layer.tif", os.urandom(64 << 10))
    young = make(stage / "copy.tif", os.urandom(64 << 10), age=0)
    nested = make(stage / "sub" / "n.bin", os.urandom(32 << 10))
    return stage, [old, young, nested], ["layer.tif", "copy.tif", "sub/n.bin"]


# --------------------------------------------------------------------------- #
# P1b-5a hashing.rename                                                        #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("mode", ["on", "shadow"])
def test_rename_carries_entries_of_a_renamed_directory(ntfs, slow_racy, mode):
    stage, files, names = stage_with_three_files(ntfs)
    target = ntfs / "x"
    with hashing.run_scope("t", mode=mode):
        digests = hashing.sha256_many(files, source="t", fresh=True)
        kinds = [entry(path).kind for path in files]
        assert kinds == ["ledger", "write", "ledger"]                  # only the would-be-racy one sealed
        before = counters()
        assert hashing.rename(stage, target) == target
        assert keys_under(stage) == [] and len(keys_under(target)) == 3
        assert [entry(target / name).kind for name in names] == kinds
        again = hashing.sha256_many([target / name for name in names], source="t")
        change = moved(before)
    assert change["moved"] == 3 and change["dropped"] == 0
    assert again == digests == [sha(target / name) for name in names]
    if mode == "on":
        assert change["real"] == 0
    else:
        assert change["checks"] == 3 and change["mismatches"] == 0


@pytest.mark.parametrize("case", ["size-after-rename", "replaced-before-stat", "mtime-before-stat",
                                  "mtime-after-stat"])
@pytest.mark.parametrize("mode", ["on", "shadow"])
def test_rename_drops_an_entry_whose_file_changed(ntfs, stat_hook, mode, case):
    stage = ntfs / ".partial-c"
    stage.mkdir()
    path = make(stage / "m.bin", os.urandom(256 << 10))
    target = ntfs / "C"
    new = target / "m.bin"

    def touch_mtime():
        st = os.stat(new)
        os.utime(new, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))

    def replace_same_size_and_mtime():
        st = os.stat(new)
        other = target / "other.tmp"
        other.write_bytes(os.urandom(st.st_size))
        os.utime(other, ns=(st.st_atime_ns, st.st_mtime_ns))
        with hashing.writing(new):
            os.replace(other, new)

    with hashing.run_scope("t", mode=mode):
        hashing.sha256_file(path, source="t")
        before = counters()
        if case == "replaced-before-stat":
            stat_hook(new, before=replace_same_size_and_mtime)
        elif case == "mtime-before-stat":
            stat_hook(new, before=touch_mtime)
        elif case == "mtime-after-stat":
            stat_hook(new, after=touch_mtime)
        hashing.rename(stage, target)
        if case == "size-after-rename":
            with open(new, "ab") as handle:
                handle.write(b"more")
        rename_change = moved(before)
        digest = hashing.sha256_file(new, source="t")
        change = moved(before)
    assert digest == sha(new)
    assert change["mismatches"] == 0 and change["real"] == os.stat(new).st_size    # a real read, no ledger answer
    if case in ("size-after-rename", "mtime-after-stat"):
        assert (rename_change["moved"], rename_change["dropped"]) == (1, 0)
        assert change["miss_stale"] == 1                                       # moved, then stale at lookup
    else:
        assert (rename_change["moved"], rename_change["dropped"]) == (0, 1)


@pytest.mark.parametrize("mode", ["on", "shadow"])
def test_rename_drops_an_entry_invalidated_between_stat_and_move(ntfs, stat_hook, mode):
    stage = ntfs / ".partial-r"
    stage.mkdir()
    path = make(stage / "m.bin", bytes([7]) * (128 << 10))
    target = ntfs / "R"
    with hashing.run_scope("t", mode=mode):
        hashing.sha256_file(path, source="t")
        before = counters()
        stat_hook(target / "m.bin", after=lambda: same_size_same_mtime_write(target / "m.bin", 8))
        hashing.rename(stage, target)
        rename_change = moved(before)
        digest = hashing.sha256_file(target / "m.bin", source="t")
        change = moved(before)
    assert (rename_change["moved"], rename_change["dropped"]) == (0, 1)
    assert digest == sha(target / "m.bin") == hashlib.sha256(bytes([8]) * (128 << 10)).hexdigest()
    assert change["mismatches"] == 0


def test_rename_does_not_move_a_file_being_written(ntfs):
    stage = ntfs / ".partial-w"
    stage.mkdir()
    path = make(stage / "m.bin", os.urandom(64 << 10))
    target = ntfs / "W"
    with hashing.run_scope("t", mode="on"):
        hashing.sha256_file(path, source="t")
        before = counters()
        with hashing.writing(target / "m.bin"):         # opened for the new path before the rename
            hashing.rename(stage, target)
            assert entry(target / "m.bin") is None
        change = moved(before)
        assert hashing.sha256_file(target / "m.bin", source="t") == sha(target / "m.bin")
        assert moved(before)["real"] == (64 << 10)
    assert (change["moved"], change["dropped"]) == (0, 1)


def test_rename_keeps_racy_and_write_state(ntfs, slow_racy):
    stage = ntfs / ".partial-k"
    stage.mkdir()
    racy = make(stage / "racy.bin", os.urandom(64 << 10), age=0)
    written = stage / "written.bin"
    target = ntfs / "K"
    with hashing.run_scope("t", mode="on"):
        hashing.sha256_file(racy, source="t")                         # a plain read right after a write: racy
        hashing.write_bytes_with_digest(written, os.urandom(32 << 10), source="t")
        racy_entry, written_entry = entry(racy), entry(written)
        assert (racy_entry.kind, racy_entry.exempt) == ("ledger", False)
        assert (written_entry.kind, written_entry.exempt) == ("write", True)
        hashing.rename(stage, target)
        assert entry(target / "racy.bin") is racy_entry and entry(target / "written.bin") is written_entry
        before = counters()
        assert hashing.sha256_file(target / "written.bin", source="t") == sha(target / "written.bin")
        assert moved(before)["real"] == 0                              # the write entry is exempt: a hit
        assert hashing.sha256_file(target / "racy.bin", source="t") == sha(target / "racy.bin")
        change = moved(before)
    assert change["miss_racy"] == 1 and change["real"] == (64 << 10)  # still racy at its new path


def _failure(call):
    try:
        call()
    except Exception as exc:  # noqa: BLE001 - type and message are the contract
        return type(exc), str(exc), exc.errno if isinstance(exc, OSError) else None
    return None


class _nothing:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def in_mode(mode):
    """A new run scope in ``mode``, or no scope for None."""
    return _nothing() if mode is None else hashing.run_scope("t", mode=mode)


@pytest.mark.parametrize("setting", ["outside", "off", "ineligible"])
def test_rename_without_ledger_is_os_rename(ntfs, monkeypatch, setting):
    if setting == "ineligible":
        monkeypatch.setattr(hashing, "_volume_eligible", lambda key, st: False)
    mode = {"outside": None, "off": "off", "ineligible": "on"}[setting]
    stage, files, names = stage_with_three_files(ntfs)
    times = [mtime(path) for path in files]
    before = counters()
    with in_mode(mode):
        digests = hashing.sha256_many(files, source="t", fresh=True)
        assert hashing.snapshot()["entries"] == 0
        assert hashing.rename(stage, ntfs / "x") == ntfs / "x"
        again = hashing.sha256_many([ntfs / "x" / name for name in names], source="t", fresh=True)
    change = moved(before)
    assert digests == again == [sha(ntfs / "x" / name) for name in names]
    assert [mtime(ntfs / "x" / name) for name in names] == times      # no seal, no nudge
    assert change["moved"] == change["dropped"] == change["sealed"] == change["write"] == 0
    assert change["real"] == 2 * sum(os.stat(ntfs / "x" / name).st_size for name in names)
    # Errors: the exception type, text and errno of Path.rename for the same arguments.
    (ntfs / "a").mkdir()
    (ntfs / "b").mkdir()
    (ntfs / "b" / "keep.bin").write_bytes(b"x")
    for source, destination in ((ntfs / "a", ntfs / "b"), (ntfs / "missing", ntfs / "c"),
                                (ntfs / "a" / "missing.bin", ntfs / "b" / "d.bin")):
        with in_mode(mode):
            ours = _failure(lambda: hashing.rename(source, destination))
        assert ours is not None and ours == _failure(lambda: Path(source).rename(destination))


def test_rename_failure_keeps_entries(ntfs):
    stage = ntfs / ".partial-f"
    stage.mkdir()
    path = make(stage / "m.bin", os.urandom(64 << 10))
    (ntfs / "F").mkdir()
    (ntfs / "F" / "other.bin").write_bytes(b"other")
    with hashing.run_scope("t", mode="on"):
        hashing.sha256_file(path, source="t")
        with pytest.raises(FileExistsError):
            hashing.rename(stage, ntfs / "F")
        before = counters()
        assert hashing.sha256_file(path, source="t") == sha(path)
        change = moved(before)
    assert change["real"] == 0 and change["moved"] == change["dropped"] == 0


def test_rename_of_a_single_file(ntfs):
    path = make(ntfs / "a.bin", os.urandom(64 << 10))
    with hashing.run_scope("t", mode="on"):
        digest = hashing.sha256_file(path, source="t")
        before = counters()
        assert hashing.rename(path, ntfs / "b.bin") == ntfs / "b.bin"
        assert hashing.sha256_file(ntfs / "b.bin", source="t") == digest == sha(ntfs / "b.bin")
        change = moved(before)
        assert entry(path) is None
    assert change["real"] == 0 and change["moved"] == 1


def test_rename_reaches_every_live_scope(ntfs):
    stage = ntfs / ".partial-s"
    stage.mkdir()
    path = make(stage / "m.bin", os.urandom(256 << 10))
    target = ntfs / "S"
    hashed, renamed, done = threading.Event(), threading.Event(), {}

    def other_thread():
        with hashing.run_scope("other", mode="on"):
            hashing.sha256_file(path, source="other")
            hashed.set()
            renamed.wait(10)
            before = hashing.snapshot()["saved"]["ledger"]["files"]
            done["digest"] = hashing.sha256_file(target / "m.bin", source="other")
            done["hit"] = hashing.snapshot()["saved"]["ledger"]["files"] - before

    thread = threading.Thread(target=other_thread)
    with hashing.run_scope("main", mode="on"):
        thread.start()
        hashed.wait(10)
        hashing.sha256_file(path, source="main")
        before = counters()
        hashing.rename(stage, target)
        change = moved(before)
        renamed.set()
        thread.join(10)
        assert hashing.sha256_file(target / "m.bin", source="main") == sha(target / "m.bin")
    assert change["moved"] == 2
    assert done["digest"] == sha(target / "m.bin") and done["hit"] == 1


# --------------------------------------------------------------------------- #
# P1b-5b fresh=                                                                #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("mode", ["on", "shadow"])
def test_fresh_seals_only_a_would_be_racy_entry(ntfs, slow_racy, mode):
    young = make(ntfs / "young.tif", os.urandom(MB), age=0)
    old = make(ntfs / "old.tif", os.urandom(MB), age=60)
    young_time, old_time = mtime(young), mtime(old)
    with hashing.run_scope("t", mode=mode):
        before = counters()
        assert hashing.sha256_file(young, source="t", fresh=True) == sha(young)
        assert hashing.sha256_file(old, source="t", fresh=True) == sha(old)
        first = moved(before)
        assert (entry(young).kind, entry(young).exempt) == ("write", True)
        assert (entry(old).kind, entry(old).exempt) == ("ledger", False)
        assert mtime(young) == young_time - hashing.MTIME_NUDGE_NS and mtime(old) == old_time
        before = counters()
        for _ in range(2):                                  # a hit is not sealed again
            assert hashing.sha256_file(young, source="t", fresh=True) == sha(young)
            assert hashing.sha256_file(old, source="t", fresh=True) == sha(old)
        again = moved(before)
    assert (first["sealed"], first["fresh_ledger"], first["fresh_refused"]) == (1, 1, 0)
    assert mtime(young) == young_time - hashing.MTIME_NUDGE_NS and mtime(old) == old_time
    assert again["sealed"] == again["fresh_ledger"] == again["write"] == 0
    if mode == "on":
        assert again["real"] == 0
    else:
        assert again["checks"] == 4 and again["mismatches"] == 0


def _inject_after_read(monkeypatch, path, action):
    """Run ``action`` once right after the real read of ``path`` (before the stat that follows it)."""
    real = hashing._read_digest
    key = hashing._norm(path)
    state = {"armed": True}

    def read(target, source, check=None, abort=None):
        digest = real(target, source, check, abort)
        if state["armed"] and hashing._norm(target) == key:
            state["armed"] = False
            action()
        return digest
    monkeypatch.setattr(hashing, "_read_digest", read)


def _inject_before_seal(monkeypatch, action):
    """Run ``action`` once when the fresh seal's writing block is entered (after the post-read stat)."""
    real = hashing._writing
    state = {"armed": True}

    def wrapped(path, scope, **kwargs):
        if state["armed"] and kwargs.get("since") is not None:
            state["armed"] = False
            action()
        return real(path, scope, **kwargs)
    monkeypatch.setattr(hashing, "_writing", wrapped)


@pytest.mark.parametrize("case", ["rewrite-after-read", "rewrite-before-seal", "writing-before-seal",
                                  "open-writing-during-seal"])
@pytest.mark.parametrize("mode", ["on", "shadow"])
def test_fresh_refuses_to_seal_a_file_changed_after_its_read(ntfs, slow_racy, monkeypatch, mode, case):
    path = make(ntfs / "w.bin", bytes([1]) * MB, age=0)
    original = sha(path)
    release, holding = threading.Event(), threading.Event()
    holder = None

    def plain_rewrite():
        time.sleep(0.02)                                    # past the clock tick: a new modification time
        with open(path, "r+b") as handle:
            handle.write(bytes([2]) * MB)

    def hold_writing():
        def run():
            with hashing.writing(path):
                holding.set()
                release.wait(10)
        nonlocal holder
        holder = threading.Thread(target=run)
        holder.start()
        holding.wait(10)

    if case == "rewrite-after-read":
        _inject_after_read(monkeypatch, path, plain_rewrite)
    elif case == "rewrite-before-seal":
        _inject_before_seal(monkeypatch, plain_rewrite)
    elif case == "writing-before-seal":
        _inject_before_seal(monkeypatch, lambda: same_size_same_mtime_write(path, 3))
    else:
        _inject_before_seal(monkeypatch, hold_writing)
    with hashing.run_scope("t", mode=mode):
        before = counters()
        assert hashing.sha256_file(path, source="t", fresh=True) == original      # the bytes that were read
        change = moved(before)
        assert entry(path) is None                                                 # nothing recorded
        if holder is not None:
            release.set()
            holder.join(10)
        before = counters()
        assert hashing.sha256_file(path, source="t", fresh=True) == sha(path)     # a real read again
        later = moved(before)
    assert (change["sealed"], change["fresh_refused"], change["write"]) == (0, 1, 0)
    assert later["real"] == MB and later["mismatches"] == 0
    if case == "rewrite-after-read":
        assert change["refused_unstable"] == 1
    elif case == "rewrite-before-seal":
        assert change["refused_seal"] == 1
    else:
        assert change["refused_touched"] == 1
    if case == "open-writing-during-seal":
        assert later["sealed"] == 1                         # unchanged file, no writer now: sealed this time


def test_fresh_batch_seals_in_pool_threads(ntfs, slow_racy, monkeypatch):
    monkeypatch.setenv(hashing.ENV_WORKERS, "4")
    files = [make(ntfs / f"p{index}.bin", bytes([index + 1]) * (3 * MB), age=0) for index in range(4)]
    assert sum(os.stat(path).st_size for path in files) >= hashing.PARALLEL_MIN_BYTES
    with hashing.run_scope("t", mode="on"):
        before = counters()
        digests = hashing.sha256_many(files, source="t", fresh=True)
        first = moved(before)
        assert [entry(path).kind for path in files] == ["write"] * 4
        before = counters()
        assert hashing.sha256_many(files, source="t") == digests == [sha(path) for path in files]
        again = moved(before)
    assert first["parallel"] == 1 and first["sealed"] == 4 and first["real"] == 12 * MB
    assert again["real"] == 0


@pytest.mark.parametrize("setting", ["outside", "off"])
def test_fresh_is_plain_hashing_when_off_or_outside_a_scope(ntfs, setting):
    files = [make(ntfs / f"f{index}.bin", os.urandom(64 << 10), age=0) for index in range(3)]
    times = [mtime(path) for path in files]
    results = {}
    with in_mode("off" if setting == "off" else None):
        for fresh in (False, True):
            before = counters()
            single = [hashing.sha256_file(path, source="t", fresh=fresh) for path in files]
            batch = hashing.sha256_many(files, source="t", fresh=fresh)
            results[fresh] = single, batch, moved(before)
    assert results[True][:2] == results[False][:2] == ([sha(p) for p in files], [sha(p) for p in files])
    assert results[True][2] == results[False][2]
    assert results[True][2]["real"] == 2 * sum(os.stat(path).st_size for path in files)
    assert [mtime(path) for path in files] == times
    for mode in (None, "off", "on"):
        with in_mode(mode):
            with pytest.raises(ValueError, match="fresh flags and paths differ in length"):
                hashing.sha256_many(files, source="t", fresh=(True, False))


def _batch_outcome(paths, fresh, *, folder, workers, on_error=None, after_fails=None, cancel_at=None):
    """(result or exception type and text, call log); the folder is masked in the text."""
    log = []

    def after(index):
        log.append(("after", index))
        if index == after_fails:
            raise RuntimeError(f"after {index}")

    calls = {"n": 0}

    def check():
        calls["n"] += 1
        if calls["n"] >= cancel_at:
            raise RuntimeError("cancelled")

    def recheck(index, exc):
        log.append(("on_error", index, type(exc).__name__))
        return on_error(index, exc) if on_error else None

    try:
        hashing.sha256_many(paths, source="t", workers=workers, fresh=fresh, after=after, on_error=recheck,
                            check=check if cancel_at else None)
        result = ("ok",)
    except Exception as exc:  # noqa: BLE001 - type and message are the contract
        result = (type(exc).__name__, str(exc).replace(folder.name, "<folder>"))
    return result, log


@pytest.mark.parametrize("workers", [1, 4])
@pytest.mark.parametrize("mode", [None, "on", "shadow"])
def test_fresh_fails_like_sha256_many(ntfs, monkeypatch, mode, workers):
    """Missing file, on_error replacement, a failing after() and a cancelling check(): same as fresh=False."""
    monkeypatch.setattr(hashing, "PARALLEL_MIN_BYTES", 0)
    for case in ("missing", "on-error", "after", "cancel"):
        outcomes = {}
        for fresh in (False, True):
            folder = ntfs / f"{case}-{fresh}-{mode}-{workers}"
            folder.mkdir()
            paths = [make(folder / f"{index}.bin", bytes([index]) * (64 << 10), age=0) for index in range(5)]
            if case in ("missing", "on-error"):
                paths[2] = folder / "missing.bin"
            on_error = (lambda index, exc: ValueError(f"replaced {index}")) if case == "on-error" else None
            with in_mode(mode):
                outcomes[fresh] = _batch_outcome(paths, fresh, folder=folder, workers=workers, on_error=on_error,
                                                 after_fails=3 if case == "after" else None,
                                                 cancel_at=3 if case == "cancel" else None)
        assert outcomes[True][0] == outcomes[False][0], (case, outcomes)
        assert outcomes[True][0][0] != "ok"
        if workers == 1:                                     # serial: the very same calls in the same order
            assert outcomes[True][1] == outcomes[False][1], (case, outcomes)


# --------------------------------------------------------------------------- #
# P1b-5c diagnostics                                                           #
# --------------------------------------------------------------------------- #

def test_scope_report_counts_miss_bytes_and_largest_misses(ntfs, slow_racy):
    sizes = [1000 + 37 * index for index in range(25)]
    files = [make(ntfs / f"s{index:02d}.bin", os.urandom(size)) for index, size in enumerate(sizes)]
    racy = make(ntfs / "racy.bin", os.urandom(5000), age=0)
    stale = make(ntfs / "stale.bin", os.urandom(7000))
    busy = make(ntfs / "busy.bin", os.urandom(9000))
    other_done = threading.Event()

    def other_scope():
        with hashing.run_scope("other", mode="on"):
            hashing.sha256_file(make(ntfs / "other.bin", os.urandom(50_000)), source="other.source")
            other_done.set()

    timer = profiling.StageTimer()
    with hashing.run_scope("t", mode="on"):
        for path in files:
            hashing.sha256_file(path, source="t.no_entry")                 # 25 no_entry misses
        hashing.sha256_file(racy, source="t.racy")
        hashing.sha256_file(racy, source="t.racy")                         # recorded racy: a racy miss
        hashing.sha256_file(stale, source="t.stale")
        with open(stale, "ab") as handle:
            handle.write(b"x")
        hashing.sha256_file(stale, source="t.stale")                       # a stale miss
        with hashing.writing(busy):
            hashing.sha256_file(busy, source="t.busy")                     # a busy miss
        thread = threading.Thread(target=other_scope)
        thread.start()
        thread.join(10)
        assert other_done.is_set()
        report = hashing.scope_report()
        hashing.note_scope_report()
    note = timer.finish()["notes"][hashing.NOTE_KEY]
    assert report["version"] == hashing.VERSION == "aester-hashing-3"
    # miss bytes are process-wide counters (like every counter of the report): the other scope's miss is in.
    # (The busy file's only miss is "busy": a file being written is not looked up further.)
    assert report["miss_bytes"]["no_entry"] == sum(sizes) + 5000 + 7000 + 50_000
    assert report["miss_bytes"]["racy"] == 5000 and report["miss_bytes"]["stale"] == 7001
    assert report["miss_bytes"]["busy"] == 9000
    largest = report["largest_misses"]
    assert len(largest) == hashing.LARGEST_MISSES
    assert [item[0] for item in largest] == sorted([item[0] for item in largest], reverse=True)
    assert largest[0][:3] == [9000, "busy", "t.busy"] and largest[0][3].endswith("/busy.bin")
    assert all(item[2] != "other.source" for item in largest)            # the other scope's misses stay there
    expected = sorted(sizes + [5000, 5000, 7000, 7001, 9000], reverse=True)[:hashing.LARGEST_MISSES]
    assert [item[0] for item in largest] == expected
    assert note["largest_misses"] == largest and note["miss_bytes"] == report["miss_bytes"]
    assert note["renamed"] == {"moved": 0, "dropped": 0} and set(note["fresh"]) == {"sealed", "ledger", "refused"}


# --------------------------------------------------------------------------- #
# No digest changes: random sequences in every mode, shadow-verified           #
# --------------------------------------------------------------------------- #

def _random_sequence(root, seed, steps=70):
    """Run a seeded random mix of writes, fresh/plain hashes, batch hashes and renames; return every digest."""
    rng = random.Random(seed)
    stage = root / ".partial-0"
    stage.mkdir(parents=True)
    names = [f"f{index}.bin" for index in range(5)]
    sizes = {name: rng.choice((4096, 65536, 200_000)) for name in names}
    for name in names:
        (stage / name).write_bytes(rng.randbytes(sizes[name]))
    out, generation = [], 0
    for step in range(steps):
        name = rng.choice(names)
        path = stage / name
        op = rng.choice(("fresh", "plain", "batch", "batch-fresh", "rewrite", "inplace", "rename", "age"))
        if op == "fresh":
            out.append((step, op, name, hashing.sha256_file(path, source="r", fresh=True)))
        elif op == "plain":
            out.append((step, op, name, hashing.sha256_file(path, source="r")))
        elif op in ("batch", "batch-fresh"):
            flags = [rng.random() < .5 for _ in names] if op == "batch-fresh" else False
            out.append((step, op, "*", tuple(hashing.sha256_many([stage / n for n in names], source="r",
                                                                  fresh=flags))))
        elif op == "rewrite":
            time.sleep(0.02)                    # a plain rewrite after the clock tick (P1's accepted residual
            path.write_bytes(rng.randbytes(sizes[name]))   # risk, same tick + same size, is excluded)
        elif op == "inplace":
            st = os.stat(path)
            with hashing.writing(path):         # a memory-map-like write: same size, same modification time
                with open(path, "r+b") as handle:
                    handle.write(rng.randbytes(st.st_size))
                os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))
        elif op == "rename":
            generation += 1
            target = root / f".partial-{generation}"
            hashing.rename(stage, target)
            stage = target
        else:
            st = os.stat(path)
            os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns - 5 * 10**9))
    for name in names:
        out.append(("end", "check", name, hashing.sha256_file(stage / name, source="r"), sha(stage / name)))
    return out


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_random_fresh_and_rename_sequences_give_the_same_digests_in_every_mode(ntfs, seed):
    runs = {}
    for mode in (None, "off", "on", "shadow"):
        root = ntfs / f"{mode}-{seed}"
        before = counters()
        with in_mode(mode):
            runs[mode] = _random_sequence(root, seed)
        runs[mode] = runs[mode], moved(before)
    reference = runs[None][0]
    for mode, (digests, change) in runs.items():
        assert digests == reference, mode
        assert change["mismatches"] == 0
    for item in reference:
        if item[0] == "end":
            assert item[3] == item[4]
    assert runs["shadow"][1]["checks"] > 0
    for mode in ("on", "shadow"):                        # both new rules were exercised
        assert runs[mode][1]["moved"] > 0 and runs[mode][1]["sealed"] > 0, (mode, runs[mode][1])
    assert runs["off"][1]["moved"] == runs["off"][1]["sealed"] == 0
