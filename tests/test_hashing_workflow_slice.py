"""P1 hashing in the workflow slice (docs/EFFICIENCY_P1_HASHING_PREREG_20260929.md).

workflow / workflow_edit / storage / trail_review_session / regions /
editing_project / mcp_server route their file hashes through
starstack.hashing: run scopes (ledger), parallel checks that fail exactly like
the old serial loops, hash-while-copy, in-place writer invalidation and the
stack job handover. The old serial helpers are kept here as references; every
case must give the same result, digest, exception type and message.
"""
from contextlib import nullcontext
from pathlib import Path
import hashlib
import json
import os
import random
import threading

import numpy as np
import pytest
import tifffile
from astropy.io import fits

from starstack import editing_project, hashing, regions, storage, trail_review_session as review
from starstack import workflow as wf, workflow_edit as edit
from test_workflow import setup, admit  # noqa: F401  (fixture)
from test_trail_default_and_extras import build


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def age(*paths, seconds=30):
    """Move modification times back, so ledger entries are not "racy" (as for files written earlier)."""
    for path in paths:
        st = os.stat(path)
        os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns - seconds * 10**9))


def outcome(function, *args, **kwargs):
    try:
        return ("ok", function(*args, **kwargs))
    except Exception as exc:  # type and message are the contract
        return (type(exc).__name__, str(exc))


def counts():
    snap = hashing.snapshot()
    return {"real": snap["real"]["files"], **{kind: value["files"] for kind, value in snap["saved"].items()},
            "refused_busy": snap["refused"]["busy"], "miss_busy": snap["misses"]["busy"],
            "mismatches": snap["shadow"]["mismatches"]}


def delta(before, after):
    return {key: after[key] - before[key] for key in before}


@pytest.fixture
def parallel(monkeypatch):
    """Use the thread pool even for the small files of these tests."""
    monkeypatch.setattr(hashing, "PARALLEL_MIN_BYTES", 0)
    monkeypatch.setenv(hashing.ENV_WORKERS, "4")


MODES = [None, "on", "shadow", "off"]


def scoped(mode):
    return nullcontext() if mode is None else hashing.run_scope("test.slice", mode=mode)


# --------------------------------------------------------------------------- #
# Old serial helpers (as of HEAD before P1), the references                    #
# --------------------------------------------------------------------------- #

def old_file(path):
    path = Path(path).expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError("需要普通文件：" + str(path))
    return {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}


def old_verify_files(entries):
    for entry in entries:
        path = Path(entry["path"])
        if not path.is_file() or path.stat().st_size != entry["bytes"] or sha(path) != entry["sha256"]:
            raise ValueError("输入或已完成阶段的文件已改变／缺失：" + str(path))


def old_verify_file(path, expected, description):
    if not expected or not Path(path).is_file() or sha(path) != expected:
        raise ValueError(f"{description}缺失、已改变或缺少内容校验")


def old_verify_inputs(inputs):
    for source in inputs:
        old_verify_file(source["path"], source["sha256"], "原始曝光")


def old_verify_delivery_files(root, receipt):
    for item in receipt["files"]:
        name = item["file"]
        if Path(name).name != name:
            raise ValueError("作品记录包含越界路径")
        path = storage._no_links(root / name).resolve(strict=True)
        if path.parent != root:
            raise ValueError("作品记录包含越界文件")
        old_verify_files([{**item, "path": str(path)}])


@pytest.fixture
def files(tmp_path):
    rng = random.Random(20260929)
    paths = []
    for index in range(6):
        path = tmp_path / f"f{index}.bin"
        path.write_bytes(bytes(rng.getrandbits(8) for _ in range(rng.randint(0, 300))))
        paths.append(path)
    folder = tmp_path / "folder"
    folder.mkdir()
    age(*paths)
    return {"paths": paths, "missing": tmp_path / "missing.bin", "folder": folder}


def entry_kinds(files):
    paths = files["paths"]
    good = [{"path": str(p), "sha256": sha(p), "bytes": p.stat().st_size} for p in paths]
    return {
        "good": lambda r: dict(r.choice(good)),
        "digest": lambda r: {**r.choice(good), "sha256": "0" * 64},
        "size": lambda r: (lambda e: {**e, "bytes": e["bytes"] + 1})(r.choice(good)),
        "missing": lambda r: {"path": str(files["missing"]), "sha256": "0" * 64, "bytes": 0},
        "folder": lambda r: {"path": str(files["folder"]), "sha256": "0" * 64, "bytes": 0},
        "no_bytes": lambda r: {k: v for k, v in r.choice(good).items() if k != "bytes"},
        "no_digest": lambda r: {k: v for k, v in r.choice(good).items() if k != "sha256"},
        "no_path": lambda r: {"sha256": "0" * 64, "bytes": 0},
    }


# --------------------------------------------------------------------------- #
# Parallel checks keep the serial result, message and first failing entry      #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("mode", MODES)
def test_verify_files_fails_like_the_serial_loop(files, parallel, mode):
    kinds = entry_kinds(files)
    names = sorted(kinds)
    rng = random.Random(7)
    with scoped(mode):
        for _ in range(150):
            entries = [kinds[rng.choice(names)](rng) for _ in range(rng.randint(1, 7))]
            if rng.random() < .5:            # mostly good entries, so late failures and repeats occur
                entries = [kinds["good"](rng) for _ in range(rng.randint(1, 5))] + entries
            assert outcome(wf._verify_files, entries) == outcome(old_verify_files, entries), entries


@pytest.mark.parametrize("mode", MODES)
def test_files_records_equal_the_serial_file_loop(files, parallel, mode):
    rng = random.Random(11)
    choices = [*map(str, files["paths"]), str(files["missing"]), str(files["folder"]), "refused"]

    def prepare(value):
        if value == "refused":
            raise ValueError("产物不在当前工作流中：refused")
        return value

    with scoped(mode):
        for _ in range(120):
            paths = [rng.choice(choices[:6]) for _ in range(rng.randint(0, 4))] + \
                    [rng.choice(choices) for _ in range(rng.randint(0, 3))]
            expected = outcome(lambda: [old_file(prepare(p)) for p in paths])
            assert outcome(wf._files, paths, prepare=prepare) == expected, paths


@pytest.mark.parametrize("mode", MODES)
def test_review_inputs_fail_like_the_serial_loop(files, parallel, mode):
    paths = files["paths"]
    good = [{"path": str(p), "sha256": sha(p), "size": p.stat().st_size} for p in paths]
    kinds = [lambda r: dict(r.choice(good)), lambda r: {**r.choice(good), "sha256": "f" * 64},
             lambda r: {**r.choice(good), "sha256": ""}, lambda r: {**r.choice(good), "sha256": None},
             lambda r: {"path": str(files["missing"]), "sha256": "a" * 64},
             lambda r: {"path": str(files["folder"]), "sha256": "a" * 64},
             lambda r: {"sha256": "a" * 64}, lambda r: {"path": str(paths[0])}]
    rng = random.Random(5)
    with scoped(mode):
        for _ in range(150):
            inputs = [rng.choice(kinds[:1] * 3 + kinds)(rng) for _ in range(rng.randint(1, 6))]
            assert outcome(review._verify_inputs, inputs) == outcome(old_verify_inputs, inputs), inputs


def test_delivery_verification_reports_the_first_failure_in_order(tmp_path, parallel):
    root = tmp_path / "delivery-1"
    root.mkdir()
    for name in ("a.tif", "b.jpg", "c.json"):
        (root / name).write_bytes(name.encode() * 50)
    good = [{"file": name, "sha256": sha(root / name), "bytes": (root / name).stat().st_size}
            for name in ("a.tif", "b.jpg", "c.json")]
    bad_digest = {**good[1], "sha256": "0" * 64}
    outside = {"file": "sub/x.tif", "sha256": "0" * 64, "bytes": 1}
    missing = {"file": "gone.tif", "sha256": "0" * 64, "bytes": 1}
    cases = [good, [good[0], bad_digest, outside], [good[0], outside, bad_digest], [missing, bad_digest],
             [bad_digest, missing], [good[2], good[0], missing], [good[0], good[1], outside, good[2]]]
    for files_ in cases:
        receipt = {"version": storage.VERSION, "files": files_}
        expected = outcome(old_verify_delivery_files, root.resolve(), receipt)
        with hashing.run_scope("test.delivery", mode="on"):
            actual = outcome(storage.verify_delivery, root, receipt=receipt)
        assert (actual[0], actual[1] if actual[0] != "ok" else None) == \
               (expected[0], expected[1] if expected[0] != "ok" else None), files_


def _remove_during_first_read(monkeypatch, first, victim):
    """While ``first`` is hashed, delete ``victim``; the victim's own read waits for that (see test_hashing)."""
    real_read = hashing._read_digest
    removed = threading.Event()

    def fake_read(path, source, check=None, abort=None):
        name = os.path.basename(os.fspath(path))
        if name == Path(first).name and not removed.is_set():
            os.remove(victim)
            removed.set()
        elif name == Path(victim).name:
            removed.wait(5)
        return real_read(path, source, check, abort)
    monkeypatch.setattr(hashing, "_read_digest", fake_read)


def _serial_with_removal(function, first, victim):
    """Run a former serial helper; ``victim`` goes away while ``first`` is hashed (via the test sha)."""
    global sha
    real = sha

    def hooked(path):
        if Path(path).name == Path(first).name and Path(victim).exists():
            os.remove(victim)
        return real(path)
    sha = hooked
    try:
        return outcome(function)
    finally:
        sha = real


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("helper", ["files", "verify_files", "delivery"])
def test_a_file_removed_while_a_check_runs_fails_like_the_serial_loop(tmp_path, parallel, monkeypatch, mode, helper):
    """Review finding: the up-front checks of a parallel check are re-run for a file whose hash fails."""
    def build(folder):
        folder.mkdir()
        for index in range(4):
            (folder / f"f{index}.bin").write_bytes(bytes([index]) * (30_000 + index))
        age(*folder.iterdir())
        paths = [folder / f"f{index}.bin" for index in range(4)]
        entries = [{"path": str(p), "sha256": sha(p), "bytes": p.stat().st_size} for p in paths]
        receipt = {"version": storage.VERSION,
                   "files": [{"file": p.name, "sha256": e["sha256"], "bytes": e["bytes"]} for p, e in zip(paths, entries)]}
        new = {"files": lambda: wf._files(paths), "verify_files": lambda: wf._verify_files(entries),
               "delivery": lambda: storage.verify_delivery(folder, receipt=receipt)}[helper]
        old = {"files": lambda: [old_file(p) for p in paths], "verify_files": lambda: old_verify_files(entries),
               "delivery": lambda: old_verify_delivery_files(folder.resolve(), receipt)}[helper]
        return paths, new, old

    paths, _, old = build(tmp_path / "run-serial")
    expected = _serial_with_removal(old, paths[0], paths[2])
    paths, new, _ = build(tmp_path / "run-batch")
    _remove_during_first_read(monkeypatch, paths[0], paths[2])
    with scoped(mode):
        actual = outcome(new)

    def same(result):
        return tuple(str(part).replace("run-serial", "run-X").replace("run-batch", "run-X") for part in result)
    assert same(actual) == same(expected)
    assert actual[0] == {"files": "FileNotFoundError", "verify_files": "ValueError", "delivery": "FileNotFoundError"}[helper]


def test_region_sources_report_frames_before_the_reference(tmp_path, parallel):
    frames = []
    for index in range(4):
        path = tmp_path / f"{index}.NEF"
        path.write_bytes(f"frame {index}".encode() * 100)
        frames.append(path)
    linear = tmp_path / "reference.fit"
    fits.writeto(linear, np.ones((3, 16, 20), np.float32))
    age(*frames, linear)
    reference = {"state": "succeeded", "source_path": str(frames[0]), "source_sha256": sha(frames[0]),
                 "linear_path": str(linear), "linear_sha256": sha(linear), "width": 20, "height": 16}
    before = counts()
    project = regions.create_project(reference, [str(p) for p in frames])
    change = delta(before, counts())
    assert [item["sha256"] for item in project["source_binding"]["input_files"]] == [sha(p) for p in frames]
    # create_project hashes the frames, then validates them again: the second pass is the ledger.
    assert change["ledger"] >= len(frames) and change["real"] == len(frames) + 1

    def edited(frame_digests=None, reference_digest=None):
        record = json.loads(json.dumps(project))
        for index, digest in (frame_digests or {}).items():
            record["source_binding"]["input_files"][index]["sha256"] = digest
        if reference_digest:
            record["source_binding"]["reference"]["linear_sha256"] = reference_digest
        return regions._seal(record)

    wrong = "0" * 64
    assert outcome(regions.validate_project, edited({2: wrong}, wrong)) == ("ValueError", "分区绑定的素材已改变")
    assert outcome(regions.validate_project, edited(None, wrong)) == ("ValueError", "分区参考图已改变")
    assert outcome(regions.validate_project, edited({3: wrong})) == ("ValueError", "分区绑定的素材已改变")
    assert outcome(regions.validate_project, edited())[0] == "ok"
    # A missing later frame loses against an earlier mismatch, and wins against a later one.
    frames[2].unlink()
    assert outcome(regions.validate_project, edited({1: wrong})) == ("ValueError", "分区绑定的素材已改变")
    assert outcome(regions.validate_project, edited({3: wrong}))[0] == "FileNotFoundError"


def test_saved_region_project_keeps_its_bytes_and_records_the_written_digest(tmp_path):
    frames = []
    (tmp_path / "raw").mkdir()
    for index in range(2):
        path = tmp_path / "raw" / f"{index}.NEF"
        path.write_bytes(f"frame {index}".encode() * 10)
        frames.append(path)
    linear = tmp_path / "reference.fit"
    fits.writeto(linear, np.ones((3, 16, 20), np.float32))
    reference = {"state": "succeeded", "source_path": str(frames[0]), "source_sha256": sha(frames[0]),
                 "linear_path": str(linear), "linear_sha256": sha(linear), "width": 20, "height": 16}
    project = regions.draw_milkyway(regions.create_project(reference, [str(p) for p in frames]), [.5, .3, .2, .1],
                                    reviewer="审阅者", rationale="手绘银河 ✓")
    with hashing.run_scope("test.regions", mode="shadow"):
        saved = regions.save_project(project, tmp_path / "out")
        path = Path(saved["project_path"])
        before = counts()
        assert regions._sha(path) == sha(path)
        assert delta(before, counts())["write"] == 1
    # The same bytes the text-mode write produced before.
    payload = json.dumps(regions._body(project) | {"revision": project["revision"]},
                         ensure_ascii=False, indent=2, allow_nan=False)
    reference_file = tmp_path / "reference.json"
    reference_file.write_text(payload, encoding="utf-8")
    assert path.read_bytes() == reference_file.read_bytes()
    assert regions.load_project(path)["revision"] == project["revision"]
    # Without a ledger (no scope, or off) the former write_text path runs: same bytes, no write digest.
    for mode in (None, "off"):
        folder = tmp_path / f"out-{mode}"
        with scoped(mode):
            before = hashing.snapshot()["memory"]["calls"]
            again = Path(regions.save_project(project, folder)["project_path"])
            assert hashing.snapshot()["memory"]["calls"] == before
        assert again.read_bytes() == reference_file.read_bytes()


# --------------------------------------------------------------------------- #
# Ledger scopes                                                                #
# --------------------------------------------------------------------------- #

def test_repeated_checks_are_answered_by_the_ledger_only_inside_a_scope(files):
    entries = [{"path": str(p), "sha256": sha(p), "bytes": p.stat().st_size} for p in files["paths"]]
    before = counts()
    wf._verify_files(entries)
    wf._verify_files(entries)
    outside = delta(before, counts())
    assert outside["real"] == 2 * len(entries) and outside["ledger"] == 0
    before = counts()
    with hashing.run_scope("test.scope", mode="on"):
        wf._verify_files(entries)
        wf._verify_files(entries)
        inside = delta(before, counts())
        # The admission form always reads, even when every file is in the ledger.
        wf._verify_files(entries, trust=hashing.TRUST_REAL)
        admission = delta(before, counts())
    assert inside["real"] == len(entries) and inside["ledger"] == len(entries)
    assert admission["real"] == 2 * len(entries)


def test_submit_resume_apply_and_region_calls_open_their_own_scope(setup, monkeypatch, tmp_path):
    labels = {}
    doctor = wf.core.doctor
    monkeypatch.setattr(wf.core, "doctor", lambda p=None: labels.setdefault("submit", hashing.snapshot()["label"])
                        and doctor(p))
    admit(setup)
    assert labels["submit"] == "workflow.submit"
    monkeypatch.setattr(edit, "_apply_body", lambda *a, **k: hashing.snapshot()["label"])
    assert edit._apply_locked(tmp_path, "draft.json", automatic=False) == "workflow_edit.apply"
    with hashing.run_scope("workflow.worker"):   # inside the worker the apply shares the worker's ledger
        assert edit._apply_locked(tmp_path, "draft.json", automatic=True) == "workflow.worker"
    seen = []
    monkeypatch.setattr(wf, "_verify", lambda record: seen.append(hashing.snapshot()["label"]) or
                        (_ for _ in ()).throw(ValueError("stop here")))
    directory = Path(wf.submit_workflow(setup["inputs"], setup["output"], idempotency_key="resume")["workflow_dir"])
    wf._update(directory, state="failed")
    with pytest.raises(ValueError, match="stop here"):
        wf.resume_workflow(directory)
    assert seen[-1] == "workflow.resume"
    for name in ("create_project", "build_masks"):
        assert getattr(regions, name).__wrapped__
    for name in ("save_project", "load_project"):
        assert getattr(editing_project, name).__wrapped__
    assert hashing.snapshot()["active"] is False


def test_mcp_region_operations_share_one_scope(tmp_path, monkeypatch):
    from starstack.mcp_server import WorkspaceTools
    inp, out = tmp_path / "input", tmp_path / "output"
    inp.mkdir(); out.mkdir()
    sources = []
    for index in range(3):
        path = inp / f"{index}.NEF"
        path.write_bytes(f"original {index}".encode() * 100)
        sources.append(str(path))
    directory = out / "reference-audit"
    directory.mkdir()
    linear = directory / "reference.fit"
    fits.writeto(linear, np.ones((3, 16, 20), dtype=np.float32))
    age(*sources, linear)
    reference = {"state": "succeeded", "source_path": sources[0], "source_sha256": sha(sources[0]),
                 "linear_path": str(linear), "linear_sha256": sha(linear),
                 "width": 20, "height": 16, "reference_dir": str(directory)}
    (directory / "reference.json").write_text(json.dumps(reference), encoding="utf8")
    engine = tmp_path / "host-siril.exe"
    engine.write_bytes(b"never run")
    service = WorkspaceTools(inp, out, engine)
    labels = []
    save = regions.save_project
    monkeypatch.setattr(regions, "save_project", lambda *a, **k: labels.append(hashing.snapshot()["label"]) or save(*a, **k))
    before = counts()
    project = service.create_regions(str(directory), sources, scene_mode="sky")
    change = delta(before, counts())
    assert labels == ["mcp.create_regions"]
    # Frames and reference are read once; every later validation of them in the call is a ledger answer.
    assert change["real"] == len(sources) + 1 and change["ledger"] >= 2 * len(sources)
    service.update_regions(project["project_path"], scene_mode="sky")
    assert labels[-1] == "mcp.update_regions"


# --------------------------------------------------------------------------- #
# Hash-while-copy                                                              #
# --------------------------------------------------------------------------- #

def old_storage_copy(source, destination, check):
    source = Path(source).resolve(strict=True)
    frozen = old_file(source)
    with source.open("rb") as incoming, destination.open("xb") as outgoing:
        while True:
            check()
            chunk = incoming.read(8 * 1024 * 1024)
            if not chunk:
                break
            outgoing.write(chunk)
        outgoing.flush()
        os.fsync(outgoing.fileno())
    check()
    copied = old_file(destination)
    if (copied["sha256"], copied["bytes"]) != (frozen["sha256"], frozen["bytes"]):
        raise ValueError("作品复制校验失败")
    old_verify_files([frozen])
    return frozen


@pytest.mark.parametrize("size", [0, 5, (8 << 20) + 3])
def test_storage_copy_matches_the_old_copy_and_its_cancellation_points(tmp_path, size):
    source = tmp_path / "source.tif"
    source.write_bytes(os.urandom(size))
    calls = {"old": 0, "new": 0}

    def check(name):
        def call():
            calls[name] += 1
        return call

    expected = old_storage_copy(source, tmp_path / "old.tif", check("old"))
    with hashing.run_scope("test.copy", mode="shadow"):
        frozen = storage._copy_verified(source, tmp_path / "new.tif", check("new"))
        # The destination is recorded as written by this process: a later check is a ledger answer.
        before = counts()
        wf._verify_files([{**frozen, "path": str(tmp_path / "new.tif")}])
        assert delta(before, counts())["write"] == 1
    assert frozen == expected and calls["new"] == calls["old"]
    assert (tmp_path / "new.tif").read_bytes() == source.read_bytes()


@pytest.mark.parametrize("mode", MODES)
def test_copies_without_a_ledger_take_the_former_path(tmp_path, monkeypatch, mode):
    """Review finding: AESTER_HASH_LEDGER=off (and no scope) restores lever 2: plain copy, source re-verified."""
    source = tmp_path / "source.tif"
    source.write_bytes(os.urandom(3000))
    age(source)
    used, verified = [], []
    real_copy, real_verify = hashing.copy_with_digest, wf._verify_files
    monkeypatch.setattr(hashing, "copy_with_digest", lambda *a, **k: used.append(1) or real_copy(*a, **k))
    monkeypatch.setattr(wf, "_verify_files", lambda entries, **k: verified.append(len(entries)) or real_verify(entries, **k))
    target = tmp_path / "stage" / "copy.tif"
    target.parent.mkdir()
    with scoped(mode):
        frozen = storage._copy_verified(source, tmp_path / "delivered.tif", lambda: None)
        edit._copy_verified(str(source), target)
    ledger = mode in ("on", "shadow")
    assert (bool(used), verified) == ((True, []) if ledger else (False, [1, 1]))
    assert frozen == old_file(source) and target.read_bytes() == source.read_bytes()
    # The source changes right after the copy: the former path reports it (the source re-verification);
    # hash-while-copy has already proved the bytes copied, so the copy stands (the accepted lever 2 change).
    freeze = wf._file

    def racing(path, **kwargs):
        record = freeze(path, **kwargs)
        if Path(path).name == "raced.tif":
            source.write_bytes(os.urandom(3000))
        return record
    monkeypatch.setattr(wf, "_file", racing)
    with scoped(mode):
        result = outcome(storage._copy_verified, source, tmp_path / "raced.tif", lambda: None)
    if ledger:
        assert result[0] == "ok"
    else:
        assert result[0] == "ValueError" and result[1].startswith("输入或已完成阶段的文件已改变／缺失：")


def test_storage_copy_errors_and_a_later_rewrite(tmp_path, monkeypatch):
    source = tmp_path / "source.tif"
    source.write_bytes(b"original bytes" * 1000)
    existing = tmp_path / "exists.tif"
    existing.write_bytes(b"x")
    with pytest.raises(FileExistsError):
        storage._copy_verified(source, existing, lambda: None)

    def stop():
        raise wf.WorkflowCancelled("用户请求停止")
    with pytest.raises(wf.WorkflowCancelled):
        storage._copy_verified(source, tmp_path / "stopped.tif", stop)
    # The source changes after it was frozen: the copied bytes are not the frozen ones.
    freeze = wf._file

    def racing(path, **kwargs):
        frozen = freeze(path, **kwargs)
        Path(path).write_bytes(b"changed bytes!" * 1000)
        return frozen
    monkeypatch.setattr(wf, "_file", racing)
    with pytest.raises(ValueError, match="^作品复制校验失败$"):
        storage._copy_verified(source, tmp_path / "raced.tif", lambda: None)
    monkeypatch.setattr(wf, "_file", freeze)
    # A same-size rewrite right after the copy is never answered from the ledger.
    with hashing.run_scope("test.copy", mode="on"):
        target = tmp_path / "copied.tif"
        frozen = storage._copy_verified(source, target, lambda: None)
        with target.open("r+b") as handle:
            handle.write(b"broken")
        with pytest.raises(ValueError, match="文件已改变／缺失"):
            wf._verify_files([{**frozen, "path": str(target)}])


def test_review_copy_overwrites_like_copyfile_and_skips_the_source_recheck(tmp_path, monkeypatch):
    source = tmp_path / "master.tif"
    source.write_bytes(b"master" * 500)
    target = tmp_path / "stage" / "master_scene_linear.tif"
    target.parent.mkdir()
    target.write_bytes(b"old content that is longer than nothing")
    with hashing.run_scope("test.copy", mode="on"):
        edit._copy_verified(str(source), target)
    assert target.read_bytes() == source.read_bytes()
    import shutil
    with pytest.raises(shutil.SameFileError):
        edit._copy_verified(str(source), source)
    with pytest.raises(FileNotFoundError):
        edit._copy_verified(str(source), tmp_path / "no-such-folder" / "x.tif")
    freeze = wf._file

    def racing(path, **kwargs):
        frozen = freeze(path, **kwargs)
        Path(path).write_bytes(b"MASTER" * 500)
        return frozen
    monkeypatch.setattr(wf, "_file", racing)
    with pytest.raises(ValueError, match="^复核成片复制校验失败$"):
        edit._copy_verified(str(source), tmp_path / "raced.tif")


# --------------------------------------------------------------------------- #
# In-place writers                                                             #
# --------------------------------------------------------------------------- #

def pair_inputs(tmp_path):
    from starstack import photo_render as pr
    meta = {"roworder": "TOP-DOWN", "scene_linear": False, "display_encoding_applied": True,
            "color_space": "encoded_sRGB"}
    b = np.full((16, 24, 3), 3500, np.uint16)
    s = b.copy()
    s[4:10, 6:9] += np.array([300, 200, 100], np.uint16)
    paths = []
    for name, array in (("base", b), ("source", s), ("finished", b + 1000)):
        path = tmp_path / (name + ".tif")
        tifffile.imwrite(path, array, photometric="rgb", iccprofile=pr.srgb_icc(), metadata=meta)
        paths.append(str(path))
    stage = tmp_path / "edit"
    stage.mkdir()
    return stage, ({"colour": {"photo": {"tiff": paths[0]}}}, {"tiff": paths[1]}, {"tiff": paths[2]})


def meddle_while_writing(monkeypatch, stage):
    """A reader hashes the memory-mapped output after it was created, before its pixels are written.

    The output's modification time is moved back first, so the entry would
    not be racy: only the writer's invalidation keeps it out of the ledger.
    """
    original = edit.add_signed_trails

    def meddling(*arrays, **kwargs):
        final = stage / "final_srgb16.tif"
        age(final)
        hashing.sha256_file(final, source="test.reader")
        return original(*arrays, **kwargs)
    monkeypatch.setattr(edit, "add_signed_trails", meddling)


def test_write_pair_invalidates_its_memory_mapped_outputs_under_shadow(tmp_path, monkeypatch):
    stage, (stack, reviewed, finished) = pair_inputs(tmp_path)
    meddle_while_writing(monkeypatch, stage)
    with hashing.run_scope("test.pair", mode="shadow"):
        before = counts()
        edit._write_pair(stage, stack, reviewed, finished)
        change = delta(before, counts())
        final = stage / "final_srgb16.tif"
        assert hashing.sha256_file(final, source="test.after") == sha(final)
    assert change["refused_busy"] + change["miss_busy"] >= 1 and change["mismatches"] == 0


def test_shadow_mode_catches_the_same_memory_map_write_without_invalidation(tmp_path, monkeypatch):
    """Control: without writing() the stale pre-pixel digest would be handed out; shadow mode reports it."""
    stage, (stack, reviewed, finished) = pair_inputs(tmp_path)
    meddle_while_writing(monkeypatch, stage)
    monkeypatch.setattr(hashing, "writing", lambda path: nullcontext())
    with hashing.run_scope("test.pair", mode="shadow"):
        edit._write_pair(stage, stack, reviewed, finished)
        with pytest.raises(hashing.HashLedgerMismatch):
            hashing.sha256_file(stage / "final_srgb16.tif", source="test.after")


# --------------------------------------------------------------------------- #
# Shadow mode: a wrong ledger answer is never swallowed by a fallback           #
# --------------------------------------------------------------------------- #

def test_a_shadow_mismatch_in_the_trail_add_back_fails_the_workflow(setup, monkeypatch):
    def mismatch(directory):
        raise hashing.HashLedgerMismatch(directory, "t", "ledger", "0" * 64, "f" * 64)
    monkeypatch.setattr(edit, "apply_default_trails", mismatch)
    record = admit(setup)
    directory = Path(record["workflow_dir"])
    with pytest.raises(hashing.HashLedgerMismatch):
        wf._automatic_trails(directory)
    done = wf._run_attempt(directory, {**wf._read(directory), "trail_addback_pending": True})
    assert done["state"] == "failed" and done["error"].startswith("HashLedgerMismatch: hash ledger mismatch")
    # Any other failure of the add-back still keeps the base photograph and succeeds.
    monkeypatch.setattr(edit, "apply_default_trails", lambda directory: (_ for _ in ()).throw(RuntimeError("boom")))
    done = wf._automatic_trails(directory)
    assert done["state"] == "succeeded" and "boom" in done["trail_review"]["automatic_addback_error"]


# --------------------------------------------------------------------------- #
# Whole synthetic workflows                                                    #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("mode", ["on", "shadow"])
def test_worker_attempt_uses_one_scope_real_admission_and_the_job_handover(setup, monkeypatch, mode):
    monkeypatch.setenv(hashing.ENV_LEDGER, mode)
    original = setup["submit"]
    seen = {}

    def submit(*args, **kwargs):
        seen["label"] = hashing.snapshot()["label"]
        token = seen["token"] = os.environ.get(wf.HANDOVER_TOKEN_ENV)

        def job_worker():  # a separate process in production: it inherits no scope, has its own ledger
            with hashing.run_scope("job.worker", mode="on"):
                job = seen["job"] = original(*args, **kwargs)
                colour = job["diagnostics"]["color_export"]
                products = [colour["master"]["path"], colour["photo"]["tiff"], colour["photo"]["jpeg"]]
                age(*products)
                for path in products:
                    hashing.sha256_file(Path(path), source="test.job")
                seen["exported"] = hashing.export_handover(None, job["job_dir"], token=token)
        thread = threading.Thread(target=job_worker)
        thread.start()
        thread.join()
        return seen["job"]
    monkeypatch.setattr(wf.core, "submit_job", submit)
    pruned = []
    monkeypatch.setattr(hashing, "prune_handovers", lambda *a, **k: pruned.append(1) or 0)
    record = admit(setup)
    calls = []
    verify = hashing.verify_entries

    def spy(entries, **kwargs):
        calls.append(kwargs.get("trust", hashing.TRUST_LEDGER))
        return verify(entries, **kwargs)
    monkeypatch.setattr(hashing, "verify_entries", spy)
    before = counts()
    done = wf.run_workflow(record["workflow_dir"])
    change = delta(before, counts())
    assert done["state"] == "succeeded", done.get("error")
    assert seen["label"] == "workflow.worker" and seen["token"] and seen["exported"] >= 3
    assert wf.HANDOVER_TOKEN_ENV not in os.environ
    assert not hashing.handover_path(seen["token"]).exists()
    # master, TIFF and JPEG at the stack checkpoint (and again when it is frozen) came from the handover.
    assert change["handover"] >= 6 and change["mismatches"] == 0
    assert calls[0] == hashing.TRUST_REAL and hashing.TRUST_LEDGER in calls[1:]
    assert hashing.snapshot()["active"] is False
    assert pruned == [1]                                      # old orphaned handovers are pruned per launch
    note = done["timings"]["notes"]["hash_ledger"]
    assert (note["label"], note["mode"]) == ("workflow.worker", mode) and note["saved"]["handover"]["files"] >= 6
    assert note["shadow"]["mismatches"] == 0 and (note["shadow"]["checks"] > 0) == (mode == "shadow")


def test_off_mode_passes_no_handover_token(setup, monkeypatch):
    monkeypatch.setenv(hashing.ENV_LEDGER, "off")
    original = setup["submit"]
    tokens = []
    monkeypatch.setattr(wf.core, "submit_job",
                        lambda *a, **k: tokens.append(os.environ.get(wf.HANDOVER_TOKEN_ENV)) or original(*a, **k))
    pruned = []
    monkeypatch.setattr(hashing, "prune_handovers", lambda *a, **k: pruned.append(1) or 0)
    record = admit(setup)
    before = counts()
    done = wf.run_workflow(record["workflow_dir"])
    change = delta(before, counts())
    assert done["state"] == "succeeded" and tokens == [None] and pruned == []
    assert change["ledger"] == change["handover"] == change["write"] == change["batch"] == 0
    assert done["timings"]["notes"]["hash_ledger"]["mode"] == "off"


def recorded_files(record):
    """Every (path, sha256, bytes) a workflow record binds, where the file still exists."""
    items = []
    for checkpoint in record["checkpoints"].values():
        items += checkpoint["files"]
    items += record.get("artifacts", []) + record.get("base_artifacts", [])
    for delivery in record.get("review_deliveries", []):
        items += delivery["artifacts"]
    items += list(record.get("publication_receipts", {}).values())
    return [item for item in items if Path(item["path"]).is_file()]


@pytest.mark.parametrize("mode", ["on", "shadow"])
def test_full_workflow_records_equal_real_digests_and_match_the_off_run(setup, monkeypatch, tmp_path, mode):
    """Stars, external delivery, meteor add-back and trail layer; recorded digests are the files' digests."""
    age(*setup["inputs"].iterdir(), setup["engine"], setup["model"], setup["mask"])
    runs = {}
    for ledger in ("off", mode):
        monkeypatch.setenv(hashing.ENV_LEDGER, ledger)
        record, _ = build(setup, monkeypatch, works=tmp_path / ("works-" + ledger), layer=True, stars=True,
                          save_starless=True)
        before = counts()
        done = wf.run_workflow(record["workflow_dir"])
        runs[ledger] = (done, delta(before, counts()))
        assert done["state"] == "succeeded", done.get("error")
        items = recorded_files(done)
        assert len(items) > 20
        for item in items:
            assert (sha(item["path"]), Path(item["path"]).stat().st_size) == (item["sha256"], item["bytes"]), item
    off, on = runs["off"][0], runs[mode][0]
    assert runs["off"][1]["ledger"] == 0 and runs[mode][1]["mismatches"] == 0
    assert runs[mode][1]["ledger"] + runs[mode][1]["write"] + runs[mode][1]["batch"] > 0
    # Each run's timings carry its ledger report (H3 evidence), the add-back's own one included.
    for ledger, record in (("off", off), (mode, on)):
        note = record["timings"]["notes"]["hash_ledger"]
        assert (note["label"], note["mode"], note["shadow"]["mismatches"]) == ("workflow.worker", ledger, 0)
        applied = [d["timings"]["notes"]["hash_ledger"] for d in record["review_deliveries"] if d.get("timings")]
        assert applied and all(item["label"] == "workflow_edit.apply" for item in applied)
    assert on["timings"]["notes"]["hash_ledger"]["shadow"]["checks"] > 0 if mode == "shadow" else True

    # Both runs deliver the same pixels (TIFF metadata names each run's own folders, so compare pixels).
    def images(record):
        return {Path(item["path"]).name: item["path"] for item in record["artifacts"]
                if Path(item["path"]).suffix in (".tif", ".jpg")}
    assert images(on).keys() == images(off).keys() and len(images(on)) >= 4
    for name, path in images(on).items():
        other = images(off)[name]
        if name.endswith(".jpg"):
            assert sha(path) == sha(other)
        else:
            assert np.array_equal(tifffile.imread(path), tifffile.imread(other), equal_nan=True), name
    assert [len(d["artifacts"]) for d in on["review_deliveries"]] == [len(d["artifacts"]) for d in off["review_deliveries"]]
