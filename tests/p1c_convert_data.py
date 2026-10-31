"""Real data for the efficiency P1c convert-stage site tests (C-1 median, C-3 sky zeroing; prereg §6.2 (b)).

The F48 reference job's 48 stage DNGs are re-converted by the job's own Siril with the job's conversion script, once
per test session (``reconverted(count)``): Siril converts each raw independently, so the first 16 converted frames
are the 16-frame set and the 48 frames the 48-frame set. The F48 job records what the product made of them: the
foreground master's digest (its ``foreground`` checkpoint) and every frame's digest after sky zeroing (``convert``).
The fixture stays read-only: its DNGs are copied, converted frames live in the session's basetemp.

``controlled_job`` is the job-level counterpart of §6.2 (c): a small masked job on a controlled engine whose
convert stage runs both pools, so that a failure injected in the median or the zeroing can be compared between a
serial and a parallel job, and resumed.
"""
from __future__ import annotations

import json
import os
import shutil
import stat
from pathlib import Path

import p1c_site_harness as h

SIRIL_ENV = "STARSTACK_SIRIL"
CONVERT_SCRIPT = ["setext fit", "set32bits", "setcompress 0", "cd stage", "convertraw light -debayer -out=../work"]
FOREGROUND_NOTE = ("Unregistered temporal median of the converted frames, taken before the sky masking; "
                   "fixed tripod only")
_CACHE: dict = {}


def siril_executable() -> str:
    """STARSTACK_SIRIL, else the main checkout's tools/siril (the F48 job's engine); missing: skip (or fail)."""
    value = os.environ.get(SIRIL_ENV, "").strip()
    if value and Path(value).is_file():
        return value
    return str(h.fixture_path("tools", "siril", "bin", "siril-cli.exe", what="Siril 1.4.4 (tools/siril)"))


def f48_record() -> dict:
    """The F48 job's job.json (read-only)."""
    return json.loads((h.f48_job() / "job.json").read_text(encoding="utf-8"))


def foreground_sha256() -> str:
    (entry,) = f48_record()["checkpoint_files"]["foreground"]
    return entry["sha256"]


def zeroed_sha256() -> dict:
    """{light_xxxxx.fit: sha256} of the F48 job's frames after sky zeroing (its convert checkpoint)."""
    return {Path(e["path"].replace("\\", "/")).name: e["sha256"]
            for e in f48_record()["checkpoint_files"]["convert"] if Path(e["path"].replace("\\", "/")).name
            .startswith("light_")}


def validity_record() -> dict:
    return json.loads((h.f48_job() / "work" / "sky_validity.json").read_text(encoding="utf-8"))


def _header_block(path) -> bytes:
    """The FITS header of ``path`` up to the end of the 2880-byte block holding its END card."""
    head = b""
    with open(path, "rb") as stream:
        while True:
            block = stream.read(2880)
            assert len(block) == 2880, path
            head += block
            if any(block[i:i + 80].startswith(b"END     ") or block[i:i + 80] == b"END".ljust(80)
                   for i in range(0, 2880, 80)):
                return head


def sha256_as_f48(path) -> str:
    """SHA-256 of a re-converted frame with the F48 frame's header: a re-conversion changes only the DATE card
    (Siril writes the file's creation time; a timestamp the comparison rules allow, prereg §6). Asserts that every
    other header byte equals the F48 frame's, then hashes the F48 header and this file's data, so the result is
    comparable with the F48 job's convert-checkpoint digest without reading the F48 frame's data."""
    import hashlib
    reference = _header_block(h.f48_job() / "work" / Path(path).name)
    mine = _header_block(path)
    assert len(mine) == len(reference), path
    cards = [(mine[i:i + 80], reference[i:i + 80]) for i in range(0, len(mine), 80)]
    different = [(a, b) for a, b in cards if a != b]
    assert all(a.startswith(b"DATE    = ") and b.startswith(b"DATE    = ") for a, b in different), different
    digest = hashlib.sha256(reference)
    with open(path, "rb") as stream:
        stream.seek(len(mine))
        for chunk in iter(lambda: stream.read(16 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reconverted(tmp_path_factory, count: int = 48) -> list:
    """The first ``count`` unzeroed converted frames (sorted paths) of a session-wide re-conversion of all 48 DNGs.
    Read only: copy them before changing them (copy_frames)."""
    if "frames" not in _CACHE:
        job = h.f48_job()
        exe = siril_executable()
        from starstack import siril
        root = h.assert_short_path(tmp_path_factory.mktemp("cv48"))
        (root / "stage").mkdir()
        (root / "work").mkdir()
        dngs = sorted((job / "stage").glob("frame*.dng"))
        assert len(dngs) == 48, dngs
        for dng in dngs:
            target = root / "stage" / dng.name
            shutil.copyfile(dng, target)
            os.chmod(target, stat.S_IREAD | stat.S_IWRITE)
        siril.run_script(exe, root, "01_convert", CONVERT_SCRIPT, lambda: None)
        frames = sorted((root / "work").glob("light_*.fit"))
        assert len(frames) == 48, frames
        for frame in frames:
            os.chmod(frame, stat.S_IREAD)          # the session's unzeroed originals: never changed in place
        _CACHE["frames"] = frames
    return _CACHE["frames"][:count]


def controlled_job(root: Path, monkeypatch, *, frames: int = 7, height: int = 200, width: int = 23, seed: int = 5):
    """A masked FITS-input stack job on a controlled engine, for the job-level failure tests (prereg §6.2 (c)).

    ``frames`` 16-bit Siril-like RGB exposures (so the median takes its integer route: 4 blocks at height 200) and a
    sky mask with a soft horizon. The engine's conversion copies the staged frames into work, its export writes a
    sequence file and its stack is the frames' mean (as tests/test_hashing_job_slice.py's controlled engine; jobs are
    admitted before the EFF-D4 contract, as tests/conftest.py's pre_contract_jobs). Returns ``run(label)``, which
    submits a job, runs it to its end and returns its job.json, and ``resume(record)``, which resumes a failed job
    and returns its job.json after the new attempt."""
    import numpy as np
    from types import SimpleNamespace

    from PIL import Image
    from starstack import core, imaging, stack_settings
    from test_foreground_median import banded_frames, siril_frame

    source = root / "sources"
    source.mkdir(parents=True)
    values = banded_frames(np.random.default_rng(seed), frames, height, width)
    for i in range(frames):
        siril_frame(source / f"exposure {i:02d}.fit", values[i])
    mask = np.full((height, width), 255, dtype=np.uint8)
    horizon = (height * 0.6 + 12 * np.sin(np.arange(width) / 2.0)).astype(int)
    for x in range(width):
        mask[horizon[x]:horizon[x] + 3, x] = 128
        mask[horizon[x] + 3:, x] = 0
    mask_png = root / "sky_mask.png"
    Image.fromarray(mask).save(mask_png)
    engine = {"available": True, "executable": str(root / "engine.exe"), "version": "1.4.4", "message": "controlled"}
    monkeypatch.setattr(core, "doctor", lambda *_: dict(engine))
    monkeypatch.setattr(core, "_spawn_worker", lambda directory: None)
    monkeypatch.setattr(core.shutil, "disk_usage", lambda _: SimpleNamespace(free=10**12))
    monkeypatch.setattr(stack_settings, "admission_field", lambda *args, **kwargs: None)

    def sequence_text(count, reference):
        lines = ["#Siril sequence file.", f"S 'light_' 1 {count} {count} 5 {reference} 6 0 0 0", "L 3"]
        lines += [f"I {index} 1" for index in range(1, count + 1)]
        for position in range(count):
            shift = 0.0 if position == reference else 1.0 * (position - reference)
            lines.append(f"R1 3.5 3.6 0.7 0 1500 1800 H 1 0 {shift} 0 1 {shift} 0 0 1")
        return "\n".join(lines) + "\n"

    def run_script(executable, directory, name, commands, check):
        check()
        work = directory / "work"
        if name == "01_convert":
            for i, path in enumerate(sorted((directory / "stage").glob("*")), 1):
                shutil.copyfile(path, work / f"light_{i:05d}.fit")
        elif name == "02_export":
            converted = sorted(work.glob("light_*.fit"))
            (work / core.SEQUENCE_FILE).write_text(sequence_text(len(converted), 1), encoding="utf-8")
        elif name == "02_stack":
            converted = sorted(work.glob("light_*.fit"))
            mean = np.mean([imaging.read_rgb(p) for p in converted], axis=0).astype(np.float32)
            imaging.write_linear(work / "sky_master.fit", mean, "Controlled engine, not Siril")
        elif name != "02_probe":
            raise AssertionError(name)
    monkeypatch.setattr(core, "run_script", run_script)

    def run(label):
        job = core.submit_job(str(source), str(root / label), sky_mask_path=str(mask_png), transient_mode="off")
        core.run_job(job["job_dir"])
        return core.load_job(job["job_dir"])

    def resume(record):
        core.resume_job(record["job_dir"])
        core.run_job(record["job_dir"])
        return core.load_job(record["job_dir"])
    return SimpleNamespace(run=run, resume=resume, frames=frames, height=height)


def job_outcome(record) -> dict:
    """What a failed job records, to compare the serial and parallel jobs (prereg §6.2 (c))."""
    return {key: record.get(key) for key in ("state", "stage", "error", "message", "checkpoints")} | {
        "transients": (record.get("diagnostics") or {}).get("transients")}


def failing_jobs(root: Path, monkeypatch, fail: dict, *, workers: int = 4) -> dict:
    """The controlled job uninterrupted with ``workers`` frame-pool threads, then with the injected failure on
    (``fail['on']``, read by the caller's patch) serially (AESTER_FRAME_WORKERS=1) and with ``workers`` threads,
    then both failed jobs resumed with the failure gone (each resume with its own thread count): {'uninterrupted',
    'serial', 'parallel', 'serial-resumed', 'parallel-resumed'} -> job.json, and 'pools': {the same names ->
    [(pool name, requested workers, granted workers or None), ...] of every frame_pool.run_ordered call}, which
    shows the pools a failed job ran (a pool that fails notes nothing in the timings, as the P1b sites)."""
    from starstack import frame_pool
    job = controlled_job(root, monkeypatch)
    out, calls = {"pools": {}}, []
    real = frame_pool.run_ordered

    def spy(fn, items, **kwargs):
        try:
            return real(fn, items, **kwargs)
        finally:
            calls.append((kwargs.get("name"), kwargs.get("workers"), (kwargs.get("stats") or {}).get("workers")))
    monkeypatch.setattr(frame_pool, "run_ordered", spy)

    def attempt(name, count, run):
        monkeypatch.setenv(frame_pool.ENV_WORKERS, str(count))
        calls.clear()
        out[name] = run()
        out["pools"][name] = list(calls)
    fail["on"] = False
    attempt("uninterrupted", workers, lambda: job.run("uninterrupted"))
    fail["on"] = True
    for name, count in (("serial", 1), ("parallel", workers)):
        attempt(name, count, lambda: job.run(name))
    fail["on"] = False
    for name, count in (("serial", 1), ("parallel", workers)):
        attempt(f"{name}-resumed", count, lambda: job.resume(out[name]))
    return out


def pool_workers(record, site: str) -> list:
    """The thread counts of a job's ``frame_pool.<site>`` timings notes (one per call)."""
    notes = ((record.get("diagnostics") or {}).get("timings") or {}).get("notes") or {}
    return [entry["workers"] for entry in notes.get(f"frame_pool.{site}", [])]


def copy_frames(frames, folder: Path) -> list:
    """Writable copies of ``frames`` in ``folder`` (sorted, same names), not flushed: a test meets the write-back of
    what it copied as a job meets the write-back of Siril's conversion."""
    folder.mkdir(parents=True, exist_ok=True)
    copies = []
    for frame in frames:
        target = folder / frame.name
        shutil.copyfile(frame, target)
        os.chmod(target, stat.S_IREAD | stat.S_IWRITE)
        copies.append(target)
    return copies
