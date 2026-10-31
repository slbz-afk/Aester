"""Efficiency P1b-5 at its call sites: renamed finishing stages, fresh read-backs, finite and range copies.

docs/EFFICIENCY_P1B_PREREG_20261001.md (P1b-5).  The four finishing stages
re-key their artifacts when the ``.partial-`` folder is renamed, so the
workflow checkpoint is answered from the ledger; the M TIFF, the masks and the
milky-way region are sealed when they are read back right after their write;
the finite and range count copies are sealed through the guarded ``fresh``
path instead of an unguarded ``register_written``.  Products, records and every
recorded digest stay those of the run without a ledger; shadow mode verifies
every ledger answer.
"""
from pathlib import Path
import hashlib
import json
import os
import shutil
import sys
import threading
import time

import numpy as np
import pytest

from starstack import (color_export as ce, finite_stack as fs, hashing, master_export as me, profiling,
                       stack_range as srange, workflow as wf)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_hashing_colour_finishing import (_chain, _colour_export, _digests, ledger_volume,  # noqa: E402
                                           scoped, tree)
from test_finite_stack import prepared, runtime, run as finite_run  # noqa: E402,F401  (fixtures)
from test_stack_range_v2 import ClipMeanEngine, engine, inputs  # noqa: E402,F401  (fixtures)
from test_workflow import setup, stub_model  # noqa: E402,F401  (fixture)
from test_workflow_regions import bind_child_mask, region_project  # noqa: E402

MODES = (None, "off", "on", "shadow")


def real(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def totals():
    snap = hashing.snapshot()
    return {"real": snap["real"]["bytes"], "checks": snap["shadow"]["checks"],
            "mismatches": snap["shadow"]["mismatches"], "moved": snap["renamed"]["moved"],
            "dropped": snap["renamed"]["dropped"], "sealed": snap["fresh"]["sealed"],
            "refused": snap["fresh"]["refused"]}


def change(before):
    after = totals()
    return {key: after[key] - before[key] for key in before}


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in (hashing.ENV_LEDGER, hashing.ENV_WORKERS):
        monkeypatch.delenv(name, raising=False)
    assert not hashing.snapshot()["active"]
    yield
    assert not hashing.snapshot()["active"]


@pytest.fixture
def slow_racy(monkeypatch):
    """A just-written file stays "racy" for 5 s, so a slow test machine cannot make the seals unnecessary."""
    monkeypatch.setattr(hashing, "RACY_SECONDS", 5.0)


def _read_log(monkeypatch):
    """Record (normalized path, source) of every real read that feeds a digest."""
    reads = []
    original = hashing._read_digest

    def read(path, source, check=None, abort=None):
        digest = original(path, source, check, abort)
        reads.append((os.path.normcase(os.path.abspath(path)), source))
        return digest
    monkeypatch.setattr(hashing, "_read_digest", read)
    return reads


# --------------------------------------------------------------------------- #
# 16. Finishing stages: artifacts answered after the stage renames             #
# --------------------------------------------------------------------------- #

STAGES = ("separation", "reduction", "layers", "milkyway", "photo")


def test_finishing_chain_artifacts_are_answered_after_stage_renames(tmp_path, slow_racy):
    results = {}
    for mode in MODES:
        root = tmp_path / str(mode)
        root.mkdir()
        with scoped(mode):
            out = _chain(root)
            artifacts = [item["path"] for stage in STAGES for item in getattr(out, stage)["artifacts"].values()]
            results_json = sorted({str(Path(path).parent / "result.json") for path in artifacts})
            before = totals()
            digests = hashing.sha256_many(artifacts, source="test.checkpoint")
            artifact_reads = change(before)
            before = totals()
            hashing.sha256_many(results_json, source="test.checkpoint")
            json_reads = change(before)
        assert digests == [real(path) for path in artifacts]
        assert not any(".partial-" in path for path in artifacts)
        results[mode] = _digests(out), artifacts, artifact_reads, json_reads
    reference = results[None][0]
    assert len(reference) >= 20
    for mode, (artifact_digests, *_rest) in results.items():
        assert artifact_digests == reference, mode
    for mode in (None, "off"):
        assert results[mode][2]["real"] == sum(Path(p).stat().st_size for p in results[mode][1])
    if not ledger_volume(tmp_path):
        return
    for mode in ("on", "shadow"):
        _, artifacts, artifact_reads, json_reads = results[mode]
        size = sum(Path(p).stat().st_size for p in artifacts)
        if mode == "on":
            assert artifact_reads["real"] == 0                       # every artifact: a ledger answer
        else:
            assert artifact_reads["checks"] == len(artifacts) and artifact_reads["mismatches"] == 0
            assert artifact_reads["real"] == size                    # the shadow verification reads only
        # result.json is written after the stage hash: its first read is a real read, exactly once.
        results_json = {Path(p).parent / "result.json" for p in artifacts}
        assert json_reads["real"] == sum(path.stat().st_size for path in results_json)


def test_stage_rename_moves_entries_only_inside_a_ledger(tmp_path, slow_racy):
    counts = {}
    for mode in MODES:
        root = tmp_path / str(mode)
        root.mkdir()
        with scoped(mode):
            before = totals()
            _chain(root)
            counts[mode] = change(before)
    assert counts[None]["moved"] == counts["off"]["moved"] == 0
    assert counts[None]["sealed"] == counts["off"]["sealed"] == 0
    if ledger_volume(tmp_path):
        for mode in ("on", "shadow"):
            assert counts[mode]["moved"] >= 20 and counts[mode]["dropped"] == 0, counts[mode]
            assert counts[mode]["mismatches"] == 0 and counts[mode]["refused"] == 0


# --------------------------------------------------------------------------- #
# 17. The M TIFF read-back is sealed; photo_render's source-master read hits   #
# --------------------------------------------------------------------------- #

def test_master_readback_is_sealed_and_photo_render_hits(tmp_path, slow_racy):
    root = tmp_path / "colour"
    runs = {}
    for mode in MODES:
        if root.exists():
            shutil.rmtree(root)
        root.mkdir()
        with scoped(mode):
            before = profiling.hashed_sources().get("photo_render.source_master", {}).get("hashed_bytes", 0)
            saved = profiling.hash_saved_sources().get("photo_render.source_master", {}).get("hash_saved_bytes", 0)
            export, artifacts, record = _colour_export(root)
            read = profiling.hashed_sources().get("photo_render.source_master", {}).get("hashed_bytes", 0) - before
            hit = (profiling.hash_saved_sources().get("photo_render.source_master", {})
                   .get("hash_saved_bytes", 0) - saved)
        master = json.loads((export / me.RECIPE_FILENAME).read_text(encoding="utf-8"))
        assert record["master"]["sha256"] == master["master_sha256"] == real(master["master"])
        assert record["photo"]["tiff_sha256"] == real(record["photo"]["tiff"])
        runs[mode] = tree(root), read, hit, os.stat(master["master"]).st_size
    reference = runs[None][0]
    for mode, (files, *_rest) in runs.items():
        assert [name for name in files if files[name] != reference[name]] == [], mode
    for mode in (None, "off"):
        assert runs[mode][1] == runs[mode][3] and runs[mode][2] == 0      # read for real, as before
    if ledger_volume(tmp_path):
        assert runs["on"][1] == 0 and runs["on"][2] == runs["on"][3]     # a ledger answer, no read
        assert runs["shadow"][2] == runs["shadow"][3]                     # answered and verified


# --------------------------------------------------------------------------- #
# 19. The workflow's masks and milky-way region are read once                  #
# --------------------------------------------------------------------------- #

def _region_workflow(setup, monkeypatch, mode, reads):
    monkeypatch.setenv(hashing.ENV_LEDGER, mode)
    bind_child_mask(setup, monkeypatch)
    stub_model(monkeypatch)
    path, files = region_project(setup)
    record = wf.submit_workflow(setup["inputs"], setup["output"], regions_path=path, input_files=files,
                                milkyway_enabled=True, stars_enabled=True, star_model_path=setup["model"],
                                background_strength=.7, backend_options={"tile": 64, "stride": 32})
    reads.clear()
    done = wf.run_workflow(record["workflow_dir"])
    assert done["state"] == "succeeded", done.get("error")
    return done


@pytest.mark.parametrize("mode", ["off", "on", "shadow"])
def test_masks_and_milkyway_region_are_read_once(setup, monkeypatch, slow_racy, mode):
    reads = _read_log(monkeypatch)
    done = _region_workflow(setup, monkeypatch, mode, reads)
    def of(path):                     # the workflow's own file (the job folder has its own sky_weight.tiff)
        key = os.path.normcase(os.path.abspath(path))
        return [source for name, source in reads if name == key]
    sky = of(done["checkpoints"]["masks"]["data"]["sky_mask"])
    region = of(done["checkpoints"]["milkyway_region"]["data"]["path"])
    note = done["timings"]["notes"]["hash_ledger"]
    assert note["version"] == "aester-hashing-3" and note["shadow"]["mismatches"] == 0
    for item in (done["checkpoints"]["masks"], done["checkpoints"]["milkyway_region"]):
        for entry in item["files"]:
            assert entry["sha256"] == real(entry["path"]) and entry["bytes"] == Path(entry["path"]).stat().st_size
    if mode == "off":
        assert len(sky) > 1 and len(region) > 1                     # as before: re-read by every later check
        return
    probe = Path(done["checkpoints"]["masks"]["data"]["sky_mask"]).parent
    if not ledger_volume(probe):
        return
    if mode == "on":
        assert sky == ["workflow._checkpoint"] and region == ["workflow._checkpoint"]
        # K4 in miniature: the checkpoints read no finishing artifact again (renamed stages carry their
        # entries); only first reads remain (result.json, masks, region, evidence and snapshot JSON).
        finishing = os.sep + "finishing" + os.sep
        again = [name for name, source in reads if source == "workflow._checkpoint" and finishing in name
                 and not name.endswith("result.json")]
        assert again == []
        assert note["renamed"]["moved"] >= 4 and note["renamed"]["dropped"] == 0
    else:                                                         # every later answer verified by a real read
        assert sky[0] == region[0] == "workflow._checkpoint" and len(sky) > 1
    assert note["fresh"]["sealed"] >= 3 and note["fresh"]["refused"] == 0
    assert set(note["miss_bytes"]) == {"no_entry", "stale", "racy", "busy"} and note["largest_misses"]


def test_a_missing_checkpoint_file_fails_with_the_same_error_when_fresh(setup, tmp_path):
    directory = tmp_path / "wf"
    directory.mkdir()
    present = directory / "present.tif"
    present.write_bytes(b"present")
    outcomes = {}
    for mode in (None, "on"):
        for fresh in (False, True):
            with scoped(mode):
                try:
                    wf._checkpoint(directory, "masks", {}, [present, directory / "missing.tiff"], fresh=fresh)
                    outcomes[mode, fresh] = "ok"
                except Exception as exc:  # noqa: BLE001 - type and message are the contract
                    outcomes[mode, fresh] = (type(exc).__name__, str(exc))
    assert outcomes[None, True] == outcomes[None, False] == outcomes["on", True] == outcomes["on", False]
    assert outcomes[None, False][0] == "FileNotFoundError"


# --------------------------------------------------------------------------- #
# 21. Finite and range count copies: fresh, guarded                            #
# --------------------------------------------------------------------------- #

def _rewrite_after_read(monkeypatch, name):
    """Plain same-size rewrite (new modification time) of the first file called ``name``, right after its read."""
    original = hashing._read_digest
    state = {"done": False}

    def read(path, source, check=None, abort=None):
        digest = original(path, source, check, abort)
        if not state["done"] and Path(path).name == name:
            state["done"] = True
            time.sleep(0.02)
            data = Path(path).read_bytes()
            with open(path, "r+b") as handle:
                handle.write(bytes(len(data)))
        return digest
    monkeypatch.setattr(hashing, "_read_digest", read)
    return state


@pytest.mark.parametrize("mode", ["on", "shadow"])
def test_finite_and_range_copies_are_hashed_fresh(prepared, runtime, inputs, engine, slow_racy, mode):
    directory, work = prepared[0], prepared[1]
    for path in work.iterdir():
        os.utime(path, ns=(time.time_ns() - 60 * 10**9,) * 2)
    with hashing.run_scope("t", mode=mode):
        before = totals()
        record = finite_run(prepared, runtime)
        finite = change(before)
        kinds = {name: hashing._current().entries[hashing._norm(work / name)].kind
                 for name in (fs.NVALID, fs.NUSED, fs.VALID)}
        range_directory, range_work, arrays = inputs
        before = totals()
        range_record = srange.run_protected_stack(engine, range_directory, range_work, lambda: None,
                                                  ClipMeanEngine(arrays))
        ranged = change(before)
        range_kind = hashing._current().entries[hashing._norm(range_work / srange.VALID)].kind
    assert record["counts"] == {fs.NVALID: real(work / fs.NVALID), fs.NUSED: real(work / fs.NUSED)}
    assert record["valid_sha256"] == real(work / fs.VALID)
    assert record["decoded_sha256"] == real(directory / record["attempt"] / "finite_master.fit")
    assert range_record["valid_sha256"] == real(range_work / srange.VALID)
    assert kinds == {fs.NVALID: "write", fs.NUSED: "write", fs.VALID: "write"}     # just-copied: sealed
    assert range_kind == "write"
    assert finite["sealed"] == 3 and ranged["sealed"] == 1
    assert finite["refused"] == ranged["refused"] == 0 and finite["mismatches"] == ranged["mismatches"] == 0


@pytest.mark.parametrize("mode", ["on", "shadow"])
def test_a_copy_rewritten_between_read_and_seal_is_not_sealed(prepared, runtime, inputs, engine, monkeypatch,
                                                               slow_racy, mode):
    directory, work = prepared[0], prepared[1]
    finite_state = _rewrite_after_read(monkeypatch, fs.NUSED)
    with hashing.run_scope("t", mode=mode):
        record = finite_run(prepared, runtime)
        assert finite_state["done"]
        assert hashing._current().entries.get(hashing._norm(work / fs.NUSED)) is None    # not sealed
        before = totals()
        assert hashing.sha256_file(work / fs.NUSED, source="t") == real(work / fs.NUSED)
        later = change(before)
    # The record holds the digest of the bytes that were read (as without fresh); the ledger kept nothing.
    assert record["counts"][fs.NUSED] != real(work / fs.NUSED)
    assert later["real"] == (work / fs.NUSED).stat().st_size and later["mismatches"] == 0
    range_directory, range_work, arrays = inputs
    range_state = _rewrite_after_read(monkeypatch, srange.VALID)
    with hashing.run_scope("t", mode=mode):
        try:
            srange.run_protected_stack(engine, range_directory, range_work, lambda: None, ClipMeanEngine(arrays))
        except ValueError:
            pass                     # a later identity check of the job may refuse the changed copy: fine here
        assert range_state["done"]
        assert hashing._current().entries.get(hashing._norm(range_work / srange.VALID)) is None
        before = totals()
        assert hashing.sha256_file(range_work / srange.VALID, source="t") == real(range_work / srange.VALID)
        assert change(before)["real"] == (range_work / srange.VALID).stat().st_size
