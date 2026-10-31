"""P1c T-5x (a): the editable layer export's 16-row blocks (16 x 4096 tiles) on the frame pool
(docs/EFFICIENCY_P1C_PREREG_20261005.md §3.5, §6.2 A5 (a)-(f)). No cancel point (none before P1c, none added).

The kept reference is the export loop at the P1c base (tests/p1c_finishing_reference.py): the five layers compare as
bytes, the two counts in order. Real size: the F48 fixture's layer pack, made by the serial code.
"""
import json
from pathlib import Path

import numpy as np
import pytest

import p1c_finishing_data as fd
import p1c_site_harness as h
from p1c_finishing_reference import _star_layers_reference
from starstack import star_layer_export as sle

SHAPE = (200, 40)                     # 13 blocks
UNITS = 13
_REFERENCE = {}


def _inputs(seed, paired):
    stack = fd.layer_stack(seed, SHAPE, paired=paired)
    writers = [np.zeros(stack["shape"], np.uint16) for _ in range(5)]
    return (stack["s"], stack["b"], stack["d"], stack["r"], stack["t"], stack["sky"], stack["protect"], paired,
            writers, stack["shape"]), writers


def reference(seed, paired=True):
    if (seed, paired) not in _REFERENCE:
        args, writers = _inputs(seed, paired)
        counts = _star_layers_reference(*args)
        _REFERENCE[(seed, paired)] = (writers, counts)
    return _REFERENCE[(seed, paired)]


def pooled(seed, paired=True, *, hook=None):
    args, writers = _inputs(seed, paired)
    if hook is not None:
        hook(args)
    counts = sle._export_rows(*args)
    return writers, counts


def reference_target(seed, paired):
    return reference(seed, paired)


def pooled_target(seed, paired):
    with fd.forced(h.Case(8, None, "reuse", 0)):
        return pooled(seed, paired)


def test_the_synthetic_layers_have_negative_stars_and_trails():
    _, counts = reference(0, True)
    assert counts[0] > 0 and counts[1] > 0


@pytest.mark.parametrize("case", h.params(h.synthetic_cases()))
def test_pool_matches_the_reference(case):
    before = h.aester_threads()
    with fd.notes() as seen, fd.forced(case):
        actual = pooled(case.seed)
    h.assert_same(reference(case.seed), actual)
    entry = fd.pool_entries(seen, "star_layers")
    assert len(entry) == 1 and entry[0]["units"] == UNITS and entry[0]["workers"] == min(case.workers, UNITS)
    assert entry[0]["unit_bytes"] == sle.layers_unit_bytes(SHAPE[1])
    h.assert_pool_quiet(before)


@pytest.mark.parametrize("workers", [1, 3, 16])
def test_unpaired_layers_match_the_reference(workers):
    with fd.forced(h.Case(workers, None, "fresh", workers)):
        h.assert_same(reference(1, False), pooled(1, False))


def test_reruns_under_two_hash_seeds_give_one_digest():
    digest = h.assert_hashseed_stable("test_p1c_site_star_layers:reference_target",
                                      "test_p1c_site_star_layers:pooled_target", 2, True)
    assert digest == h.canonical_digest(reference(2, True))


@pytest.mark.parametrize("case", h.params(h.real_cases()))
def test_real_size_publishes_the_f48_layers(tmp_path, case):
    f48 = fd.F48()
    published = f48.record(f48.layers)
    before = h.aester_threads()
    with fd.notes() as seen, fd.forced(case):
        record = sle.export_star_layers(f48.separation, tmp_path / "layers")
    folder = Path(record["processing_directory"])
    for name in sle.LAYER_NAMES:
        assert (folder / name).read_bytes() == (f48.layers / name).read_bytes(), name
    fd.assert_record_like({k: v for k, v in published.items() if k != "artifacts"},
                          {k: v for k, v in record.items() if k != "artifacts"})
    assert list(record["artifacts"]) == list(published["artifacts"])
    assert all(record["artifacts"][n]["sha256"] == item["sha256"] for n, item in published["artifacts"].items())
    entry = fd.pool_entries(seen, "star_layers")[0]
    assert entry["units"] == 345 and entry["workers"] == (case.workers or entry["requested"])
    h.assert_pool_quiet(before)


def _forbidden_change(k):
    def hook(args):
        s, b, d, r, t, sky, protect = args[:7]
        y = k * 16 + 2
        sky[y, 30] = .5                                   # not pure sky, yet the diffuse layer differs there
        d[y, 30] = np.uint16(int(b[y, 30, 0]) - 7) if b[y, 30, 0] > 7 else d[y, 30]
        d[y, 30, 1:] = b[y, 30, 1:]
        r[y, 30] = b[y, 30].astype(np.int32) - d[y, 30].astype(np.int32)
    return hook


@pytest.mark.parametrize("k", h.failure_positions(UNITS))
@pytest.mark.parametrize("workers", [2, 4, 8, 16])
def test_a_failing_block_ends_like_the_serial_loop(k, workers):
    before = h.aester_threads()

    def run(case):
        def body(delivered):
            with fd.forced(case, delivered=delivered):
                return pooled(4, hook=_forbidden_change(k))
        return h.outcome(body)

    serial = run(h.Case(1, None, "reuse", None))
    parallel = run(h.Case(workers, None, "reuse", k))
    assert serial.error_type is ValueError and serial.message == "diffuse layer changed outside its permitted scope"
    assert serial.delivered == list(range(k))
    h.assert_like_serial(serial, parallel)
    h.assert_pool_quiet(before)


@pytest.mark.parametrize("workers", [4, 16])
def test_a_failed_export_leaves_the_serial_folder(tmp_path, monkeypatch, workers):
    """export_star_layers' failed attempt: the same record and the same staging folder as the serial loop. The pool
    may have written rows past the failing block into the five staged maps (§5 item 2); the staged pixels are dropped
    once the maps are closed (§3.5), so either folder holds its failed result.json only."""
    stack = fd.layer_stack(5, (600, 40), paired=True)
    result = fd.make_separation(tmp_path / "sep", stack)
    rows = stack["d"][20 * 16:21 * 16]
    original = sle.compose_layer_codes

    def compose(diffuse, *args):
        if diffuse.shape == rows.shape and np.array_equal(diffuse, rows):     # block 20, identified by content
            raise ValueError("injected layer failure in block 20")
        return original(diffuse, *args)

    monkeypatch.setattr(sle, "compose_layer_codes", compose)
    states = {}
    for label, count in (("serial", 1), ("parallel", workers)):
        with fd.forced(h.Case(count, None, "reuse", 2)), pytest.raises(ValueError) as error:
            sle.export_star_layers(result, tmp_path / label)
        assert str(error.value) == "injected layer failure in block 20"
        stage = next((tmp_path / label).iterdir())
        assert stage.name.startswith(".partial-star-layers-")
        assert sorted(p.name for p in stage.rglob("*")) == ["result.json"]
        recorded = json.loads((stage / "result.json").read_text(encoding="utf-8"))
        assert recorded["state"] == "failed" and recorded["error"] == "injected layer failure in block 20"
        states[label] = {k: v for k, v in recorded.items() if k not in ("seconds", "processing_directory")}
    assert states["serial"] == states["parallel"]


def test_the_pool_peak_stays_within_its_declared_bound(tmp_path, monkeypatch):
    f48 = fd.F48()
    original, rises = sle._export_rows, []

    def measured(*args):
        with h.PeakSampler(trace=False) as peak:       # the layer maps are file-backed: only the units' memory rises
            result = original(*args)
        rises.append(peak.rise)
        return result

    monkeypatch.setattr(sle, "_export_rows", measured)
    with fd.notes() as seen, fd.forced(h.Case(8, 16, "reuse", 0), head_last=True, head_seconds=.5):
        record = sle.export_star_layers(f48.separation, tmp_path / "layers")
    assert (Path(record["processing_directory"]) / "00_background.tif").read_bytes() == \
        (f48.layers / "00_background.tif").read_bytes()
    entry = fd.pool_entries(seen, "star_layers")[0]
    limit = h.pool_memory_limit(entry, blas=False)
    assert entry["workers"] == 8 and entry["unit_bytes"] == sle.LAYERS_UNIT_BYTES
    if rises[0] is not None:
        assert rises[0] <= limit, (rises[0], limit)
    print(f"star_layers: rise {rises[0]} limit {limit}")
