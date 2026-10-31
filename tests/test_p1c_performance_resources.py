"""Efficiency P1c-0 (docs/EFFICIENCY_P1C_PREREG_20261005.md §2.2): accumulated pool notes (R19), the governor's rows
for the P1c pool sites, and the job note's 8 KiB cap with every new pool decision in it."""
import json

import pytest

from starstack import frame_pool, performance, profiling, resources

GB, MB, MiB, GiB = 10 ** 9, 10 ** 6, 1 << 20, 1 << 30
PIXELS = resources.REFERENCE_PIXELS
P1C_ROWS = ("py.median", "py.sky_zero", "py.native_mask", "py.strict_exports", "py.finite_write", "py.finite_inventory",
            "py.trail_crops", "py.milkyway_render", "py.star_reduction", "py.trail_pair", "py.addback_boxes")
LAPTOP = "total=15.4G,avail=9G,lps=16,cores=8,siril_pool=16,commit=24G,commit_avail=14G,io=nvme"


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    for name in (resources.MODE_ENV, resources.FACTS_ENV, "AESTER_FRAME_WORKERS", "STARSTACK_TRANSIENT_WORKERS",
                 "AESTER_RENDER_WORKERS", "AESTER_HASH_WORKERS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(resources, "_scope", None)
    yield
    resources._scope = None


# ================================================================== R19: accumulated notes

def test_note_append_accumulates_one_entry_per_call_in_every_live_timer():
    a, b = profiling.StageTimer(), profiling.StageTimer()
    value = {"workers": 4, "bad": float("inf")}
    profiling.note_append("frame_pool.demo", value)
    value["workers"] = 99                                          # a copy was stored
    profiling.note_append("frame_pool.demo", {"workers": 1})
    profiling.note("frame_pool.coverage", {"workers": 8})          # plain notes still overwrite
    profiling.note("frame_pool.coverage", {"workers": 6})
    ra, rb = a.finish(), b.finish()
    for report in (ra, rb):
        assert report["notes"]["frame_pool.demo"] == [{"workers": 4, "bad": None}, {"workers": 1}]
        assert report["notes"]["frame_pool.coverage"] == {"workers": 6}
    profiling.note_append("frame_pool.demo", {"late": True})        # after finish(): ignored
    assert a.finish()["notes"]["frame_pool.demo"] == ra["notes"]["frame_pool.demo"]


def test_note_append_is_bounded_and_counts_what_it_drops():
    timer = profiling.StageTimer()
    for index in range(7):
        profiling.note_append("frame_pool.many", {"call": index}, limit=5)
    notes = timer.finish()["notes"]
    assert [entry["call"] for entry in notes["frame_pool.many"]] == list(range(5))
    assert notes["frame_pool.many.dropped"] == 2
    assert profiling.NOTE_LIST_LIMIT == 64


def test_note_append_never_raises_and_needs_no_timer():
    profiling.note_append("frame_pool.none", {"x": 1})             # no live timer: a no-op
    timer = profiling.StageTimer()
    profiling.note_append("frame_pool.odd", object())               # stored as text, like note()
    profiling.note_append("frame_pool.odd", {"x": 1}, limit="many")  # a bad limit: dropped, never raised
    entries = timer.finish()["notes"]["frame_pool.odd"]
    assert len(entries) == 1 and isinstance(entries[0], str)


def test_note_pool_records_each_run_of_a_site_under_timings():
    timer = profiling.StageTimer()
    for workers in (1, 3):
        pool_stats = {}
        frame_pool.run_ordered(lambda x, u: x, range(6), workers=workers, name="t-note", stats=pool_stats,
                               unit_bytes=MiB)
        performance.note_pool("demo", pool_stats)
    report = timer.finish()
    entries = report["notes"]["frame_pool.demo"]
    assert [entry["workers"] for entry in entries] == [1, 3] and [entry["serial"] for entry in entries] == [True, False]
    for entry in entries:
        assert entry["unit_bytes"] == MiB and "cpu_user_seconds" in entry and "max_unit_seconds" in entry
        assert "fp_env_applied" in entry
    assert set(report) >= {"notes"} and "frame_pool.demo" not in json.dumps({k: v for k, v in report.items()
                                                                              if k != "notes"})


# ================================================================== resources rows

def _one(stage, **kwargs):
    fp = resources.FOOTPRINTS[stage]
    one = resources.footprint_bytes(fp, 1, **kwargs)["commit"]
    zero = resources.footprint_bytes(fp, 0, **kwargs)["commit"]
    return one[0] - zero[0], one[1] - zero[1]


def test_the_p1c_rows_exist_with_their_provisional_unit_sizes():
    assert resources.FOOTPRINTS_VERSION == "footprints-2026-10-05a"
    for stage in P1C_ROWS:
        fp = resources.FOOTPRINTS[stage]
        assert fp.pool == "frame" and fp.override_env == "AESTER_FRAME_WORKERS", stage
        assert fp.provenance.startswith(("est", "measured")), stage   # P1c sites replace "est" once measured
    assert _one("py.median", frames=48, width=8280) == pytest.approx((60 * MB, 80 * MB), abs=2)
    assert _one("py.median", frames=24, width=8280)[0] == pytest.approx(30 * MB, abs=2)       # n x W
    assert _one("py.median", frames=48, width=4140)[0] == pytest.approx(30 * MB, abs=2)
    assert _one("py.median")[0] == pytest.approx(60 * MB, abs=2)                              # 48 x 8280 default
    assert _one("py.sky_zero", width=8280) == (256 * 8280 * 16, 256 * 8280 * 30)              # 34 / 63.6 MB
    assert _one("py.native_mask", width=8280) == pytest.approx((190 * MiB, 200 * MiB), abs=2)      # G-4t measured
    assert _one("py.strict_exports", frame_pixels=PIXELS) == pytest.approx((226 * MiB, 256 * MiB), abs=2)  # F-2
    assert _one("py.finite_write", frame_pixels=PIXELS) == pytest.approx((0.68 * GB, 0.68 * GB), abs=2)
    assert _one("py.finite_inventory", frame_pixels=PIXELS)[0] == pytest.approx(2 * PIXELS + 8 * MiB, abs=2)
    assert _one("py.trail_crops") == (32 * MiB, 32 * MiB)
    assert _one("py.milkyway_render", width=8280)[0] == pytest.approx(130 * MB, abs=2)
    assert _one("py.star_reduction", width=8280) == pytest.approx((45 * MB, 60 * MB), abs=2)
    assert _one("py.trail_pair", width=8280)[0] == pytest.approx(210 * MB, abs=2)
    caps = {stage: resources.FOOTPRINTS[stage].cap for stage in P1C_ROWS}
    assert caps == {"py.median": 4, "py.sky_zero": 4, "py.native_mask": 4, "py.strict_exports": 4,
                    "py.finite_write": 4, "py.finite_inventory": 4, "py.trail_crops": 8, "py.milkyway_render": 8,
                    "py.star_reduction": 8, "py.trail_pair": 8, "py.addback_boxes": 2}
    for stage in ("py.native_mask", "py.strict_exports"):    # F-2, G-4t: units measured, caps open (idle sweeps, §3.2)
        assert "cap 4 [provisional" in resources.FOOTPRINTS[stage].provenance, stage


def test_convert_phase_no_longer_counts_the_two_write_copies():
    fp = resources.FOOTPRINTS["py.convert_phase"]
    stage = resources.footprint_bytes(fp, 0, frame_pixels=PIXELS)["commit"][0]
    assert stage / GB == pytest.approx(1.88 - 1.04, abs=0.005)


def test_the_median_row_scales_through_shadow_decisions(monkeypatch):
    facts = resources.parse_whatif("total=16G,avail=3G,lps=16,cores=8,commit=24G,commit_avail=6G")
    budget = resources.budget(facts, resident={"ws": 0, "commit": 0})
    small = resources.pool_workers("py.median", items=87, facts=facts, budget=budget, frames=6, width=8280)
    large = resources.pool_workers("py.median", items=87, facts=facts, budget=budget, frames=400, width=8280)
    assert small.predicted["commit"][0] < large.predicted["commit"][0]
    assert small.workers >= large.workers


@pytest.mark.parametrize("env, expected", [
    ({"STARSTACK_TRANSIENT_WORKERS": "3"}, (3, None)),
    ({"STARSTACK_TRANSIENT_WORKERS": "3", "AESTER_FRAME_WORKERS": "5"}, (3, None)),
    ({"AESTER_FRAME_WORKERS": "5"}, (5, None)),
    ({"STARSTACK_TRANSIENT_WORKERS": "auto", "AESTER_FRAME_WORKERS": "2"}, (2, None)),
    ({"AESTER_FRAME_WORKERS": "max"}, ("max", None)),
    ({"STARSTACK_TRANSIENT_WORKERS": "0", "AESTER_FRAME_WORKERS": "5"}, (None, "STARSTACK_TRANSIENT_WORKERS")),
    ({}, (None, None))])
def test_transient_scan_overrides_follow_the_t2_order(monkeypatch, env, expected):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    workers, error = resources._override(resources.FOOTPRINTS["py.transient_scan"])
    want_workers, want_error = expected
    if want_workers == "max":            # T-2: the scan's own maximum, max(floor, ceiling), not the frame pool's 16
        want_workers = max(resources.scan_process_bounds(resources.FOOTPRINTS["py.transient_scan"].cap))
    assert workers == want_workers
    assert (error is None) if want_error is None else (want_error in error)


# ================================================================== the job note's 8 KiB cap

def stored_bytes(note, depth):
    """As tests/test_resources_shadow.py: measured by nesting the note ``depth`` deep, indent 2, CRLF."""
    def text(value):
        for level in range(depth):
            value = {f"level{level}": value}
        return json.dumps(value, ensure_ascii=False, indent=2).replace("\n", "\r\n").encode("utf-8")
    return len(text(note)) - len(text(None)) + len("null")


LAUNCHES = {"convert": ["cd stage", "convertraw light -debayer -out=../work"],
            "probe": ["cd work", "register light -2pass"],
            "export": ["cd work", "seqapplyreg light -filter-included"],
            "stack": ["cd work", "stack r_light rej w 3 3 -norm=addscale -32b"],
            "finite": ["cd work", "stack q rej w 3 3 -nan_missing_v1"]}
# an F job's job-process pools after P1c (the T-5 pools run in the workflow process, under the workflow scope)
F_JOB_POOLS = ("py.median", "py.sky_zero", "py.native_mask", "py.strict_exports", "py.finite_write",
               "py.finite_inventory", "py.trail_crops", "py.addback_boxes", "py.transient_scan")
F_JOB = (F_JOB_POOLS, ("convert", "probe", "export", "export", "finite"))
EVERYTHING = (P1C_ROWS + ("py.coverage", "py.transient_scan"), ("convert", "probe", "export", "stack", "export", "finite"))


@pytest.mark.parametrize("mode", ["shadow", "shadow-full"])
@pytest.mark.parametrize("job", ["f", "everything"])
def test_a_job_note_with_the_p1c_pools_stays_within_8_kib(monkeypatch, tmp_path, mode, job):
    """Prereg §2.2: new pool decisions may push the job note over 8 KiB; the compact form (P1b), then trimmed pool
    entries (P1c-0 step 4b), then the oldest decisions keep it within the cap. An F job at the product default
    (shadow) keeps every Siril decision; actual / recommended / whatif of every pool always stay (M6)."""
    stages, launches = F_JOB if job == "f" else EVERYTHING
    monkeypatch.setenv(resources.MODE_ENV, mode)
    monkeypatch.setenv(resources.FACTS_ENV, LAPTOP)
    monkeypatch.setattr(resources, "aester_resident", lambda **_: {"ws": 10 ** 9, "commit": 2 * 10 ** 9, "members": 3})
    timer = profiling.StageTimer()
    resources.shadow_start(resources.JOB_SCOPE, timer=timer, job={"inputs": [{"path": "C:\\x.dng"}] * 48,
                                                                  "disk_estimate": {"frame_pixels": PIXELS}})
    names = [f"{index:02d}_{kind}" for index, kind in enumerate(launches, 1)]
    for name, kind in zip(names, launches):
        resources.shadow_siril(name, tmp_path, LAUNCHES[kind], "siril")
    for stage in stages:
        resources.shadow_pool(stage, actual=4, items=48, width=8280)
    resources.shadow_pool("py.median", actual=4, items=87, width=8280, frames=48)     # a second call aggregates
    note = timer.finish()["notes"]["governor"]
    depth = resources.NOTE_DEPTH[resources.JOB_SCOPE]
    assert stored_bytes(note, depth) == note["bytes"] <= resources.JOB_NOTE_CAP_BYTES
    kept = [d["name"] for d in note["decisions"]]
    assert note["dropped"] + len(kept) == len(names) and kept == names[len(names) - len(kept):]   # the newest kept
    if (job, mode) == ("f", "shadow"):
        assert note["dropped"] == 0
    pools = {stage: resources.note_view(entry) for stage, entry in note["pools"].items()}
    assert set(pools) >= set(stages)
    for stage in stages:
        assert {"actual", "recommended", "whatif"} <= set(pools[stage]), (stage, pools[stage])
    assert pools["py.median"]["calls"] == 2 and note["footprints"] == resources.FOOTPRINTS_VERSION
    print(f"{job} {mode}: {note['bytes']} B at depth {depth}, compact={note.get('compact', False)}, "
          f"pools_trimmed={note.get('pools_trimmed', False)}, dropped={note['dropped']}")


def test_pool_trimming_keeps_p1b_sized_notes_untouched(monkeypatch, tmp_path):
    """A P1b-sized job note (three pools) fits before step 4b: no pools_trimmed, every pool field kept."""
    monkeypatch.setenv(resources.MODE_ENV, "shadow")
    timer = profiling.StageTimer()
    resources.shadow_start(resources.JOB_SCOPE, timer=timer, job={"inputs": [{"path": "C:\\x.dng"}] * 48,
                                                                  "disk_estimate": {"frame_pixels": PIXELS}})
    for kind in ("convert", "export", "stack"):
        resources.shadow_siril(kind, tmp_path, LAUNCHES[kind], "siril")
    for stage in ("py.coverage", "py.transient_scan"):
        resources.shadow_pool(stage, actual=4, items=48)
    note = timer.finish()["notes"]["governor"]
    assert "pools_trimmed" not in note and note["dropped"] == 0
    for entry in note["pools"].values():
        assert {"calls", "actual", "recommended", "bound_by", "planned_action", "predicted_commit_gb",
                "over_budget"} <= set(resources.note_view(entry))
