"""Governor pool decisions (efficiency P1b-2 §3.5, cross-check X1): overrides, bounds, io classes, frame-pool clamp."""
import pytest

from starstack import frame_pool, resources

GiB = 1024 ** 3
PIXELS = resources.REFERENCE_PIXELS
KNOBS = ("AESTER_HASH_WORKERS", "STARSTACK_TRANSIENT_WORKERS", "AESTER_RENDER_WORKERS", "AESTER_FRAME_WORKERS",
         "AESTER_POOL_WORKERS")


@pytest.fixture(autouse=True)
def no_knobs(monkeypatch):
    for name in KNOBS:
        monkeypatch.delenv(name, raising=False)


def machine(*, avail=200 * GiB, cores=96, lps=192, io="nvme"):
    facts = resources.parse_whatif(f"total={max(avail, 256 * GiB)},avail={avail},lps={lps},cores={cores},"
                                   f"commit={max(avail, 256 * GiB) + 16 * GiB},commit_avail={avail + 8 * GiB},io={io}")
    return facts, resources.budget(facts)


def decide(stage, items, facts_budget, **kwargs):
    facts, budget = facts_budget
    return resources.pool_workers(stage, items=items, facts=facts, budget=budget, **kwargs)


@pytest.mark.parametrize("stage, env, value, expected", [
    ("py.hash", "AESTER_HASH_WORKERS", "3", 3), ("py.hash", "AESTER_HASH_WORKERS", "99", 32),
    ("py.transient_scan", "STARSTACK_TRANSIENT_WORKERS", "2", 2),
    ("py.render_rows", "AESTER_RENDER_WORKERS", "5", 5),
    ("py.coverage", "AESTER_FRAME_WORKERS", "1", 1), ("py.coverage", "AESTER_FRAME_WORKERS", "12", 12),
    ("py.median", "AESTER_FRAME_WORKERS", "7", 7)])
def test_existing_overrides_keep_precedence(monkeypatch, stage, env, value, expected):
    monkeypatch.setenv(env, value)
    decision = decide(stage, 48, machine(), frame_pixels=PIXELS, width=8280)
    assert decision.bound_by == "override" and decision.workers == expected


def test_frame_workers_max_is_the_frame_pool_ceiling(monkeypatch):
    monkeypatch.setenv("AESTER_FRAME_WORKERS", "max")
    decision = decide("py.coverage", 48, machine(), frame_pixels=PIXELS)
    assert decision.workers == min(frame_pool.MAX_FRAME_WORKERS, frame_pool.usable_cpus())
    assert decision.workers <= frame_pool.MAX_FRAME_WORKERS


@pytest.mark.parametrize("env, value", [("AESTER_FRAME_WORKERS", "17"), ("AESTER_FRAME_WORKERS", "many"),
                                        ("AESTER_RENDER_WORKERS", "9"), ("STARSTACK_TRANSIENT_WORKERS", "0"),
                                        ("AESTER_HASH_WORKERS", "x")])
def test_a_malformed_override_is_noted_and_ignored_never_raised(monkeypatch, env, value):
    stage = {"AESTER_FRAME_WORKERS": "py.coverage", "AESTER_RENDER_WORKERS": "py.render_rows",
             "STARSTACK_TRANSIENT_WORKERS": "py.transient_scan", "AESTER_HASH_WORKERS": "py.hash"}[env]
    monkeypatch.setenv(env, value)
    decision = decide(stage, 48, machine(), frame_pixels=PIXELS, width=8280)
    assert decision.bound_by != "override" and "override_error" in decision.basis


def test_aester_pool_workers_is_not_a_knob_in_p1b(monkeypatch):
    """X1: dropped from P1b (AESTER_FRAME_WORKERS is the one verification knob for frame pools)."""
    before = decide("py.coverage", 48, machine(), frame_pixels=PIXELS)
    monkeypatch.setenv("AESTER_POOL_WORKERS", "all=1")
    after = decide("py.coverage", 48, machine(), frame_pixels=PIXELS)
    assert before == after and after.workers == 8


@pytest.mark.parametrize("stage, items, expected_bound, expected", [
    ("py.coverage", 48, "cap", 8), ("py.coverage", 3, "items", 3), ("py.median", 48, "cap", 4),
    ("py.render_rows", 44, "cap", 8), ("py.transient_scan", 48, "cap", 8), ("py.hash", 48, "io", 4)])
def test_bound_by_on_a_roomy_machine(stage, items, expected_bound, expected):
    decision = decide(stage, items, machine(), frame_pixels=PIXELS, width=8280)
    assert (decision.bound_by, decision.workers) == (expected_bound, expected)
    assert decision.planned_action == "run" and not decision.over_budget


def test_cpu_bound_and_gil_is_not_forced_to_one():
    decision = decide("py.coverage", 48, machine(cores=4, lps=8), frame_pixels=PIXELS)
    assert decision.bound_by == "cpu" and decision.workers == 3                 # usable cores - 1
    gil = decide("py.median", 48, machine(), frame_pixels=PIXELS)
    assert resources.FOOTPRINTS["py.median"].kind == "gil" and gil.workers == 4  # X1: the cap, not 1


@pytest.mark.parametrize("avail_gib, workers, bound", [(12, 4, "ram"), (8, 2, "ram"), (5, 1, "ram")])
def test_memory_bounds_the_scan_processes(avail_gib, workers, bound):
    """A 16 GB computer: 1.8 GB a scan process at 45.7 MP (P1c T-2's measured worst case; the row charged P1a's
    typical 0.55 GB before), so at most 4 processes within the 8 GiB allowance."""
    facts = resources.parse_whatif(f"total=16G,avail={avail_gib}G,lps=16,cores=8,commit=24G,commit_avail=20G")
    budget = resources.budget(facts, resident={"ws": 0, "commit": 0})
    decision = resources.pool_workers("py.transient_scan", items=48, frame_pixels=PIXELS, facts=facts, budget=budget)
    assert decision.workers == workers
    assert decision.ram_allowance == budget.ram_allowance()
    if bound == "ram":
        assert decision.bound_by == "ram" and decision.planned_action == "run_reduced"


def test_over_budget_runs_one_and_warns():
    facts = resources.parse_whatif("total=8G,avail=1G,lps=8,cores=4,commit=10G,commit_avail=1G")
    decision = resources.pool_workers("py.transient_scan", items=48, frame_pixels=PIXELS, facts=facts,
                                      budget=resources.budget(facts))
    assert decision.over_budget and decision.workers == 1 and decision.planned_action == "warn_run"


@pytest.mark.parametrize("io, workers", [("nvme", 4), ("ssd", 2), ("hdd", 1), ("network", 1), ("removable", 1),
                                         ("unknown", 2)])
def test_io_stages_follow_the_source_volume_class(io, workers):
    decision = decide("py.hash", 48, machine(io=io))
    assert decision.bound_by == "io" and decision.workers == workers
    finite = decide("py.finite_write", 48, machine(io=io), frame_pixels=PIXELS)
    assert finite.workers == min(workers, 4)


def test_frame_pool_stages_are_clamped_to_the_process_budget():
    for key, fp in resources.FOOTPRINTS.items():
        if fp.pool == "frame":
            decision = decide(key, 500, machine(), frame_pixels=PIXELS)
            assert decision.workers <= frame_pool.MAX_FRAME_WORKERS


def test_no_items_no_workers():
    decision = decide("py.coverage", 0, machine(), frame_pixels=PIXELS)
    assert decision.workers == 0 and not decision.over_budget
