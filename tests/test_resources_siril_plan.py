"""Governor Siril plans (efficiency P1b-2 §3.6-§3.7, A6): routes, masks, record neutrality, today's plan."""
import dataclasses

import pytest

from starstack import core, finite_stack, resources, siril, stack_range

GiB, GB = 1024 ** 3, 10 ** 9
PIXELS = resources.REFERENCE_PIXELS
LAPTOP = "total=15.4G,avail=9G,lps=16,cores=8,siril_pool=16,commit=24G,commit_avail=14G,io=nvme"
CONVERT = ["setext fit", "set32bits", "setcompress 0", "cd stage", "convertraw light -debayer -out=../work"]
PROBE = list(core.PROBE_SCRIPT)
EXPORT = ["cd work", "seqapplyreg light -filter-included"]
L_STACK = ["cd work", "seqapplyreg light -filter-included", core.STACK_COMMAND]
L_STACK_ONLY = ["cd work", "seqclean r_light -stat", core.STACK_COMMAND]
RANGE = ["setext fit", "set32bits", "setcompress 0", "seqclean r_light -stat", stack_range.STACK]
FINITE = ["offline", "setext fit", "set32bits", "setcompress 0", "setref q 3", finite_stack.STACK]
GOVERNABLE = (CONVERT, PROBE, EXPORT, FINITE)
UNCAPPABLE = (L_STACK, L_STACK_ONLY, RANGE, core.stack_script(None))


def laptop(resident=(0.55e9, 1.7e9)):
    facts = resources.parse_whatif(LAPTOP)
    return facts, resources.budget(facts, resident={"ws": int(resident[0]), "commit": int(resident[1])})


def dev_facts(*, affinity=None, omp=None, avail=230e9):
    cores = [(g, 0, (2 * i, 2 * i + 1), True) for g in range(3) for i in range(32)]
    topology = resources.build_topology(cores, [64, 64, 64], [0], affinity, usable_lps=192 if not affinity else 16,
                                        omp=omp)
    memory = resources.MemorySnapshot(int(274.6e9), int(avail), int(291.8e9), int(248e9), None, None, 0.0)
    return resources.MachineFacts(memory, topology, "system_managed", {}, (), "light")


def today(commands, facts, **kwargs):
    kwargs.setdefault("pin", None)
    kwargs.setdefault("environment", {})
    kwargs.setdefault("siril_ini", {})
    return resources.siril_today(commands, "siril-cli.exe", facts, frames=48, frame_pixels=PIXELS, **kwargs)


# --------------------------------------------------------------------------- affinity masks

def test_mask_takes_one_lp_per_core_before_any_sibling():
    cores = [(0, 0, (2 * i, 2 * i + 1)) for i in range(8)]
    assert resources.affinity_mask(cores, 4, group=0) == 0b01010101
    assert resources.affinity_mask(cores, 8, group=0) == 0x5555
    assert resources.affinity_mask(cores, 9, group=0) == 0x5557          # a sibling only after every core
    assert resources.affinity_mask(cores, 16, group=0) == 0xFFFF
    for n in range(1, 17):
        mask = resources.affinity_mask(cores, n, group=0)
        bits = [b for b in range(16) if mask >> b & 1]
        assert len(bits) == n and len({b // 2 for b in bits}) == min(n, 8)


def test_mask_prefers_the_highest_efficiency_class_and_stays_in_its_group():
    hybrid = [(0, 0, (0,)), (0, 0, (1,)), (0, 1, (2, 3)), (0, 1, (4, 5)), (1, 1, (0, 1))]
    assert resources.affinity_mask(hybrid, 2, group=0) == 0b010100     # the two P-cores (class 1) first
    assert resources.affinity_mask(hybrid, 4, group=0) == 0b010111     # then the E-cores, siblings last
    assert resources.affinity_mask(hybrid, 99, group=0) == 0b111111    # never outside group 0
    assert resources.affinity_mask(hybrid, 4, group=1) == 0b11
    big = [(0, 0, (i,)) for i in range(64)]
    assert resources.affinity_mask(big, 200, group=0) == (1 << 64) - 1   # never more than 64
    assert resources.affinity_mask(big, 0, group=0) == 0


# --------------------------------------------------------------------------- routes

@pytest.mark.parametrize("commands", GOVERNABLE)
@pytest.mark.parametrize("omp", [1, 4])
def test_route_a_never_emits_script_commands_and_sets_the_configured_omp(commands, omp):
    facts = dev_facts()
    budget = resources.budget(facts)
    plan = resources.plan_siril(commands, "x", facts, budget, frames=48, frame_pixels=PIXELS, route="A", omp=omp,
                                today=today(commands, facts))
    assert plan.route == "A" and plan.script_commands == ()
    assert plan.environment == {"OMP_NUM_THREADS": str(omp)} and plan.pool == omp
    assert plan.record_neutral is True and not plan.over_budget
    assert plan.affinity and bin(plan.affinity["mask"]).count("1") == plan.threads


def test_route_b_clamps_setcpu_to_the_lps_the_process_may_use():
    facts = dev_facts(affinity=0xFFFF)                                   # a 16-LP affinity: Siril sees 16
    budget = resources.budget(facts)
    plan = resources.plan_siril(PROBE, "x", facts, budget, frames=48, frame_pixels=PIXELS, route="B",
                                today=today(PROBE, facts))
    assert plan.route == "B" and plan.threads == 16 and plan.script_commands == ("setcpu 16",)
    assert plan.environment == {"OMP_NUM_THREADS": "16"} and plan.affinity is None
    assert plan.record_neutral is False


def test_route_b_caps_the_allowance_when_only_that_fits():
    facts, budget = laptop()
    plan = resources.plan_siril(FINITE, "x", facts, budget, frames=48, frame_pixels=PIXELS, route="B",
                                today=today(FINITE, facts, whatif=True))
    assert plan.route == "B" and plan.script_commands[0].startswith("setcpu ")
    assert plan.script_commands[1:] == ("set core.mem_mode=1", "set core.mem_amount=5")
    assert not plan.over_budget and plan.record_neutral is False


@pytest.mark.parametrize("parts, route, neutral", [
    (("convert",), "A", True), (("probe",), "A", True), (("export",), "A", True), (("stack_finite",), "A", True),
    (("stack_rej",), "A", False), (("export_stack",), "A", False), (("probe", "export_stack"), "A", False),
    (("convert",), "B", False), (("stack_finite",), "A+B", False), (("stack_rej",), "B", False),
    (("stack_rej",), "today", True), (("other",), "A", None), (("other",), "today", None)])
def test_record_neutrality(parts, route, neutral):
    assert resources.record_neutral(parts, route) is neutral


@pytest.mark.parametrize("commands", UNCAPPABLE)
def test_stock_rejection_stacks_are_never_changed(commands):
    facts = dev_facts()
    plan = resources.recommend_siril(commands, "x", facts, resources.budget(facts), frames=48, frame_pixels=PIXELS,
                                     today=today(commands, facts))
    assert plan.route == "today" and plan.record_neutral is True and plan.script_commands == ()
    assert plan.threads == 64 and plan.planned_action == "run" and not plan.over_budget


# --------------------------------------------------------------------------- today

def test_today_reproduces_the_unpinned_dev_pc():
    facts = dev_facts()
    plan = today(CONVERT, facts)
    assert (plan.threads, plan.pool) == (64, 64)
    assert plan.allowance_bytes == int(0.9 * 230e9)
    assert plan.predicted["commit"][0] / GB == pytest.approx(8.62 + 0.86 * 48, rel=0.01)   # pool(64) + 48 units


def test_today_reproduces_a_pin():
    facts = dev_facts()
    pin = {"cpu": 16, "mem": 16}
    environment = siril.siril_environment(".", pin, base={})
    plan = today(L_STACK, facts, pin=pin, environment=environment)
    assert (plan.threads, plan.allowance_bytes) == (16, 16 * GiB)
    assert plan.pool == resources.siril_pool_setting(environment)[0]        # whatever the pinned block carries
    plan = today(L_STACK, facts, pin=pin, environment={"OMP_NUM_THREADS": "16"})
    assert plan.pool == 16
    assert plan.script_commands == ("setcpu 16", "set core.mem_mode=1", "set core.mem_amount=16")


def test_today_reproduces_the_16_lp_proxy_with_its_pool_of_64():
    facts = dev_facts(affinity=0xFFFF)
    plan = today(EXPORT, facts, pin={"mem": 9})
    assert (plan.threads, plan.pool, plan.allowance_bytes) == (16, 64, 9 * GiB)
    assert plan.predicted["bound_by"] == "allowance"


def test_today_follows_omp_in_sirils_environment():
    facts = dev_facts()
    plan = today(CONVERT, facts, environment={"OMP_NUM_THREADS": "1"})
    assert plan.pool == 1 and plan.basis["pool_source"] == "OMP_NUM_THREADS"
    real = resources.siril_today(CONVERT, "x", facts, frames=48, frame_pixels=PIXELS, folder=".", pin=None)
    expected = resources.siril_pool_setting(siril.siril_environment(".", None, commands=CONVERT))[0] or 64
    assert expected == int(siril.siril_omp_threads(CONVERT))   # a RAW-decoding script gets the decode pool
    assert real.pool == expected                       # X14: whatever Siril's own environment block says


@pytest.mark.parametrize("environment, pool, source", [
    ({"OMP_NUM_THREADS": "1"}, 1, "OMP_NUM_THREADS"),                                      # U-G3 (siril.py)
    ({"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "8"}, 8, "OMP_NUM_THREADS"),
    ({"OPENBLAS_NUM_THREADS": "1"}, 64, "group"),          # a U-G3 written as OpenBLAS's knob would not shrink it
    ({"OMP_NUM_THREADS": "0"}, 64, "group"),
    ({}, 64, "group")])
def test_the_siril_pool_is_read_from_the_block_siril_receives(environment, pool, source):
    """GOV-7: OMP_NUM_THREADS of the block Siril receives, else the group's LPs (the dev PC: 64). This Siril's
    OpenBLAS is an OpenMP build: OPENBLAS_NUM_THREADS does not size its pool (siril.py U-G3, SIRIL-1 fix)."""
    plan = today(CONVERT, dev_facts(), environment=environment)
    assert (plan.pool, plan.basis["pool_source"]) == (pool, source)
    assert resources.siril_pool_setting(environment) == ((pool, source) if source != "group" else (None, None))


def test_today_reads_the_jobs_siril_ini(tmp_path):
    facts = dev_facts()
    (tmp_path / "siril.ini").write_text("[core]\nmem_mode=1\nmem_ratio=0.90000000000000002\nmem_amount=16\n[gui]\n",
                                        encoding="utf-8")
    plan = resources.siril_today(EXPORT, "x", facts, frames=48, frame_pixels=PIXELS, folder=tmp_path, pin=None,
                                 environment={})
    assert plan.allowance_bytes == 16 * GiB and plan.basis["allowance_source"] == "siril.ini"
    (tmp_path / "siril.ini").write_text("[core]\nmem_mode=0\nmem_ratio=0.5\n", encoding="utf-8")
    plan = resources.siril_today(EXPORT, "x", facts, frames=48, frame_pixels=PIXELS, folder=tmp_path, pin=None,
                                 environment={})
    assert plan.allowance_bytes == int(0.5 * 230e9)
    assert resources.parse_siril_ini("[gui]\nmem_mode=1\n") == {}


def test_a_malformed_pin_is_recorded_not_raised(monkeypatch):
    monkeypatch.setenv(siril.SIRIL_PIN_ENV, "cpu=lots")
    plan = resources.siril_today(CONVERT, "x", dev_facts(), frames=48, frame_pixels=PIXELS, folder=".")
    assert "pin_error" in plan.basis and plan.threads == 64


# --------------------------------------------------------------------------- the laptop (A6)

@pytest.mark.parametrize("commands", GOVERNABLE)
def test_laptop_governable_launches_fit_or_say_so(commands):
    facts, budget = laptop()
    plan = resources.recommend_siril(commands, "x", facts, budget, frames=48, frame_pixels=PIXELS,
                                     today=today(commands, facts, whatif=True))
    fits = plan.predicted["ws"][0] <= budget.ram_allowance() and plan.predicted["commit"][0] <= budget.commit_allowance()
    assert fits or (plan.over_budget and plan.planned_action in resources.PLANNED_ACTIONS)
    assert plan.threads <= 64
    if plan.route == "A":
        assert plan.record_neutral is True
        bits = [b for b in range(16) if plan.affinity["mask"] >> b & 1]
        assert len({b // 2 for b in bits}) == min(len(bits), 8)        # no packed siblings while cores are free


def test_laptop_numbers_of_the_design():
    facts, budget = laptop()
    convert = resources.recommend_siril(CONVERT, "x", facts, budget, frames=48, frame_pixels=PIXELS,
                                        today=today(CONVERT, facts, whatif=True))
    assert convert.threads == 8 and convert.route == "A"
    assert convert.predicted["commit"][0] / GB == pytest.approx(7.1, abs=0.1)
    assert convert.predicted["ws"][0] / GB == pytest.approx(6.2, abs=0.1)
    export = resources.recommend_siril(EXPORT, "x", facts, budget, frames=48, frame_pixels=PIXELS,
                                       today=today(EXPORT, facts, whatif=True))
    assert export.threads == 3 and export.predicted["commit"][0] / GB == pytest.approx(6.2, abs=0.1)
    finite = resources.recommend_siril(FINITE, "x", facts, budget, frames=48, frame_pixels=PIXELS,
                                       today=today(FINITE, facts, whatif=True))
    assert finite.route == "A+B" and finite.script_commands[-1] in ("set core.mem_amount=5", "set core.mem_amount=6")
    assert finite.record_neutral is False and not finite.over_budget


@pytest.mark.parametrize("commands", [L_STACK, L_STACK_ONLY, RANGE])
def test_laptop_l_and_r_stacks_need_u6(commands):
    facts, budget = laptop()
    current = today(commands, facts, whatif=True)
    assert (current.threads, current.pool) == (16, 16)
    assert current.allowance_bytes == int(0.9 * 9 * GiB)
    plan = resources.recommend_siril(commands, "x", facts, budget, frames=48, frame_pixels=PIXELS, today=current)
    assert plan.route == "today" and plan.over_budget and plan.planned_action == "needs_U6"
    assert plan.predicted["ws"][0] > budget.ram_allowance()          # the predicted overrun is recorded


def test_with_omp_1_the_laptop_l_stack_saves_commit_but_not_ram():
    facts, budget = laptop()
    current = today(L_STACK, facts, whatif=True)
    with_u3 = today(L_STACK, facts, whatif=True, environment={"OMP_NUM_THREADS": "1"})
    assert (current.predicted["commit"][0] - with_u3.predicted["commit"][0]) / GB == pytest.approx(2.0, abs=0.05)
    assert current.predicted["ws"] == with_u3.predicted["ws"]


def test_wait_when_another_aester_task_holds_the_budget():
    facts, _ = laptop()
    busy = resources.budget(facts, resident={"ws": int(8.0e9), "commit": int(8e9)})
    plan = resources.recommend_siril(PROBE, "x", facts, busy, frames=48, frame_pixels=PIXELS,
                                     today=today(PROBE, facts, whatif=True))
    assert plan.over_budget and plan.planned_action == "wait"
    tiny = dataclasses.replace(facts, memory=resources.MemorySnapshot(int(4 * GiB), int(1 * GiB), int(5 * GiB),
                                                                      int(1 * GiB), None, None, 0.0))
    plan = resources.recommend_siril(PROBE, "x", tiny, resources.budget(tiny), frames=48, frame_pixels=PIXELS,
                                     today=today(PROBE, tiny, whatif=True))
    assert plan.over_budget and plan.planned_action == "warn_run"
