"""Governor budgets (efficiency P1b-2 §3.3): RAM (working set) and commit, each with its own reserve."""
import dataclasses

import pytest

from starstack import resources

GiB, GB = 1024 ** 3, 10 ** 9


def facts(total, avail, commit, commit_avail, *, lps=16, cores=8, usable_cores=None, affinity=None):
    cpu = resources.CpuTopology(logical_total=lps, groups=(lps,), physical_cores=cores, smt=lps > cores,
                                efficiency_classes={0: cores}, cores=(), process_affinity=affinity,
                                python_usable_lps=lps, python_usable_cores=usable_cores or cores,
                                siril_group_lps=min(64, lps), siril_threads_today=min(64, lps),
                                siril_pool_today=min(64, lps))
    memory = resources.MemorySnapshot(total, avail, commit, commit_avail, None, None, 0.0)
    return resources.MachineFacts(memory, cpu, "system_managed", {}, (), "light")


@pytest.mark.parametrize("machine, resident, ram_gb, commit_gb", [
    # the three worked examples of governor final.md §3.3
    (facts(int(15.4 * GiB), 9 * GiB, 24 * GiB, 14 * GiB), (0.55e9, 1.7e9), 6.99, 14.15),
    (facts(int(31.9 * GiB), 22 * GiB, 36 * GiB, 26 * GiB, lps=16), (0.55e9, 1.7e9), 17.1, 25.7),
    (facts(274.6e9, 236.2e9, 291.8e9, 248.5e9, lps=192, cores=96), (2e9, 2e9), 137.3, 221.3),
])
def test_worked_examples(machine, resident, ram_gb, commit_gb):
    budget = resources.budget(machine, resident={"ws": int(resident[0]), "commit": int(resident[1])})
    assert budget.ram_total / GB == pytest.approx(ram_gb, abs=0.06)
    assert budget.commit_total / GB == pytest.approx(commit_gb, abs=0.06)


def test_laptop_allowances_are_total_minus_resident():
    budget = resources.budget(facts(int(15.4 * GiB), 9 * GiB, 24 * GiB, 14 * GiB),
                              resident={"ws": int(0.55e9), "commit": int(1.7e9)})
    assert budget.ram_allowance() / GB == pytest.approx(6.44, abs=0.02)
    assert budget.commit_allowance() / GB == pytest.approx(12.45, abs=0.02)
    assert budget.cpu_lps == 14 and budget.python_cpu_cores == 7
    assert budget.basis["ram_bound"] == "available"


def test_units_are_not_mixed():
    base = facts(32 * GiB, 20 * GiB, 40 * GiB, 30 * GiB)
    a = resources.budget(base, resident={"ws": 1 * GiB, "commit": 5 * GiB})
    b = resources.budget(base, resident={"ws": 1 * GiB, "commit": 9 * GiB})
    assert a.ram_total == b.ram_total                      # RAM uses working-set terms only
    assert b.commit_total - a.commit_total == 4 * GiB      # commit uses commit terms only
    c = resources.budget(base, resident={"ws": 2 * GiB, "commit": 5 * GiB})
    assert c.commit_total == a.commit_total


def test_monotone_in_available_and_commit_available():
    previous = None
    for avail in range(2, 30, 3):
        budget = resources.budget(facts(32 * GiB, avail * GiB, 40 * GiB, 30 * GiB))
        assert previous is None or budget.ram_total >= previous
        previous = budget.ram_total
    previous = None
    for commit_avail in range(1, 40, 4):
        budget = resources.budget(facts(32 * GiB, 20 * GiB, 40 * GiB, commit_avail * GiB))
        assert previous is None or budget.commit_total >= previous
        previous = budget.commit_total


def test_presets_and_no_floors():
    machine = facts(16 * GiB, 2 * GiB, 18 * GiB, 1 * GiB)
    budget = resources.budget(machine)
    assert budget.ram_total < 0 and budget.commit_total < 0           # no floor: over budget is reported
    assert budget.ram_allowance() < 0
    roomy = facts(64 * GiB, 60 * GiB, 80 * GiB, 70 * GiB)
    save, balanced, full = (resources.budget(roomy, p).ram_total for p in ("save", "balanced", "full"))
    assert save < balanced < full
    with pytest.raises(ValueError):
        resources.budget(roomy, "turbo")


def test_python_cores_follow_the_usable_cores():
    all_groups = facts(64 * GiB, 60 * GiB, 80 * GiB, 70 * GiB, lps=192, cores=96, usable_cores=96)
    masked = facts(64 * GiB, 60 * GiB, 80 * GiB, 70 * GiB, lps=192, cores=96, usable_cores=8,
                   affinity={"group": 0, "mask": 0xFFFF})
    assert resources.budget(all_groups).python_cpu_cores == 95
    assert resources.budget(masked).python_cpu_cores == 7
    assert resources.budget(all_groups).cpu_lps == 62                  # Siril: its group's 64 LPs - 2


def test_unknown_memory_gives_unknown_budgets():
    unknown = dataclasses.replace(facts(GiB, GiB, GiB, GiB),
                                  memory=resources.MemorySnapshot(None, None, None, None, None, None, 0.0))
    budget = resources.budget(unknown)
    assert budget.ram_total is None and budget.commit_total is None
    assert budget.ram_allowance() is None and budget.commit_allowance() is None


def test_over_budget_path_gives_zero_units():
    budget = resources.budget(facts(16 * GiB, 2 * GiB, 18 * GiB, 3 * GiB), resident={"ws": 2 * GiB, "commit": 2 * GiB})
    fp = resources.FOOTPRINTS["py.transient_scan"]
    assert resources.units_within(fp, budget.ram_allowance(), budget.commit_allowance(), cap_items=8) == 0
