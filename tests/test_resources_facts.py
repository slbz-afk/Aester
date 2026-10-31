"""Governor machine facts (efficiency P1b-2): memory, topology parsing, volumes, page file, usable CPUs (X2).

Every reader must return unknowns rather than raise, parse variable-length Win32 buffers from bytes under bounds
checks, never query a remote volume beyond its drive type, and never block longer than its timeout."""
import os
import struct
import subprocess
import sys
import time

import pytest

from starstack import frame_pool, resources

WINDOWS = os.name == "nt"
windows = pytest.mark.skipif(not WINDOWS, reason="Windows facts")


# --------------------------------------------------------------------------- memory

@windows
def test_memory_snapshot_is_consistent_and_fast():
    snapshot = resources.memory_snapshot()
    assert 0 < snapshot.available_physical <= snapshot.total_physical
    assert 0 <= snapshot.commit_available <= snapshot.commit_limit
    assert snapshot.own_private and snapshot.own_working_set
    started = time.perf_counter()
    for _ in range(200):
        resources.memory_snapshot()
    assert (time.perf_counter() - started) / 200 < 0.2e-3        # A3: <= 0.2 ms


class _Failing:
    """kernel32 stand-in whose every call fails (returns 0) or raises."""

    def __init__(self, mode="zero"):
        self.mode = mode

    def __getattr__(self, name):
        def call(*args, **kwargs):
            if self.mode == "raise":
                raise OSError(f"{name} denied")
            return 0
        return call


@pytest.mark.parametrize("mode", ["zero", "raise"])
def test_every_reader_returns_unknowns_with_failing_kernel32(monkeypatch, mode):
    monkeypatch.setattr(resources, "_kernel32", lambda: _Failing(mode))
    monkeypatch.setattr(resources, "_topology_cache", None)
    monkeypatch.setattr(resources, "_volume_cache", {})
    snapshot = resources.memory_snapshot()
    assert snapshot.total_physical is None and snapshot.commit_limit is None and snapshot.own_private is None
    topology = resources.cpu_topology(refresh=True)
    assert topology.physical_cores is None and topology.cores == ()
    assert resources.power_state() == {"on_mains": None, "has_battery": None}
    resident = resources.aester_resident(detail=True)
    assert resident["ws"] is None and resident["commit"] is None
    volume = resources.volume_facts("C:\\data\\x.dng", "source")
    assert volume.io_class in ("unknown", "fixed") and volume.status in ("ok", "error")
    facts = resources.machine_facts(level="full", paths=[("work", "C:\\x")])
    budget = resources.budget(facts)
    assert budget.ram_total is None and budget.commit_total is None
    monkeypatch.setattr(resources, "_topology_cache", None)


# --------------------------------------------------------------------------- topology

def _core(flags, efficiency, groups_masks):
    """One SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX record of RelationProcessorCore."""
    body = struct.pack("<BB20xH", flags, efficiency, len(groups_masks))
    for group, mask in groups_masks:
        body += struct.pack("<QH6x", mask, group)
    return struct.pack("<II", 0, 8 + len(body)) + body


def test_parse_smt_cores_two_classes_and_two_groups():
    raw = (_core(1, 1, [(0, 0b11)]) + _core(1, 1, [(0, 0b1100)]) + _core(0, 0, [(0, 0b10000)])
           + _core(1, 0, [(1, 0b11)]))
    cores = resources.parse_processor_cores(raw)
    assert cores == [(0, 1, (0, 1), True), (0, 1, (2, 3), True), (0, 0, (4,), False), (1, 0, (0, 1), True)]
    topology = resources.build_topology(cores, [5, 2], [0], 0b11111, usable_lps=7)
    assert topology.physical_cores == 4 and topology.smt and topology.logical_total == 7
    assert topology.efficiency_classes == {0: 2, 1: 2}
    assert topology.process_affinity is None and topology.python_usable_cores == 4
    assert topology.siril_group_lps == 5 and topology.siril_threads_today == 5 and topology.siril_pool_today == 5


def test_topology_with_an_explicit_affinity_counts_the_masks_cores():
    cores = [(0, 0, (2 * i, 2 * i + 1), True) for i in range(32)]
    topology = resources.build_topology(cores, [64], [0], 0xFFFF, usable_lps=16, omp="16")
    assert topology.process_affinity == {"group": 0, "mask": 0xFFFF}
    assert topology.python_usable_lps == 16 and topology.python_usable_cores == 8
    assert topology.siril_threads_today == 16 and topology.siril_group_lps == 64
    assert topology.siril_pool_today == 16                       # OMP_NUM_THREADS of this process


@pytest.mark.parametrize("omp, pool", [(None, 64), ("1", 1), ("8", 8), ("zero", 64), ("0", 64)])
def test_siril_pool_today_follows_omp_num_threads(omp, pool):
    cores = [(0, 0, (2 * i, 2 * i + 1), True) for i in range(32)]
    assert resources.build_topology(cores, [64], [0], None, usable_lps=64, omp=omp).siril_pool_today == pool


@pytest.mark.parametrize("raw", [b"", b"\x00" * 7, struct.pack("<II", 0, 4), struct.pack("<II", 0, 400) + b"\x00" * 8,
                                 _core(1, 0, [(0, 3)])[:-5], struct.pack("<II", 0, 48) + b"\x00" * 40,
                                 os.urandom(97), None, "text"])
def test_truncated_or_garbage_core_buffers_are_unknown_and_never_raise(raw):
    assert resources.parse_processor_cores(raw) is None


def test_other_relations_are_skipped():
    other = struct.pack("<II", 2, 16) + b"\x00" * 8          # RelationCache-like record
    cores = resources.parse_processor_cores(other + _core(0, 0, [(0, 1)]))
    assert cores == [(0, 0, (0,), False)]


@windows
def test_real_topology_sums_match_the_os():
    topology = resources.cpu_topology(refresh=True)
    assert sum(topology.groups) == os.cpu_count() == topology.logical_total
    assert all(group <= 64 for group in topology.groups)
    assert 1 <= topology.physical_cores <= topology.logical_total
    assert sum(len(lps) for _, _, lps in topology.cores) == topology.logical_total
    assert topology.siril_group_lps <= 64 and topology.siril_threads_today <= 64
    assert topology.python_usable_lps == frame_pool.usable_cpus()          # X2: one definition


# --------------------------------------------------------------------------- X2: usable CPUs on this PC

_X2_PROBE = ("import json, sys; sys.path.insert(0, sys.argv[1]);"
             "from starstack import frame_pool, resources;"
             "t = resources.cpu_topology();"
             "print(json.dumps({'usable': frame_pool.usable_cpus(), 'lps': t.python_usable_lps,"
             " 'cores': t.python_usable_cores, 'affinity': t.process_affinity, 'groups': list(t.groups),"
             " 'siril_threads': t.siril_threads_today}))")


@windows
def test_x2_usable_cpus_default_process_and_a_16_lp_affinity(tmp_path):
    """X2: record (a) a default process (group count, mask, result) and (b) the bench's 16-LP affinity (16)."""
    import json
    src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
    env = dict(os.environ)
    default = json.loads(subprocess.run([sys.executable, "-c", _X2_PROBE, src], capture_output=True, text=True,
                                        env=env, timeout=60, check=True).stdout)
    assert default["usable"] == default["lps"]
    assert default["affinity"] is None
    assert default["usable"] in (os.cpu_count(), *default["groups"])       # all groups, or the primary group
    print("X2 default process:", default)
    if os.cpu_count() < 16:
        pytest.skip("fewer than 16 logical processors")
    # (b) the bench's 16-LP affinity: a child started from a process restricted to 16 LPs inherits it
    script = ("import ctypes, subprocess, sys, json;"
              "k = ctypes.WinDLL('kernel32', use_last_error=True); k.GetCurrentProcess.restype = ctypes.c_void_p;"
              "k.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t];"
              "assert k.SetProcessAffinityMask(k.GetCurrentProcess(), 0xFFFF);"
              "print(subprocess.run([sys.executable, '-c', sys.argv[1], sys.argv[2]], capture_output=True, text=True,"
              " check=True).stdout)")
    out = subprocess.run([sys.executable, "-c", script, _X2_PROBE, src], capture_output=True, text=True, env=env,
                         timeout=60, check=True).stdout
    restricted = json.loads(out.strip().splitlines()[-1])
    assert restricted["usable"] == restricted["lps"] == 16
    assert restricted["affinity"] == {"group": restricted["affinity"]["group"], "mask": 0xFFFF}
    assert restricted["siril_threads"] == 16 and restricted["cores"] in (8, 16)
    print("X2 16-LP affinity:", restricted)


# --------------------------------------------------------------------------- volumes

def test_storage_descriptor_and_seek_penalty_parsing():
    descriptor = struct.pack("<IIBBBBIIIII", 40, 40, 0, 0, 0, 1, 0, 0, 0, 0, 17) + b"\x00" * 4
    assert resources.parse_storage_descriptor(descriptor) == "nvme"
    sata = bytearray(descriptor)
    struct.pack_into("<I", sata, 28, 11)
    assert resources.parse_storage_descriptor(bytes(sata)) == "sata"
    assert resources.parse_storage_descriptor(descriptor[:20]) is None
    assert resources.parse_storage_descriptor(struct.pack("<II", 40, 4) + b"\x00" * 30) is None
    assert resources.parse_seek_penalty(struct.pack("<II?", 12, 12, False)) is False
    assert resources.parse_seek_penalty(struct.pack("<II?", 12, 12, True)) is True
    assert resources.parse_seek_penalty(b"\x00" * 5) is None
    assert resources.parse_seek_penalty(None) is None


@pytest.mark.parametrize("drive, bus, seek, cls", [
    ("remote", None, None, "network"), ("removable", None, None, "removable"), ("fixed", "usb", False, "removable"),
    ("fixed", "nvme", False, "nvme"), ("fixed", "sata", False, "ssd"), ("fixed", "sata", True, "hdd"),
    ("fixed", None, None, "fixed"), ("unknown", None, None, "unknown"), ("ramdisk", None, None, "ssd")])
def test_io_class(drive, bus, seek, cls):
    assert resources.io_class(drive, bus, seek) == cls
    assert resources.IO_WORKERS[cls] >= 1


@windows
@pytest.mark.parametrize("path, root", [("C:\\a\\b.dng", "C:\\"), ("d:/x", "D:\\"),
                                        ("\\\\server\\share\\a\\b", "\\\\server\\share\\"),
                                        ("\\\\?\\UNC\\server\\share\\x", "\\\\server\\share\\"),
                                        ("\\\\?\\C:\\x", "C:\\")])
def test_volume_root(path, root):
    assert resources.volume_root(path) == root


def test_a_remote_volume_is_never_queried_beyond_its_drive_type(monkeypatch):
    monkeypatch.setattr(resources, "_volume_cache", {})
    monkeypatch.setattr(resources, "_drive_type", lambda root: resources.DRIVE_REMOTE)

    def forbidden(*args, **kwargs):
        raise AssertionError("a remote volume was queried")

    monkeypatch.setattr(resources, "_volume_query", forbidden)
    monkeypatch.setattr(resources, "_storage_property", forbidden)
    facts = resources.volume_facts("T:\\bench\\x", "output", level="full")
    assert (facts.status, facts.io_class, facts.drive_type) == ("skipped_remote", "network", "remote")


@windows
def test_a_unc_root_is_remote_without_any_api_call(monkeypatch):
    monkeypatch.setattr(resources, "_volume_cache", {})
    monkeypatch.setattr(resources, "_kernel32", lambda: (_ for _ in ()).throw(AssertionError("API used")))
    facts = resources.volume_facts("\\\\fileserver\\share\\x", "source", level="full")
    assert facts.status == "skipped_remote" and facts.io_class == "network"


def test_a_sleeping_volume_query_times_out_quickly(monkeypatch):
    monkeypatch.setattr(resources, "_volume_cache", {})
    monkeypatch.setattr(resources, "_drive_type", lambda root: 3)
    monkeypatch.setattr(resources, "_volume_query", lambda root, level: time.sleep(2))
    started = time.perf_counter()
    facts = resources.volume_facts("C:\\sleepy\\x", "source", timeout=0.2)
    assert time.perf_counter() - started < 0.3
    assert facts.status == "timeout" and facts.io_class == "fixed"


@windows
def test_light_level_never_opens_a_device_handle(monkeypatch, tmp_path):
    monkeypatch.setattr(resources, "_volume_cache", {})
    monkeypatch.setattr(resources, "_storage_property", lambda *a: (_ for _ in ()).throw(AssertionError("IOCTL")))
    facts = resources.volume_facts(tmp_path, "work", level="light")
    assert facts.status == "ok" and facts.bus_type is None and facts.seek_penalty is None
    assert facts.free_bytes and facts.file_system


@windows
def test_full_level_reads_the_bus_type_of_this_volume(monkeypatch, tmp_path):
    monkeypatch.setattr(resources, "_volume_cache", {})
    facts = resources.volume_facts(tmp_path, "work", level="full")
    assert facts.status == "ok"
    assert facts.io_class in ("nvme", "ssd", "hdd", "removable", "fixed")
    started = time.perf_counter()
    resources.machine_facts(level="full", paths=[("work", tmp_path)])
    assert time.perf_counter() - started < 0.05      # A3 (10 ms first call here; generous for a loaded machine)


# --------------------------------------------------------------------------- page file, power

@pytest.mark.parametrize("entries, mode", [
    (["?:\\pagefile.sys"], "system_managed"), (["C:\\pagefile.sys 0 0"], "system_managed"),
    (["C:\\pagefile.sys 4096 8192"], "fixed"), ([], "none"), ([""], "none"), ("", "none"),
    (["C:\\pagefile.sys"], "system_managed"), (["D:\\pagefile.sys 1024 1024", "?:\\pagefile.sys"], "system_managed"),
    (None, "unknown")])
def test_pagefile_mode(entries, mode):
    assert resources.pagefile_mode(entries) == mode


@windows
def test_real_pagefile_and_power_are_read_without_raising():
    assert resources.read_pagefile_mode() in ("system_managed", "fixed", "none", "unknown")
    power = resources.power_state()
    assert set(power) == {"on_mains", "has_battery"}


# --------------------------------------------------------------------------- what-if facts

def test_whatif_facts_describe_the_laptop():
    facts = resources.parse_whatif("total=15.4G,avail=9G,lps=16,cores=8,siril_pool=16,commit=24G,commit_avail=14G,"
                                   "io=nvme,resident_ws=0.55G,resident_commit=1.7G")
    assert facts.level == "whatif" and facts.cpu.logical_total == 16 and facts.cpu.physical_cores == 8
    assert facts.cpu.siril_threads_today == 16 and facts.cpu.siril_pool_today == 16 and facts.cpu.smt
    assert facts.memory.total_physical == int(15.4 * 1024 ** 3)
    assert facts.memory.commit_limit == 24 * 1024 ** 3 and facts.memory.commit_available == 14 * 1024 ** 3
    assert {v.io_class for v in facts.volumes} == {"nvme"}
    assert facts.override["resident_ws"] == int(0.55 * 1024 ** 3)
    assert resources.affinity_mask(facts.cpu.cores, 8, group=0) == 0x5555
    assert resources.parse_whatif("") is None and resources.parse_whatif(None) is None


@pytest.mark.parametrize("text", ["total=16G", "total=16G,avail=9G,lps=16", "total=16G,avail=20G,lps=16,cores=8",
                                  "total=16G,avail=9G,lps=8,cores=16", "total=16G,avail=9G,lps=16,cores=8,io=floppy",
                                  "total=16X,avail=9G,lps=16,cores=8", "total=16G,avail=9G,lps=16,cores=8,bogus=1",
                                  "total=16G,total=8G,avail=1G,lps=1,cores=1"])
def test_malformed_whatif_facts_raise(text):
    with pytest.raises(ValueError):
        resources.parse_whatif(text)
