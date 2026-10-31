"""EFF-D4 stack-settings contract (efficiency P1c §4.2, KD-1): unit and contract tests, controlled engines only.

The real-Siril checks are in tests/test_legacy_stack_split_siril.py and scripts/kd5_contract_stack_repeat.py (KD-5).
"""
from __future__ import annotations

from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import re
import sys

import numpy as np
import pytest

from starstack import core, imaging, siril, stack_settings as ss
from starstack import finite_stack as fs, registered_support as rs, stack_range as sr
import d4_fakes
from test_legacy_stack_split import FRAMES, world  # noqa: F401  (the controlled L world, a fixture)
from test_stack_range_v2 import ClipMeanEngine, engine, inputs  # noqa: F401  (R's independent engine and fixtures)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = {"version": "siril-stack-settings-1", "engine_version": "1.4.4",
            "rule": "setcpu equals normalization workers; explicit core.mem_amount",
            "applies_to": ["legacy", "strict-legacy", "positive_power2_v1"]}


# --------------------------------------------------------------------------- Siril's arithmetic, modelled

def _siril_normalization_workers(memory_gib, max_thread, frames, per_frame_mb):
    """normalization.c: nb = max_memory_MB / memory_per_image_MB, capped by com.max_thread, then by the frames."""
    nb = (memory_gib * 1024) // per_frame_mb
    nb = min(nb, max_thread)
    return min(nb, frames)


def _compute_thread_distribution(workers, maximum):
    """core/processing.c compute_thread_distribution."""
    base, rem = divmod(maximum, workers)
    out = []
    for _ in range(workers):
        value = base + (1 if rem > 0 else 0)
        rem -= 1 if rem > 0 else 0
        out.append(value or 1)
    return out


def test_contract_and_constants():
    assert ss.contract() == CONTRACT and ss.contract() is not ss.contract()
    assert ss.FINITE_MODE == fs.MODE
    assert sr.MODE in ss.APPLIES_TO and ss.NOTE_KEY == "siril_stack_settings"
    assert siril.STACK_SETTINGS_OMP_THREADS == "1"
    assert siril.STACK_SETTINGS_UNSET == ("OMP_DYNAMIC", "OMP_THREAD_LIMIT", "OMP_MAX_ACTIVE_LEVELS")
    engine = {"version": "1.4.4"}
    assert ss.admission_field(engine, "legacy") == CONTRACT
    assert ss.admission_field(engine, sr.MODE) == CONTRACT
    assert ss.admission_field(engine, fs.MODE) is None
    assert ss.admission_field({"version": "1.4.5"}, "legacy") is None
    assert ss.admission_field(None, "legacy") is None


@pytest.mark.parametrize("width, height, layers, bitpix, expected", [
    (8280, 5520, 3, -32, 871),     # 45.7 MP RGB float: 8280*5520*(3+2)*4 / 2**20 = 871.75
    (8280, 5520, 1, -32, 523),     # mono float: *(1+2)*4
    (8280, 5520, 3, 16, 697),      # RGB 16-bit: *((3+1)*2 + 2*4)
    (8280, 5520, 1, 16, 523),      # mono 16-bit: *((1+1)*2 + 8)
    (8280, 5520, 3, 8, 697),       # BYTE_IMG reads as DATA_USHORT
    (8280, 5520, 3, -64, 871),     # DOUBLE_IMG reads as DATA_FLOAT
    (8280, 5520, 3, 32, 871),      # LONG_IMG reads as DATA_FLOAT
    (24, 18, 3, -32, 1),           # below one MB: at least 1
    (900, 600, 3, -32, 10),        # the real-Siril synthetic frames (10.3 MB)
])
def test_per_frame_mb_is_normalization_c(width, height, layers, bitpix, expected):
    assert ss.frame_mb(width, height, layers, bitpix) == expected


def test_unsupported_bitpix_is_refused():
    with pytest.raises(ValueError, match="BITPIX=64"):
        ss.frame_mb(10, 10, 3, 64)


def test_threads_property_grid():
    """T' = min(cap, n, B*1024//p) gives Siril's normalization exactly T' workers of one inner thread each."""
    checked = 0
    for cap in (1, 4, 10, 16, 32, 64):
        for frames in (1, 2, 3, 6, 10, 16, 48, 300):
            for memory in (1, 4, 8, 9, 16, 153):
                for per_frame in (1, 10, 523, 697, 871, 2000):
                    if memory * 1024 // per_frame < 1:
                        with pytest.raises(ValueError, match="不足以容纳一帧"):
                            ss.threads(cap, frames, memory, per_frame)
                        continue
                    count = ss.threads(cap, frames, memory, per_frame)
                    assert count == min(cap, frames, memory * 1024 // per_frame) >= 1
                    workers = _siril_normalization_workers(memory, count, frames, per_frame)
                    assert workers == count
                    assert _compute_thread_distribution(workers, count) == [1] * count
                    checked += 1
    assert checked > 1000


def test_thread_cap_pin_groups_and_ceiling():
    assert ss.thread_cap({"cpu": 12, "mem": 3}, usable=192, groups=(64, 64, 64)) == (12, "pin")
    assert ss.thread_cap({"cpu": 99}, usable=4, groups=(2,)) == (99, "pin")        # a pin is taken as given
    assert ss.thread_cap(None, usable=192, groups=(64, 64, 64)) == (64, "machine")
    assert ss.thread_cap({"mem": 9}, usable=96, groups=(64, 32)) == (32, "machine")   # the smallest active group
    assert ss.thread_cap(None, usable=16, groups=(64, 64, 64)) == (16, "machine")      # affinity 16
    assert ss.thread_cap(None, usable=128, groups=()) == (64, "machine")              # groups unknown: <= 64
    assert ss.thread_cap(None, usable=8, groups=None)[0] <= 8                         # this machine's groups
    cap, source = ss.thread_cap(None)
    assert 1 <= cap <= 64 and source == "machine"


def test_memory_budget_follows_the_runtime_memory(monkeypatch):
    GiB = 1 << 30
    assert ss.memory_gib({"mem": 9}, total=16 * GiB, available=1 * GiB) == (9, "pin")
    assert ss.memory_gib(None, total=256 * GiB, available=203 * GiB) == (153, "runtime")   # 0.6 x 256 = 153.6
    budget = ss.memory_gib(None, total=16 * GiB, available=5 * GiB)                        # min(9, 4)
    assert budget == (4, "runtime")
    assert ss.threads(64, 48, budget[0], 871) == 4                                          # B <= 4 -> T' <= 4
    assert ss.memory_gib(None, total=int(15.4 * GiB), available=9 * GiB) == (8, "runtime")
    from starstack import performance
    monkeypatch.setattr(performance, "available_memory", lambda: None)        # available memory unreadable
    assert ss.memory_gib(None, total=16 * GiB) == (4, "fallback")
    monkeypatch.setattr(performance, "available_memory", lambda: 5 * GiB)     # the one source of available memory
    assert ss.memory_gib(None, total=16 * GiB) == (4, "runtime")
    monkeypatch.setattr(ss, "_last_total", None)
    monkeypatch.setattr(ss, "_total_memory", lambda: None)
    assert ss.memory_gib(None, available=5 * GiB) == (4, "fallback")


def test_unreadable_total_uses_the_total_read_before(monkeypatch):
    GiB = 1 << 30
    monkeypatch.setattr(ss, "_last_total", None)     # first, so that teardown restores this process's real total
    assert ss._read_total_memory() > GiB                                     # this machine
    monkeypatch.setattr(ss, "_read_total_memory", lambda: 32 * GiB)
    assert ss.memory_gib(None, available=20 * GiB) == (18, "runtime")                        # min(19, 18)
    monkeypatch.setattr(ss, "_read_total_memory", lambda: None)
    assert ss._last_total == 32 * GiB
    assert ss.memory_gib(None, available=20 * GiB) == (18, "runtime")       # the total read before
    monkeypatch.setattr(ss, "_last_total", None)
    assert ss.memory_gib(None, available=20 * GiB) == (4, "fallback")


def test_processor_groups_are_read_once():
    groups = ss.processor_groups()
    assert groups and all(g >= 1 for g in groups) and ss.processor_groups() is groups


def test_budget_below_one_frame_is_refused():
    with pytest.raises(ValueError, match="871 MB"):
        ss.threads(64, 48, 0, 871)
    with pytest.raises(ValueError, match="不足以容纳一帧"):
        ss.threads(64, 48, 1, 2000)
    assert ss.threads(64, 48, 1, 1024) == 1


def _frames_folder(folder: Path, count=3, *, shape=(18, 24, 3), selected=None) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(1, count + 1):
        imaging.write_linear(folder / f"r_light_{i:05d}.fit", np.full(shape, .02 * i, np.float32), "frame",
                             extra={"STACKCNT": 1})
    (folder / ss.SEQUENCE_NAME).write_text(d4_fakes.sequence_text(count, selected), encoding="utf-8")
    return folder


def test_plan_reads_the_sequence_text_and_the_first_selected_frame(tmp_path, monkeypatch):
    from starstack import hashing
    folder = _frames_folder(tmp_path / "work", 4, selected=[2, 3, 4])
    (folder / "r_light_00002.fit").unlink()
    imaging.write_linear(folder / "r_light_00002.fit", np.zeros((600, 900, 3), np.float32), "first selected")
    for helper in ("sha256_file", "sha256_many"):        # "from the text of r_light_.seq, not the ledger"
        monkeypatch.setattr(hashing, helper, lambda *a, **k: pytest.fail("the plan must not hash"))
    plan = ss.plan(folder, pin=None, usable=16, groups=(64,), total=256 << 30, available=200 << 30)
    assert plan == {"version": ss.VERSION, "script": "02_stack", "frames": 3, "frame_mb": 10, "threads": 3,
                    "memory_gib": 153, "thread_cap": 16, "thread_cap_source": "machine", "memory_source": "runtime"}
    pinned = ss.plan(folder, pin={"cpu": 2, "mem": 1}, script="x")
    assert (pinned["threads"], pinned["memory_gib"], pinned["thread_cap_source"], pinned["memory_source"]) == (
        2, 1, "pin", "pin")
    (folder / ss.SEQUENCE_NAME).write_text(d4_fakes.sequence_text(4, []), encoding="utf-8")
    with pytest.raises(ValueError, match="选片"):
        ss.plan(folder)


def test_script_lines_go_before_seqclean_after_the_pin_lines():
    settings = {"threads": 10, "memory_gib": 9}
    commands = ss.with_settings(core.LEGACY_STACK_SCRIPT, settings)
    assert commands == ["cd work", "setcpu 10", "set core.mem_mode=1", "set core.mem_amount=9",
                        "seqclean r_light -stat", core.STACK_COMMAND]
    assert ss.without_settings(commands) == (list(core.LEGACY_STACK_SCRIPT), (10, 9))
    assert ss.without_settings(core.LEGACY_STACK_SCRIPT) == (list(core.LEGACY_STACK_SCRIPT), None)
    text = siril.script_text(commands, {"cpu": 16, "mem": 16}).splitlines()
    assert text == ["requires 1.4.4 1.5.0", "setcpu 16", "set core.mem_mode=1", "set core.mem_amount=16",
                    "cd work", "setcpu 10", "set core.mem_mode=1", "set core.mem_amount=9",
                    "seqclean r_light -stat", core.STACK_COMMAND]     # the later lines win
    trial = ss.with_settings(["setext fit", "set32bits", "setcompress 0", "seqclean r_light -stat", sr.STACK], settings)
    assert trial[3:6] == ss.script_lines(settings) and trial[6] == "seqclean r_light -stat"
    with pytest.raises(ValueError, match="seqclean"):
        ss.with_settings(["cd work", core.STACK_COMMAND], settings)


def test_contract_environment_fixes_omp_to_one():
    base = {"OMP_NUM_THREADS": "7", "OMP_DYNAMIC": "true", "omp_thread_limit": "3", "OMP_MAX_ACTIVE_LEVELS": "2",
            "Omp_Num_Threads": "5", "PATH": "x"}
    pinned = siril.siril_environment("f", {"cpu": 16}, base, windows=False, stack_settings=True)
    assert pinned == {"PATH": "x", "OMP_NUM_THREADS": "1"}
    plain = siril.siril_environment("f", {"cpu": 16}, base, windows=False)
    assert plain["OMP_NUM_THREADS"] == "16" and plain["OMP_DYNAMIC"] == "true" and plain["omp_thread_limit"] == "3"
    windows = siril.siril_environment("f", None, base, windows=True, stack_settings=True)
    assert windows["OMP_NUM_THREADS"] == "1" and windows["DBUS_SESSION_BUS_ADDRESS"] == "disabled:"
    assert not {"OMP_DYNAMIC", "OMP_THREAD_LIMIT", "OMP_MAX_ACTIVE_LEVELS"} & {k.upper() for k in windows}


class _Process:
    returncode = 0

    def poll(self):
        return 0


def _launch_recorder(monkeypatch):
    launches = []

    @contextmanager
    def guarded(args, **kwargs):
        launches.append((list(args), kwargs["env"]))
        yield _Process()
    monkeypatch.setattr(siril, "guarded_process", guarded)
    return launches


def test_run_script_takes_the_configuration_and_the_contract_environment(tmp_path, monkeypatch):
    launches = _launch_recorder(monkeypatch)
    monkeypatch.setenv(siril.SIRIL_PIN_ENV, "cpu=16,mem=16")
    monkeypatch.setenv("OMP_DYNAMIC", "true")
    config = tmp_path / "work" / ss.CONFIG_NAME
    config.parent.mkdir()
    siril.run_script("siril-cli.exe", tmp_path, "02_stack", ["cd work", "setcpu 4", "seqclean r_light -stat"],
                     lambda: None, config=config, stack_settings=True)
    args, env = launches[-1]
    assert args[1:3] == ["-i", str(config)] and config.is_file() and not (tmp_path / "siril.ini").exists()
    assert env["OMP_NUM_THREADS"] == "1" and "OMP_DYNAMIC" not in {k.upper() for k in env}
    log = (tmp_path / "02_stack.log").read_text(encoding="utf-8")
    assert "OMP_NUM_THREADS=1" in log and "OMP_NUM_THREADS=16" not in log and ss.CONFIG_NAME in log
    assert not ss.MEMORY_LINE.search(log) and not ss.THREADS_LINE.search(log)
    siril.run_script("siril-cli.exe", tmp_path, "02_export", ["cd work"], lambda: None)       # unchanged default
    args, env = launches[-1]
    assert args[1:3] == ["-i", str(tmp_path / "siril.ini")] and env["OMP_NUM_THREADS"] == "16"
    assert env["OMP_DYNAMIC"] == "true"


# --------------------------------------------------------------------------- the checks after the stack

PLAN = {"frames": 16, "frame_mb": 871, "threads": 16, "memory_gib": 153}
GOOD = "Memory per image: 871 MB. Max memory: 156672 MB\nallocating data for 16 threads (each 1 MB)\n"


@pytest.mark.parametrize("earlier, current, passes", [
    ("", GOOD, True),
    ("", GOOD.replace("\n", "\r\n"), True),                                            # CRLF
    ("", "log: noise\n" + GOOD + "stacking done\n", True),
    ("", GOOD.replace("16 threads", "64 threads"), False),                             # threads differ
    ("", GOOD.replace("871 MB.", "870 MB."), False),                                   # per-frame MB differs
    ("", GOOD.replace("156672", "157286"), False),                                     # budget differs
    ("", "", False),                                                                   # both lines missing
    ("", GOOD.splitlines()[0] + "\n", False),                                          # thread line missing
    ("", GOOD + GOOD.splitlines()[0] + "\n", False),                                   # two memory lines
    ("", GOOD + GOOD.splitlines()[1] + "\n", False),                                   # two thread lines
    (GOOD, "", False),                                                                 # only an earlier attempt
    (GOOD.replace("16 threads", "64 threads"), GOOD, True),                            # earlier attempt ignored
])
def test_log_check(tmp_path, earlier, current, passes):
    log = tmp_path / "02_stack.log"
    log.write_bytes(earlier.encode())
    offset = ss.log_offset(log)
    with log.open("ab") as handle:
        handle.write(current.encode())
    if passes:
        ss.verify_log(log, offset, PLAN)
    else:
        with pytest.raises(RuntimeError, match="堆栈设置核对失败"):
            ss.verify_log(log, offset, PLAN)


def test_log_check_of_a_missing_log(tmp_path):
    assert ss.log_offset(tmp_path / "none.log") == 0
    with pytest.raises(RuntimeError):
        ss.verify_log(tmp_path / "none.log", 0, PLAN)


@pytest.mark.parametrize("value, passes", [(16, True), (15, False), (None, False), (True, False)])
def test_stack_count_check(tmp_path, value, passes):
    master = tmp_path / "sky_master.fit"
    imaging.write_linear(master, np.zeros((4, 5, 3), np.float32), "master",
                         extra={} if value is None else {"STACKCNT": value})
    if passes:
        ss.verify_stack_count(master, PLAN)
    else:
        with pytest.raises(RuntimeError, match="STACKCNT"):
            ss.verify_stack_count(master, PLAN)


# --------------------------------------------------------------------------- admission and records

@pytest.mark.parametrize("change", ["version", "rule", "extra", "applies_to", "finite", "engine"])
def test_a_changed_field_is_refused(change):
    job = {"stack_range_mode": "legacy", "engine": {"version": "1.4.4"},
           "admission_options": {"stack_settings": ss.contract()}}
    assert ss.job_contract(job) == CONTRACT
    assert ss.job_contract({"engine": {"version": "1.4.4"}}) is None                  # a job without the field
    field = job["admission_options"]["stack_settings"]
    if change == "version":
        field["version"] = "siril-stack-settings-2"
    elif change == "rule":
        field["rule"] = "setcpu 16"
    elif change == "extra":
        field["threads"] = 16
    elif change == "applies_to":
        field["applies_to"].append("finite_validity_v1")
    elif change == "finite":
        job["stack_range_mode"] = fs.MODE
    else:
        job["engine"]["version"] = "1.4.5"
    with pytest.raises(ValueError, match="堆栈设置合同已改变"):
        ss.job_contract(job)


def test_every_message_of_the_module_has_english(tmp_path):
    """The English UI shows each refusal and failed check without Chinese (translations_messages.py).  REFUSED is
    raised through a name, which tests/test_message_translation.py's scanner cannot see, so it is checked here."""
    from astropy.io import fits
    from starstack.i18n import translate_message
    master = tmp_path / "sky_master.fit"
    imaging.write_linear(master, np.zeros((4, 5, 3), np.float32), "master", extra={"STACKCNT": 15})
    sequence = tmp_path / ss.SEQUENCE_NAME
    sequence.write_text(d4_fakes.sequence_text(3, selected=[]), encoding="utf-8")
    flat = tmp_path / "flat.fit"
    fits.PrimaryHDU(np.zeros(5, np.float32)).writeto(flat)
    changed = {"engine": {"version": "1.4.5"}, "admission_options": {"stack_settings": CONTRACT}}
    calls = [lambda: ss.job_contract(changed),
             lambda: ss.data_type(64), lambda: ss.threads(64, 48, 1, 2000), lambda: ss.threads(64, 0, 1, 1),
             lambda: ss.selected_frames(sequence), lambda: ss.frame_geometry(flat),
             lambda: ss.verify_log(tmp_path / "none.log", 0, PLAN), lambda: ss.verify_stack_count(master, PLAN)]
    cjk = re.compile(r"[　-〿㐀-䶿一-鿿豈-﫿＀-￯]")
    for call in calls:
        with pytest.raises((ValueError, RuntimeError)) as raised:
            call()
        message = str(raised.value)
        assert cjk.search(message), message                         # the backend message stays Chinese
        english = translate_message(message, "en")
        assert not cjk.search(english), english
    assert translate_message(f"RuntimeError: 堆栈作业未成功：ValueError: {ss.REFUSED}", "en") == (
        "RuntimeError: Stacking did not finish: ValueError: The stack settings contract has changed. "
        "Please create a new job.")


def _l_job(world, key="d4", **options):
    return core.submit_job(str(world.source), str(world.tmp / "admit"), idempotency_key=key, transient_mode="off",
                           **options)


def test_admission_writes_the_field_for_l_and_r_only(world):
    mp = world.monkeypatch
    job = _l_job(world)
    assert job["admission_options"]["stack_settings"] == CONTRACT == job["stack_settings"]
    core._verify_admission_options(job)
    mp.setattr(sr, "require_engine", lambda *_: None)
    r_job = _l_job(world, "d4-r", stack_range_mode=sr.MODE)
    assert r_job["admission_options"]["stack_settings"] == CONTRACT
    mp.setattr(rs, "require_engine", lambda *_: None)
    mp.setattr(fs, "describe_engine", lambda path: {"executable": str(path), "sha256": "0" * 64})
    mask = world.tmp / "sky.png"
    mask.write_bytes(b"mask")
    engine = world.tmp / "finite-siril-cli.exe"
    engine.write_bytes(b"stand-in")
    f_job = _l_job(world, "d4-f", support_mode=rs.MODE, sky_mask_path=str(mask), stack_range_mode=fs.MODE,
                   stack_engine_path=str(engine))
    assert "stack_settings" not in f_job and "stack_settings" not in f_job.get("admission_options", {})
    strict = _l_job(world, "d4-s", support_mode=rs.MODE, sky_mask_path=str(mask))
    assert strict["admission_options"]["stack_settings"] == CONTRACT
    mp.setattr(core, "doctor", lambda *_: {"available": True, "executable": "x", "version": "1.4.5", "message": "m"})
    newer = _l_job(world, "d4-145")
    assert "stack_settings" not in newer and newer["admission_options"] == {"transient_mode": "off"}
    for record in (job, r_job):
        edited = json.loads(json.dumps(record))
        edited["stack_settings"] = {**CONTRACT, "version": "x"}
        with pytest.raises(ValueError):
            core._verify_admission_options(edited)
        edited["admission_options"]["stack_settings"] = edited["stack_settings"]
        edited["admission_options_sha256"] = core._options_digest(edited["admission_options"])
        with pytest.raises(ValueError, match="堆栈设置合同已改变"):
            core._verify_admission_options(edited)


def _verifier():
    spec = importlib.util.spec_from_file_location("verify_workflow_equivalence_d4",
                                                  ROOT / "scripts" / "verify_workflow_equivalence.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_the_verifier_recomputes_the_signature_of_contract_jobs(world):
    """KD-1: scripts/verify_workflow_equivalence.py rebuilds the job signature of an L and an R contract job (the
    field is in admission_options, which its preimage already carries): no derived-digest failure."""
    verifier = _verifier()
    world.monkeypatch.setattr(sr, "require_engine", lambda *_: None)
    for key, options in (("d4-l", {}), ("d4-r", {"stack_range_mode": sr.MODE})):
        job = _l_job(world, key, **options)
        folder = Path(job["job_dir"])
        assert job["admission_options"]["stack_settings"] == CONTRACT
        entry = verifier.fingerprint_file(folder / "job.json", "{JOB}/job.json")
        side = verifier.Side({"files": {"{JOB}/job.json": entry}, "roots": {"{JOB}": str(folder)}}, "a")
        checks = [k for k in side.derived if k[3] == "job signature"]
        assert len(checks) == 1 and side.derived_failures() == []
        preimage = verifier._job_signature_preimage(job)
        assert preimage["admission_options"]["stack_settings"] == CONTRACT
        field = preimage["admission_options"]["stack_settings"]
        assert not [path for path in _keys(field) if verifier.excluded_key_reason(path[-1])]   # no excluded key name


def _keys(value, path=()):
    if isinstance(value, dict):
        for key, item in value.items():
            yield (*path, key)
            yield from _keys(item, (*path, key))
    elif isinstance(value, list):
        for item in value:
            yield from _keys(item, path)


def test_a_submission_admitted_before_the_contract_keeps_it(world):
    """submit_job matches the pre-D4 preimage: the old job is returned (not refused), resumed with its old script,
    and a forced restart writes exactly the record the earlier version wrote."""
    with d4_fakes.admitted_before_contract():
        old = _l_job(world, "kept")
    assert "stack_settings" not in old and old["admission_options"] == {"transient_mode": "off"}
    again = _l_job(world, "kept")                       # the same submission today: the existing job, not a refusal
    assert again["signature"] == old["signature"] and again["job_dir"] == old["job_dir"]
    with pytest.raises(ValueError, match="不同素材或参数"):
        core.submit_job(str(world.source), str(world.tmp / "admit"), idempotency_key="kept", transient_mode="review")
    folder = Path(old["job_dir"])
    record = json.loads((folder / "job.json").read_text(encoding="utf-8"))
    record["state"] = "failed"
    (folder / "job.json").write_text(json.dumps(record), encoding="utf-8")
    restarted = core.submit_job(str(world.source), str(world.tmp / "admit"), idempotency_key="kept",
                                transient_mode="off", force_restart=True)
    assert restarted["signature"] == old["signature"] and "stack_settings" not in restarted
    expected = {k: v for k, v in old.items() if k not in ("created_at", "updated_at", "state")}
    got = {k: v for k, v in restarted.items() if k not in ("created_at", "updated_at", "state", "restarted_from")}
    assert list(got) == list(expected) and got == expected
    # with an option the old record has admission_options: the field is removed from it, key order kept
    with d4_fakes.admitted_before_contract():
        old_rotated = _l_job(world, "kept-rot", image_rotation=90)
    record = core._without_stack_settings(_l_job(world, "kept-rot-new", image_rotation=90),
                                          old_rotated["admission_options"], old_rotated["signature"])
    assert [k for k in record if k not in ("job_dir",)] == [k for k in old_rotated if k not in ("job_dir",)]
    assert record["admission_options"] == {"transient_mode": "off", "image_rotation": 90}
    assert record["admission_options_sha256"] == old_rotated["admission_options_sha256"]


# --------------------------------------------------------------------------- contract L jobs, controlled engine

def _succeeded(job):
    assert job["state"] == "succeeded", job.get("error")
    return job


def _expected_plan(job):
    """What the product planned for this job's 02_stack (from its timings note)."""
    notes = job["diagnostics"]["timings"]["notes"][ss.NOTE_KEY]
    assert len(notes) == 1
    return notes[0]


@pytest.mark.parametrize("path", ["3a", "3b", "3c"])
def test_contract_l_job_runs_the_settings_lines(world, path):
    job = _succeeded(world.run(path, output=f"d4-{path}", contract=True))
    folder = Path(job["job_dir"])
    plan = _expected_plan(job)
    selected = FRAMES - 1 if path == "3b" else FRAMES
    assert plan == {"version": ss.VERSION, "script": "02_stack", "frames": selected, "frame_mb": 1,
                    "threads": min(plan["thread_cap"], selected), "memory_gib": plan["memory_gib"],
                    "thread_cap": plan["thread_cap"], "thread_cap_source": "machine",
                    "memory_source": plan["memory_source"]}
    assert plan["memory_source"] in ("runtime", "fallback")
    stack = world.engine.commands("02_stack")
    assert stack == ss.with_settings(core.LEGACY_STACK_SCRIPT, plan)
    assert dict(world.engine.options)["02_stack"] == {"config": folder / "work" / ss.CONFIG_NAME, "stack_settings": True}
    assert all(options == {} for name, options in world.engine.options if name != "02_stack")
    assert (folder / "02_stack.ssf").read_text(encoding="utf-8") == siril.script_text(stack)
    assert job["admission_options"]["stack_settings"] == CONTRACT
    assert job["diagnostics"]["stacked_count"] == selected
    assert not (folder / "work" / ss.CONFIG_NAME).exists()
    outside = json.dumps({k: v for k, v in job.items() if k != "diagnostics"}) + json.dumps(
        {k: v for k, v in job["diagnostics"].items() if k not in ("timings", "timing_attempts")})
    assert "memory_gib" not in outside and "frame_mb" not in outside          # T', B, p, n: timings only


@pytest.mark.parametrize("path", ["3a", "3c"])
def test_a_job_admitted_before_the_contract_keeps_its_scripts_byte_for_byte(world, path):
    job = _succeeded(world.run(path, output=f"old-{path}"))
    folder = Path(job["job_dir"])
    assert world.engine.commands("02_stack") == list(core.LEGACY_STACK_SCRIPT)
    assert all(options == {} for _, options in world.engine.options)
    assert (folder / "02_stack.ssf").read_text(encoding="utf-8") == siril.script_text(core.LEGACY_STACK_SCRIPT)
    assert ss.NOTE_KEY not in job["diagnostics"]["timings"].get("notes", {})


def test_combined_layout_is_refused_for_a_contract_job(world):
    job = world.run("3a", output="d4-combined", layout="combined", contract=True)
    assert job["state"] == "failed" and core.L_LAYOUT_ENV in job["error"] and ss.VERSION in job["error"]
    assert world.engine.scripts == []                     # before any Siril launch


def test_the_stack_configuration_is_separate_from_the_job_configuration(world):
    seen = {}

    def after(name, directory):
        if name == "01_convert":
            (directory / "siril.ini").write_text("[core]\nmem_mode=0\nmem_ratio=0.9\n", encoding="utf-8")
        if name == "02_stack":
            config = dict(world.engine.options)["02_stack"]["config"]
            seen["copy"] = config.read_text(encoding="utf-8")
            config.write_text("[core]\nmem_mode=1\nmem_amount=153\n", encoding="utf-8")   # Siril persists them
    world.engine.after = after
    job = _succeeded(world.run("3c", output="d4-ini", contract=True))
    folder = Path(job["job_dir"])
    assert seen["copy"] == "[core]\nmem_mode=0\nmem_ratio=0.9\n"
    assert (folder / "siril.ini").read_text(encoding="utf-8") == "[core]\nmem_mode=0\nmem_ratio=0.9\n"
    assert not (folder / "work" / ss.CONFIG_NAME).exists()


@pytest.mark.parametrize("change", ["threads", "memory", "stackcnt"])
def test_a_failed_check_fails_before_the_checkpoint_and_a_resume_stacks_again(world, change):
    if change == "threads":
        world.engine.stack_log = {"threads": 64}
    elif change == "memory":
        world.engine.stack_log = {"memory_mb": (1, 1024)}
    else:
        original = world.engine._stack

        def short(work):
            original(work)
            from astropy.io import fits
            with fits.open(work / "sky_master.fit", mode="update") as hdus:
                hdus[0].header["STACKCNT"] = FRAMES - 1
        world.engine._stack = short
    job = world.run("3a", output=f"d4-fail-{change}", contract=True)
    assert job["state"] == "failed" and "堆栈设置核对失败" in job["error"], job.get("error")
    assert "stack" not in job["checkpoints"]
    world.engine.scripts.clear()
    world.engine.stack_log = {}
    if change == "stackcnt":
        world.engine._stack = original
    core.resume_job(job["job_dir"])
    core.run_job(job["job_dir"])
    resumed = _succeeded(core.load_job(job["job_dir"]))
    assert world.engine.names() == ["02_export", "02_stack"]                  # re-exported and stacked again
    assert len(resumed["diagnostics"]["timings"]["notes"][ss.NOTE_KEY]) == 1


def test_a_cancel_inside_the_contract_stack_resumes_cleanly(world):
    def after(name, directory):
        if name == "02_stack" and not after.done:
            after.done = True
            core.cancel_job(str(directory))
            raise siril.Cancelled("用户取消")
    after.done = False
    world.engine.after = after
    job = world.run("3c", output="d4-cancel", contract=True)
    assert job["state"] == "cancelled" and "stack" not in job["checkpoints"]
    assert not (Path(job["job_dir"]) / "work" / ss.CONFIG_NAME).exists()
    world.engine.after = None
    world.engine.scripts.clear()
    core.resume_job(job["job_dir"])
    core.run_job(job["job_dir"])
    _succeeded(core.load_job(job["job_dir"]))
    assert world.engine.names() == ["02_export", "02_stack"]


def test_a_budget_below_one_frame_fails_before_the_stack(world):
    world.monkeypatch.setattr(ss, "memory_gib", lambda *a, **k: (0, "runtime"))
    job = world.run("3c", output="d4-memory", contract=True)
    assert job["state"] == "failed" and "不足以容纳一帧" in job["error"]
    assert world.engine.names() == ["01_convert", "02_probe", "02_export"]   # no stack process


def test_a_pinned_contract_job_plans_with_the_pin(world):
    world.monkeypatch.setenv(siril.SIRIL_PIN_ENV, "cpu=2,mem=1")
    job = _succeeded(world.run("3c", output="d4-pin", contract=True))
    plan = _expected_plan(job)
    assert (plan["threads"], plan["memory_gib"], plan["thread_cap_source"], plan["memory_source"]) == (2, 1, "pin", "pin")
    assert world.engine.commands("02_stack")[1:4] == ["setcpu 2", "set core.mem_mode=1", "set core.mem_amount=1"]


# --------------------------------------------------------------------------- R: the wrapped trial runner

def _r_job():
    return {"stack_range_mode": sr.MODE, "engine": {"version": "1.4.4"},
            "admission_options": {"stack_settings": ss.contract()}}


def test_range_runner_wraps_only_the_trial_stack(tmp_path):
    trial = _frames_folder(tmp_path / "range-0123456789ab" / "trial1", 3)
    calls, notes = [], []

    def runner(executable, folder, name, commands, cancel_check, **options):
        calls.append((name, list(commands), options))
        d4_fakes.stack_log(folder, name, commands)

    wrapped = core._range_runner(runner, _r_job(), note=notes.append)
    commands = ["setext fit", "set32bits", "setcompress 0", "seqclean r_light -stat", sr.STACK]
    wrapped("siril", trial, "stack", list(commands), lambda: None)
    name, sent, options = calls[-1]
    plan = notes[-1]
    assert options == {"stack_settings": True} and sent == ss.with_settings(commands, plan)
    assert plan["frames"] == 3 and plan["frame_mb"] == 1 and plan["script"] == "range-0123456789ab/trial1/stack"
    wrapped("siril", trial, "other", ["x"], lambda: None)
    assert calls[-1] == ("other", ["x"], {}) and len(notes) == 1
    assert core._range_runner(runner, {"stack_range_mode": sr.MODE, "engine": {"version": "1.4.4"}}) is runner

    def wrong(executable, folder, name, commands, cancel_check, **options):
        d4_fakes.stack_log(folder, name, commands, threads=64)
    with pytest.raises(RuntimeError, match="堆栈设置核对失败"):
        core._range_runner(wrong, _r_job())("siril", trial, "stack", list(commands), lambda: None)


def test_contract_r_trials_are_planned_and_checked(inputs, engine):
    """The unchanged stack_range.run_protected_stack with the wrapped runner: every trial stack (here an upper-endpoint
    retry, so two) gets its own plan from the trial folder, its lines and the contract environment, and is checked."""
    directory, work, arrays = inputs

    def once(runner, out):
        if len(runner.calls) == 1:
            out[0, 1, 1] = 1
        return out
    clip = ClipMeanEngine(arrays, once)
    seen, notes = [], []

    def contract_engine(executable, folder, name, commands, check, **options):
        seen.append((name, options, list(commands)))
        clip(executable, folder, name, commands, check)
        d4_fakes.stack_log(folder, name, commands)
    job = {"stack_range_mode": sr.MODE, "engine": engine, "admission_options": {"stack_settings": ss.contract()}}
    record = sr.run_protected_stack(engine, directory, work, lambda: None,
                                    core._range_runner(contract_engine, job, note=notes.append))
    assert record["divisor"] == 4 and [t["state"] for t in record["trials"]] == ["upper_limit_retry", "checked"]
    assert [n["script"].split("/")[1:] for n in notes] == [["trial1", "stack"], ["trial2", "stack"]]
    plain = ["setext fit", "set32bits", "setcompress 0", "seqclean r_light -stat", sr.STACK]
    for (name, options, commands), plan in zip(seen, notes, strict=True):
        assert name == "stack" and options == {"stack_settings": True}
        assert commands == ss.with_settings(plain, plan) and plan["frames"] == 3 and plan["frame_mb"] == 1


def test_range_runner_checks_only_this_trial_s_log_section(tmp_path):
    trial = _frames_folder(tmp_path / "range-x" / "trial2", 3)
    (trial / "stack.log").write_text("Memory per image: 1 MB. Max memory: 1024 MB\nallocating data for 1 threads\n",
                                     encoding="utf-8")

    def silent(executable, folder, name, commands, cancel_check, **options):
        pass
    with pytest.raises(RuntimeError):
        core._range_runner(silent, _r_job())("siril", trial, "stack", ["seqclean r_light -stat", sr.STACK],
                                             lambda: None)


def test_kd5_script_cuts_the_sequence_to_its_first_frames(tmp_path):
    from starstack.siril_sequence import parse_seq_text
    spec = importlib.util.spec_from_file_location("kd5_contract_stack_repeat",
                                                  ROOT / "scripts" / "kd5_contract_stack_repeat.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = ("#Siril sequence file.\n#S 'sequence_name' ...\nS 'r_light_' 1 4 3 5 2 6 0 0 0\nL 3\n"
              "I 1 1\nI 2 0\nI 3 1\nI 4 1\n" + "".join(f"R1 3.5 3.6 0.7 0 1500 {i} H 1 0 0 0 1 0 0 0 1\n" for i in range(4))
              + "M0-0 540000 1 2 3\nM0-1 540000 1 2 3\n")
    text = module.sequence_subset(source, 3)
    parsed = parse_seq_text(text)
    assert parsed["header"]["nb_images"] == 3 and parsed["header"]["nb_selected"] == 3
    assert parsed["header"]["reference_image"] == 2 and all(i["selected"] for i in parsed["images"])
    assert len(parsed["regdata"][1]) == 3 and parsed["stats"] == []
    assert module.sequence_subset(source, 2).splitlines()[2] == "S 'r_light_' 1 2 2 5 0 6 0 0 0"   # reference cut off
    (tmp_path / "r_light_.seq").write_text(text, encoding="utf-8")
    assert ss.selected_frames(tmp_path / "r_light_.seq") == (3, "r_light_00001")
    with pytest.raises(ValueError):
        module.sequence_subset(source, 5)


def test_stack_range_module_is_not_changed_for_the_contract():
    """stack_range.py stays outside the contract (frozen holdout closure): no reference to the new module."""
    source = (ROOT / "src" / "starstack" / "stack_range.py").read_text(encoding="utf-8")
    assert "stack_settings" not in source
    siril_source = (ROOT / "src" / "starstack" / "siril.py").read_text(encoding="utf-8")
    assert "import stack_settings" not in siril_source and "from . import stack_settings" not in siril_source
