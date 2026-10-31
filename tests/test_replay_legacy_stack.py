"""scripts/replay_legacy_stack.py (efficiency P1b-3 phase replay) with a stand-in Siril: never runs Siril."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import sys
import time

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from starstack import core, imaging, process_guard, siril, stack_settings  # noqa: E402
from starstack import registration_calibration as rc  # noqa: E402
import d4_fakes  # noqa: E402
import test_core_two_pass_reference as two_pass  # noqa: E402

FRAMES = 4


def _module():
    spec = importlib.util.spec_from_file_location("replay_legacy_stack", ROOT / "scripts" / "replay_legacy_stack.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _frames(count=FRAMES):
    return [np.full((18, 24, 3), .03 + i * .001, dtype=np.float32) for i in range(count)]


def _source_job(root: Path, *, pinned=False, own=True, contract=False, stack_lines=None, frames=FRAMES,
                exported=False) -> Path:
    """A finished L job folder as a kept bench slot leaves it (own registration, path 3a).  ``contract``: admitted
    under the EFF-D4 stack-settings contract; ``stack_lines``: its 02_stack.ssf commands; ``exported``: its
    work/r_light_.seq and r_light frames are there (the replay's n and p before any run)."""
    job = root / "job-0123456789abcdef01234567"
    work = job / "work"
    work.mkdir(parents=True)
    for i, data in enumerate(_frames(frames), 1):
        imaging.write_linear(work / f"light_{i:05d}.fit", data, "light")
        if exported:
            imaging.write_linear(work / f"r_light_{i:05d}.fit", data, "registered light")
    if exported:
        (work / "r_light_.seq").write_text(d4_fakes.sequence_text(frames), encoding="utf-8")
    two_pass.JobDiagnosticsTest._fake_calibration(work, {f"light_{i:05d}" for i in range(1, frames + 1)}, count=frames)
    imaging.write_linear(work / "sky_master_undistorted.fit", np.mean(_frames(frames), axis=0).astype(np.float32),
                         "master")
    (job / "siril.ini").write_text(f"[core]\nwd={str(job).replace(chr(92), chr(92) * 2)}\nmem_mode=0\n", encoding="utf-8")
    notes = {"siril_pin": {"cpu": 16}} if pinned else {}
    record = {"state": "succeeded", "support_mode": "legacy", "stack_range_mode": "legacy", "signature": "abc",
              "engine": {"version": "1.4.4"},
              "diagnostics": {"registration_source": core.REGISTRATION_SOURCE_OWN if own else core.REGISTRATION_SOURCE_SIRIL,
                              "timings": {"notes": notes}}}
    if contract:
        record["admission_options"] = {"stack_settings": stack_settings.contract()}
    (job / "job.json").write_text(json.dumps(record), encoding="utf-8")
    if stack_lines is not None:
        (job / "02_stack.ssf").write_text(siril.script_text(stack_lines), encoding="utf-8")
    return job


class FakeSiril:
    """Siril's file contract for the replayed scripts; publishes Job Object accounting like the guard."""

    def __init__(self):
        self.launches = []
        self.options = []        # (name, run_script keyword options, AESTER_SIRIL_PIN at launch)
        self.threads = {"A": 8, "B": 8}
        self.master_offset = {"A": 0.0, "B": 0.0}
        self.peak = {"02_stack/A": 100, "02_export/B": 100, "02_stack/B": 90}
        self.sleep = {"A": 0.0, "B": 0.0}
        self.fail_at = None
        self.d4_change = {}      # T' -> f(master): a Siril whose result depends on the thread count
        self.d4_threads = None   # printed instead of T' (a Siril that ran something else)

    def __call__(self, executable, folder, name, commands, cancel_check, **options):
        self.launches.append((name, list(commands)))
        self.options.append((name, options, os.environ.get(siril.SIRIL_PIN_ENV)))
        settings = d4_fakes.settings_of(commands)
        assert (settings is not None) == bool(options.get("stack_settings")), (commands, options)
        if settings is not None:
            assert options["config"].is_file() and options["config"].name == stack_settings.CONFIG_NAME
            assert os.environ.get(siril.SIRIL_PIN_ENV) is None, "a contract replay run is never pinned"
        commands = stack_settings.without_settings(commands)[0]
        layout = "A" if core.STACK_COMMAND in commands and "seqapplyreg light -filter-included" in commands else "B"
        if self.fail_at == len(self.launches):
            raise RuntimeError(f"Siril {name} 失败（3）\nlog tail")
        work = folder / "work"
        (folder / f"{name}.ssf").write_text(siril.script_text(commands), encoding="utf-8")
        log = [f"aester: Siril launch priority=normal\nlog: 并行计算已启用：使用 16 个逻辑处理器。\n"]
        if "seqapplyreg light -filter-included" in commands:
            sequence = core.read_seq(work / core.SEQUENCE_FILE)
            assert "D1 2 distortion.wcs" in (work / core.SEQUENCE_FILE).read_text(encoding="utf-8")
            selected = [image["stem"] for image in sequence["images"] if image["selected"]]
            assert not list(work.glob("r_light_*.fit")), "the reset removed the earlier export"
            for number, stem in enumerate(selected, 1):
                shutil.copyfile(work / f"{stem}.fit", work / f"r_light_{number:05d}.fit")
            (work / "r_light_.seq").write_text(d4_fakes.sequence_text(len(selected)), encoding="utf-8")
            (work / core.SEQUENCE_FILE).write_text("re-saved by Siril\n", encoding="utf-8")
            log.append("running command seqapplyreg\n")
        if core.STACK_COMMAND in commands:
            frames = sorted(work.glob("r_light_*.fit"))
            assert frames
            pin = siril.siril_pin()
            threads = settings[0] if settings else (pin or {}).get("cpu")
            data = np.mean([imaging.read_rgb(p) for p in frames], axis=0).astype(np.float32) + self.master_offset[layout]
            data = self.d4_change.get(threads, lambda master: master)(data)
            imaging.write_linear(work / "sky_master.fit", data, "master", extra={"STACKCNT": len(frames)})
            if settings or (pin and "cpu" in pin):        # what Siril prints for a contract stack / a pinned control
                memory = settings[1] if settings else pin["mem"]
                log.append(f"running command stack\nMemory per image: 1 MB. Max memory: {memory * 1024} MB\n"
                           f"allocating data for {self.d4_threads or threads} threads (each 1 MB)\n")
            else:
                log.append("running command stack\nMemory per image: 2 MB. Max memory: 1000 MB\n"
                           f"allocating data for {self.threads[layout]} threads (each 1 MB)\n")
        with (folder / f"{name}.log").open("a", encoding="utf-8") as handle:
            handle.write("".join(log))
        (folder / "siril.ini").write_text("[core]\nwd=" + str(folder).replace("\\", "\\\\") + "\nmem_mode=0\n",
                                          encoding="utf-8")
        time.sleep(self.sleep[layout] if name == "02_stack" else 0.0)
        process_guard._publish_accounting({"wall_seconds": .01, "cpu_user_seconds": .02, "cpu_kernel_seconds": .01,
                                           "peak_job_memory_bytes": self.peak[f"{name}/{layout}"],
                                           "descendant_processes": 2, "returncode": 0})


@pytest.fixture
def replay(tmp_path, monkeypatch):
    module = _module()
    fake = FakeSiril()
    monkeypatch.setattr(siril, "run_script", fake)
    monkeypatch.setattr(module, "set_affinity", lambda logical: (1 << logical) - 1)
    monkeypatch.delenv(siril.SIRIL_PIN_ENV, raising=False)
    exe = tmp_path / "siril-cli.exe"
    exe.write_bytes(b"stand-in")
    job = _source_job(tmp_path / "slot")

    def run(profile="dev", blocks=1, *extra):
        report = tmp_path / f"report-{profile}-{len(fake.launches)}.json"
        code = module.main(["--replay", str(tmp_path / "replay"), "--job", str(job), "--profile", profile,
                            "--blocks", str(blocks), "--siril", str(exe), "--report", str(report), *extra])
        return code, json.loads(report.read_text(encoding="utf-8")) if report.exists() else None

    return module, fake, job, run, tmp_path


def test_dry_run_plans_abba_and_both_layouts(replay, capsys):
    module, fake, job, _, tmp_path = replay
    assert module.main(["--replay", str(tmp_path / "r"), "--profile", "hx", "--blocks", "2", "--dry-run"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["order"] == list("ABBAABBA")
    assert plan["scripts"]["A"] == core.stack_script(core.DISTORTION_WCS, own_registration=True)
    assert plan["scripts"]["B"] == [core.export_script(core.DISTORTION_WCS, True), list(core.LEGACY_STACK_SCRIPT)]
    assert plan["settings"] == {"affinity": 16, "pin": "mem=9", "max_delta_seconds": 2.0}
    assert not (tmp_path / "r").exists() and fake.launches == []


def test_the_replay_passes_when_the_layouts_agree(replay, monkeypatch):
    module, fake, job, run, tmp_path = replay
    code, report = run("dev", 2)
    assert code == 0, report
    assert report["passed"] and report["error"] is None
    assert [r["layout"] for r in report["runs"]] == list("ABBAABBA")
    assert [name for name, _ in fake.launches] == ["02_stack", "02_export", "02_stack", "02_export", "02_stack",
                                                   "02_stack"] * 2
    assert ["r_light_digest" in r for r in report["runs"]] == [True] * 4 + [False] * 4    # first block only
    assert len(report["pairs"]) == 4 and all(p["peak_ratio"] == 1.0 for p in report["pairs"])
    assert report["i"]["matches_expected"] is True                    # the source job's sky_master_undistorted.fit
    assert report["ii"]["A"] == report["ii"]["B"]
    assert report["runs"][0]["stack_log"] == {"memory_per_image_mb": 2, "max_memory_mb": 1000, "threads": 8,
                                              "processors": 16, "normalization_workers": FRAMES}
    b = report["runs"][1]
    assert [p["name"] for p in b["processes"]] == ["02_export", "02_stack"] and b["peak_job_memory_bytes"] == 100
    prepared = json.loads((tmp_path / "replay" / "prepared.json").read_text(encoding="utf-8"))
    assert prepared["frames"] == FRAMES and prepared["source_job"] == str(job.resolve())
    assert sorted(p.name for p in (job / "work").iterdir()) == sorted(
        [f"light_{i:05d}.fit" for i in range(1, FRAMES + 1)] + [rc.DISTORTION_JSON, rc.DISTORTION_WCS,
                                                                 "sky_master_undistorted.fit"])   # source untouched
    # a second profile reuses the prepared folder, with its pin set for the Siril runs only
    seen = []

    def observed(*args):
        seen.append(os.environ.get(siril.SIRIL_PIN_ENV))
        return fake(*args)
    monkeypatch.setattr(siril, "run_script", observed)
    code, report = run("hx", 1)
    assert code == 0 and report["environment"] == {"affinity_mask": hex(0xFFFF), siril.SIRIL_PIN_ENV: "mem=9"}
    assert set(seen) == {"mem=9"} and siril.SIRIL_PIN_ENV not in os.environ


@pytest.mark.parametrize("change, criterion", [
    ("master", "i"), ("threads", "ii"), ("slower", "iii"), ("memory", "iv")])
def test_each_criterion_fails_on_its_own_difference(replay, monkeypatch, change, criterion):
    module, fake, job, run, _ = replay
    if change == "master":
        fake.master_offset["B"] = 1e-7
    elif change == "threads":
        fake.threads["B"] = 4
    elif change == "slower":
        monkeypatch.setitem(module.PROFILES["dev"], "max_delta_seconds", 0.05)
        fake.sleep["B"] = 0.2
    else:
        fake.peak["02_export/B"] = 120
    code, report = run("dev", 1)
    assert code == 1 and not report["passed"]
    assert report[criterion]["passed"] is False
    assert all(report[other]["passed"] for other in ("i", "ii", "iii", "iv") if other != criterion)


def test_a_siril_failure_stops_the_replay_without_a_retry(replay):
    module, fake, job, run, _ = replay
    fake.fail_at = 2                     # the first B run's 02_export exits non-zero
    code, report = run("dev", 1)
    assert code == 1 and report["passed"] is False
    assert "失败（3）" in report["error"]
    assert len(report["runs"]) == 1 and len(fake.launches) == 2
    assert "i" not in report


def test_unfit_sources_and_folders_are_refused(replay, capsys):
    module, fake, job, run, tmp_path = replay
    pinned = _source_job(tmp_path / "pinned", pinned=True)
    fallback = _source_job(tmp_path / "fallback", own=False)
    for source in (pinned, fallback):
        assert module.main(["--replay", str(tmp_path / "x"), "--job", str(source), "--profile", "dev"]) == 2
        assert not (tmp_path / "x").exists()
    assert "pinned" in capsys.readouterr().err
    assert module.main(["--replay", str(job / "inner"), "--job", str(job), "--profile", "dev"]) == 2
    assert module.main(["--replay", str(tmp_path / "missing"), "--profile", "dev"]) == 2      # not prepared
    assert fake.launches == []


def test_the_expected_master_can_come_from_a_workflow_manifest(replay, tmp_path):
    module, *_ = replay
    master = tmp_path / "m.fit"
    imaging.write_linear(master, np.ones((4, 6, 3), np.float32), "m")
    digest = module.fits_digest(master)["data"]
    manifest = {"files": {"{WF}/stack/job-0123456789abcdef01234567/work/sky_master_undistorted.fit":
                          {"content": {"hdus": [{"data": {"digest": digest[0]}}]}},
                          "{WF}/stack/job-0123456789abcdef01234567/work/sky_master.fit": {"content": {"hdus": []}}}}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert module.manifest_master(path) == digest


# --------------------------------------------------------------------------- --contract d4 (EFF-D4 KD-2)

D4_STACK = stack_settings.with_settings(core.LEGACY_STACK_SCRIPT, {"threads": 4, "memory_gib": 153})


@pytest.fixture
def d4(replay):
    module, fake, _, _, tmp_path = replay
    job = _source_job(tmp_path / "k-slot", contract=True, stack_lines=D4_STACK)
    exe = tmp_path / "siril-cli.exe"

    def run(*extra, settings="4:9,2:9,1:1", replay_name="replay-d4", job_=job):
        report = tmp_path / f"report-d4-{len(fake.launches)}.json"
        code = module.main(["--contract", "d4", "--replay", str(tmp_path / replay_name), "--job", str(job_),
                            "--profile", "dev", "--siril", str(exe), "--settings", settings, "--report", str(report),
                            *extra])
        return code, json.loads(report.read_text(encoding="utf-8")) if report.exists() else None
    return module, fake, job, run, tmp_path


def test_d4_dry_run_plans_the_settings_and_the_control(replay, capsys):
    module, fake, job, _, tmp_path = replay
    assert module.main(["--contract", "d4", "--replay", str(tmp_path / "r"), "--profile", "dev", "--dry-run"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["settings"] == [[48, 153], [16, 153], [10, 9], [8, 8], [4, 4]] and plan["control"] == [48, 153]
    assert plan["repeat"] == 2 and plan["stack_script"][1:4] == ["setcpu T'", "set core.mem_mode=1",
                                                                  "set core.mem_amount=B"]
    assert not (tmp_path / "r").exists() and fake.launches == []
    for bad in (["--settings", "4"], ["--settings", "0:9"], ["--repeat", "0"], ["--expect-differing", "3"]):
        with pytest.raises(SystemExit):
            module.main(["--contract", "d4", "--replay", str(tmp_path / "r"), "--profile", "dev", *bad])
    with pytest.raises(SystemExit):        # d4 options without --contract d4
        module.main(["--replay", str(tmp_path / "r"), "--profile", "dev", "--control", "4:4"])


def test_d4_replay_gives_one_master_with_its_own_plans(d4):
    module, fake, job, run, tmp_path = d4
    code, report = run()
    assert code == 0, report
    assert report["schema"] == module.D4_SCHEMA and report["passed"] and not report["stop_p1c"]
    kinds = [(r["kind"], r["settings"]) for r in report["runs"]]
    assert kinds == [("contract", [4, 9]), ("contract", [2, 9]), ("contract", [1, 1]), ("control", [4, 9])] * 2
    assert all(r["own_plan"] for r in report["runs"] if r["kind"] == "contract")
    assert report["runs"][1]["stack_log"] == {"memory": [[1, 9216]], "threads": [2]}
    assert report["one_master"]["matches_expected"] is True and report["control"]["equal_to_contract"] == [True, True]
    assert all(r["ulp_to_first"]["differing"] == 0 for r in report["runs"][1:] if r["kind"] == "contract")
    stacks = [(options, pin, commands) for (name, options, pin), (_, commands) in zip(fake.options, fake.launches)
              if name == "02_stack"]
    assert [pin for _, pin, _ in stacks] == [None, None, None, "cpu=4,mem=9"] * 2
    assert stacks[0][2] == stack_settings.with_settings(core.LEGACY_STACK_SCRIPT, {"threads": 4, "memory_gib": 9})
    assert stacks[3][0] == {} and stacks[3][2] == list(core.LEGACY_STACK_SCRIPT)
    assert all(not options for name, options, _ in fake.options if name == "02_export")
    assert not (tmp_path / "replay-d4" / "work" / stack_settings.CONFIG_NAME).exists()
    assert siril.SIRIL_PIN_ENV not in os.environ


def test_d4_sources_are_checked(d4, capsys):
    module, fake, job, run, tmp_path = d4
    assert module.main(["--replay", str(tmp_path / "x"), "--job", str(job), "--profile", "dev"]) == 2   # A/B refuses
    assert "--contract d4" in capsys.readouterr().err
    missing = _source_job(tmp_path / "k-missing", contract=True, stack_lines=list(core.LEGACY_STACK_SCRIPT))
    pinned = _source_job(tmp_path / "k-pinned", contract=True,
                         stack_lines=["setcpu 16", "set core.mem_mode=1", "set core.mem_amount=16", *D4_STACK])
    for source in (missing, pinned):
        assert module.main(["--contract", "d4", "--replay", str(tmp_path / "y"), "--job", str(source),
                            "--profile", "dev"]) == 2
    assert not (tmp_path / "y").exists() and fake.launches == []
    old = _source_job(tmp_path / "pre-d4")                       # a job admitted before the contract is accepted
    code, report = run("--control", "none", "--repeat", "1", job_=old, replay_name="replay-old")
    assert code == 0 and [r["kind"] for r in report["runs"]] == ["contract"] * 3


def test_d4_refuses_a_pair_that_is_not_a_contract_plan(d4):
    module, fake, job, run, tmp_path = d4
    code, report = run(settings="8:9")       # T' = 8 > n = 4 frames; the source keeps no export: the first run decides
    assert code == 1 and "not a contract plan" in report["error"] and report["runs"] == []
    assert report["pairs_checked_before_runs"] is None


def _documented_command(module, start: str) -> list:
    """The arguments of the docstring example that starts with ``start`` (continuation lines joined)."""
    lines = [line.strip() for line in module.__doc__.splitlines()]
    index = next(i for i, line in enumerate(lines) if line.startswith(start))
    command = lines[index]
    while command.endswith("\\"):
        index += 1
        command = command[:-1].rstrip() + " " + lines[index]
    return shlex.split(command)[1:]


def test_d4_documented_eight_frame_command_runs(replay):
    """KD-2's 8-frame fixture command exactly as the docstring gives it (paths and the digest prefix substituted)."""
    module, fake, _, _, tmp_path = replay
    job = _source_job(tmp_path / "job8", frames=8, exported=True)
    mean = np.mean(_frames(8), axis=0).astype(np.float32)
    probe = tmp_path / "probe.fit"
    imaging.write_linear(probe, mean, "the stand-in Siril's master")
    reference = mean.copy()
    flat = reference.reshape(-1)
    flat[:16] = np.nextafter(flat[:16], np.float32(1))         # the pre-contract master: 16 values one ULP above
    old = tmp_path / "old8" / "sky_master_undistorted.fit"
    old.parent.mkdir()
    imaging.write_linear(old, reference, "the master of the job admitted before the contract")
    args = _documented_command(module, "replay_legacy_stack.py --contract d4 --replay DIR8")
    prefix = module.fits_digest(probe)["data"][0][:8]
    substitute = {"DIR8": str(tmp_path / "replay8"), "JOB8": str(job), "b4f7d0b6": prefix,
                  "OLD8/sky_master_undistorted.fit": str(old)}
    assert set(substitute) <= set(args)
    report_path = tmp_path / "report8.json"
    code = module.main([substitute.get(arg, arg) for arg in args]
                       + ["--siril", str(tmp_path / "siril-cli.exe"), "--report", str(report_path)])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert code == 0, report
    assert report["passed"] and report["error"] is None and not report["stop_p1c"]
    assert report["pairs_checked_before_runs"] == {"frames": 8, "frame_mb": 1}
    assert report["prepared"]["stack_sequence"] == {"frames": 8, "frame_mb": 1}
    tiers = [("contract", [8, 153]), ("contract", [8, 9]), ("contract", [8, 8]), ("contract", [4, 4]),
             ("control", [8, 153])]
    assert [(r["kind"], r["settings"]) for r in report["runs"]] == tiers * 2
    assert all(r["own_plan"] and r["plan"]["frames"] == 8 for r in report["runs"] if r["kind"] == "contract")
    assert report["one_master"]["matches_expected"] is True
    assert report["reference"]["passed"] and report["reference_comparison"]["differing"] == 16


def test_d4_pairs_are_checked_before_any_siril_run(replay, capsys):
    module, fake, _, _, tmp_path = replay
    exe = str(tmp_path / "siril-cli.exe")
    job = _source_job(tmp_path / "job8", frames=8, exported=True)
    base = ["--contract", "d4", "--profile", "dev", "--siril", exe]
    # KD-2's default (48-frame) tiers against the source's export: a usage error before the frames are copied
    assert module.main([*base, "--replay", str(tmp_path / "r1"), "--job", str(job)]) == 2
    err = capsys.readouterr().err
    assert "48:153 is not a contract plan for n=8, p=1 MB (T' would be 8)" in err and "10:9 is not" in err
    assert "8:8 is not" not in err and "4:4 is not" not in err
    assert not (tmp_path / "r1").exists() and fake.launches == []
    assert module.main([*base, "--replay", str(tmp_path / "r1"), "--job", str(job), "--settings", "8:8",
                        "--control", "16:153"]) == 2                    # the control is checked too
    assert "16:153 is not a contract plan" in capsys.readouterr().err and fake.launches == []
    # a source without its export: the first run's export decides; that export then checks a rerun of the folder
    bare = _source_job(tmp_path / "job8-bare", frames=8)
    report = tmp_path / "bare.json"
    assert module.main([*base, "--replay", str(tmp_path / "r2"), "--job", str(bare), "--report", str(report)]) == 1
    first = json.loads(report.read_text(encoding="utf-8"))
    assert "48:153 is not a contract plan" in first["error"] and first["runs"] == []
    assert first["pairs_checked_before_runs"] is None and first["prepared"]["stack_sequence"] is None
    assert [name for name, _ in fake.launches] == ["02_export"]
    assert module.main([*base, "--replay", str(tmp_path / "r2")]) == 2
    assert "48:153 is not a contract plan" in capsys.readouterr().err
    assert [name for name, _ in fake.launches] == ["02_export"]


@pytest.mark.parametrize("change, failed", [("one_ulp", "one_master"), ("wild", "no_wild_difference"),
                                            ("threads", "own_plan")])
def test_d4_criteria_fail_on_their_own_difference(d4, change, failed):
    module, fake, job, run, tmp_path = d4
    if change == "one_ulp":
        fake.d4_change = {2: lambda master: np.nextafter(master, np.float32(1))}     # every value one ULP up
    elif change == "wild":
        fake.d4_change = {2: lambda master: master + np.float32(1e-3)}
    else:
        fake.d4_threads = 7
    code, report = run("--control", "none", "--repeat", "1")
    assert code == 1 and not report["passed"] and report[failed]["passed"] is False
    assert report["stop_p1c"] is (change == "wild")


def test_d4_reference_master_counts_differing_values(d4):
    module, fake, job, run, tmp_path = d4
    mean = np.mean(_frames(), axis=0).astype(np.float32)
    reference = mean.copy()
    flat = reference.reshape(-1)
    flat[:3] = np.nextafter(flat[:3], np.float32(1))             # three values one ULP above
    path = tmp_path / "old-master.fit"
    imaging.write_linear(path, reference, "an older master")
    code, report = run("--control", "none", "--repeat", "1", "--reference-master", str(path),
                       "--expect-differing", "3")
    assert code == 0, report
    assert report["reference_comparison"] == {"comparable": True, "values": mean.size, "differing": 3, "max_ulp": 1,
                                              "nonfinite_differences": 0}
    code, report = run("--control", "none", "--repeat", "1", "--reference-master", str(path),
                       "--expect-differing", "4", replay_name="replay-d4-b")
    assert code == 1 and report["reference"]["passed"] is False
