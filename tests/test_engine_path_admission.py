"""Efficiency P1a S-8: refuse an F/R job whose stack engine could not write its outputs.

Both pinned Siril builds write a file only when its full path is at most 259
UTF-16 units. F (finite_validity_v1) and R (positive_power2_v1) jobs have the
engine write 45 units below the job directory, so a job directory of 215 units
or more could never finish: it used to fail at the stack, after the whole
conversion and export. Such a submission (a job, or a workflow for its child
stack job) is now refused at admission, before anything is converted, renamed
or created; L jobs are never checked. Admitted jobs are unchanged.

The boundary tests place the root so that the job directory lies at the
boundary. When the test's own temporary folder is too long for that, the limit
is raised by the difference, which moves the boundary by the same amount and
keeps every relation (the real 259 is pinned by the pure tests and, with the
local finite runtime, by the engine itself). An admitted job at the boundary
writes its job.json through a temporary name 46 units below the job directory,
so those steps need Windows long paths (LongPathsEnabled) and are skipped
without them; the refusals never need them.
"""
import hashlib
import os
import subprocess
import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits

from starstack import core, finite_stack as fs, imaging, process_guard, registered_support as rs, siril
from starstack import stack_range as sr, workflow as wf
from starstack.i18n import translate_message

KEY = "k"                    # job-<sha256(key)[:24]>: job directory = root + 29 units
JOB_EXTRA = 29
WORKFLOW_EXTRA = 69          # root\workflow-<24 hex>\stack\job-<24 hex>
ENGINE_EXTRA = 45            # job\finite-<12 hex>\finite_master_nvalid.fit, job\range-<12 hex>\trial3\encoded_master.fit
FINITE_ENGINE = (Path(__file__).resolve().parents[1] / "tools" / "finite-siril" / "finite-validity-v1"
                 / "siril-cli.exe")


# --------------------------------------------------------------------------- the rule

def test_path_units_count_utf16_code_units():
    assert siril.path_units("abc") == 3
    assert siril.path_units("星空") == 2
    assert siril.path_units("\U0001F320") == 2                 # an astral character is a surrogate pair
    assert siril.path_units(Path("C:/a")) == len(str(Path("C:/a")))


def test_limit_and_message():
    assert siril.ENGINE_PATH_LIMIT == 259                       # both pinned Siril builds (measured 2026-09-29)
    root = "D:\\星空 缓存"
    assert siril.engine_path_refusal(root, "C:\\" + "a" * 256) is None                 # 259 units: written
    assert siril.engine_path_refusal(root, "C:\\" + "星" * 256) is None
    message = siril.engine_path_refusal(root, "C:\\" + "a" * 256 + "\U0001F320")      # 261 units
    assert message == ("缓存或结果文件夹路径过长：在 D:\\星空 缓存 中，堆栈引擎需要写入长 261 个字符的文件路径，"
                       "比 Windows 路径上限（259 个字符）多 2 个字符。请换一个更短的缓存或结果文件夹后重新提交；"
                       "本次提交尚未开始处理")
    assert translate_message(message, "en") == (
        "The cache or output folder path is too long: in D:\\星空 缓存, the stacking engine would need to write a file "
        "path of 261 characters, 2 more than the Windows path limit of 259 characters. Choose a shorter cache or "
        "output folder and submit again; this submission has not started processing.")


def test_longest_engine_paths_are_45_units_below_the_job(tmp_path):
    job = tmp_path / "job-x"
    finite, ranged = fs.longest_engine_path(job), sr.longest_engine_path(job)
    assert finite == job.resolve() / "finite-000000000000" / "finite_master_nvalid.fit"
    assert ranged == job.resolve() / "range-000000000000" / "trial3" / "encoded_master.fit"
    for longest in (finite, ranged):
        assert siril.path_units(longest) == siril.path_units(job.resolve()) + ENGINE_EXTRA
    # the placeholders have the length of the attempt names the engines use, and the names are the engines' own
    assert len("finite-" + uuid.uuid4().hex[:12]) == len("finite-000000000000")
    assert len("range-" + uuid.uuid4().hex[:12]) == len("range-000000000000")
    assert "-out=finite_master.fit" in fs.STACK and "-nan_missing_v1" in fs.STACK
    assert "-out=encoded_master.fit" in sr.STACK and sr.contract()["max_attempts"] == 3
    for mode, expected in ((fs.MODE, finite), (sr.MODE, ranged)):
        refusal = core.stack_engine_path_refusal("r", job, mode)
        assert refusal == siril.engine_path_refusal("r", expected)
    assert core.stack_engine_path_refusal("r", tmp_path / ("p" * 300), "legacy") is None      # L: never checked


# --------------------------------------------------------------------------- admission

@pytest.fixture
def env(tmp_path, monkeypatch):
    source = tmp_path / "s"
    source.mkdir()
    for n in range(3):
        imaging.write_linear(source / f"f{n}.fit", np.full((18, 24, 3), .03 + n * .001, dtype=np.float32), "t")
    mask = tmp_path / "sky.png"
    mask.write_bytes(b"admission reads only the name and digest")
    engine = tmp_path / "finite-siril-cli.exe"
    engine.write_bytes(b"stand-in")
    (tmp_path / "e.exe").write_bytes(b"stand-in")               # a workflow hashes the Siril executable
    info = {"available": True, "executable": str(tmp_path / "e.exe"), "version": "1.4.4", "message": "t"}
    monkeypatch.setattr(core, "doctor", lambda *_: dict(info))
    monkeypatch.setattr(sr, "require_engine", lambda *_: None)
    monkeypatch.setattr(rs, "require_engine", lambda *_: None)
    monkeypatch.setattr(fs, "describe_engine", lambda path: {"executable": str(path), "sha256": "0" * 64})
    monkeypatch.setattr(core.shutil, "disk_usage", lambda _: SimpleNamespace(free=10 ** 12))
    launched = []
    monkeypatch.setattr(core, "_spawn_worker", lambda d: launched.append(d))
    monkeypatch.setattr(wf, "_spawn", lambda d: launched.append(d))
    base = tmp_path.resolve()
    state = SimpleNamespace(extra=None)

    def root(offset, extra, tag):
        """A root whose job (extra 29) or child job (extra 69) directory has ``longest + offset`` units.

        One layout per test: the first call raises the limit when ``base`` is too long for it (module docstring)."""
        if state.extra is None:
            state.extra = extra
            shift = max(0, siril.path_units(base) + 5 - (siril.ENGINE_PATH_LIMIT - ENGINE_EXTRA - extra))
            if shift:
                monkeypatch.setattr(siril, "ENGINE_PATH_LIMIT", siril.ENGINE_PATH_LIMIT + shift)
        assert extra == state.extra
        folder = base / tag
        pad = siril.ENGINE_PATH_LIMIT - ENGINE_EXTRA + offset - extra - siril.path_units(folder) - 1
        assert pad >= 1
        return folder / ("p" * pad)

    def need_long_paths():
        probe = base / ("q" * max(1, 265 - siril.path_units(base) - 1))
        try:
            probe.write_bytes(b"")
            probe.unlink()
        except OSError:
            pytest.skip("an admitted job at the boundary needs Windows long paths (LongPathsEnabled)")

    return SimpleNamespace(source=source, mask=mask, engine=engine, launched=launched, root=root,
                           longest=lambda: siril.ENGINE_PATH_LIMIT - ENGINE_EXTRA, need_long_paths=need_long_paths,
                           finite={"support_mode": rs.MODE, "sky_mask_path": str(mask),
                                   "stack_range_mode": fs.MODE, "stack_engine_path": str(engine)})


def submit(env, root, **kwargs):
    return core.submit_job(str(env.source), str(root), idempotency_key=KEY, **kwargs)


def assert_refused(env, root, call):
    with pytest.raises(ValueError) as info:
        call()
    message = str(info.value)
    assert message.startswith("缓存或结果文件夹路径过长：在 " + str(root) + " 中")
    assert "多 1 个字符" in message and message.endswith("本次提交尚未开始处理")
    assert not env.launched
    return message


@pytest.mark.parametrize("mode", ["R", "F"])
def test_job_at_the_boundary(env, mode):
    options = {"stack_range_mode": sr.MODE} if mode == "R" else env.finite
    longest_path = sr.longest_engine_path if mode == "R" else fs.longest_engine_path
    refused = env.root(1, JOB_EXTRA, "b")
    assert_refused(env, refused, lambda: submit(env, refused, **options))
    assert sorted(p.name for p in refused.iterdir()) == [".admission.lock"]     # no job directory, no job.json
    env.need_long_paths()
    ok = env.root(0, JOB_EXTRA, "a")
    job = submit(env, ok, **options)
    assert siril.path_units(job["job_dir"]) == env.longest()
    assert siril.path_units(longest_path(job["job_dir"])) == siril.ENGINE_PATH_LIMIT
    assert env.launched == [Path(job["job_dir"])] and job["stack_range_mode"] == options["stack_range_mode"]


def test_legacy_job_on_the_same_root_is_admitted(env):
    root = env.root(1, JOB_EXTRA, "l")
    env.need_long_paths()
    job = submit(env, root)
    assert siril.path_units(job["job_dir"]) == env.longest() + 1 and job["stack_range_mode"] == "legacy"
    assert env.launched == [Path(job["job_dir"])]


def test_existing_job_is_returned_and_force_restart_keeps_the_liveness_refusals_first(env, monkeypatch):
    root = env.root(1, JOB_EXTRA, "x")
    env.need_long_paths()
    with monkeypatch.context() as patch:
        patch.setattr(core, "stack_engine_path_refusal", lambda *_: None)     # a job admitted by older code
        job = submit(env, root, stack_range_mode=sr.MODE)
    env.launched.clear()
    again = submit(env, root, stack_range_mode=sr.MODE)
    assert again["job_dir"] == job["job_dir"] and again["state"] == "queued" and not env.launched
    with pytest.raises(ValueError, match="仍在运行或排队"):
        submit(env, root, stack_range_mode=sr.MODE, force_restart=True)
    core._update(Path(job["job_dir"]), state="failed")
    before = sorted(p.name for p in root.iterdir())
    assert_refused(env, root, lambda: submit(env, root, stack_range_mode=sr.MODE, force_restart=True))
    assert sorted(p.name for p in root.iterdir()) == before                    # nothing renamed or created
    assert core.load_job(job["job_dir"])["state"] == "failed"


def test_workflow_refuses_before_creating_its_folder(env):
    bad = env.root(1, WORKFLOW_EXTRA, "v")
    assert_refused(env, bad, lambda: wf.submit_workflow(str(env.source), str(bad), stack_range_mode=sr.MODE,
                                                         idempotency_key=KEY))
    assert not list(bad.glob("workflow-*"))
    env.need_long_paths()
    status = wf.submit_workflow(str(env.source), str(bad), idempotency_key=KEY)        # L: never checked
    assert Path(status["workflow_dir"]).is_dir() and env.launched
    env.launched.clear()
    ok = env.root(0, WORKFLOW_EXTRA, "w")
    status = wf.submit_workflow(str(env.source), str(ok), stack_range_mode=sr.MODE, idempotency_key=KEY)
    directory = Path(status["workflow_dir"])
    record = wf._read(directory)
    child = wf._stack_job_dir(directory, record["signature"])
    key = "workflow-stack-" + record["signature"]
    # the folder core.submit_job creates for the key _wait_for_stack passes
    assert child == directory.resolve() / "stack" / ("job-" + hashlib.sha256(key.encode()).hexdigest()[:24])
    assert siril.path_units(child) == env.longest() and env.launched == [directory]


def test_finite_workflow_is_checked_too(env):
    options = {"sky_mask_path": str(env.mask), "stack_engine_path": str(env.engine)}
    bad = env.root(1, WORKFLOW_EXTRA, "f")
    assert_refused(env, bad, lambda: wf.submit_workflow(str(env.source), str(bad), idempotency_key=KEY, **options))
    assert not list(bad.glob("workflow-*"))
    env.need_long_paths()
    ok = env.root(0, WORKFLOW_EXTRA, "g")
    status = wf.submit_workflow(str(env.source), str(ok), idempotency_key=KEY, **options)
    assert wf._read(Path(status["workflow_dir"]))["request"]["stack_range_mode"] == fs.MODE


def test_child_of_a_workflow_admitted_earlier_is_refused_by_core(env, monkeypatch):
    """A workflow admitted before this check now fails when it submits its child, not at the stack."""
    bad = env.root(1, WORKFLOW_EXTRA, "c")
    env.need_long_paths()
    with monkeypatch.context() as patch:
        patch.setattr(core, "stack_engine_path_refusal", lambda *_: None)
        status = wf.submit_workflow(str(env.source), str(bad), stack_range_mode=sr.MODE, idempotency_key=KEY)
    directory = Path(status["workflow_dir"])
    record = wf._read(directory)
    env.launched.clear()
    with pytest.raises(ValueError, match="路径过长"):
        core.submit_job(record["request"]["input_dir"], str(directory / "stack"),
                        idempotency_key="workflow-stack-" + record["signature"], stack_range_mode=sr.MODE)
    assert not list((directory / "stack").glob("job-*")) and not env.launched


# --------------------------------------------------------------------------- the engine itself

def _q_frames(folder):
    """Three small inputs as finite_stack._write_input leaves them (BOTTOM-UP, DATAMIN/DATAMAX, NaN = missing)."""
    rng = np.random.default_rng(259)
    for number in (1, 2, 3):
        data = (.2 + rng.normal(0, .01, (3, 64, 96))).astype(np.float32)
        data[:, 0, :2] = np.nan                                   # independently known missing coverage
        header = fits.Header()
        header["ROWORDER"] = "BOTTOM-UP"
        header["DATAMIN"], header["DATAMAX"] = float(np.nanmin(data)), float(np.nanmax(data))
        fits.PrimaryHDU(data, header).writeto(folder / f"q_{number:05d}.fit")


@pytest.mark.skipif(os.name != "nt" or not FINITE_ENGINE.is_file(), reason="needs the local pinned finite runtime")
@pytest.mark.parametrize("total, written", [(259, True), (260, False)])
def test_the_finite_engine_writes_its_longest_file_only_up_to_the_limit(tmp_path, total, written):
    base = tmp_path.resolve()
    name = "finite_master_nvalid.fit"
    attempt = base / ("p" * (total - siril.path_units(base) - 2 - len(name)))
    attempt.mkdir()
    assert siril.path_units(attempt / name) == total
    _q_frames(attempt)
    (attempt / "siril.ini").touch()
    (attempt / "stack.ssf").write_text(siril.script_text(
        ["offline", "setext fit", "set32bits", "setcompress 0", "set gui.icc_autoassignment=0",
         "set gui.icc_autoconversion=0", "setref q 1", fs.STACK]), encoding="utf-8")
    # MSYSTEM and a private XDG_DATA_HOME keep Siril's Python venv work away from the user's %LOCALAPPDATA%\siril;
    # the disabled bus keeps it off the user's D-Bus session bus (S-4d; tests/siril_guard.py R2).
    environment = {**os.environ, "MSYSTEM": "MINGW64", "XDG_DATA_HOME": str(base),
                   "DBUS_SESSION_BUS_ADDRESS": "disabled:"}
    with (base / "stack.log").open("ab") as log, process_guard.guarded_process(
            [str(FINITE_ENGINE), "-i", str(attempt / "siril.ini"), "-d", str(attempt), "-s", str(attempt / "stack.ssf")],
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), env=environment) as process:
        returncode = process.wait(timeout=120)
    assert (returncode == 0) is written, (base / "stack.log").read_text(encoding="utf-8", errors="replace")
    assert (attempt / name).is_file() is written
    assert (siril.engine_path_refusal(base, attempt / name) is None) is written
