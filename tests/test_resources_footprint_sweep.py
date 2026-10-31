"""scripts/bench_footprints.py (efficiency P1b-2 §2.6, cross-check M10/X12): plan, refusals, a stubbed sweep.

The real sweep runs in P2 preparation; here Siril is replaced at the one launch point the tool may use
(siril.run_script), and its Job Object accounting is published exactly as process_guard does."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest

from starstack import process_guard as pg, resources, siril

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "bench_footprints.py"
spec = importlib.util.spec_from_file_location("bench_footprints_tested", SCRIPT)
sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep)


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "Local"))
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    frames = []
    for number in range(1, 4):
        path = inputs / f"frame{number:05d}.dng"
        path.write_bytes(os.urandom(2048 + number))
        frames.append({"frame": number, "dng_path": f"x/{path.name}", "bytes": path.stat().st_size,
                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"frames": frames}), encoding="utf-8")
    siril = tmp_path / "siril-cli.exe"
    siril.write_bytes(b"stand-in")
    return {"inputs": inputs, "manifest": manifest, "root": tmp_path / "sweep", "siril": siril, "tmp": tmp_path}


def argv(fx, *extra):
    return ["--inputs-dir", str(fx["inputs"]), "--manifest", str(fx["manifest"]), "--root", str(fx["root"]),
            "--siril", str(fx["siril"]), "--frames", "3", *extra]


def received_pool(folder, commands=None):
    """The pool of the environment block Siril would receive for a run of ``commands`` in ``folder`` now
    (siril._started's block)."""
    environment = siril.siril_environment(folder, siril.siril_pin(), commands=commands)
    pool, source = resources.siril_pool_setting(environment)
    return (pool, source) if pool else (resources.cpu_topology().siril_group_lps, "group")


@pytest.fixture(params=["product", "openblas"])
def pool_variant(request, monkeypatch):
    """'product': siril.siril_environment as it is (U-G3: OMP_NUM_THREADS=1). 'openblas': a U-G3 written as
    OPENBLAS_NUM_THREADS=1 that leaves OMP_NUM_THREADS inherited (a cpu pin still sets OMP_NUM_THREADS=N), with
    OMP_NUM_THREADS=8 in this process. This Siril's OpenBLAS is an OpenMP build, so the pool Siril gets -- and the
    sweep must record and predict -- is the inherited OMP_NUM_THREADS, not 1."""
    if request.param == "openblas":
        real = siril.siril_environment

        def variant(folder, pin=None, base=None, *, windows=None, commands=None):
            env = real(folder, pin, base, windows=windows, commands=commands)
            parent = dict(os.environ if base is None else base)
            env.pop("OMP_NUM_THREADS", None)
            if "OMP_NUM_THREADS" in parent:
                env["OMP_NUM_THREADS"] = parent["OMP_NUM_THREADS"]
            env.update(siril.pin_environment(pin))
            env["OPENBLAS_NUM_THREADS"] = "1"
            return env

        monkeypatch.setattr(siril, "siril_environment", variant)
        monkeypatch.setenv("OMP_NUM_THREADS", "8")
    return request.param


def test_dry_run_prints_the_plan_and_creates_nothing(fixture, capsys):
    assert sweep.main(argv(fixture, "--dry-run", "--threads", "2,4", "--mem", "none,4")) == 0
    out = json.loads(capsys.readouterr().out)
    plan = out["plan"]
    assert out["dry_run"] and out["inputs_verified"] == 3
    assert [c["id"] for c in plan["configs"]] == ["A-t2-mnone", "A-t2-m4", "A-t4-mnone", "A-t4-m4"]
    assert plan["configs"][1]["pin"] == "mem=4" and plan["configs"][0]["pin"] is None
    assert "omp" not in plan["configs"][0] and plan["launches"] == 16
    # GOV-7: the pool each configuration's Siril receives (siril_environment), not a knob of this tool
    expected = received_pool(fixture["root"] / "cfg-A-t2-mnone")
    assert all((c["pool"], c["pool_source"]) == expected for c in plan["configs"])
    assert plan["disk_per_config_bytes"] == 3 * 12 * sweep.DEFAULT_FRAME_PIXELS * 2
    assert not fixture["root"].exists()


@pytest.mark.parametrize("extra, message", [
    (["--threads", "0"], "--threads"), (["--threads", "65"], "--threads"), (["--mem", "lots"], "--mem"),
    (["--stages", "stack,convert"], "--stages"), (["--stages", "convert,unknown"], "--stages"),
    (["--route", "C"], "--route"), (["--omp", "0"], "--omp"), (["--omp", "8"], "--omp"),
    (["--route", "B", "--omp", "4"], "--omp")])
def test_bad_options_are_refused(fixture, capsys, extra, message):
    assert sweep.main(argv(fixture, "--dry-run", *extra)) == 2
    assert message in capsys.readouterr().err


def test_route_b_pins_threads_and_memory(fixture, capsys, pool_variant):
    assert sweep.main(argv(fixture, "--dry-run", "--route", "B", "--threads", "8", "--mem", "4")) == 0
    config = json.loads(capsys.readouterr().out)["plan"]["configs"][0]
    assert config["pin"] == "cpu=8,mem=4" and "omp" not in config
    environment = siril.siril_environment(fixture["root"] / f"cfg-{config['id']}", siril.siril_pin("cpu=8,mem=4"))
    assert (config["pool"], config["pool_source"]) == resources.siril_pool_setting(environment)
    if pool_variant == "openblas":                       # the pin's OMP_NUM_THREADS=8 sizes it; OpenBLAS's knob not
        assert (config["pool"], config["pool_source"]) == (8, "OMP_NUM_THREADS")
        assert config["pool_environment"] == {"OMP_NUM_THREADS": "8", "OPENBLAS_NUM_THREADS": "1"}


def test_a_root_inside_the_users_aester_folder_is_refused(fixture, capsys):
    local = Path(os.environ["LOCALAPPDATA"]) / "Aester" / "sweep"
    args = argv(fixture, "--dry-run")
    args[args.index("--root") + 1] = str(local)
    assert sweep.main(args) == 2
    assert "outside" in capsys.readouterr().err


def test_a_stubbed_sweep_records_one_point_per_launch_and_cleans_up(fixture, monkeypatch, pool_variant):
    seen = []
    parent_omp = os.environ.get("OMP_NUM_THREADS")

    def fake_run_script(executable, folder, name, commands, cancel_check):
        assert os.environ.get("OMP_NUM_THREADS") == parent_omp             # the tool never sets a pool knob
        seen.append((Path(folder).name, name, list(commands), received_pool(folder, commands),
                     os.environ.get("AESTER_SIRIL_PIN")))
        stage = Path(folder) / "stage"
        assert sorted(p.name for p in stage.iterdir()) == ["frame00001.dng", "frame00002.dng", "frame00003.dng"]
        pg._publish_accounting({"peak_job_memory_bytes": 123, "child_peak_working_set_bytes": 45,
                                "child_peak_commit_bytes": 67, "returncode": 0, "descendant_processes": 1})

    args = sweep.build_parser().parse_args(argv(fixture, "--threads", "2", "--mem", "none,4",
                                                "--stages", "convert,probe"))
    plan = sweep.plan_sweep(args)
    monkeypatch.setattr(sweep, "_Affinity", lambda: type("A", (), {"set": lambda self, m: None,
                                                                    "restore": lambda self: None})())
    before = dict(os.environ)
    result = sweep.run_sweep(plan, run_script=fake_run_script)
    assert dict(os.environ) == before                                   # knobs restored
    assert [(f, n) for f, n, *_ in seen] == [("cfg-A-t2-mnone", "sweep_convert"), ("cfg-A-t2-mnone", "sweep_probe"),
                                             ("cfg-A-t2-m4", "sweep_convert"), ("cfg-A-t2-m4", "sweep_probe")]
    assert [pin for *_, pin in seen] == [None, None, "mem=4", "mem=4"]
    assert seen[0][2][-1].startswith("convertraw") and seen[1][2][-1].startswith("register")
    points = result["points"]
    assert len(points) == 4 and all(p["error"] is None for p in points)
    assert points[0]["peak_job_commit_bytes"] == 123 and points[0]["child_peak_working_set_bytes"] == 45
    assert points[0]["parts"] == ["convert"] and points[1]["parts"] == ["probe"]
    assert points[2]["predicted"]["allowance"] == 4 * resources.GiB
    assert points[0]["mask"] is not None
    # GOV-7 / cross-1: each point records, and predicts with, the pool of the block Siril received
    for point, (_, _, _, pool, _) in zip(points, seen):
        assert (point["pool"], point["pool_source"]) == pool
        if pool_variant == "product":
            # U-G3: one pool thread, except the decode pool of a RAW-decoding script (SIRIL-1 fix)
            assert point["pool"] == int(siril.siril_omp_threads(seen[points.index(point)][2]))
        else:                                   # OPENBLAS_NUM_THREADS=1 is recorded but the inherited OMP=8 sizes it
            assert (point["pool"], point["pool_source"]) == (8, "OMP_NUM_THREADS")
            assert point["pool_environment"]["OPENBLAS_NUM_THREADS"] == "1"
        expected = resources.siril_demand(point["parts"], threads=2, pool=point["pool"], frames=3,
                                          frame_pixels=sweep.DEFAULT_FRAME_PIXELS,
                                          allowance=point["predicted"]["allowance"])
        assert point["predicted"]["commit"] == expected["commit"]
    assert json.loads(Path(plan["out"]).read_text(encoding="utf-8"))["footprints_version"] == resources.FOOTPRINTS_VERSION
    assert not any(p.name.startswith("cfg-") for p in fixture["root"].iterdir())   # configurations deleted
    assert (fixture["root"] / "inputs").is_dir()


def test_a_failing_launch_is_recorded_not_retried(fixture, monkeypatch):
    calls = []

    def failing(executable, folder, name, commands, cancel_check):
        calls.append(name)
        raise RuntimeError("Siril sweep_convert 失败（3）")

    args = sweep.build_parser().parse_args(argv(fixture, "--threads", "2", "--mem", "none"))
    plan = sweep.plan_sweep(args)
    monkeypatch.setattr(sweep, "_Affinity", lambda: type("A", (), {"set": lambda self, m: None,
                                                                    "restore": lambda self: None})())
    result = sweep.run_sweep(plan, run_script=failing)
    assert calls == ["sweep_convert"]
    assert result["points"][0]["error"].startswith("RuntimeError")


def test_remove_config_refuses_anything_but_a_marked_configuration(fixture):
    root = fixture["root"]
    root.mkdir()
    (root / "cfg-x").mkdir()
    with pytest.raises(sweep.SweepError, match="marker"):
        sweep.remove_config(root / "cfg-x", root)
    (root / sweep.MARKER).write_text(json.dumps({"schema": sweep.SCHEMA}), encoding="utf-8")
    with pytest.raises(sweep.SweepError, match="not a configuration"):
        sweep.remove_config(root / "inputs", root)
    sweep.remove_config(root / "cfg-x", root)
    assert not (root / "cfg-x").exists()


def test_siril_is_started_only_through_run_script():
    """X12: no subprocess/Popen/guarded_process in the tool; Siril only via starstack.siril.run_script."""
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    imports = {alias.name for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))
               for alias in node.names}
    assert "subprocess" not in imports and "subprocess" not in names
    assert not {"Popen", "guarded_process", "system", "startfile", "spawnv"} & attributes
    assert "run_script" in attributes
