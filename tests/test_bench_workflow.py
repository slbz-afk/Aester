"""scripts/bench_workflow.py without a real Siril run: plan, input manifest, asset copy,
safe deletion, import hook, and the Windows process-tree sampler on tiny children."""
import contextlib
import hashlib
import importlib.util
import inspect
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from starstack import regions

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "bench_workflow.py"
spec = importlib.util.spec_from_file_location("bench_workflow_tested", SCRIPT)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)
windows = pytest.mark.skipif(os.name != "nt", reason="Windows process APIs")
# The P1c knobs' pins when none is given (efficiency P1c prereg 6.1)
P1C_PINS = {"parallel": None, "overlap": None, "trails": "on", "memory_view_gib": None, "commit_view_gib": None,
            "python_openblas": None}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def strings(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)
    elif isinstance(value, str):
        yield value


def tree_state(root):
    """Every file and directory below root with size, mtime and digest."""
    state = {}
    for path in sorted(Path(root).rglob("*")):
        info = path.stat()
        state[str(path)] = (path.is_dir(), info.st_size, info.st_mtime_ns, None if path.is_dir() else sha(path))
    return state


@pytest.fixture
def world(tmp_path, monkeypatch):
    """Fake inputs + baseline manifest + a user cache holding a region drawing, like the real one."""
    local = tmp_path / "Local"
    monkeypatch.setenv("LOCALAPPDATA", str(local))
    monkeypatch.delenv("OPENSSL_ia32cap", raising=False)
    inputs = tmp_path / "baseline" / "inputs"
    inputs.mkdir(parents=True)
    frames = []
    for number in range(1, 4):
        path = inputs / f"frame{number:05d}.dng"
        path.write_bytes(os.urandom(4096 + number))
        frames.append({"frame": number, "dng_path": f"outputs/x/inputs/{path.name}", "bytes": path.stat().st_size,
                       "sha256": sha(path)})
    manifest = tmp_path / "baseline.json"
    manifest.write_text(json.dumps({"version": "development-baseline-1", "frames": frames}), encoding="utf-8")
    cache = local / "Aester" / "Cache"
    reference_dir = cache / "reference-abc" / "work"
    reference_dir.mkdir(parents=True)
    ref = reference_dir / "ref_00001.fit"
    fits.PrimaryHDU(np.zeros((3, 24, 32), np.float32)).writeto(ref)
    project = regions.create_project({"state": "succeeded", "source_path": str(inputs / "frame00001.dng"),
                                      "source_sha256": frames[0]["sha256"], "linear_path": str(ref),
                                      "linear_sha256": sha(ref), "width": 32, "height": 24},
                                     [inputs / "frame00001.dng", inputs / "frame00002.dng"])
    project = regions.revise(project, operations=[{"kind": "horizon", "points": [[0, .7], [.5, .6], [1, .75]]}],
                             scene_mode="landscape")
    project = regions.draw_milkyway(project, [.5, .3, .2, .5], angle_degrees=30, feather=.5)
    regions.save_project(project, cache)
    regions.build_masks(project, cache / "region-assets")
    tools = tmp_path / "tools"
    tools.mkdir()
    for name in ("siril-cli.exe", "finite-cli.exe", "nox.onnx"):
        (tools / name).write_bytes(b"stand-in")
    monkeypatch.setattr(bench, "SIRIL", tools / "siril-cli.exe")
    monkeypatch.setattr(bench, "NOX_MODEL", tools / "nox.onnx")
    monkeypatch.setitem(bench.MODES, "F", dict(bench.MODES["F"], stack_engine_path=tools / "finite-cli.exe"))
    monkeypatch.setattr(bench, "MAX_CACHE_CHARS", 1000)      # pytest's temporary paths are long
    return {"tmp": tmp_path, "local": local, "cache": cache, "inputs": inputs, "manifest": manifest,
            "revision": project["revision"], "frames": frames, "bench": tmp_path / "bench", "ref": ref}


def prepare(world, **kwargs):
    return bench.prepare_assets(world["bench"], source_cache=world["cache"], revision=world["revision"],
                                manifest_path=world["manifest"], inputs_dir=world["inputs"], src=REPO / "src",
                                project_frames=(2, 3), **kwargs)


def common(world):
    return ["--bench-root", str(world["bench"]), "--inputs-dir", str(world["inputs"]),
            "--baseline-manifest", str(world["manifest"])]


# --------------------------------------------------------------------------- assets

def test_prepare_assets_copies_without_writing_the_source_and_rewrites_paths(world):
    before = tree_state(world["local"])
    record = prepare(world)
    assert tree_state(world["local"]) == before            # no byte, time or new file in the user's cache
    assets = world["bench"] / "assets"
    assert record["mask_check"] == {"sky_png": True, "milkyway_tif": True}
    for rel, entry in record["files"].items():
        copy = assets / rel
        assert copy.is_file() and sha(copy) == entry["sha256"] and bench._inside(entry["source"], world["cache"])
    reference_copy = assets / "reference" / "ref_00001.fit"
    assert sha(reference_copy) == sha(world["ref"])
    for count in (2, 3):
        project = json.loads((assets / f"regions-f{count:02d}.json").read_text(encoding="utf-8"))
        assert project["source_binding"]["reference"]["linear_path"] == str(reference_copy)
        paths = [str(world["inputs"] / f"frame{n:05d}.dng") for n in range(1, count + 1)]
        regions.validate_project(project, input_files=paths, verify_sources=True)
        assert not any(bench._inside(text, world["local"]) for text in strings(project) if os.path.isabs(text))
    # The drawing itself (geometry) is unchanged; only the binding moved.
    user = json.loads((world["cache"] / f"regions-{world['revision']}.json").read_text(encoding="utf-8"))
    derived = json.loads((assets / "regions-f02.json").read_text(encoding="utf-8"))
    for key in ("operations", "milkyway", "scene_mode", "coordinate_system", "schema"):
        assert derived[key] == user[key]
    # Second call only verifies.
    assert prepare(world)["files"] == record["files"] and tree_state(world["local"]) == before


def test_prepare_assets_refuses_a_destination_inside_the_users_aester_folder(world):
    with pytest.raises(bench.BenchError, match="outside"):
        bench.prepare_assets(world["local"] / "Aester" / "bench", source_cache=world["cache"],
                             revision=world["revision"], manifest_path=world["manifest"],
                             inputs_dir=world["inputs"], project_frames=(2,))


def test_changed_asset_copy_is_detected(world):
    prepare(world)
    target = world["bench"] / "assets" / "reference" / "ref_00001.fit"
    data = bytearray(target.read_bytes())
    data[-1] ^= 1
    target.write_bytes(bytes(data))
    with pytest.raises(bench.BenchError, match="changed"):
        bench.verify_assets(world["bench"])


# --------------------------------------------------------------------------- inputs

def test_input_manifest_mismatch_is_an_error(world, capsys):
    prepare(world)
    victim = world["inputs"] / "frame00002.dng"
    victim.write_bytes(b"x" * victim.stat().st_size)              # same size, other content
    assert bench.main(["--dry-run", "--mode", "L", "--frames", "2", *common(world)]) == 2
    assert "SHA-256 differs" in capsys.readouterr().err
    assert bench.main(["--dry-run", "--mode", "L", "--frames", "1", *common(world)]) == 0   # first frame still fine


def test_missing_or_resized_inputs_are_errors(world):
    (world["inputs"] / "frame00003.dng").write_bytes(b"short")
    with pytest.raises(bench.BenchError, match="size differs"):
        bench.verify_inputs(world["manifest"], world["inputs"], 3)
    (world["inputs"] / "frame00003.dng").unlink()
    with pytest.raises(bench.BenchError, match="missing"):
        bench.verify_inputs(world["manifest"], world["inputs"], 3)
    with pytest.raises(bench.BenchError, match="1..3"):
        bench.verify_inputs(world["manifest"], world["inputs"], 4)


# --------------------------------------------------------------------------- plan

@pytest.mark.parametrize("mode, range_mode, engine", [("L", "legacy", False), ("F", "finite_validity_v1", True),
                                                      ("R", "positive_power2_v1", False)])
def test_dry_run_resolves_every_path_inside_the_bench_root(world, capsys, mode, range_mode, engine):
    prepare(world)
    code = bench.main(["--dry-run", "--mode", mode, "--frames", "3", "--label", "R9", "--pin-siril", "cpu=16,mem=16",
                       "--pin-ort", "8", "--affinity", "16", "--mask-sha", *common(world)])
    assert code == 0
    report = json.loads(capsys.readouterr().out)
    plan = report["plan"]
    root = world["bench"]
    assert report["dry_run"] and report["assets"] == "prepared" and report["inputs_verified"] == 3
    for name, path in plan["paths"].items():
        assert bench._inside(path, root), name
        for forbidden in bench.forbidden_roots():
            assert not bench._inside(path, forbidden)
    submit = plan["submit"]
    assert bench._inside(submit["delivery_root"], plan["paths"]["slot"])
    assert submit["regions_path"] == str(root / "assets" / "regions-f03.json")
    assert submit["stack_range_mode"] == range_mode and bool(submit["stack_engine_path"]) is engine
    assert submit["backend_options"]["intra_op_threads"] == 8 and submit["backend_options"]["channel_order"] == "bgr"
    assert submit["idempotency_key"] == f"aester-bench-{mode}3"
    assert (submit["trail_default"], submit["milkyway_strength"], submit["background_strength"]) == ("meteor", .75, .85)
    assert submit["color_adjustments"] == {"warmth": -5.0, "tint": -6.0, "exposure_ev": .3}
    assert submit["save_starless"] and submit["save_star_layer"] and submit["delivery_format"] == "aester-delivery-2"
    env = plan["environment"]
    assert env["AESTER_SIRIL_PIN"] == "cpu=16,mem=16" and env["OPENSSL_ia32cap"] == bench.SHA_MASK
    assert bench._inside(env["AESTER_CACHE_ROOT"], root) and bench._inside(env["AESTER_COMPONENT_ROOT"], root)
    assert plan["pins"] == {"siril": {"cpu": 16, "mem": 16}, "ort_intra_op_threads": 8, "affinity_logical": 16,
                            "mask_sha": True, "render_workers": None, "frame_workers": None,
                            "governor": "shadow-full", "governor_facts": None, **P1C_PINS}
    assert env["AESTER_RENDER_WORKERS"] is None
    assert env["XDG_DATA_HOME"] == plan["paths"]["siril_xdg"] and bench._inside(env["XDG_DATA_HOME"], plan["paths"]["slot"])
    assert env["MSYSTEM"] and plan["siril_launch"]["MSYSTEM"] == env["MSYSTEM"]
    assert plan["simulation"]["label"] == "16 logical processors, no SHA, Siril mem 16 GiB (other memory unlimited)"
    assert plan["simulation"]["status"] == "ad hoc knobs" and plan["profile"] is None
    # P1c: the import hook is in every run, inside the slot (imports, memory views, OpenBLAS log, thread audit)
    assert plan["run_id"].endswith(f"-{mode}3-R9") and plan["hook"] and not plan["code_root_given"]
    assert env["PYTHONPATH"].split(os.pathsep) == [plan["code_src"], plan["paths"]["hook"]]
    for name in (bench.HOOK_LOG_ENV, bench.HOOK_EVENTS_ENV, bench.THREAD_LOG_ENV):
        assert bench._inside(env[name], plan["paths"]["hook"])
    assert env[bench.BLAS_LOG_ENV] == "1" and env[bench.HOOK_SHIM_ENV] is None
    assert plan["explicit_knobs"] == {"pin_siril": "cpu=16,mem=16", "pin_ort": 8, "affinity": 16, "mask_sha": True}
    assert not (root / "runs").exists() and not (root / "results").exists()      # a dry run creates nothing


def test_default_run_is_unpinned_and_clears_an_inherited_pin(world, capsys):
    prepare(world)
    assert bench.main(["--dry-run", "--mode", "L", "--frames", "2", *common(world)]) == 0
    plan = json.loads(capsys.readouterr().out)["plan"]
    assert plan["environment"]["AESTER_SIRIL_PIN"] is None and plan["environment"]["OPENSSL_ia32cap"] is None
    assert plan["environment"]["AESTER_RENDER_WORKERS"] is None          # an inherited override is cleared
    assert "AESTER_FRAME_WORKERS" in plan["environment"] and plan["environment"]["AESTER_FRAME_WORKERS"] is None
    assert plan["pins"]["frame_workers"] is None
    assert plan["submit"]["backend_options"]["intra_op_threads"] is None
    assert plan["simulation"] is None and plan["profile"] is None
    assert bench._apply_environment({"AESTER_BENCH_TEST_VAR": None}) == {}


def test_an_inherited_l_siril_layout_is_cleared(world, capsys, monkeypatch):
    """P1b X9: the diagnostics-only AESTER_L_SIRIL_LAYOUT never reaches a bench run from the shell."""
    prepare(world)
    monkeypatch.setenv("AESTER_L_SIRIL_LAYOUT", "combined")
    assert bench.main(["--dry-run", "--mode", "L", "--frames", "2", *common(world)]) == 0
    plan = json.loads(capsys.readouterr().out)["plan"]
    assert "AESTER_L_SIRIL_LAYOUT" in plan["environment"] and plan["environment"]["AESTER_L_SIRIL_LAYOUT"] is None
    assert bench._apply_environment({"AESTER_L_SIRIL_LAYOUT": None}) == {"AESTER_L_SIRIL_LAYOUT": "<removed 'combined'>"}
    assert "AESTER_L_SIRIL_LAYOUT" not in os.environ


@pytest.mark.parametrize("inherited, expected, source", [(None, "MINGW64", "bench default"),
                                                         ("UCRT64", "UCRT64", "inherited")])
def test_siril_launch_hygiene_keeps_an_inherited_msystem_and_a_private_xdg(world, capsys, monkeypatch,
                                                                           inherited, expected, source):
    """P1a harness hygiene: Siril skips its Python venv work and never touches %LOCALAPPDATA%\\siril."""
    prepare(world)
    if inherited is None:
        monkeypatch.delenv("MSYSTEM", raising=False)
    else:
        monkeypatch.setenv("MSYSTEM", inherited)
    monkeypatch.setenv("XDG_DATA_HOME", str(world["local"]))          # an inherited value is replaced
    assert bench.main(["--dry-run", "--mode", "F", "--frames", "2", *common(world)]) == 0
    plan = json.loads(capsys.readouterr().out)["plan"]
    env, launch = plan["environment"], plan["siril_launch"]
    assert env["MSYSTEM"] == launch["MSYSTEM"] == expected and launch["MSYSTEM_source"] == source
    assert env["XDG_DATA_HOME"] == launch["XDG_DATA_HOME"] == plan["paths"]["siril_xdg"]
    assert Path(env["XDG_DATA_HOME"]) == Path(plan["paths"]["slot"]) / "x"
    for forbidden in bench.forbidden_roots():
        assert not bench._inside(env["XDG_DATA_HOME"], forbidden)
    assert not bench._inside(env["XDG_DATA_HOME"], world["local"])
    assert bench.RELEVANT_ENV.match("MSYSTEM") and bench.RELEVANT_ENV.match("XDG_DATA_HOME")
    assert not bench.RELEVANT_ENV.match("XDG_CONFIG_HOME")
    assert "mode" not in launch                                         # only --siril-desktop names a mode
    # P1b review SIRIL-2: an old code root's stock Siril must not start or join the user's session bus (SF-2)
    assert env["DBUS_SESSION_BUS_ADDRESS"] == launch["DBUS_SESSION_BUS_ADDRESS"] == "disabled:"


def test_every_run_disables_the_session_bus_whatever_the_shell_says(world, capsys, monkeypatch):
    """P1b review SIRIL-2: the same value the product's siril_environment gives Siril (S-4d), for every process of the
    run, so a pre-S-4d --code-root (E1, B1 bisection) cannot end the user's GLib programs; recorded and applied."""
    from starstack import siril
    prepare(world)
    monkeypatch.setenv("DBUS_SESSION_BUS_ADDRESS", "autolaunch:")
    assert bench.main(["--dry-run", "--mode", "L", "--frames", "2", *common(world)]) == 0
    plan = json.loads(capsys.readouterr().out)["plan"]
    assert plan["environment"]["DBUS_SESSION_BUS_ADDRESS"] == bench.NO_SESSION_BUS == "disabled:"
    assert bench.NO_SESSION_BUS == siril._NO_SESSION_BUS["DBUS_SESSION_BUS_ADDRESS"]
    assert plan["siril_launch"]["DBUS_SESSION_BUS_ADDRESS"] == "disabled:"
    applied = bench._apply_environment({"DBUS_SESSION_BUS_ADDRESS": plan["environment"]["DBUS_SESSION_BUS_ADDRESS"]})
    assert applied == {"DBUS_SESSION_BUS_ADDRESS": "disabled:"} and os.environ["DBUS_SESSION_BUS_ADDRESS"] == "disabled:"


def test_siril_desktop_runs_like_an_explorer_gui_with_a_sentinel_data_folder(world, capsys, monkeypatch, tmp_path):
    """P1b W5: MSYSTEM removed for the bench and every child, XDG_DATA_HOME = the sentinel; nothing else changes."""
    prepare(world)
    monkeypatch.setenv("MSYSTEM", "MINGW64")
    desktop = tmp_path / "desktop-F"
    assert bench.main(["--dry-run", "--mode", "F", "--frames", "2", "--siril-desktop", str(desktop), *common(world)]) == 0
    plan = json.loads(capsys.readouterr().out)["plan"]
    assert plan["environment"]["MSYSTEM"] is None and plan["environment"]["XDG_DATA_HOME"] == str(desktop)
    assert plan["environment"]["DBUS_SESSION_BUS_ADDRESS"] == "disabled:"   # kept: old code roots run here (E1)
    assert plan["siril_launch"] == {"mode": "desktop-sentinel", "MSYSTEM": None,
                                    "MSYSTEM_source": "removed (--siril-desktop)", "XDG_DATA_HOME": str(desktop),
                                    "DBUS_SESSION_BUS_ADDRESS": "disabled:",
                                    "note": "end-to-end check: the summary's siril_desktop listing must not change"}
    assert bench.main(["--dry-run", "--mode", "F", "--frames", "2", *common(world)]) == 0
    default = json.loads(capsys.readouterr().out)["plan"]
    same = {k: v for k, v in plan["environment"].items() if k not in ("MSYSTEM", "XDG_DATA_HOME")}
    assert same == {k: v for k, v in default["environment"].items() if k not in ("MSYSTEM", "XDG_DATA_HOME")}
    assert (plan["submit"], plan["pins"]) == (default["submit"], default["pins"])
    applied = bench._apply_environment({"MSYSTEM": None})                # removed for this process and children
    assert applied == {"MSYSTEM": "<removed 'MINGW64'>"} and "MSYSTEM" not in os.environ
    assert not desktop.exists()                                          # a dry run creates nothing


def test_siril_desktop_refuses_the_users_real_data_folder(world, capsys, monkeypatch, tmp_path):
    prepare(world)
    known = tmp_path / "KnownLocal"
    monkeypatch.setattr(bench, "known_local_appdata", lambda: known)
    for refused, message in ((world["local"], "outside"),        # %LOCALAPPDATA% holds the user's Aester folder
                             (known, "sentinel"), (known / ".", "sentinel"),
                             (world["local"] / "Aester" / "desktop", "outside")):
        assert bench.main(["--dry-run", "--mode", "F", "--frames", "2", "--siril-desktop", str(refused),
                           *common(world)]) == 2
        assert message in capsys.readouterr().err, refused
    monkeypatch.setattr(bench, "forbidden_roots", lambda: [])
    assert bench.main(["--dry-run", "--mode", "F", "--frames", "2", "--siril-desktop", str(world["local"]),
                       *common(world)]) == 2
    assert "sentinel" in capsys.readouterr().err                        # LOCALAPPDATA itself, by the second rule
    assert bench.main(["--dry-run", "--mode", "F", "--frames", "2", "--siril-desktop", str(known / "sentinel"),
                       *common(world)]) == 0                             # a folder below it is a sentinel, not it


def _junction(link: Path, target: Path):
    """A directory junction (no privilege needed, unlike a symbolic link)."""
    import _winapi
    _winapi.CreateJunction(str(target), str(link))
    assert os.path.samefile(link, target)


def test_siril_desktop_refuses_other_spellings_of_the_real_data_folder(world, monkeypatch, tmp_path):
    """P1b review SIRIL-3: extended-length, device and UNC spellings are refused by their form (they named the real
    LocalAppData past the path comparisons); the real folder itself, also by identity (st_dev/st_ino)."""
    real = bench.known_local_appdata()                  # the machine's real LocalAppData (None off Windows)
    known = tmp_path / "KnownLocal"
    monkeypatch.setattr(bench, "known_local_appdata", lambda: known)
    for spelling in ("\\\\?\\" + str(known), "\\\\?\\" + str(tmp_path / "elsewhere"), "\\\\.\\" + str(known),
                     "\\\\?\\UNC\\localhost\\C$\\Users", "\\\\localhost\\C$\\Users\\x\\AppData\\Local",
                     "\\\\127.0.0.1\\c$\\x", "//localhost/C$/x", "\\\\server\\share\\desktop"):
        with pytest.raises(bench.BenchError, match="plain local folder"):
            bench.siril_desktop_folder(spelling)
    if real:                                            # the reviewer's spellings; refused by form, never opened
        drive, rest = os.path.splitdrive(str(real))
        for spelling in ("\\\\?\\" + str(real), "\\\\?\\UNC\\localhost\\" + drive[:1] + "$" + rest,
                         "\\\\localhost\\" + drive[:1] + "$" + rest, "\\\\127.0.0.1\\" + drive[:1] + "$" + rest):
            with pytest.raises(bench.BenchError, match="plain local folder"):
                bench.siril_desktop_folder(spelling)


@windows
def test_siril_desktop_refuses_the_real_folders_through_junctions(world, monkeypatch, tmp_path):
    """P1b review SIRIL-3: a junction (like a subst drive or an 8.3 name) is the same folder under another name."""
    known = tmp_path / "KnownLocal"
    (known / "siril").mkdir(parents=True)
    (world["local"] / "Aester").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(bench, "known_local_appdata", lambda: known)
    links = tmp_path / "links"
    links.mkdir()
    _junction(links / "local", known)                                   # DIR is LocalAppData
    (links / "other").mkdir()
    _junction(links / "other" / "siril", known / "siril")              # DIR\siril is the real siril folder
    _junction(links / "inside", known / "siril")                       # DIR is below the real siril folder
    _junction(links / "aester", world["local"] / "Aester")             # DIR is below %LOCALAPPDATA%\Aester
    for refused, message in ((links / "local", "sentinel"), (links / "other", "sentinel"),
                             (links / "inside" / "desk", "outside"), (links / "aester" / "desk", "outside")):
        with pytest.raises(bench.BenchError, match=message):
            bench.siril_desktop_folder(refused)
    for accepted in (tmp_path / "desktop", links / "plain", known / "sentinel"):   # missing folders: no identity
        assert bench.siril_desktop_folder(accepted) == accepted
    (tmp_path / "desktop" / "siril").mkdir(parents=True)               # a sentinel with its own siril\ is fine
    assert bench.siril_desktop_folder(tmp_path / "desktop") == tmp_path / "desktop"


def test_folder_listing_sees_every_change_below_the_sentinel(tmp_path):
    desktop = tmp_path / "desktop"
    assert bench.folder_listing(desktop) == {}
    (desktop / "siril" / "venv" / "Scripts").mkdir(parents=True)
    (desktop / "siril" / "venv" / "Scripts" / "python.exe").write_bytes(b"")
    before = bench.folder_listing(desktop)
    assert before == {"siril": ["dir"], "siril/venv": ["dir"], "siril/venv/Scripts": ["dir"],
                      "siril/venv/Scripts/python.exe": [0, hashlib.sha256(b"").hexdigest()]}
    (desktop / "siril" / ".python_module.tmp.4242").mkdir()
    assert set(bench.folder_listing(desktop)) - set(before) == {"siril/.python_module.tmp.4242"}
    (desktop / "siril" / "venv" / "Scripts" / "python.exe").write_bytes(b"x")
    assert bench.folder_listing(desktop)["siril/venv/Scripts/python.exe"] != before["siril/venv/Scripts/python.exe"]


def test_old_code_root_gets_the_hook_and_pin_shim(world, capsys, tmp_path):
    prepare(world)
    old = tmp_path / "old-checkout" / "src" / "starstack"
    old.mkdir(parents=True)
    (old / "__init__.py").write_text("", encoding="utf-8")
    (old / "siril.py").write_text("def run_script(*a, **k):\n    pass\n", encoding="utf-8")
    assert bench.main(["--dry-run", "--mode", "L", "--frames", "2", "--pin-siril", "cpu=4",
                       "--code-root", str(old.parents[1]), *common(world)]) == 0
    plan = json.loads(capsys.readouterr().out)["plan"]
    assert plan["hook"] and plan["pin_shim"] and not plan["native_pin_support"]
    assert plan["environment"]["PYTHONPATH"].split(os.pathsep) == [str(old.parent), plan["paths"]["hook"]]
    assert plan["environment"]["AESTER_BENCH_PIN_SHIM"] == "1"
    assert bench.main(["--dry-run", "--mode", "L", "--frames", "2", "--code-root", str(tmp_path), *common(world)]) == 2


@pytest.mark.parametrize("argv, message", [
    (["--mode", "L", "--pin-siril", "cpu=0"], "pin-siril"),
    (["--mode", "L", "--label", "../x"], "label"),
    (["--mode", "L", "--affinity", "65"], "affinity"),
    (["--mode", "L", "--pin-ort", "0"], "pin-ort"),
    (["--mode", "L", "--render-workers", "9"], "render-workers"),
    (["--mode", "L", "--render-workers", "0"], "render-workers"),
    (["--mode", "L", "--frame-workers", "0"], "frame-workers"),
    (["--mode", "L", "--frame-workers", "17"], "frame-workers"),
    (["--mode", "L", "--frame-workers", "eight"], "frame-workers"),
])
def test_invalid_options_are_refused(world, capsys, argv, message):
    assert bench.main(["--dry-run", *argv, *common(world)]) == 2
    assert message in capsys.readouterr().err


def test_ordinary16_profile_is_labelled_as_no_sha_and_unlimited_memory(world, capsys):
    prepare(world)
    assert bench.main(["--dry-run", "--mode", "L", "--frames", "2", "--profile", "ordinary16", "--label", "R6",
                       *common(world)]) == 0
    plan = json.loads(capsys.readouterr().out)["plan"]
    assert plan["pins"] == {"siril": None, "ort_intra_op_threads": 8, "affinity_logical": 16, "mask_sha": True,
                            "render_workers": None, "frame_workers": None, "governor": "shadow-full",
                            "governor_facts": None, **P1C_PINS}
    env = plan["environment"]
    assert env["OPENSSL_ia32cap"] == bench.SHA_MASK and env["AESTER_SIRIL_PIN"] is None
    assert env["AESTER_RENDER_WORKERS"] is None
    simulation = plan["simulation"]
    assert simulation["label"] == "16 logical processors, no SHA, unlimited memory"
    assert simulation["profile"] == "ordinary16" and simulation["status"].startswith("prereg R6")
    assert simulation["sha_extensions"] == "masked" and simulation["memory_limits"] == []
    notes = " ".join(simulation["notes"])
    assert "not a 6900HX stand-in" in notes and "事后: memory is not limited" in notes and "--pin-ort 8" in notes
    assert "memory truth" in notes


@pytest.mark.parametrize("mode, legacy_note", [("L", True), ("R", True), ("F", False)])
def test_6900hx_proxy_profile_keeps_sha_and_limits_siril_memory_and_render_workers(world, capsys, mode, legacy_note):
    prepare(world)
    assert bench.main(["--dry-run", "--mode", mode, "--frames", "2", "--profile", "6900hx-proxy", *common(world)]) == 0
    plan = json.loads(capsys.readouterr().out)["plan"]
    assert plan["pins"] == {"siril": {"mem": 9}, "ort_intra_op_threads": 8, "affinity_logical": 16, "mask_sha": False,
                            "render_workers": 3, "frame_workers": None, "governor": "shadow-full",
                            "governor_facts": bench.LAPTOP_GOVERNOR_FACTS, **P1C_PINS}
    env = plan["environment"]
    assert env["OPENSSL_ia32cap"] is None and env["AESTER_SIRIL_PIN"] == "mem=9" and env["AESTER_RENDER_WORKERS"] == "3"
    simulation = plan["simulation"]
    assert simulation["label"] == ("16 logical processors, SHA on, Siril mem 9 GiB, render workers 3 "
                                   "(other memory unlimited)")
    assert simulation["status"].startswith("事后") and simulation["sha_extensions"] == "host"
    notes = " ".join(simulation["notes"])
    assert "6900HX stand-in" not in notes and "not limited or modelled" in notes and "memory truth" in notes
    assert ("float32 ULP" in notes) is legacy_note


@pytest.mark.parametrize("extra, ok", [
    (["--profile", "6900hx-proxy", "--mask-sha"], False),
    (["--profile", "ordinary16", "--affinity", "8"], False),
    (["--profile", "6900hx-proxy", "--pin-siril", "cpu=16,mem=9"], False),
    (["--profile", "6900hx-proxy", "--render-workers", "4"], False),
    (["--profile", "ordinary16", "--affinity", "16", "--mask-sha", "--pin-ort", "8"], True),
    (["--profile", "6900hx-proxy", "--pin-siril", " MEM=9 "], True),
])
def test_explicit_knobs_must_agree_with_the_profile(world, capsys, extra, ok):
    prepare(world)
    code = bench.main(["--dry-run", "--mode", "L", "--frames", "2", *extra, *common(world)])
    captured = capsys.readouterr()
    assert code == (0 if ok else 2), captured.err
    if not ok:
        assert "conflicts with --profile" in captured.err


# --------------------------------------------------------------------------- governor (efficiency P1b-2)

def test_governor_defaults_to_shadow_full_and_clears_inherited_knobs(world, capsys, monkeypatch):
    """X9: AESTER_GOVERNOR / AESTER_GOVERNOR_FACTS are always set or cleared and recorded in the pins."""
    prepare(world)
    monkeypatch.setenv("AESTER_GOVERNOR", "off")
    monkeypatch.setenv("AESTER_GOVERNOR_FACTS", "total=1G,avail=1G,lps=1,cores=1")
    assert bench.main(["--dry-run", "--mode", "L", "--frames", "2", *common(world)]) == 0
    plan = json.loads(capsys.readouterr().out)["plan"]
    env = plan["environment"]
    assert env["AESTER_GOVERNOR"] == "shadow-full" and env["AESTER_GOVERNOR_FACTS"] is None
    assert plan["pins"]["governor"] == "shadow-full" and plan["pins"]["governor_facts"] is None
    assert bench.RELEVANT_ENV.match("AESTER_GOVERNOR") and bench.RELEVANT_ENV.match("AESTER_GOVERNOR_FACTS")
    applied = bench._apply_environment({"AESTER_GOVERNOR": env["AESTER_GOVERNOR"],
                                        "AESTER_GOVERNOR_FACTS": env["AESTER_GOVERNOR_FACTS"]})
    assert os.environ["AESTER_GOVERNOR"] == "shadow-full" and "AESTER_GOVERNOR_FACTS" not in os.environ
    assert applied["AESTER_GOVERNOR_FACTS"].startswith("<removed")


@pytest.mark.parametrize("extra, governor, facts", [
    (["--governor", "off"], "off", None),
    (["--governor", " Shadow "], "shadow", None),
    (["--governor-facts", "total=16G,avail=9G,lps=16,cores=8"], "shadow-full", "total=16G,avail=9G,lps=16,cores=8"),
    (["--profile", "6900hx-proxy"], "shadow-full", None),
    (["--profile", "ordinary16"], "shadow-full", None),
])
def test_governor_knobs_are_recorded(world, capsys, extra, governor, facts):
    prepare(world)
    assert bench.main(["--dry-run", "--mode", "F", "--frames", "2", *extra, *common(world)]) == 0
    plan = json.loads(capsys.readouterr().out)["plan"]
    if "6900hx-proxy" in extra:
        facts = bench.LAPTOP_GOVERNOR_FACTS
    assert plan["pins"]["governor"] == plan["environment"]["AESTER_GOVERNOR"] == governor
    assert plan["pins"]["governor_facts"] == plan["environment"]["AESTER_GOVERNOR_FACTS"] == facts


@pytest.mark.parametrize("extra, message", [
    (["--governor", "full"], "--governor must be"),
    (["--governor-facts", "total=16G;avail=9G"], "--governor-facts must look like"),
    (["--governor", "off", "--governor-facts", "total=16G,avail=9G,lps=16,cores=8"], "needs --governor"),
    (["--profile", "6900hx-proxy", "--governor-facts", "total=32G,avail=20G,lps=16,cores=8"], "conflicts with --profile"),
])
def test_governor_knobs_are_validated(world, capsys, extra, message):
    prepare(world)
    assert bench.main(["--dry-run", "--mode", "L", "--frames", "2", *extra, *common(world)]) == 2
    assert message in capsys.readouterr().err


def test_laptop_governor_facts_parse_in_the_product():
    from starstack import resources
    facts = resources.parse_whatif(bench.LAPTOP_GOVERNOR_FACTS)
    assert facts.cpu.logical_total == 16 and facts.cpu.physical_cores == 8 and facts.cpu.siril_pool_today == 16
    assert facts.memory.available_physical == 9 * 1024 ** 3 and facts.volumes[0].io_class == "nvme"
    assert bench.PROFILES["6900hx-proxy"]["knobs"]["governor_facts"] == bench.LAPTOP_GOVERNOR_FACTS
    assert "governor_facts" not in bench.PROFILES["ordinary16"]["knobs"]


def test_render_workers_view_and_the_laptop_numbers_behind_the_notes(monkeypatch):
    from starstack import performance
    assert bench.MAX_RENDER_WORKERS == performance.MAX_RENDER_WORKERS
    # The notes' figures: a 45.7 MP frame on 16 logical processors is CPU-capped at 8 with ample RAM,
    # and memory-capped at 3 with about 10 GiB free.
    monkeypatch.delenv("AESTER_RENDER_WORKERS", raising=False)
    assert performance.render_workers(*bench.FRAME_HW, cpu_count=16, memory_bytes=200 * bench.GIB) == 8
    assert performance.render_workers(*bench.FRAME_HW, cpu_count=16, memory_bytes=10 * bench.GIB) == 3
    view = bench.render_workers_view()
    assert view["frame_hw"] == [5520, 8280] and view["override"] is None
    assert view["workers"] == performance.render_workers(*bench.FRAME_HW)
    monkeypatch.setenv("AESTER_RENDER_WORKERS", "3")
    view = bench.render_workers_view()
    assert view["override"] == "3" and 1 <= view["workers"] <= 3


@pytest.mark.parametrize("value", ["1", "auto", "max", "16"])
def test_frame_workers_reaches_the_child_environment_and_the_pins(world, capsys, monkeypatch, value):
    """P1b-1: --frame-workers sets AESTER_FRAME_WORKERS for the run (an inherited value is replaced) and is recorded."""
    from starstack import frame_pool
    assert (bench.FRAME_WORKERS_ENV, bench.MAX_FRAME_WORKERS) == (frame_pool.ENV_WORKERS, frame_pool.MAX_FRAME_WORKERS)
    prepare(world)
    monkeypatch.setenv("AESTER_FRAME_WORKERS", "3")
    assert bench.main(["--dry-run", "--mode", "L", "--frames", "2", "--frame-workers", value, *common(world)]) == 0
    plan = json.loads(capsys.readouterr().out)["plan"]
    assert plan["environment"]["AESTER_FRAME_WORKERS"] == value and plan["pins"]["frame_workers"] == value
    monkeypatch.setenv("AESTER_FRAME_WORKERS", value)
    frame_pool.validate_environment()                   # every accepted value is one the job worker accepts


def test_a_cache_path_near_max_path_is_refused_early(world, capsys, monkeypatch):
    monkeypatch.setattr(bench, "MAX_CACHE_CHARS", 20)
    assert bench.main(["--dry-run", "--mode", "F", "--frames", "2", *common(world)]) == 2
    assert "shorter --bench-root" in capsys.readouterr().err


def test_bench_root_under_the_users_aester_folder_is_refused(world, capsys):
    argv = ["--dry-run", "--mode", "L", "--frames", "2", "--bench-root", str(world["local"] / "Aester" / "bench"),
            "--inputs-dir", str(world["inputs"]), "--baseline-manifest", str(world["manifest"])]
    assert bench.main(argv) == 2
    assert "outside" in capsys.readouterr().err


@windows
def test_real_run_without_prepared_assets_stops_before_creating_anything(world, capsys):
    assert bench.main(["--mode", "L", "--frames", "2", *common(world)]) == 2
    assert "prepare-assets" in capsys.readouterr().err
    assert not (world["bench"] / "runs").exists() and not (world["bench"] / "results").exists()


def test_inherited_sha_mask_without_the_flag_is_refused(world, capsys, monkeypatch):
    prepare(world)
    monkeypatch.setenv("OPENSSL_ia32cap", bench.SHA_MASK)
    assert bench.main(["--mode", "L", "--frames", "2", *common(world)]) == 2
    assert "OPENSSL_ia32cap" in capsys.readouterr().err


# --------------------------------------------------------------------------- safe deletion

def make_slot(runs, name, marker=True, slot_name=None):
    slot = runs / name
    (slot / "c" / "workflow-x" / "stack").mkdir(parents=True)
    (slot / "c" / "workflow-x" / "stack" / "light_00001.fit").write_bytes(b"data")
    readonly = slot / "d" / "final.jpg"
    readonly.parent.mkdir()
    readonly.write_bytes(b"jpeg")
    os.chmod(readonly, 0o444)
    if marker:
        (slot / bench.SLOT_MARKER).write_text(json.dumps({"slot": slot_name or name}), encoding="utf-8")
    return slot


def test_safe_remove_deletes_only_a_marked_slot_inside_runs(tmp_path):
    runs = tmp_path / "bench" / "runs"
    slot = make_slot(runs, "L20")
    bench.safe_remove(slot, runs)
    assert not slot.exists() and runs.is_dir()


@pytest.mark.parametrize("case", ["outside", "runs-itself", "nested", "bad-name", "no-marker", "wrong-marker",
                                  "not-named-runs"])
def test_safe_remove_refuses_everything_else(tmp_path, case):
    runs = tmp_path / "bench" / "runs"
    runs.mkdir(parents=True)
    target = {
        "outside": lambda: make_slot(tmp_path / "elsewhere", "L20"),
        "runs-itself": lambda: runs,
        "nested": lambda: make_slot(runs / "L20", "F48"),
        "bad-name": lambda: make_slot(runs, "results"),
        "no-marker": lambda: make_slot(runs, "F48", marker=False),
        "wrong-marker": lambda: make_slot(runs, "F48", slot_name="L20"),
        "not-named-runs": lambda: make_slot(tmp_path / "bench" / "other", "L20"),
    }[case]()
    runs_root = target.parent if case == "not-named-runs" else runs
    before = tree_state(tmp_path)
    with pytest.raises(bench.BenchError, match="refusing"):
        bench.safe_remove(target, runs_root)
    assert tree_state(tmp_path) == before


def test_safe_remove_refuses_the_users_aester_folder(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "Local"))
    runs = tmp_path / "Local" / "Aester" / "runs"
    slot = make_slot(runs, "L20")
    with pytest.raises(bench.BenchError, match="refusing"):
        bench.safe_remove(slot, runs)
    assert slot.is_dir()


@windows
def test_safe_remove_refuses_a_slot_containing_a_junction(tmp_path):
    import _winapi
    runs = tmp_path / "bench" / "runs"
    slot = make_slot(runs, "L20")
    precious = tmp_path / "precious"
    precious.mkdir()
    (precious / "keep.txt").write_text("keep", encoding="utf-8")
    _winapi.CreateJunction(str(precious), str(slot / "c" / "link"))
    with pytest.raises(bench.BenchError, match="junction"):
        bench.safe_remove(slot, runs)
    assert (precious / "keep.txt").read_text(encoding="utf-8") == "keep" and slot.is_dir()
    os.rmdir(slot / "c" / "link")


# --------------------------------------------------------------------------- after-run helpers

def test_pin_check_reads_every_script_header(tmp_path):
    work = tmp_path / "wf"
    (work / "stack" / "finite-1").mkdir(parents=True)
    (work / "stack" / "01_convert.ssf").write_text("requires 1.4.4 1.5.0\nsetcpu 16\nset core.mem_mode=1\n"
                                                   "set core.mem_amount=16\ncd work\n", encoding="utf-8")
    (work / "stack" / "finite-1" / "stack.ssf").write_text("requires 1.4.4 1.5.0\noffline\nsetcpu 16\n"
                                                           "set core.mem_mode=1\nset core.mem_amount=16\nstack q\n",
                                                           encoding="utf-8")
    (work / "stack" / "01_convert.log").write_bytes(b"aester: AESTER_SIRIL_PIN applied: setcpu 16\nlog: ok\n")
    result = bench.pin_check(work, {"cpu": 16, "mem": 16})
    assert result["ok"] and result["pinned"] == 2 and result["unpinned"] == [] and result["logs_with_note"] == 1
    assert not bench.pin_check(work, {"cpu": 8})["ok"]
    assert not bench.pin_check(work, None)["ok"]            # an unpinned run must not carry header pins
    (work / "stack" / "02_stack.ssf").write_text("requires 1.4.4 1.5.0\ncd work\n", encoding="utf-8")
    assert bench.pin_check(work, {"cpu": 16, "mem": 16})["unpinned"] == [str(Path("stack") / "02_stack.ssf")]


def test_collect_timings_finds_blocks_and_seconds_without_touching_digests():
    record = {"signature": "abc", "data_sha256": "def", "diagnostics": {"timings": {"total_seconds": 3.0}},
              "checkpoints": {"stack": {"data": {"seconds": 1.5, "sha256": "x"}}}, "finishing": [{"render_seconds": 2}]}
    paths = {item["path"] for item in bench.collect_timings(record)}
    assert paths == {"$.diagnostics.timings", "$.checkpoints.stack.data.seconds", "$.finishing[0].render_seconds"}


def test_fingerprint_is_skipped_with_a_note_when_the_verifier_is_absent(tmp_path):
    result = bench.fingerprint_run(tmp_path, tmp_path / "manifest.json", tool=tmp_path / "missing.py")
    assert result["status"] == "skipped" and "not present" in result["note"]
    assert not (tmp_path / "manifest.json").exists()


def test_fingerprint_prefers_the_verifiers_command_line_in_process(tmp_path):
    tool = tmp_path / "verify_workflow_equivalence.py"
    tool.write_text("import json, sys\n"
                    "def main(argv=None):\n"
                    "    assert argv[0] == 'fingerprint' and argv[2] == '--out'\n"
                    "    open(argv[3], 'x').write(json.dumps({'files': {}}))\n"
                    "    print('{\"files\": 0}'); print('progress', file=sys.stderr)\n"
                    "    return 0\n", encoding="utf-8")
    result = bench.fingerprint_run(tmp_path / "wf", tmp_path / "manifest.json", tool=tool)
    assert result["status"] == "ok" and result["method"].startswith("main") and '"files": 0' in result["report"]
    failing = tmp_path / "failing.py"
    failing.write_text("import sys\ndef main(argv=None):\n    print('boom', file=sys.stderr)\n    return 2\n",
                       encoding="utf-8")
    result = bench.fingerprint_run(tmp_path / "wf", tmp_path / "other.json", tool=failing)
    assert result["status"] == "error" and "boom" in result["stderr"]


def test_fingerprint_uses_the_verifiers_function_when_present(tmp_path):
    tool = tmp_path / "verify_workflow_equivalence.py"
    tool.write_text("def fingerprint(workflow_dir, *, progress=print):\n"
                    "    return {'version': 'fake', 'workflow_dir': workflow_dir, 'files': {'a': 1}}\n", encoding="utf-8")
    result = bench.fingerprint_run(tmp_path / "wf", tmp_path / "manifest.json", tool=tool)
    assert result["status"] == "ok" and result["files"] == 1
    assert json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))["version"] == "fake"


# --------------------------------------------------------------------------- import hook

def test_hook_records_the_imported_code_root_in_child_processes(tmp_path):
    code = tmp_path / "checkout" / "src"
    (code / "starstack").mkdir(parents=True)
    (code / "starstack" / "__init__.py").write_text("", encoding="utf-8")
    hook = tmp_path / "h"
    bench.write_hook(hook)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(code), str(hook)]),
               AESTER_BENCH_HOOK_LOG=str(hook / "imports"))
    env.pop("AESTER_BENCH_PIN_SHIM", None)
    subprocess.run([sys.executable, "-c", "import starstack"], env=env, check=True, timeout=120)
    result = bench.hook_imports(hook / "imports", code)
    assert result["ok"] and result["processes"] == 1
    other = bench.hook_imports(hook / "imports", REPO / "src")
    assert not other["ok"] and len(other["outside_code_root"]) == 1


# --------------------------------------------------------------------------- Windows measurement

@windows
def test_sampler_follows_children_and_grandchildren(tmp_path):
    win = bench.Win()
    sampler = bench.Sampler(win, tmp_path / "timeline.jsonl", interval=.2, stage=lambda: "test").start()
    grandchild = "import time; x = bytearray(24 << 20); time.sleep(1.5)"
    child = ("import subprocess, sys, time; x = bytearray(64 << 20); "
             f"subprocess.run([sys.executable, '-c', {grandchild!r}]); time.sleep(.5)")
    subprocess.run([sys.executable, "-c", child], check=True, timeout=120)
    deadline = time.monotonic() + 30
    while sampler.descendants_alive() and time.monotonic() < deadline:
        time.sleep(.2)
    summary = sampler.stop()
    me = os.getpid()
    # A venv python.exe is a launcher that starts the base interpreter, so each
    # Python level may be two processes; count depth along parent links.
    parent = {p["pid"]: p["ppid"] for p in summary["processes"]}

    def depth(pid):
        steps = 0
        while pid != me and pid in parent and steps < 20:
            pid, steps = parent[pid], steps + 1
        return steps if pid == me else None

    descendants = [p for p in summary["processes"] if not p["self"] and depth(p["pid"])]
    big = sorted((depth(p["pid"]), p["peak_private"] or 0) for p in descendants if (p["peak_private"] or 0) >= 24 << 20)
    assert len(big) >= 2 and big[0][1] >= 64 << 20 and big[-1][0] > big[0][0], summary["processes"]
    assert summary["samples"] >= 3 and summary["tree_cpu_s"] > 0 and summary["peak_tree_private_bytes"] > 64 << 20
    rows = [json.loads(line) for line in (tmp_path / "timeline.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[0]["stage"] == "test" and all("sys_avail" in row for row in rows)


@windows
def test_job_accounting_covers_the_whole_tree(tmp_path):
    code = (f"import importlib.util, json, subprocess, sys\n"
            f"spec = importlib.util.spec_from_file_location('b', {str(SCRIPT)!r}); b = importlib.util.module_from_spec(spec)\n"
            f"spec.loader.exec_module(b)\n"
            f"w = b.Win(); job = w.own_job()\n"
            f"subprocess.run([sys.executable, '-c', 'x = bytearray(80 << 20)'], check=True)\n"
            f"print(json.dumps(w.job_accounting(job)))\n")
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stderr
    accounting = json.loads(run.stdout)
    assert accounting["processes_total"] >= 1 and accounting["peak_process_commit_bytes"] >= 80 << 20
    assert accounting["cpu_user_s"] + accounting["cpu_kernel_s"] > 0


@windows
def test_check_environment_verifies_affinity_inheritance_and_the_sha_mask():
    run = subprocess.run([sys.executable, "-X", "utf8", str(SCRIPT), "--check-environment", "--affinity", "2",
                          "--mask-sha"], capture_output=True, text=True, timeout=300)
    assert run.returncode == 0, run.stderr
    report = json.loads(run.stdout)
    assert report["affinity"]["inherited"] and report["affinity"]["mask"] == "0x3"
    assert len(report["affinity"]["descendant_masks"]) == 2
    assert report["sha"]["digests_match"] and report["sha"]["masked"]["hashlib_ok"]


# --------------------------------------------------------------------------- whole run() against a fake code root

FAKE_SIRIL = '''"""Pre-pin style adapter: the bench hook must add the pin in the worker process."""
import subprocess
import sys
from contextlib import contextmanager

PROBE = "import os,sys; open(sys.argv[1],'w').write(os.environ.get('OMP_NUM_THREADS','-'))"


@contextmanager
def guarded_process(args, **kwargs):
    process = subprocess.Popen([sys.executable, "-c", PROBE, args[-1] + ".omp"], stdout=kwargs.get("stdout"),
                               stderr=subprocess.STDOUT)
    try:
        yield process
    finally:
        process.wait()


def run_script(executable, job_dir, name, commands, cancel_check, **timeouts):
    script = job_dir / f"{name}.ssf"
    script.write_text("requires 1.4.4 1.5.0\\n" + "\\n".join(commands) + "\\n", encoding="utf-8")
    with (job_dir / f"{name}.log").open("ab", buffering=0) as output, guarded_process(
            [executable, "-s", str(script)], stdin=subprocess.DEVNULL, stdout=output,
            stderr=subprocess.STDOUT, creationflags=0) as process:
        process.wait()
'''
FAKE_WORKFLOW = '''"""Stand-in for starstack.workflow: same API, a detached worker, no Siril."""
import json
import os
import subprocess
import sys
from pathlib import Path

TERMINAL = {"succeeded", "failed", "cancelled"}


def _write(directory, **fields):
    path = Path(directory) / "workflow.json"
    record = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    record.update(fields)
    temp = path.with_name("workflow.tmp")
    temp.write_text(json.dumps(record), encoding="utf-8")
    os.replace(temp, path)
    return record


def submit_workflow(input_dir, output_root, *, idempotency_key, input_files, delivery_root, **kwargs):
    directory = Path(output_root) / ("workflow-" + idempotency_key)
    directory.mkdir()
    record = _write(directory, workflow_dir=str(directory), state="queued", stage="queued", created_at="t0",
                    request={"input_dir": input_dir, "input_files": input_files, "kwargs": kwargs})
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1]) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.Popen([sys.executable, "-m", "starstack.fakeworker", str(directory), delivery_root], env=env,
                     stdin=subprocess.DEVNULL, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return record


def workflow_status(directory):
    return json.loads((Path(directory) / "workflow.json").read_text(encoding="utf-8"))


def cancel_workflow(directory):
    (Path(directory) / "cancel.request").touch()
'''
FAKE_WORKER = '''import json
import os
import sys
import time
from pathlib import Path

from starstack import siril, workflow

directory, delivery = Path(sys.argv[1]), Path(sys.argv[2])
workflow._write(directory, state="running", stage="stack")
job = directory / "stack" / "job-1"
job.mkdir(parents=True)
siril.run_script("fake-siril", job, "01_convert", ["cd work", "convert light"], lambda: None)
siril.run_script("fake-siril", job, "02_stack", ["offline", "stack q"], lambda: None)
ballast = bytearray(48 << 20)
time.sleep(1.5)
omp = {p.name: p.read_text() for p in job.glob("*.omp")}
xdg = os.environ.get("XDG_DATA_HOME")
(job / "job.json").write_text(json.dumps({"state": "succeeded", "created_at": "a", "omp_seen": omp,
                                          "pin_env": os.environ.get("AESTER_SIRIL_PIN"),
                                          "siril_launch": {"MSYSTEM": os.environ.get("MSYSTEM"), "XDG_DATA_HOME": xdg,
                                                           "xdg_is_dir": bool(xdg) and Path(xdg).is_dir(),
                                                           "bus": os.environ.get("DBUS_SESSION_BUS_ADDRESS")},
                                          "diagnostics": {"timings": {"total_seconds": 1.5}}}), encoding="utf-8")
base = delivery / "delivery-1"
base.mkdir(parents=True)
(base / "final.jpg").write_bytes(b"jpeg")
workflow._write(directory, state="running", stage="trail_addback", base_delivery_dir=str(base),
                stack_job_dir=str(job), trail_addback_pending=True)
time.sleep(1.0)
workflow._write(directory, state="succeeded", stage="complete", trail_addback_pending=False,
                trail_review={"state": "needs_review", "kept_count": 1}, updated_at="t9")
'''
FAKE_REGIONS = '''import hashlib
import json


def _revision(project):
    body = {k: v for k, v in project.items() if k not in ("revision", "project_path")}
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()
'''
DRIVER = '''import importlib.util, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location("b", sys.argv[1])
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)
b.MAX_CACHE_CHARS = 1000
b.SIRIL, b.NOX_MODEL, b.VERIFIER = Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
sys.exit(b.main(sys.argv[5:]))
'''


@windows
def test_whole_run_on_a_fake_old_code_root_measures_pins_and_cleans_up(world, tmp_path):
    prepare(world)
    code = tmp_path / "old-checkout" / "src" / "starstack"
    code.mkdir(parents=True)
    for name, text in (("__init__.py", ""), ("siril.py", FAKE_SIRIL), ("workflow.py", FAKE_WORKFLOW),
                       ("fakeworker.py", FAKE_WORKER), ("regions.py", FAKE_REGIONS)):
        (code / name).write_text(text, encoding="utf-8")
    tools = world["tmp"] / "tools"
    argv = ["--mode", "L", "--frames", "2", "--label", "fake", "--pin-siril", "cpu=3,mem=2", "--pin-ort", "2",
            "--affinity", "2", "--code-root", str(code.parents[1]), "--sample-seconds", "0.2",
            "--poll-seconds", "0.1", *common(world)]
    env = {k: v for k, v in os.environ.items()
           if k not in ("PYTHONPATH", "AESTER_SIRIL_PIN", "OPENSSL_ia32cap", "MSYSTEM", "XDG_DATA_HOME")}
    env["LOCALAPPDATA"] = str(world["local"])
    run = subprocess.run([sys.executable, "-X", "utf8", "-c", DRIVER, str(SCRIPT), str(tools / "siril-cli.exe"),
                          str(tools / "nox.onnx"), str(tmp_path / "no-verifier.py"), *argv],
                         capture_output=True, text=True, timeout=600, env=env)
    assert run.returncode == 0, run.stdout + run.stderr
    results = list((world["bench"] / "results").iterdir())
    assert len(results) == 1 and results[0].name.endswith("-L2-fake")
    summary = json.loads((results[0] / "summary.json").read_text(encoding="utf-8"))
    assert summary["state"] == "succeeded" and summary["deleted"] and not (world["bench"] / "runs" / "L2").exists()
    wall = summary["wall"]
    assert 0 < wall["submit_to_base_s"] < wall["submit_to_final_s"] and wall["addback_s"] > .5
    assert summary["transitions"][-1]["stage"] == "complete"
    job = json.loads((results[0] / "records" / "job.json").read_text(encoding="utf-8"))
    assert job["pin_env"] == "cpu=3,mem=2"
    assert job["omp_seen"] == {"01_convert.ssf.omp": "3", "02_stack.ssf.omp": "3"}   # set for the Siril stand-in only
    xdg = summary["plan"]["paths"]["siril_xdg"]               # Siril launch hygiene reaches the workflow's processes
    assert Path(xdg).parent.name == "L2" and Path(xdg).name == "x"
    assert job["siril_launch"] == {"MSYSTEM": "MINGW64", "XDG_DATA_HOME": xdg, "xdg_is_dir": True, "bus": "disabled:"}
    assert summary["environment_applied"]["MSYSTEM"] == "MINGW64"
    assert summary["environment_applied"]["XDG_DATA_HOME"] == xdg == summary["plan"]["siril_launch"]["XDG_DATA_HOME"]
    assert summary["environment_applied"]["DBUS_SESSION_BUS_ADDRESS"] == "disabled:"      # SIRIL-2, recorded
    assert summary["plan"]["siril_launch"]["MSYSTEM_source"] == "bench default"
    check = summary["siril_pin_check"]
    assert check["ok"] and check["pinned"] == 2 and check["logs_with_note"] == 2
    assert summary["code_root_check"]["ok"] and summary["code_root_check"]["processes"] >= 2
    sampler = summary["sampler"]
    assert sampler["affinity_check"]["mismatched"] == [] and sampler["affinity_check"]["checked"] >= 3
    assert sampler["samples"] >= 5 and sampler["peak_tree_private_bytes"] > 48 << 20
    assert summary["job_accounting"]["state"] == "ok" and summary["job_accounting"]["processes_total"] >= 3
    assert summary["fingerprint"]["status"] == "skipped"
    assert summary["plan"]["submit"]["backend_options"]["intra_op_threads"] == 2
    assert summary["simulation"]["label"] == "2 logical processors, SHA on, Siril mem 2 GiB (other memory unlimited)"
    assert summary["render_workers_at_start"]["status"] == "unavailable"     # the old checkout has no performance.py
    assert json.loads(run.stdout.strip().splitlines()[-1])["simulation"] == summary["simulation"]["label"]
    assert "$.diagnostics.timings" in {t["path"] for t in summary["record_timings"]["job"]}
    rows = [json.loads(row) for row in (results[0] / "timeline.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(row.get("stage") == "running/trail_addback" for row in rows)


@windows
def test_siril_desktop_whole_run_lists_the_sentinel_before_and_after(world, tmp_path):
    """W5 through the real bench run path: the fake old code behaves like a Siril launch before S-4b and removes the
    sentinel's venv, which the summary must show (the E2E-2 positive control, in miniature)."""
    prepare(world)
    code = tmp_path / "old-checkout" / "src" / "starstack"
    code.mkdir(parents=True)
    worker = FAKE_WORKER.replace('siril.run_script("fake-siril", job, "01_convert"',
                                 'import shutil; shutil.rmtree(Path(xdg) / "siril" / "venv")\n'
                                 'siril.run_script("fake-siril", job, "01_convert"')
    worker = worker.replace('job.mkdir(parents=True)\n', 'job.mkdir(parents=True)\nxdg = os.environ.get("XDG_DATA_HOME")\n')
    assert worker.count("rmtree") == 1
    for name, text in (("__init__.py", ""), ("siril.py", FAKE_SIRIL), ("workflow.py", FAKE_WORKFLOW),
                       ("fakeworker.py", worker), ("regions.py", FAKE_REGIONS)):
        (code / name).write_text(text, encoding="utf-8")
    desktop = tmp_path / "desktop"
    (desktop / "siril" / "venv" / "Scripts").mkdir(parents=True)
    (desktop / "siril" / "venv" / "Scripts" / "python.exe").write_bytes(b"")
    (desktop / "siril" / ".python_module").mkdir()
    tools = world["tmp"] / "tools"
    argv = ["--mode", "L", "--frames", "2", "--label", "desk", "--code-root", str(code.parents[1]),
            "--siril-desktop", str(desktop), "--sample-seconds", "0.2", "--poll-seconds", "0.1", *common(world)]
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "AESTER_SIRIL_PIN", "OPENSSL_ia32cap")}
    env.update(LOCALAPPDATA=str(world["local"]), MSYSTEM="MINGW64")
    run = subprocess.run([sys.executable, "-X", "utf8", "-c", DRIVER, str(SCRIPT), str(tools / "siril-cli.exe"),
                          str(tools / "nox.onnx"), str(tmp_path / "no-verifier.py"), *argv],
                         capture_output=True, text=True, timeout=600, env=env)
    assert run.returncode == 0, run.stdout + run.stderr
    (results,) = list((world["bench"] / "results").iterdir())
    summary = json.loads((results / "summary.json").read_text(encoding="utf-8"))
    job = json.loads((results / "records" / "job.json").read_text(encoding="utf-8"))
    assert job["siril_launch"] == {"MSYSTEM": None, "XDG_DATA_HOME": str(desktop), "xdg_is_dir": True,
                                   "bus": "disabled:"}
    listing = summary["siril_desktop"]
    assert listing["path"] == str(desktop) and listing["identical"] is False
    assert set(listing["before"]) - set(listing["after"]) == {"siril/venv", "siril/venv/Scripts",
                                                              "siril/venv/Scripts/python.exe"}
    assert summary["environment_applied"]["MSYSTEM"] == "<removed 'MINGW64'>"


def test_submitter_hashing_is_the_change_around_the_submit(tmp_path):
    """Efficiency P1 H4: the submitting process's own hashing is in no record, so the bench stores it."""
    from starstack import hashing
    before = bench.submitter_hashing()
    assert "hashed_bytes" in before and "hash_saved_bytes" in before and "hashed_memory_bytes" in before
    path = tmp_path / "input.bin"
    path.write_bytes(b"x" * 5000)
    hashing.sha256_file(path, source="test.submitter")
    hashing.sha256_bytes(b"y" * 300, source="test.submitter")
    change = bench.submitter_hashing(before)
    assert change["hashed_bytes"] == 5000 and change["hashed_files"] == 1
    assert change["hashed_memory_bytes"] == 300 and change["hash_saved_bytes"] == 0


# --------------------------------------------------------------------------- efficiency P1c (prereg 6.1)

def dry_plan(world, capsys, *extra, mode="L"):
    code = bench.main(["--dry-run", "--mode", mode, "--frames", "2", *extra, *common(world)])
    captured = capsys.readouterr()
    assert code == 0, captured.err
    return json.loads(captured.out)["plan"]


def test_parallel_constants_match_the_product():
    from starstack import frame_pool, hashing, performance, transient_stage
    assert bench.TRANSIENT_WORKERS_ENV == transient_stage.WORKERS_ENV
    assert (bench.HASH_WORKERS_ENV, bench.MAX_HASH_WORKERS) == (hashing.ENV_WORKERS, hashing.MAX_WORKERS)
    assert bench.MAX_RENDER_WORKERS == performance.MAX_RENDER_WORKERS
    assert bench.MAX_SCAN_WORKERS == getattr(transient_stage, "SCAN_WORKER_CAP", 8)        # T-2's cap once it exists
    assert set(bench.PARALLEL_TABLE) == set(bench.PARALLEL_MODES)
    assert all(set(values) == set(bench.PARALLEL_VARIABLES) for values in bench.PARALLEL_TABLE.values())
    assert frame_pool.ENV_WORKERS == bench.FRAME_WORKERS_ENV


@pytest.mark.parametrize("mode", ["serial", "default", "max"])
def test_parallel_sets_every_variable_and_the_product_reads_them(world, capsys, monkeypatch, mode):
    """--parallel: the prereg's table, every value one the product accepts; an inherited value is replaced."""
    from starstack import frame_pool, hashing, performance, transient_stage
    prepare(world)
    for name in bench.PARALLEL_VARIABLES:
        monkeypatch.setenv(name, "3")                                   # inherited from the shell: replaced or cleared
    plan = dry_plan(world, capsys, "--parallel", mode)
    env = plan["environment"]
    assert {name: env[name] for name in bench.PARALLEL_VARIABLES} == bench.PARALLEL_TABLE[mode]
    assert plan["pins"]["parallel"] == mode and plan["parallel"]["effective"] == mode
    assert plan["parallel"]["environment"] == bench.PARALLEL_TABLE[mode]
    expected = {"serial": ("1", "1", "1", "1", "off"), "default": (None, None, "auto", None, "auto"),
                "max": ("max", "8", "8", "32", "auto")}[mode]
    assert tuple(env[name] for name in bench.PARALLEL_VARIABLES) == expected
    bench._apply_environment({name: env[name] for name in bench.PARALLEL_VARIABLES})
    frame_pool.validate_environment()
    assert performance.render_workers(*bench.FRAME_HW, cpu_count=64, memory_bytes=200 * bench.GIB) == \
        {"serial": 1, "default": 8, "max": 8}[mode]
    assert hashing.resolve_workers(4) == {"serial": 1, "default": 4, "max": 32}[mode]
    expected = {"serial": 1, "max": 8}.get(mode)
    if expected is None:        # P1c T-2 automatic: today's floor up to the scan's ceiling, as memory allows
        from starstack import resources
        floor, ceiling = resources.scan_process_bounds(transient_stage.SCAN_WORKER_CAP)
        assert floor <= transient_stage.default_workers() <= max(floor, ceiling)
    else:
        assert transient_stage.default_workers() == expected


def test_without_parallel_the_variables_are_cleared_or_set_by_their_knobs(world, capsys, monkeypatch):
    prepare(world)
    for name in bench.PARALLEL_VARIABLES:
        monkeypatch.setenv(name, "2")
    plan = dry_plan(world, capsys)
    assert all(plan["environment"][name] is None for name in bench.PARALLEL_VARIABLES)       # X9: none leaks in
    assert plan["parallel"] == {"mode": None, "effective": "default", "environment": {n: None for n in
                                                                                      bench.PARALLEL_VARIABLES},
                                "knobs": [], "tier": []}
    plan = dry_plan(world, capsys, "--frame-workers", "4", "--overlap", "off")
    assert plan["environment"]["AESTER_FRAME_WORKERS"] == "4" and plan["environment"]["AESTER_OVERLAP"] == "off"
    assert plan["parallel"]["effective"] == "custom" and plan["parallel"]["knobs"] == ["frame_workers", "overlap"]
    plan = dry_plan(world, capsys, "--overlap", "on")
    assert plan["environment"]["AESTER_OVERLAP"] == "auto" and plan["pins"]["overlap"] == "on"
    # a knob written at its documented default is the default, not a fixed parameter
    assert plan["parallel"]["effective"] == "default" and plan["parallel"]["knobs"] == []
    plan = dry_plan(world, capsys, "--frame-workers", "auto")
    assert plan["environment"]["AESTER_FRAME_WORKERS"] == "auto" and plan["parallel"]["effective"] == "default"
    plan = dry_plan(world, capsys, "--frame-workers", "auto", "--overlap", "off")
    assert plan["parallel"]["effective"] == "custom" and plan["parallel"]["knobs"] == ["overlap"]


@pytest.mark.parametrize("extra, message", [
    (["--parallel", "serial", "--frame-workers", "4"], "--frame-workers (AESTER_FRAME_WORKERS=4) conflicts with --parallel serial"),
    (["--parallel", "max", "--render-workers", "3"], "conflicts with --parallel max"),
    (["--parallel", "default", "--overlap", "off"], "or omit --parallel"),
    (["--parallel", "serial", "--overlap", "on"], "conflicts with --parallel serial"),
    (["--parallel", "serial", "--profile", "6900hx-proxy-v2"], "--profile 6900hx-proxy-v2 (AESTER_RENDER_WORKERS=3)"),
    (["--parallel", "fast"], "invalid choice"),
])
def test_parallel_conflicts_are_refused(world, capsys, extra, message):
    prepare(world)
    with pytest.raises(SystemExit) if "fast" in extra else contextlib.nullcontext():
        assert bench.main(["--dry-run", "--mode", "L", "--frames", "2", *extra, *common(world)]) == 2
    assert message in capsys.readouterr().err


def test_parallel_agreeing_knobs_and_a_profiles_default_are_accepted(world, capsys):
    prepare(world)
    plan = dry_plan(world, capsys, "--parallel", "max", "--frame-workers", "max", "--render-workers", "8")
    assert plan["environment"]["AESTER_FRAME_WORKERS"] == "max" and plan["environment"]["AESTER_RENDER_WORKERS"] == "8"
    plan = dry_plan(world, capsys, "--parallel", "default", "--frame-workers", "auto", "--overlap", "on")
    assert plan["environment"]["AESTER_FRAME_WORKERS"] is None and plan["environment"]["AESTER_OVERLAP"] == "auto"
    # under 'default' a profile's own knob stands: the simulated laptop's default render worker count
    plan = dry_plan(world, capsys, "--parallel", "default", "--profile", "6900hx-proxy-v2")
    assert plan["environment"]["AESTER_RENDER_WORKERS"] == "3" and plan["parallel"]["tier"] == ["render_workers"]
    assert plan["parallel"]["effective"] == "default"
    plan = dry_plan(world, capsys, "--profile", "6900hx-proxy-v2")
    assert plan["parallel"]["effective"] == "default" and plan["parallel"]["tier"] == ["render_workers"]


def test_trails_off_is_the_gui_default(world, capsys):
    from starstack import workflow
    prepare(world)
    plan = dry_plan(world, capsys, "--trails", "off")
    assert plan["submit"]["trails_enabled"] is False and "trail_default" not in plan["submit"]
    assert plan["pins"]["trails"] == "off" and plan["explicit_knobs"] == {"trails": "off"}
    assert dry_plan(world, capsys)["submit"]["trail_default"] == "meteor"                    # unchanged default
    assert "trail_default" in inspect.signature(workflow.submit_workflow).parameters


def test_memory_views_and_python_openblas_reach_the_environment(world, capsys):
    prepare(world)
    plan = dry_plan(world, capsys, "--memory-view", "9", "--commit-view", "14.5", "--python-openblas", "Haswell:16")
    env = plan["environment"]
    assert env[bench.MEMORY_VIEW_ENV] == str(9 * bench.GIB) and env[bench.COMMIT_VIEW_ENV] == str(int(14.5 * bench.GIB))
    assert (env["OPENBLAS_CORETYPE"], env["OPENBLAS_NUM_THREADS"]) == ("Haswell", "16")
    assert env[bench.PYTHON_ONLY_ENV] == "OPENBLAS_CORETYPE,OPENBLAS_NUM_THREADS"
    assert (plan["pins"]["memory_view_gib"], plan["pins"]["commit_view_gib"]) == (9.0, 14.5)
    assert plan["pins"]["python_openblas"] == {"OPENBLAS_CORETYPE": "Haswell", "OPENBLAS_NUM_THREADS": "16"}
    assert plan["simulation"]["memory_limits"] == ["memory view 9 GiB", "commit view 14.5 GiB"]
    plain = dry_plan(world, capsys)["environment"]
    assert all(plain[name] is None for name in (bench.MEMORY_VIEW_ENV, bench.COMMIT_VIEW_ENV, "OPENBLAS_CORETYPE",
                                                "OPENBLAS_NUM_THREADS", bench.PYTHON_ONLY_ENV))


@pytest.mark.parametrize("extra, message", [
    (["--memory-view", "0"], "--memory-view must be"), (["--commit-view", "lots"], "--commit-view must be"),
    (["--python-openblas", "Haswell"], "CORETYPE:THREADS"), (["--python-openblas", "Haswell:0"], "CORETYPE:THREADS"),
    (["--profile", "6900hx-proxy-v2", "--memory-view", "8"], "conflicts with --profile"),
    (["--profile", "6900hx-proxy-v2", "--python-openblas", "Zen:16"], "conflicts with --profile"),
    (["--cancel-when", "stack:stage=x"], "RECORD:FIELD=VALUE"), (["--cancel-when", "job:stage"], "RECORD:FIELD=VALUE"),
    (["--cancel-when", "job:stage=stack", "--keep"], "drop --keep"),
])
def test_p1c_knobs_are_validated(world, capsys, extra, message):
    prepare(world)
    assert bench.main(["--dry-run", "--mode", "L", "--frames", "2", *extra, *common(world)]) == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize("mode", ["L", "F"])
def test_6900hx_proxy_v2_adds_the_views_and_the_laptops_openblas(world, capsys, mode):
    prepare(world)
    plan = dry_plan(world, capsys, "--profile", "6900hx-proxy-v2", mode=mode)
    assert plan["pins"] == {"siril": {"mem": 9}, "ort_intra_op_threads": None, "affinity_logical": 16,
                            "mask_sha": False, "render_workers": 3, "frame_workers": None, "governor": "shadow-full",
                            "governor_facts": bench.LAPTOP_GOVERNOR_FACTS, "parallel": None, "overlap": None,
                            "trails": "on", "memory_view_gib": 9.0, "commit_view_gib": 14.0,
                            "python_openblas": {"OPENBLAS_CORETYPE": "Haswell", "OPENBLAS_NUM_THREADS": "16"}}
    # the views are the governor facts' laptop (avail=9G, commit_avail=14G)
    facts = dict(part.split("=") for part in bench.LAPTOP_GOVERNOR_FACTS.split(","))
    assert (facts["avail"], facts["commit_avail"]) == ("9G", "14G")
    assert plan["submit"]["backend_options"]["intra_op_threads"] is None            # EFF-NOX writes 8 at admission
    simulation = plan["simulation"]
    assert simulation["label"] == ("16 logical processors, SHA on, Siril mem 9 GiB, render workers 3, memory view 9 GiB, "
                                   "commit view 14 GiB (other memory unlimited)")
    assert simulation["status"].startswith("prereg P1c 6.1") and simulation["python_openblas"]["OPENBLAS_CORETYPE"] == "Haswell"
    notes = " ".join(simulation["notes"])
    assert "available_memory / available_commit" in notes and "OPENBLAS_CORETYPE=Haswell" in notes
    assert "not limited or modelled" not in notes                                   # v1's note is replaced
    assert plan["explicit_knobs"] == {"profile": "6900hx-proxy-v2"}
    # v1 is unchanged
    v1 = dry_plan(world, capsys, "--profile", "6900hx-proxy", mode=mode)
    assert v1["pins"]["memory_view_gib"] is None and v1["pins"]["ort_intra_op_threads"] == 8
    assert v1["simulation"]["label"] == ("16 logical processors, SHA on, Siril mem 9 GiB, render workers 3 "
                                         "(other memory unlimited)")


def test_effective_environment_ignores_the_case_of_names():
    plan = {"environment": {"OPENSSL_ia32cap": None, "AESTER_X": "1"}}
    env = bench.effective_environment(plan, {"OPENSSL_IA32CAP": ":~0x20000000", "MSYSTEM": "MINGW64", "PATH": "x"})
    assert env == {"MSYSTEM": "MINGW64", "AESTER_X": "1"}


# --------------------------------------------------------------------------- the hook's P1c parts

FAKE_PERFORMANCE = '''
RESERVED = 0


def available_memory():
    return 64 * 1024 ** 3


def render_workers(*a, **k):
    return 1
'''
FAKE_PERFORMANCE_RAW = '''
RESERVED = 2 * 1024 ** 3


def _available_memory_raw():
    return 64 * 1024 ** 3


def available_memory():
    raw = _available_memory_raw()
    return None if raw is None else raw - RESERVED


def available_commit():
    return 100 * 1024 ** 3
'''
FAKE_SIRIL_ENV = '''import os


def siril_environment(folder, pin=None, base=None, **kwargs):
    env = dict(os.environ if base is None else base)
    env["OMP_NUM_THREADS"] = "1"
    return env
'''
HOOK_CHILD = '''import json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy
from starstack import performance, siril
out = {"memory": performance.available_memory(),
       "commit": performance.available_commit() if hasattr(performance, "available_commit") else None,
       "siril_blas": siril.siril_environment(".").get("OPENBLAS_CORETYPE"),
       "own_blas": os.environ.get("OPENBLAS_CORETYPE")}
with ThreadPoolExecutor(3, thread_name_prefix="aester-median") as pool, \\
        ThreadPoolExecutor(2, thread_name_prefix="aester-render") as render, \\
        ThreadPoolExecutor(1, thread_name_prefix="aester-hash") as hashes:
    futures = [pool.submit(time.sleep, 2.6) for _ in range(3)] + [render.submit(time.sleep, 2.6) for _ in range(2)]
    futures.append(hashes.submit(time.sleep, 2.6))
    [f.result() for f in futures]
print(json.dumps(out))
'''


def _hook_child(tmp_path, performance_source, siril_source, extra_files=None, **extra_env):
    code = tmp_path / "checkout" / "src"
    (code / "starstack").mkdir(parents=True)
    (code / "starstack" / "__init__.py").write_text("", encoding="utf-8")
    (code / "starstack" / "performance.py").write_text(performance_source, encoding="utf-8")
    (code / "starstack" / "siril.py").write_text(siril_source, encoding="utf-8")
    for name, source in (extra_files or {}).items():
        (code / "starstack" / name).write_text(source, encoding="utf-8")
    hook = tmp_path / "h"
    bench.write_hook(hook)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("AESTER_", "OPENBLAS_", "PYTHONPATH"))}
    env.update(PYTHONPATH=os.pathsep.join([str(code), str(hook)]), AESTER_BENCH_HOOK_LOG=str(hook / "imports"),
               AESTER_BENCH_HOOK_EVENTS=str(hook / "events"), AESTER_BENCH_THREAD_LOG=str(hook / "threads"),
               AESTER_BENCH_BLAS_LOG="1", **extra_env)
    run = subprocess.run([sys.executable, "-c", HOOK_CHILD], env=env, capture_output=True, text=True, timeout=300)
    assert run.returncode == 0, run.stderr
    return json.loads(run.stdout), hook, code


def test_hook_applies_the_views_strips_siril_blas_logs_openblas_and_audits_threads(tmp_path):
    out, hook, code = _hook_child(tmp_path, FAKE_PERFORMANCE, FAKE_SIRIL_ENV,
                                  AESTER_BENCH_MEMORY_VIEW=str(9 * bench.GIB), AESTER_BENCH_COMMIT_VIEW=str(14 * bench.GIB),
                                  OPENBLAS_CORETYPE="Haswell", OPENBLAS_NUM_THREADS="16",
                                  AESTER_BENCH_PYTHON_ONLY="OPENBLAS_CORETYPE,OPENBLAS_NUM_THREADS")
    assert out["memory"] == 9 * bench.GIB and out["commit"] is None          # this code root has no available_commit
    assert out["own_blas"] == "Haswell" and out["siril_blas"] is None         # Python yes, Siril's block no
    events = bench.hook_events(hook / "events", {"pins": {"memory_view_gib": 9.0, "commit_view_gib": 14.0}})
    views = events["views"]
    assert list(views["by_function"]) == ["available_commit", "available_memory"]
    assert set(views["by_function"]["available_memory"]) == {"wrapped"} and views["absent"] == ["available_commit"]
    assert views["ok"] and views["never_seen"] == [] and len(views["processes"]) == 1
    (blas,) = events["openblas"]
    assert "OpenBLAS" in blas["config"] and blas["corename"] == "Haswell" and blas["num_threads"] == 16
    assert blas["environment"] == {"OPENBLAS_CORETYPE": "Haswell", "OPENBLAS_NUM_THREADS": "16"}
    assert events["siril_environment"] == [{"pid": blas["pid"], "status": "wrapped",
                                            "names": ["OPENBLAS_CORETYPE", "OPENBLAS_NUM_THREADS"]}]
    audit = bench.thread_audit(hook / "threads")
    (process,) = audit["processes"]
    assert process["frame_pool_max"] == 3 and process["render_max"] == 2
    assert process["blas_capable_max"] == 1 + 3 + 2                        # main + frame pool + render; not the hasher
    assert process["max_by_prefix"] == {"aester-median": 3, "aester-render": 2, "aester-hash": 1}
    assert process["samples"] >= 2 and audit["frame_pool_max"] == 3
    assert bench.hook_imports(hook / "imports", code)["ok"]
    later = time.time_ns()
    assert bench.hook_events(hook / "events", since_ns=later)["events"] == 0      # a later segment sees none of these
    assert bench.thread_audit(hook / "threads", since_ns=later)["processes"] == []


def test_hook_puts_the_view_below_an_in_process_reservation(tmp_path):
    """A code root with _available_memory_raw gets the view on the raw reading, so its reservation still counts."""
    out, hook, _ = _hook_child(tmp_path, FAKE_PERFORMANCE_RAW, FAKE_SIRIL_ENV,
                               AESTER_BENCH_MEMORY_VIEW=str(9 * bench.GIB), AESTER_BENCH_COMMIT_VIEW=str(14 * bench.GIB))
    assert out["memory"] == 7 * bench.GIB and out["commit"] == 14 * bench.GIB and out["siril_blas"] is None
    (event,) = [e for e in (json.loads(p.read_text(encoding="utf-8")) for p in (hook / "events").glob("*.json"))
                if e["event"] == "view"]
    assert event["applied"]["available_memory"] == {"status": "wrapped", "function": "_available_memory_raw",
                                                    "view_bytes": 9 * bench.GIB, "reservations": "below the view",
                                                    "frame_pool_reserved": False}
    assert event["applied"]["available_commit"]["function"] == "available_commit"
    assert event["applied"]["available_commit"]["reservations"] == "none"         # this code root reserves nothing


def test_hook_without_its_variables_changes_nothing(tmp_path):
    out, hook, _ = _hook_child(tmp_path, FAKE_PERFORMANCE, FAKE_SIRIL_ENV)
    assert out["memory"] == 64 * bench.GIB
    assert [e["event"] for e in (json.loads(p.read_text(encoding="utf-8")) for p in (hook / "events").glob("*.json"))] \
        == ["openblas"]


REAL_PRODUCT_CHILD = '''import json, time
from starstack import frame_pool, performance, siril
from starstack import imaging                                   # numpy and the rest of a job worker's imports
out = {"memory": performance.available_memory(), "siril": siril.siril_environment(".", commands=["stack r"]),
       "commit": performance.available_commit() if hasattr(performance, "available_commit") else "absent"}
frame_pool.run_ordered(lambda payload, unit: time.sleep(1.4), list(range(4)), workers=4, name="median")
reserved = None
if hasattr(frame_pool, "reserved"):                             # P1c-0: a reservation is deducted from the view too
    with frame_pool.reserved(1024 ** 3):
        reserved = performance.available_memory()
print(json.dumps({"memory": out["memory"], "commit": out["commit"], "siril_blas": out["siril"].get("OPENBLAS_CORETYPE"),
                  "siril_omp": out["siril"].get("OMP_NUM_THREADS"), "reserved_memory": reserved}))
'''


def test_hook_on_the_real_product_modules(tmp_path):
    """The hook against this checkout's starstack: the view reaches performance (and so frame_pool's budget), Siril's
    environment block loses the Python-only OpenBLAS names and keeps its own pool, a real frame pool is audited."""
    from starstack import performance
    hook = tmp_path / "h"
    bench.write_hook(hook)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("AESTER_", "OPENBLAS_", "PYTHONPATH"))}
    env.update(PYTHONPATH=os.pathsep.join([str(REPO / "src"), str(hook)]), AESTER_BENCH_HOOK_LOG=str(hook / "imports"),
               AESTER_BENCH_HOOK_EVENTS=str(hook / "events"), AESTER_BENCH_THREAD_LOG=str(hook / "threads"),
               AESTER_BENCH_BLAS_LOG="1", AESTER_BENCH_MEMORY_VIEW=str(2 * bench.GIB),
               AESTER_BENCH_COMMIT_VIEW=str(3 * bench.GIB), AESTER_BENCH_PYTHON_ONLY="OPENBLAS_CORETYPE,OPENBLAS_NUM_THREADS",
               OPENBLAS_CORETYPE="Haswell", OPENBLAS_NUM_THREADS="16")
    run = subprocess.run([sys.executable, "-c", REAL_PRODUCT_CHILD], env=env, capture_output=True, text=True, timeout=300)
    assert run.returncode == 0, run.stderr
    out = json.loads(run.stdout)
    real = performance.available_memory()
    assert out["memory"] == min(real, 2 * bench.GIB) and out["siril_blas"] is None and out["siril_omp"] == "1"
    if out["reserved_memory"] is not None and real > 3 * bench.GIB:     # min(raw, view) - reserved, never min(net, view)
        assert out["reserved_memory"] == bench.GIB
    events = bench.hook_events(hook / "events", {"pins": {"memory_view_gib": 2.0, "commit_view_gib": 3.0}})
    assert events["views"]["ok"] and "wrapped" in events["views"]["by_function"]["available_memory"]
    if hasattr(performance, "available_commit"):           # P1c's new reading: wrapped once it exists
        assert "wrapped" in events["views"]["by_function"]["available_commit"] and out["commit"] <= 3 * bench.GIB
    else:
        assert events["views"]["absent"] == ["available_commit"] and out["commit"] == "absent"
    assert events["siril_environment"][0]["status"] == "wrapped"
    assert {b["corename"] for b in events["openblas"]} == {"Haswell"}
    (process,) = bench.thread_audit(hook / "threads")["processes"]
    assert process["frame_pool_max"] == 4 and process["max_by_prefix"] == {"aester-median": 4}
    assert bench.hook_imports(hook / "imports", REPO / "src")["ok"]


def test_hook_stop_ends_every_write(tmp_path, monkeypatch):
    monkeypatch.setenv(bench.THREAD_LOG_ENV, str(tmp_path / "threads"))
    monkeypatch.setenv(bench.HOOK_EVENTS_ENV, str(tmp_path / "events"))
    for name in (bench.HOOK_LOG_ENV, bench.MEMORY_VIEW_ENV, bench.COMMIT_VIEW_ENV, bench.BLAS_LOG_ENV,
                 bench.PYTHON_ONLY_ENV, bench.HOOK_SHIM_ENV):
        monkeypatch.delenv(name, raising=False)
    hook = bench.load_hook(bench.write_hook(tmp_path / "h"))
    deadline = time.monotonic() + 10
    while not list((tmp_path / "threads").glob("*.json")) and time.monotonic() < deadline:
        time.sleep(.05)
    hook.stop()
    shutil.rmtree(tmp_path / "threads")
    hook._event("view", applied={})
    hook._AUDIT["write"]()
    time.sleep(1.5)                                       # the audit loop has ended and writes nothing more
    assert not (tmp_path / "threads").exists() and not (tmp_path / "events").exists()


def test_the_thread_audit_counts_o1_s_helper_as_blas_capable_but_not_as_a_pool_thread(tmp_path, monkeypatch):
    """Efficiency P1c O-1: the Beside helper is a plain thread named aester-beside-<name> (no _<k> suffix): it is not a
    frame-pool worker (the 16-thread bound) but may call BLAS (the scan's in-process part), so no executor prefix is
    needed for it; the pools it starts (trail crops) count as frame-pool threads like the coverage pool's."""
    for name in (bench.THREAD_LOG_ENV, bench.HOOK_EVENTS_ENV, bench.HOOK_LOG_ENV, bench.MEMORY_VIEW_ENV,
                 bench.COMMIT_VIEW_ENV, bench.BLAS_LOG_ENV, bench.PYTHON_ONLY_ENV, bench.HOOK_SHIM_ENV):
        monkeypatch.delenv(name, raising=False)
    hook = bench.load_hook(bench.write_hook(tmp_path / "h"))
    counts, prefixes = hook.thread_counts(["MainThread", "aester-beside-o1-scan", "aester-trail_crops_0",
                                           "aester-trail_crops_1", "aester-coverage_0", "aester-peak-sampler"])
    assert counts["frame_pool"] == 3 and counts["aester_other"] == 2
    assert counts["blas_capable"] == 1 + 3 + 1                    # main + pool threads + the helper; not the sampler
    assert prefixes == {"aester-beside-o1-scan": 1, "aester-trail_crops": 2, "aester-coverage": 1,
                        "aester-peak-sampler": 1}


# --------------------------------------------------------------------------- --cancel-when

def test_cancel_when_parses_records_fields_values_and_delays():
    assert bench.parse_cancel_when("job:stage=stack+2.5") == {"text": "job:stage=stack+2.5", "record": "job",
                                                              "field": "stage", "value": "stack", "delay_s": 2.5}
    assert bench.parse_cancel_when("workflow:checkpoints.stack=*")["delay_s"] == 0
    assert bench.parse_cancel_when(" workflow:progress=70 ")["value"] == "70"
    for bad in ("stack:stage=x", "job:stage", "job:=x", "job:stage=x+", "job:stage=x+-1"):
        with pytest.raises(bench.BenchError, match="RECORD:FIELD=VALUE"):
            bench.parse_cancel_when(bad)
    record = {"stage": "stack", "progress": 70, "done": False, "list": [{"x": None}], "checkpoints": {"stack": {}}}
    assert bench.field_matches(bench.record_field(record, "stage"), "stack")
    assert bench.field_matches(bench.record_field(record, "progress"), "70")
    assert bench.field_matches(bench.record_field(record, "done"), "false")
    assert bench.field_matches(bench.record_field(record, "list.0.x"), "null")
    assert not bench.field_matches(bench.record_field(record, "list.0.x"), "*")
    assert bench.field_matches(bench.record_field(record, "checkpoints.stack"), "*")
    assert not bench.field_matches(bench.record_field(record, "checkpoints.transients"), "*")
    assert bench.record_field(record, "list.3") is bench._MISSING


def test_cancel_watch_waits_for_the_delay_and_reports_both_latencies(tmp_path):
    workflow_dir, job_dir = tmp_path / "wf", tmp_path / "wf" / "stack" / "job-1"
    job_dir.mkdir(parents=True)
    (workflow_dir / "workflow.json").write_text(json.dumps({"stage": "stack"}), encoding="utf-8")
    watch = bench.CancelWatch(bench.parse_cancel_when("job:stage=stack+1.5"), bench._read_json_retrying)
    status = {"state": "running"}
    assert not watch.due(workflow_dir, status, 1.0)                     # no stack job yet
    status["stack_job_dir"] = str(job_dir)
    assert not watch.due(workflow_dir, status, 2.0)                     # no job.json yet
    (job_dir / "job.json").write_text(json.dumps({"stage": "convert", "state": "running"}), encoding="utf-8")
    assert not watch.due(workflow_dir, status, 3.0)
    (job_dir / "job.json").write_text(json.dumps({"stage": "stack", "state": "running"}), encoding="utf-8")
    assert not watch.due(workflow_dir, status, 4.0) and watch.condition_t == 4.0
    (job_dir / "job.json").write_text(json.dumps({"stage": "compose", "state": "running"}), encoding="utf-8")
    assert watch.due(workflow_dir, status, 5.5)                          # the delay counts from the first match
    watch.requested_t = 5.5
    assert not watch.due(workflow_dir, status, 9.0)                      # once only
    watch.observe({"state": "cancel_requested", "stack_job_dir": str(job_dir)}, 5.7, lambda: 4)
    (job_dir / "job.json").write_text(json.dumps({"state": "cancelled"}), encoding="utf-8")
    watch.observe({"state": "cancel_requested", "stack_job_dir": str(job_dir)}, 6.1, lambda: 3)
    watch.observe({"state": "cancelled", "stack_job_dir": str(job_dir)}, 6.4, lambda: 2)
    watch.quiet_t = 7.0
    report = watch.report("cancelled")
    assert report["valid"] and report["invalid_reasons"] == [] and report["value_seen"] == "stack"
    assert report["latency"] == {"to_cancelled_s": .9, "to_job_cancelled_s": .6, "to_no_descendants_s": 1.5,
                                 "cancelled_to_no_descendants_s": .6}
    assert report["descendants_at_cancelled"] == 2 and report["condition_at_s"] == 4.0
    watch.quiet_t = None
    report = watch.report("cancelled")
    assert not report["valid"] and "Job Object" in report["invalid_reasons"][0]
    assert any("not 'cancelled'" in r for r in watch.report("succeeded")["invalid_reasons"])
    never = bench.CancelWatch(bench.parse_cancel_when("workflow:stage=x"), bench._read_json_retrying).report("succeeded")
    assert not never["triggered"] and not never["valid"] and "never held" in never["invalid_reasons"][0]


def test_foreign_cpu_is_the_machine_minus_the_tree():
    def reading(t, busy_kernel, idle, user, job_cpu):
        return {"t": t, "system": {"idle": idle, "kernel": busy_kernel, "user": user, "processors": 192,
                                   "method": "per-group"},
                "job": {"cpu_user_s": job_cpu, "cpu_kernel_s": 0.0}, "bench_cpu_s": 1.0}
    start, end = reading(0.0, 1000.0, 900.0, 50.0, 10.0), reading(100.0, 1500.0, 1300.0, 350.0, 360.0)
    result = bench.foreign_cpu(start, end)
    assert result["system_busy_cpu_s"] == 400.0 and result["tree_cpu_s"] == 350.0 and result["foreign_cpu_s"] == 50.0
    assert result["foreign_cores_mean"] == .5 and not result["over_one_core"] and result["tree_source"] == "job accounting"
    end["system"]["user"] = 500.0
    assert bench.foreign_cpu(start, end)["over_one_core"]
    no_job = bench.foreign_cpu({**start, "job": None}, {**end, "job": None}, sampler_tree_cpu_s=201.0)
    assert no_job["tree_cpu_s"] == 200.0 and no_job["tree_source"].startswith("sampler")
    assert bench.foreign_cpu({**start, "system": None}, end) == {"status": "unavailable"}


@windows
def test_system_times_cover_every_processor_group():
    times = bench.Win().system_times()
    assert times["method"] == "per-group" and times["processors"] == os.cpu_count()
    assert times["kernel"] >= times["idle"] > 0 and bench.Win.busy_seconds(times) > 0


# --------------------------------------------------------------------------- --resume

def _kept_slot(world, *, resumable=True, cancel_valid=True, knobs=None, workflow=True):
    bench_root = world["bench"]
    slot = bench_root / "runs" / "L2"
    (slot / "c" / "workflow-x").mkdir(parents=True)
    if workflow:
        (slot / "c" / "workflow-x" / "workflow.json").write_text("{}", encoding="utf-8")
    (slot / bench.SLOT_MARKER).write_text(json.dumps({"slot": "L2", "run_id": "R1", "segments": ["R1"],
                                                      "resumable": resumable,
                                                      "cancel": {"invalid_reasons": ["boom"]}}), encoding="utf-8")
    results = bench_root / "results" / "R1"
    results.mkdir(parents=True)
    (results / "plan.json").write_text(json.dumps({"explicit_knobs": knobs or {}}), encoding="utf-8")
    (results / "summary.json").write_text(json.dumps({"cancel": {"valid": cancel_valid},
                                                      "workflow_dir": str(slot / "c" / "workflow-x")}), encoding="utf-8")
    return slot


def test_resume_source_needs_a_valid_cancel_slot(world):
    with pytest.raises(bench.BenchError, match="no kept slot"):
        bench.load_resume_source(world["bench"], "L", 2)
    slot = _kept_slot(world, resumable=False)
    with pytest.raises(bench.BenchError, match="not resumable: boom"):
        bench.load_resume_source(world["bench"], "L", 2)
    shutil.rmtree(world["bench"])
    _kept_slot(world, cancel_valid=False)
    with pytest.raises(bench.BenchError, match="no valid cancel record"):
        bench.load_resume_source(world["bench"], "L", 2)
    shutil.rmtree(world["bench"])
    _kept_slot(world, workflow=False)
    with pytest.raises(bench.BenchError, match="is not in"):
        bench.load_resume_source(world["bench"], "L", 2)
    shutil.rmtree(world["bench"])
    slot = _kept_slot(world)
    source = bench.load_resume_source(world["bench"], "L", 2)
    assert source["run_id"] == "R1" and source["workflow_dir"] == str(slot / "c" / "workflow-x")


def test_replay_fills_the_first_segments_knobs_and_refuses_other_values():
    args = bench.build_parser().parse_args(["--mode", "L", "--resume", "--pin-ort", "8"])
    first = {"explicit_knobs": {"pin_ort": 8, "pin_siril": "cpu=16,mem=16", "parallel": "max", "memory_view": "9"}}
    assert bench.replay_knobs(args, first) == {"pin_siril": "cpu=16,mem=16", "parallel": "max", "memory_view": "9"}
    assert (args.pin_siril, args.parallel, args.memory_view, args.pin_ort) == ("cpu=16,mem=16", "max", "9", 8)
    args = bench.build_parser().parse_args(["--mode", "L", "--resume", "--parallel", "serial"])
    with pytest.raises(bench.BenchError, match="--parallel 'serial' differs"):
        bench.replay_knobs(args, first)
    args = bench.build_parser().parse_args(["--mode", "L", "--resume", "--pin-siril", " CPU=16,mem=16"])
    bench.replay_knobs(args, first)                                       # the same pin, spelled otherwise


def test_resume_differences_allow_only_the_code_root(world, capsys):
    prepare(world)
    first = dry_plan(world, capsys, "--parallel", "max")
    inherited = {"MSYSTEM": "MINGW64", "AESTER_HASH_LEDGER": "on"}
    summary = {"inherited_environment": inherited}
    second = json.loads(json.dumps(first))
    second["environment"]["PYTHONPATH"] = os.pathsep.join(["C:\\other\\src", first["paths"]["hook"]])
    second["environment"][bench.HOOK_SHIM_ENV] = "1"
    second["code_root"] = "C:\\other"
    assert bench.resume_differences(first, summary, second, dict(inherited, AESTER_BENCH_REEXEC="1")) == []
    assert any("AESTER_HASH_LEDGER" in p for p in bench.resume_differences(first, summary, second, {"MSYSTEM": "MINGW64"}))
    changed = json.loads(json.dumps(second))
    changed["pins"]["ort_intra_op_threads"] = 3
    assert any(p.startswith("pins:") for p in bench.resume_differences(first, summary, changed, inherited))
    changed = json.loads(json.dumps(second))
    changed["environment"]["PYTHONPATH"] = os.pathsep.join(["C:\\other\\src", "C:\\elsewhere\\h"])
    assert "PYTHONPATH: the hook entries differ" in bench.resume_differences(first, summary, changed, inherited)


# --------------------------------------------------------------------------- whole runs (fake code roots)

CANCEL_WORKFLOW = '''"""Stand-in for starstack.workflow with cancel and resume; no Siril."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

TERMINAL = {"succeeded", "failed", "cancelled"}


def _write(directory, **fields):
    path = Path(directory) / "workflow.json"
    for _ in range(200):
        try:
            record = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            break
        except (OSError, ValueError):
            time.sleep(.01)
    record.update(fields)
    temp = path.with_name("workflow.%d.tmp" % os.getpid())
    temp.write_text(json.dumps(record), encoding="utf-8")
    for _ in range(200):
        try:
            os.replace(temp, path)
            break
        except PermissionError:
            time.sleep(.01)
    return record


def _spawn(directory, delivery_root, *flags):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1]) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.Popen([sys.executable, "-m", "starstack.fakeworker", str(directory), delivery_root, *flags], env=env,
                     stdin=subprocess.DEVNULL, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def submit_workflow(input_dir, output_root, *, idempotency_key, input_files, delivery_root, **kwargs):
    directory = Path(output_root) / ("workflow-" + idempotency_key)
    directory.mkdir()
    record = _write(directory, workflow_dir=str(directory), state="queued", stage="queued", attempt=1,
                    request={"input_files": input_files, "kwargs": kwargs, "delivery_root": delivery_root})
    _spawn(directory, delivery_root)
    return record


def workflow_status(directory):
    return json.loads((Path(directory) / "workflow.json").read_text(encoding="utf-8"))


def cancel_workflow(directory):
    record = _write(directory, state="cancel_requested")        # before the request: the worker writes last
    (Path(directory) / "cancel.request").touch()
    return record


def resume_workflow(directory):
    directory = Path(directory)
    record = workflow_status(directory)
    if record["state"] != "cancelled":
        raise ValueError("only a cancelled workflow resumes here")
    (directory / "cancel.request").unlink()
    record = _write(directory, state="queued", stage="resume", attempt=record["attempt"] + 1)
    _spawn(directory, record["request"]["delivery_root"], "--resume")
    return record
'''
CANCEL_WORKER = '''import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy
from starstack import performance, workflow

directory, delivery, resumed = Path(sys.argv[1]), Path(sys.argv[2]), "--resume" in sys.argv
job = directory / "stack" / "job-1"
job.mkdir(parents=True, exist_ok=True)
seen = {"memory": performance.available_memory(), "blas": os.environ.get("OPENBLAS_CORETYPE"),
        "frame_workers": os.environ.get("AESTER_FRAME_WORKERS"), "overlap": os.environ.get("AESTER_OVERLAP")}
(job / "job.json").write_text(json.dumps({"state": "running", "stage": "stack", "seen": seen}), encoding="utf-8")
workflow._write(directory, state="running", stage="stack", stack_job_dir=str(job))
with ThreadPoolExecutor(2, thread_name_prefix="aester-strict_support") as pool:   # > 2 audit periods (1 s + writes)
    [f.result() for f in [pool.submit(time.sleep, 2.5) for _ in range(2)]]
if not resumed:
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    deadline = time.time() + 60
    while time.time() < deadline:
        if (directory / "cancel.request").exists():
            child.kill()
            child.wait()
            (job / "job.json").write_text(json.dumps({"state": "cancelled", "stage": "stack", "seen": seen}),
                                          encoding="utf-8")
            workflow._write(directory, state="cancelled", stage="stack", message="cancelled")
            sys.exit(0)
        time.sleep(.05)
    sys.exit(3)
(job / "job.json").write_text(json.dumps({"state": "succeeded", "stage": "complete", "seen": seen,
                                          "diagnostics": {"timings": {"total_seconds": 1.0}}}), encoding="utf-8")
base = delivery / "delivery-1"
base.mkdir(parents=True, exist_ok=True)
(base / "final.jpg").write_bytes(b"jpeg")
workflow._write(directory, state="succeeded", stage="complete", base_delivery_dir=str(base),
                trail_addback_pending=False, trail_review={"state": "not_requested"})
'''


def _fake_root(tmp_path, name, worker=CANCEL_WORKER):
    code = tmp_path / name / "src" / "starstack"
    code.mkdir(parents=True)
    for file, text in (("__init__.py", ""), ("siril.py", FAKE_SIRIL), ("workflow.py", CANCEL_WORKFLOW),
                       ("fakeworker.py", worker), ("regions.py", FAKE_REGIONS), ("performance.py", FAKE_PERFORMANCE)):
        (code / file).write_text(text, encoding="utf-8")
    return code.parents[1]


def _drive(world, tmp_path, argv):
    tools = world["tmp"] / "tools"
    env = {k: v for k, v in os.environ.items()
           if k not in ("PYTHONPATH", "AESTER_SIRIL_PIN", "OPENSSL_ia32cap", "MSYSTEM", "XDG_DATA_HOME")
           and not k.startswith(("AESTER_", "STARSTACK_", "OPENBLAS_"))}
    env["LOCALAPPDATA"] = str(world["local"])
    return subprocess.run([sys.executable, "-X", "utf8", "-c", DRIVER, str(SCRIPT), str(tools / "siril-cli.exe"),
                           str(tools / "nox.onnx"), str(tmp_path / "no-verifier.py"), *argv, *common(world)],
                          capture_output=True, text=True, timeout=600, env=env)


@windows
def test_cancel_when_then_resume_with_another_code_root(world, tmp_path):
    """C-runs in miniature: cancel at a record field, both latencies, the kept slot; then --resume replays the first
    segment's knobs under another code root, refuses a changed knob, and records both segments."""
    prepare(world)
    first_root, second_root = _fake_root(tmp_path, "k-checkout"), _fake_root(tmp_path, "p1c-checkout")
    base = ["--mode", "L", "--frames", "2", "--sample-seconds", "0.2", "--quiesce-seconds", "60"]
    run = _drive(world, tmp_path, [*base, "--label", "C1a", "--code-root", str(first_root), "--parallel", "max",
                                   "--memory-view", "9", "--python-openblas", "Haswell:16",
                                   "--cancel-when", "job:stage=stack+1.5"])
    assert run.returncode == 0, run.stdout + run.stderr
    (first,) = list((world["bench"] / "results").iterdir())
    summary = json.loads((first / "summary.json").read_text(encoding="utf-8"))
    cancel = summary["cancel"]
    assert summary["state"] == "cancelled" and cancel["valid"] and cancel["triggered"], cancel
    assert cancel["value_seen"] == "stack" and cancel["requested_at_s"] >= cancel["condition_at_s"] + 1.5
    latency = cancel["latency"]
    assert 0 < latency["to_cancelled_s"] < 10 and latency["to_no_descendants_s"] >= latency["to_cancelled_s"]
    assert latency["to_job_cancelled_s"] is not None and cancel["record_reads"] >= 1
    assert summary["kept"]["reason"] == "--cancel-when: kept for --resume" and not summary["deleted"]
    marker = json.loads((world["bench"] / "runs" / "L2" / bench.SLOT_MARKER).read_text(encoding="utf-8"))
    assert marker["resumable"] and marker["segments"] == [summary["run_id"]] and marker["cancel"]["valid"]
    job = json.loads((first / "records" / "job.json").read_text(encoding="utf-8"))
    assert job["seen"] == {"memory": 9 * bench.GIB, "blas": "Haswell", "frame_workers": "max", "overlap": "auto"}
    views = summary["hook_events"]["views"]
    assert views["ok"] and len(views["by_function"]["available_memory"]["wrapped"]) >= 2      # bench + worker
    assert {b["corename"] for b in summary["hook_events"]["openblas"]} == {"Haswell"}
    assert summary["thread_audit"]["frame_pool_max"] == 2
    assert summary["foreign_cpu"]["status"] == "ok" and summary["foreign_cpu"]["tree_source"] == "job accounting"
    rows = [json.loads(r) for r in (first / "timeline.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(row.get("threads") for row in rows) and any(row.get("sys_busy_cores") is not None for row in rows)
    # a changed knob is refused before anything runs
    refused = _drive(world, tmp_path, [*base, "--label", "C1x", "--resume", "--code-root", str(second_root),
                                       "--pin-ort", "3", "--dry-run"])
    assert refused.returncode == 2 and "pins" in refused.stderr
    run = _drive(world, tmp_path, [*base, "--label", "C1b", "--resume", "--code-root", str(second_root)])
    assert run.returncode == 0, run.stdout + run.stderr
    (second,) = [p for p in (world["bench"] / "results").iterdir() if p.name.endswith("C1b")]
    summary2 = json.loads((second / "summary.json").read_text(encoding="utf-8"))
    assert summary2["state"] == "succeeded" and summary2["deleted"] and not (world["bench"] / "runs" / "L2").exists()
    resume = summary2["resume"]
    assert resume["first_run_id"] == summary["run_id"] and resume["first_cancel"]["valid"]
    assert resume["segments"] == [summary["run_id"], summary2["run_id"]]
    assert resume["replayed_knobs"] == {"parallel": "max", "memory_view": "9", "python_openblas": "Haswell:16"}
    assert summary2["plan"]["pins"] == summary["plan"]["pins"] and summary2["plan"]["cancel_when"] is None
    assert json.loads((second / "segments" / "1-summary.json").read_text(encoding="utf-8"))["run_id"] == summary["run_id"]
    assert summary2["code_root_check"]["ok"]                              # only this segment's processes, new root
    job2 = json.loads((second / "records" / "job.json").read_text(encoding="utf-8"))
    assert job2["state"] == "succeeded" and job2["seen"]["memory"] == 9 * bench.GIB    # the view replayed
    gone = _drive(world, tmp_path, [*base, "--label", "C1c", "--resume", "--dry-run"])
    assert gone.returncode == 2 and "no kept slot" in gone.stderr


@windows
def test_cancel_that_never_triggers_fails_the_run_but_keeps_its_result(world, tmp_path):
    prepare(world)
    worker = CANCEL_WORKER.replace("if not resumed:", "if False:")
    root = _fake_root(tmp_path, "checkout", worker)
    run = _drive(world, tmp_path, ["--mode", "L", "--frames", "2", "--label", "nt", "--code-root", str(root),
                                   "--cancel-when", "workflow:stage=never", "--trails", "off"])
    assert run.returncode == 1, run.stdout + run.stderr
    (result,) = list((world["bench"] / "results").iterdir())
    summary = json.loads((result / "summary.json").read_text(encoding="utf-8"))
    assert summary["state"] == "succeeded" and not summary["cancel"]["triggered"] and not summary["cancel"]["valid"]
    assert summary["deleted"]                                              # a normal finished run: fingerprint, delete
    assert summary["plan"]["submit"]["trails_enabled"] is False


@windows
def test_an_invalid_cancel_keeps_its_slot_but_resume_refuses_it(world, tmp_path):
    """The bench's own write path of an invalid cancel: the worker ignores cancel.request and finishes, so the cancel
    is invalid (the workflow ended 'succeeded'); the slot is kept with a non-resumable marker and --resume refuses it."""
    prepare(world)
    worker = CANCEL_WORKER.replace("if not resumed:", "if False:")
    assert worker.count("if False:") == 1
    root = _fake_root(tmp_path, "checkout", worker)
    run = _drive(world, tmp_path, ["--mode", "L", "--frames", "2", "--label", "bad", "--code-root", str(root),
                                   "--sample-seconds", "0.2", "--quiesce-seconds", "30",
                                   "--cancel-when", "job:stage=stack+0.3"])
    assert run.returncode == 1, run.stdout + run.stderr
    (result,) = list((world["bench"] / "results").iterdir())
    summary = json.loads((result / "summary.json").read_text(encoding="utf-8"))
    cancel = summary["cancel"]
    assert cancel["triggered"] and not cancel["valid"] and summary["state"] == "succeeded"
    assert cancel["invalid_reasons"] == ["the workflow ended 'succeeded', not 'cancelled'"]
    assert not summary["deleted"] and summary["kept"]["reason"].startswith("--cancel-when: invalid cancel")
    marker = json.loads((world["bench"] / "runs" / "L2" / bench.SLOT_MARKER).read_text(encoding="utf-8"))
    assert marker["resumable"] is False and marker["cancel"]["invalid_reasons"] == cancel["invalid_reasons"]
    assert marker["segments"] == [summary["run_id"]]
    refused = _drive(world, tmp_path, ["--mode", "L", "--frames", "2", "--resume", "--code-root", str(root),
                                       "--dry-run"])
    assert refused.returncode == 2 and "not resumable: the workflow ended 'succeeded'" in refused.stderr


# --------------------------------------------------------------------------- the hook's views and reservations

FAKE_FRAME_POOL = '''
def reserved_bytes():
    return 2 * 1024 ** 3
'''
FAKE_PERFORMANCE_STATUS = '''"""P1c-0's shape: the readings before reservations in _memory_status, both public figures net of them."""
from . import frame_pool


def _memory_status():
    return 64 * 1024 ** 3, 100 * 1024 ** 3


def _net(value):
    return None if value is None else max(0, int(value) - frame_pool.reserved_bytes())


def available_memory():
    return _net(_memory_status()[0])


def available_commit():
    return _net(_memory_status()[1])
'''
FAKE_PERFORMANCE_PUBLIC_RESERVED = '''"""No reading before reservations: only the public function, already net of them."""
from . import frame_pool


def available_memory():
    return 64 * 1024 ** 3 - frame_pool.reserved_bytes()
'''


def _view_event(hook):
    (event,) = [e for e in (json.loads(p.read_text(encoding="utf-8")) for p in (hook / "events").glob("*.json"))
                if e["event"] == "view"]
    return event


def test_hook_puts_both_views_on_the_memory_status_below_the_reservations(tmp_path):
    """P1c-0 code (performance._memory_status, frame_pool.reserved_bytes): the in-process reservations are deducted
    from the 9 GiB / 14 GiB views, as on the laptop (prereg 6.1, R20)."""
    out, hook, _ = _hook_child(tmp_path, FAKE_PERFORMANCE_STATUS, FAKE_SIRIL_ENV,
                               extra_files={"frame_pool.py": FAKE_FRAME_POOL},
                               AESTER_BENCH_MEMORY_VIEW=str(9 * bench.GIB), AESTER_BENCH_COMMIT_VIEW=str(14 * bench.GIB))
    assert out["memory"] == 7 * bench.GIB and out["commit"] == 12 * bench.GIB
    applied = _view_event(hook)["applied"]
    assert applied == {"available_memory": {"status": "wrapped", "function": "_memory_status[0]",
                                            "view_bytes": 9 * bench.GIB, "reservations": "below the view"},
                       "available_commit": {"status": "wrapped", "function": "_memory_status[1]",
                                            "view_bytes": 14 * bench.GIB, "reservations": "below the view"}}
    # only a memory view: the commit reading passes through
    out, hook, _ = _hook_child(tmp_path / "second", FAKE_PERFORMANCE_STATUS, FAKE_SIRIL_ENV,
                               extra_files={"frame_pool.py": FAKE_FRAME_POOL},
                               AESTER_BENCH_MEMORY_VIEW=str(9 * bench.GIB))
    assert out["memory"] == 7 * bench.GIB and out["commit"] == 98 * bench.GIB
    assert list(_view_event(hook)["applied"]) == ["available_memory"]


def test_hook_deducts_the_reservations_from_a_view_on_the_public_function(tmp_path):
    """No reading before reservations: min(real - reserved, view - reserved) = min(raw, view) - reserved, never
    min(real - reserved, view), in which the reservation would vanish below a small view."""
    out, hook, _ = _hook_child(tmp_path, FAKE_PERFORMANCE_PUBLIC_RESERVED, FAKE_SIRIL_ENV,
                               extra_files={"frame_pool.py": FAKE_FRAME_POOL},
                               AESTER_BENCH_MEMORY_VIEW=str(9 * bench.GIB), AESTER_BENCH_COMMIT_VIEW=str(14 * bench.GIB))
    assert out["memory"] == 7 * bench.GIB and out["commit"] is None
    applied = _view_event(hook)["applied"]
    assert applied["available_memory"] == {"status": "wrapped", "function": "available_memory",
                                           "view_bytes": 9 * bench.GIB, "reservations": "deducted from the view",
                                           "frame_pool_reserved": True}
    assert applied["available_commit"] == {"status": "absent"}
    views = bench.hook_events(hook / "events", {"pins": {"memory_view_gib": 9.0, "commit_view_gib": 14.0}})["views"]
    assert views["ok"] and views["processes"][0]["applied"] == applied     # what check_p1c_run judges


# --------------------------------------------------------------------------- --job strict-legacy (K-SL20, W19-W21)

def test_job_runs_submit_the_strict_l_cli_job(world, capsys):
    prepare(world)
    plan = dry_plan(world, capsys, "--job", "strict-legacy", "--parallel", "max")
    assert plan["job"] == "strict-legacy" and plan["slot"] == "SL2" and plan["paths"]["slot"].endswith("SL2")
    sky = world["bench"] / "assets" / "source" / f"region-assets-{world['revision']}" / "sky.png"
    assert plan["submit"] == {"siril_path": str(bench.SIRIL), "sky_mask_path": str(sky),
                              "idempotency_key": "aester-bench-SL2", "support_mode": "strict_siril_1_4_4",
                              "stack_range_mode": "legacy"}
    assert plan["environment"]["AESTER_FRAME_WORKERS"] == "max" and plan["parallel"]["effective"] == "max"
    from starstack import core, registered_support
    assert registered_support.MODE == "strict_siril_1_4_4"
    parameters = inspect.signature(core.submit_job).parameters
    assert set(plan["submit"]) | {"input_files", "transient_mode"} <= set(parameters)
    off = dry_plan(world, capsys, "--job", "strict-legacy", "--trails", "off")
    assert off["submit"]["transient_mode"] == "off" and off["pins"]["trails"] == "off"
    assert bench.SLOT_NAME.fullmatch("SL20") and not bench.SLOT_NAME.fullmatch("SF20")
    assert dry_plan(world, capsys)["job"] is None                       # a workflow run, as before


@pytest.mark.parametrize("extra, message", [
    (["--mode", "F", "--job", "strict-legacy"], "--job strict-legacy runs --mode L"),
    (["--mode", "L", "--job", "strict-legacy", "--cancel-when", "job:stage=stack"], "no --cancel-when / --resume"),
    (["--mode", "L", "--job", "strict-legacy", "--resume"], "no --cancel-when / --resume"),
    (["--mode", "L", "--job", "strict-legacy", "--pin-ort", "8"], "runs none"),
    (["--mode", "L", "--frames", "0"], "--frames must be 1..99"),
])
def test_job_run_refusals(world, capsys, extra, message):
    prepare(world)
    args = ["--dry-run", "--frames", "2", *extra, *common(world)]
    assert bench.main(args) == 2
    assert message in capsys.readouterr().err


def test_a_job_run_needs_the_copied_sky_mask(world, capsys):
    assert bench.main(["--dry-run", "--mode", "L", "--frames", "2", "--job", "strict-legacy", *common(world)]) == 2
    assert "run --prepare-assets first" in capsys.readouterr().err


FAKE_CORE = '''"""Stand-in for starstack.core's job API (submit_job, load_job, cancel_job): a detached worker, no Siril."""
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

TERMINAL = {"succeeded", "failed", "cancelled"}


def _write(directory, **fields):
    path = Path(directory) / "job.json"
    for _ in range(200):
        try:
            record = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            break
        except (OSError, ValueError):
            time.sleep(.01)
    record.update(fields)
    temp = path.with_name("job.%d.tmp" % os.getpid())
    temp.write_text(json.dumps(record), encoding="utf-8")
    for _ in range(200):
        try:
            os.replace(temp, path)
            break
        except PermissionError:
            time.sleep(.01)
    return record


def submit_job(input_dir, output_root, siril_path=None, sky_mask_path=None, idempotency_key=None,
               support_mode="legacy", stack_range_mode="legacy", input_files=None, transient_mode=None):
    directory = Path(output_root) / ("job-" + hashlib.sha256(idempotency_key.encode()).hexdigest()[:24])
    directory.mkdir()
    _write(directory, job_dir=str(directory), state="queued", stage="queued", support_mode=support_mode,
           stack_range_mode=stack_range_mode, mask={"path": sky_mask_path}, inputs=[{"path": p} for p in input_files],
           admission_options={"input_selection": {"files": list(input_files)},
                              "stack_settings": {"version": "siril-stack-settings-1", "engine_version": "1.4.4"}})
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1]) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.Popen([sys.executable, "-m", "starstack.fakejobworker", str(directory)], env=env,
                     stdin=subprocess.DEVNULL, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return load_job(directory)


def load_job(directory):
    return json.loads((Path(directory) / "job.json").read_text(encoding="utf-8"))


def cancel_job(directory):
    (Path(directory) / "cancel.request").touch()
'''
FAKE_JOB_WORKER = '''import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from starstack import core, siril

directory = Path(sys.argv[1])
core._write(directory, state="running", stage="support")
with ThreadPoolExecutor(2, thread_name_prefix="aester-strict_support") as pool:
    [f.result() for f in [pool.submit(time.sleep, 2.5) for _ in range(2)]]
work = directory / "work"
work.mkdir()
(work / "interpolation_support.json").write_text(json.dumps({"implementation": "numpy_buffered_kernel_2",
                                                             "state": "complete"}), encoding="utf-8")
siril.run_script("fake-siril", directory, "02_stack",
                 ["cd work", "setcpu 2", "set core.mem_mode=1", "set core.mem_amount=4", "stack r_light"], lambda: None)
entry = {"requested": 2, "workers": 2, "units": 2, "serial": False, "unit_bytes": 1000, "cpu_user_s": .1,
         "cpu_kernel_s": .01, "max_unit_s": 2.5, "fp_env_applied": False}
notes = {"hash_ledger": {"version": "aester-hashing-3", "miss_bytes": {}, "largest_misses": []},
         "governor": {"mode": "shadow-full"}, "siril_stack_settings": [{"setcpu": 2, "mem_amount": 4}],
         "frame_pool.strict_support": [entry]}
core._write(directory, state="succeeded", stage="complete", progress=100,
            diagnostics={"timings": {"notes": notes, "resources": {"operations": {
                "script:02_stack": {"children": {"processes": 1, "descendant_processes": 1}}}}}})
'''


def _job_root(tmp_path):
    code = tmp_path / "job-checkout" / "src" / "starstack"
    code.mkdir(parents=True)
    for file, text in (("__init__.py", ""), ("siril.py", FAKE_SIRIL), ("core.py", FAKE_CORE),
                       ("fakejobworker.py", FAKE_JOB_WORKER), ("regions.py", FAKE_REGIONS)):
        (code / file).write_text(text, encoding="utf-8")
    return code.parents[1]


@windows
def test_strict_l_job_runs_are_fingerprinted_compared_and_checked(world, tmp_path):
    """K-SL20 and W21 in miniature: two --job runs (default, then --parallel max) on a fake code root; the job folder
    is fingerprinted by the real verifier as {JOB}; its compare passes with strict ids, and check_p1c_run reads the
    results like a workflow's and counts the run as SL."""
    prepare(world)
    root = _job_root(tmp_path)
    verifier = REPO / "scripts" / "verify_workflow_equivalence.py"
    tools = world["tmp"] / "tools"
    env = {k: v for k, v in os.environ.items()
           if k not in ("PYTHONPATH", "AESTER_SIRIL_PIN", "OPENSSL_ia32cap", "MSYSTEM", "XDG_DATA_HOME")
           and not k.startswith(("AESTER_", "STARSTACK_", "OPENBLAS_"))}
    env["LOCALAPPDATA"] = str(world["local"])
    folders = {}
    for label, extra in (("K-SL", []), ("W21", ["--parallel", "max"])):
        argv = ["--mode", "L", "--frames", "2", "--job", "strict-legacy", "--label", label, "--code-root", str(root),
                "--sample-seconds", "0.2", "--poll-seconds", "0.1", *extra, *common(world)]
        run = subprocess.run([sys.executable, "-X", "utf8", "-c", DRIVER, str(SCRIPT), str(tools / "siril-cli.exe"),
                              str(tools / "nox.onnx"), str(verifier), *argv], capture_output=True, text=True,
                             timeout=600, env=env)
        assert run.returncode == 0, run.stdout + run.stderr
        (folders[label],) = [p for p in (world["bench"] / "results").iterdir() if p.name.endswith(label)]
        summary = json.loads((folders[label] / "summary.json").read_text(encoding="utf-8"))
        assert summary["state"] == "succeeded" and summary["deleted"] and not (world["bench"] / "runs" / "SL2").exists()
        assert summary["fingerprint"]["status"] == "ok", summary["fingerprint"]
        assert summary["job_dir"].endswith("job-" + hashlib.sha256(b"aester-bench-SL2").hexdigest()[:24])
        assert "workflow_dir" not in summary and set(summary["record_times"]) == {"job"}
        assert sorted(p.name for p in (folders[label] / "records").iterdir()) == ["job.json"]
        assert summary["thread_audit"]["frame_pool_max"] == 2
    manifest = json.loads((folders["W21"] / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["roots"]) == {"{JOB}", "{CACHE}", "{INPUT}"} and manifest["workflow_dir"] is None
    assert {"{JOB}/job.json", "{JOB}/02_stack.ssf", "{JOB}/work/interpolation_support.json"} <= set(manifest["files"])
    spec = importlib.util.spec_from_file_location("verifier_tested", verifier)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = tmp_path / "compare.json"
    with open(os.devnull, "w", encoding="utf-8") as sink, contextlib.redirect_stdout(sink):
        code = module.main(["compare", str(folders["K-SL"] / "manifest.json"), str(folders["W21"] / "manifest.json"),
                            "-o", str(report)])
    loaded = json.loads(report.read_text(encoding="utf-8"))
    assert code == 0 and loaded["pass"] and loaded["strict_ids"]["on"], loaded
    assert loaded["strict_ids"]["same_base_roots"] and loaded["counts"].get("IDENTICAL_BYTES", 0) >= 3
    spec = importlib.util.spec_from_file_location("check_p1c_tested", REPO / "scripts" / "check_p1c_run.py")
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    result = checker.check(folders["W21"], folders["K-SL"], report, expect_parallel="max")
    assert result["passed"], {k: result[k]["problems"] for k in result["sections"] if result[k]["problems"]}
    assert result["coverage"]["mode"] == "SL" and result["coverage"]["parallel"] == "max"
    assert result["contract"]["scripts"]["{JOB}/02_stack.ssf"]["blocks"] == [{"line": 2, "setcpu": 2, "mem_amount": 4}]
    assert result["parallel"]["sites"]["strict_support"]["max_workers"] == 2
