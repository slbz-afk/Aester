"""Efficiency P1b-3 with the real stock Siril: L's export and stack in two processes give the same results.

The product path (core.submit_job + core.run_job, every launch through siril.run_script) is forced onto the two
fallback layouts that no bench run takes: 3b (fallback registration with one quality-gate exclusion, then export)
and 3c (register + seqapplyreg in the export process).  Each runs split (default) and with the diagnostics-only
AESTER_L_SIRIL_LAYOUT=combined (the pre-P1b single script); masters, exported frames, both sequence files,
siril.ini and the job diagnostics must be identical.  A cancel between the two processes and a resume, a stale
statistics rerun with seqclean, and Siril's persistence of memory settings (a guard for P2) are checked too.

Any non-zero Siril exit fails the job and the test, with the log tail in the message; nothing is retried.

EFF-D4 (efficiency P1c §4.2): the split/combined comparisons run jobs admitted before the stack-settings contract (the
combined layout is refused for a contract job); contract jobs are run on both fallback paths, cancelled between the
two processes and resumed, and the contract stack is repeated with other thread/memory settings: one master.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
import warnings

import numpy as np
import pytest

from starstack import core, imaging, siril, stack_settings
from starstack import registration_calibration as rc

STOCK_SIRIL = Path(__file__).resolve().parents[1] / "tools" / "siril" / "bin" / "siril-cli.exe"
pytestmark = pytest.mark.skipif(os.name != "nt" or not STOCK_SIRIL.is_file(), reason="needs the local Siril runtime")

FRAMES, HEIGHT, WIDTH, STARS = 10, 600, 900, 300
EXCLUDED = "light_00004"
DROPPED_CARDS = {"DATE", "CHECKSUM", "DATASUM"}


def synthesize(folder: Path, seed: int = 7) -> None:
    """Deterministic star fields: sub-pixel shifts, +-0.3 degree rotation, one streak (so rejection does work)."""
    folder.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    sx, sy = rng.uniform(30, WIDTH - 30, STARS), rng.uniform(30, HEIGHT - 30, STARS)
    flux = np.exp(rng.normal(-3.0, 1.0, STARS))
    colour = rng.uniform(.7, 1.3, (STARS, 3))
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float32)
    for index in range(FRAMES):
        angle = np.deg2rad(rng.uniform(-.3, .3))
        dx, dy = rng.uniform(-5, 5, 2)
        cx, cy = WIDTH / 2, HEIGHT / 2
        px = cx + (sx - cx) * np.cos(angle) - (sy - cy) * np.sin(angle) + dx
        py = cy + (sx - cx) * np.sin(angle) + (sy - cy) * np.cos(angle) + dy
        image = np.empty((HEIGHT, WIDTH, 3), np.float32)
        base = .02 + .01 * (xx / WIDTH) + .005 * (yy / HEIGHT)
        for channel in range(3):
            image[..., channel] = base * (1 + .05 * channel)
        for k in range(STARS):
            x0, y0 = int(px[k]), int(py[k])
            if not (8 <= x0 < WIDTH - 8 and 8 <= y0 < HEIGHT - 8):
                continue
            ys, xs = slice(y0 - 7, y0 + 8), slice(x0 - 7, x0 + 8)
            g = np.exp(-((xx[ys, xs] - px[k]) ** 2 + (yy[ys, xs] - py[k]) ** 2) / (2 * 1.6 ** 2))
            image[ys, xs] += (flux[k] * colour[k])[None, None, :] * g[..., None]
        image += rng.normal(0, .002, image.shape).astype(np.float32)
        if index == 3:
            image[300:303, 80:820] += .3
        np.clip(image, 0, 1, out=image)
        imaging.write_linear(folder / f"frame{index + 1:05d}.fit", image, "P1b-3 synthetic star field")


def fallback_calibration(path: str):
    """registration_calibration.calibrate forced to a non-fitted result: 3b with one gate exclusion, 3c without."""
    def calibrate(work, **kwargs):
        record = {"status": "no_improvement", "reason": "test: forced fallback (P1b-3 paths 3b/3c)"}
        if path == "3b":
            record["quality_gate"] = {"applied": True, "excluded_frames": [EXCLUDED],
                                      "excluded": [{"stem": EXCLUDED, "reasons": ["test: forced exclusion"]}]}
        core.atomic_json(Path(work) / rc.DISTORTION_JSON, record)
        return record
    return calibrate


class Launches:
    """core.run_script wrapped: the real siril.run_script, its names recorded, an optional hook after each."""

    def __init__(self, after=None):
        self.names, self.after, self.options = [], after, {}

    def __call__(self, executable, folder, name, commands, cancel_check, **options):
        self.names.append(name)
        self.options[name] = options
        siril.run_script(executable, folder, name, commands, cancel_check, **options)
        if self.after:
            self.after(name, folder)


@pytest.fixture(scope="module")
def real(tmp_path_factory):
    root = tmp_path_factory.mktemp("lsplit-real")
    sources = root / "frames"
    synthesize(sources)
    engine = {"available": True, "executable": str(STOCK_SIRIL.resolve()), "version": "1.4.4",
              "message": "stock Siril (test)"}
    cache, options = {}, {}       # options: job_dir -> {script name: run_script keyword options}

    @contextmanager
    def patched(path, layout=None, after=None, contract=False):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(core, "doctor", lambda *_: dict(engine))
            mp.setattr(core, "_spawn_worker", lambda directory: None)
            mp.setattr(rc, "calibrate", fallback_calibration(path))
            mp.delenv(siril.SIRIL_PIN_ENV, raising=False)
            if not contract:   # admitted as before EFF-D4: the P1b scripts, combined layout allowed
                mp.setattr(stack_settings, "admission_field", lambda *args, **kwargs: None)
            if layout is None:
                mp.delenv(core.L_LAYOUT_ENV, raising=False)
            else:
                mp.setenv(core.L_LAYOUT_ENV, layout)
            launches = Launches(after)
            mp.setattr(core, "run_script", launches)
            yield launches

    def run(path, layout=None, label=None, after=None, contract=False):
        label = label or f"{path}-{layout or 'split'}{'-d4' if contract else ''}"
        if label not in cache:
            with patched(path, layout, after, contract) as launches:
                job = core.submit_job(str(sources), str(root / label), idempotency_key="lsplit-real",
                                      transient_mode="off")
                core.run_job(job["job_dir"])
            cache[label] = (core.load_job(job["job_dir"]), launches.names)
            options[job["job_dir"]] = launches.options
        return cache[label]

    return SimpleNamespace(root=root, sources=sources, run=run, patched=patched, options=options)


def _succeeded(job):
    assert job["state"] == "succeeded", job.get("error")     # a Siril failure carries its exit code and log tail
    return job


def _fits(path: Path):
    from astropy.io import fits
    with fits.open(path, memmap=False, do_not_scale_image_data=True) as hdus:
        cards = [(card.keyword, str(card.value)) for card in hdus[0].header.cards if card.keyword not in DROPPED_CARDS]
        return hashlib.sha256(hdus[0].data.tobytes()).hexdigest(), cards


def _text(path: Path, folder: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    return text.replace(str(folder).replace("\\", "\\\\"), "{JOB}").replace(str(folder), "{JOB}")


def _without_timers(value):
    if isinstance(value, dict):
        return {k: _without_timers(v) for k, v in value.items()
                if k not in ("timings", "timing_attempts", "seconds") and not k.endswith(("_seconds", "_at"))}
    if isinstance(value, list):
        return [_without_timers(v) for v in value]
    return value


def fingerprint(job) -> dict:
    folder = Path(job["job_dir"])
    work = folder / "work"
    out = {"sky_master.fit": _fits(work / "sky_master.fit")}
    frames = sorted(work.glob("r_light_*.fit"))
    assert frames
    out.update({p.name: _fits(p) for p in frames})
    for name in ("r_light_.seq", "light_.seq"):
        out[name] = _text(work / name, folder)
    # Siril caches each frame's statistics into the sequence from its parallel normalization workers, and allocates
    # the per-layer cache array without a lock (algos/statistics.c add_stats_to_stats): concurrent first entries can
    # lose one (seen 2026-10-05: one M0-n line missing from r_light_.seq in one of two identical runs; the master and
    # every other line equal).  The statistics lines present on both sides must be equal, and at most one line per
    # layer may be missing on one side (_differing): that is not a difference of the product, and it is reported as a
    # warning so the race stays visible.
    lines = out["r_light_.seq"].splitlines()
    out["r_light_.seq"] = "\n".join(line for line in lines if not line.startswith("M"))
    out["r_light_.seq M"] = {line.split()[0]: line for line in lines if line.startswith("M")}
    out["siril.ini"] = _text(folder / "siril.ini", folder)
    diagnostics = json.dumps(_without_timers(job["diagnostics"]), ensure_ascii=False, sort_keys=True)
    out["diagnostics"] = diagnostics.replace(json.dumps(str(folder))[1:-1], "{JOB}")
    out["checkpoints"] = job["checkpoints"]
    # The recorded digests cover Siril's FITS DATE card (write time), so only the checkpointed paths are compared
    # here; the content of the FITS files is compared above without DATE/CHECKSUM/DATASUM.
    out["checkpoint_paths"] = {stage: [r["path"] for r in records] for stage, records in job["checkpoint_files"].items()}
    return out


def _differing(a: dict, b: dict) -> list:
    def same(key):
        if key.endswith(" M") and isinstance(a.get(key), dict) and isinstance(b.get(key), dict):   # see fingerprint
            missing = sorted(set(a[key]) ^ set(b[key]))          # tags M<layer>-<image> present on one side only
            layers = [tag.split("-")[0] for tag in missing]
            if missing:
                warnings.warn(f"{key}: statistics lines on one side only (Siril's unlocked cache): {missing}")
            return (len(layers) == len(set(layers))
                    and all(a[key][tag] == b[key][tag] for tag in set(a[key]) & set(b[key])))
        return a.get(key) == b.get(key)
    return sorted(key for key in set(a) | set(b) if not same(key))


EXPECTED_SPLIT = {"3b": ["01_convert", "02_probe", "02_register_fallback", "02_export", "02_stack"],
                  "3c": ["01_convert", "02_probe", "02_export", "02_stack"]}
EXPECTED_COMBINED = {"3b": ["01_convert", "02_probe", "02_register_fallback", "02_stack"],
                     "3c": ["01_convert", "02_probe", "02_stack"]}


@pytest.mark.parametrize("path", ["3b", "3c"])
def test_real_siril_split_equals_combined_on_the_product_path(real, path):
    split, split_names = real.run(path)
    combined, combined_names = real.run(path, "combined")
    _succeeded(split)
    _succeeded(combined)
    assert split_names == EXPECTED_SPLIT[path]
    assert combined_names == EXPECTED_COMBINED[path]
    a, b = fingerprint(split), fingerprint(combined)
    assert _differing(a, b) == []
    exported = [key for key in a if key.startswith("r_light_") and key.endswith(".fit")]
    assert len(exported) == (FRAMES - 1 if path == "3b" else FRAMES)     # the gate exclusion reached the export
    assert split["signature"] == combined["signature"]
    folder = Path(split["job_dir"])
    assert (folder / "02_export.ssf").read_text(encoding="utf-8") == siril.script_text(
        ["cd work", "seqapplyreg light -filter-included"] if path == "3b" else core.export_script(None))
    assert (folder / "02_stack.ssf").read_text(encoding="utf-8") == siril.script_text(core.LEGACY_STACK_SCRIPT)
    assert (Path(combined["job_dir"]) / "02_stack.ssf").read_text(encoding="utf-8") == siril.script_text(
        ["cd work", "seqapplyreg light -filter-included", core.STACK_COMMAND] if path == "3b" else core.stack_script(None))
    assert not (Path(combined["job_dir"]) / "02_export.ssf").exists()
    export_log = (folder / "02_export.log").read_text(encoding="utf-8", errors="replace")
    stack_log = (folder / "02_stack.log").read_text(encoding="utf-8", errors="replace")
    # Siril's untranslated stdout lines (the translated ones follow the system language)
    assert ("running command register" in export_log) is (path == "3c")
    assert "running command seqapplyreg" in export_log and "running command stack" not in export_log
    assert "running command seqclean" in stack_log and "running command stack" in stack_log
    assert "running command seqapplyreg" not in stack_log
    if path == "3b":
        assert split["diagnostics"]["quality_gate"]["in_effect"]
    assert split["diagnostics"]["registration_source"] == core.REGISTRATION_SOURCE_SIRIL
    assert "l_siril_layout" not in split["diagnostics"]["timings"].get("notes", {})
    assert combined["diagnostics"]["timings"]["notes"]["l_siril_layout"] == "combined"


@pytest.mark.parametrize("contract", [False, True], ids=["pre-d4", "d4"])
def test_real_siril_cancel_between_processes_then_resume(real, contract):
    reference, _ = real.run("3c", contract=contract)
    _succeeded(reference)

    def cancel_after_export(name, folder):
        if name == "02_export":
            core.cancel_job(str(folder))
    output = real.root / f"3c-cancelled{'-d4' if contract else ''}"
    with real.patched("3c", after=cancel_after_export, contract=contract) as first:
        job = core.submit_job(str(real.sources), str(output), idempotency_key="lsplit-real", transient_mode="off")
        core.run_job(job["job_dir"])
    cancelled = core.load_job(job["job_dir"])
    assert cancelled["state"] == "cancelled", cancelled.get("error")
    assert first.names == ["01_convert", "02_probe", "02_export"]          # no stack process after the cancel
    assert "stack" not in cancelled["checkpoints"]
    with real.patched("3c", contract=contract) as second:
        core.resume_job(job["job_dir"])
        core.run_job(job["job_dir"])
    resumed = _succeeded(core.load_job(job["job_dir"]))
    assert second.names == ["02_export", "02_stack"]
    assert _differing(fingerprint(resumed), fingerprint(reference)) == []
    export_log = (Path(job["job_dir"]) / "02_export.log").read_text(encoding="utf-8", errors="replace")
    assert export_log.count("aester: Siril launch priority=") == 2          # appended across the two attempts
    assert len(resumed["diagnostics"]["timing_attempts"]) == 1


def _script_job(real, name: str) -> Path:
    """A job-like folder converted and exported by Siril (3c layout), through siril.run_script."""
    folder = real.root / name
    (folder / "stage").mkdir(parents=True)
    (folder / "work").mkdir()
    for source in sorted(real.sources.glob("*.fit")):
        shutil.copyfile(source, folder / "stage" / source.name)
    with pytest.MonkeyPatch.context() as mp:
        mp.delenv(siril.SIRIL_PIN_ENV, raising=False)
        siril.run_script(str(STOCK_SIRIL), folder, "01_convert",
                         ["setext fit", "set32bits", "setcompress 0", "cd stage", "convert light -out=../work"],
                         lambda: None)
        siril.run_script(str(STOCK_SIRIL), folder, "02_export", core.export_script(None), lambda: None)
    return folder


def _stack_only(folder: Path, commands) -> tuple:
    master = folder / "work" / "sky_master.fit"
    master.unlink(missing_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        mp.delenv(siril.SIRIL_PIN_ENV, raising=False)
        siril.run_script(str(STOCK_SIRIL), folder, "02_stack", list(commands), lambda: None)
    return _fits(master)


def test_real_siril_seqclean_resets_stale_statistics(real):
    folder = _script_job(real, "seqclean")
    sequence = folder / "work" / "r_light_.seq"
    assert not any(line.startswith("M") for line in sequence.read_text(encoding="utf-8").splitlines())
    fresh = _stack_only(folder, core.LEGACY_STACK_SCRIPT)
    assert any(line.startswith("M") for line in sequence.read_text(encoding="utf-8").splitlines())   # cached stats
    stale_cleaned = _stack_only(folder, core.LEGACY_STACK_SCRIPT)
    assert stale_cleaned == fresh                    # a stack-only rerun with seqclean equals the fresh stack
    stale_kept = _stack_only(folder, ["cd work", core.STACK_COMMAND])
    # Information only (data-dependent): whether normalizing with the cached, rounded statistics changes the master.
    print(f"P1b-3 info: stack rerun without seqclean {'differs from' if stale_kept != fresh else 'equals'} the fresh stack")


def test_siril_persists_memory_settings_into_the_job_config(real):
    """Siril-behaviour guard for P2: 'set core.mem_*' in one script reaches later Siril processes of the job through
    siril.ini, so a governed export must not write them into the job's own configuration."""
    folder = real.root / "persist"
    folder.mkdir()
    with pytest.MonkeyPatch.context() as mp:
        mp.delenv(siril.SIRIL_PIN_ENV, raising=False)
        siril.run_script(str(STOCK_SIRIL), folder, "02_export", ["set core.mem_mode=1", "set core.mem_amount=3"],
                         lambda: None)
        settings = (folder / "siril.ini").read_text(encoding="utf-8").splitlines()
        assert "mem_mode=1" in settings and "mem_amount=3" in settings
        siril.run_script(str(STOCK_SIRIL), folder, "02_stack", ["setext fit"], lambda: None)
    settings = (folder / "siril.ini").read_text(encoding="utf-8").splitlines()
    assert "mem_mode=1" in settings and "mem_amount=3" in settings          # still there after a later process


# --------------------------------------------------------------------------- EFF-D4 stack-settings contract

@pytest.mark.parametrize("path", ["3b", "3c"])
def test_real_siril_contract_job_on_the_product_path(real, path, record_property):
    job, names = real.run(path, contract=True)
    _succeeded(job)
    assert names == EXPECTED_SPLIT[path]
    folder = Path(job["job_dir"])
    plan = job["diagnostics"]["timings"]["notes"][stack_settings.NOTE_KEY]
    assert len(plan) == 1
    plan = plan[0]
    selected = FRAMES - 1 if path == "3b" else FRAMES
    assert plan["frames"] == selected and plan["frame_mb"] == 10                       # 900 x 600 x (3+2) x 4 bytes
    assert plan["threads"] == min(plan["thread_cap"], selected) and plan["memory_source"] == "runtime"
    assert (folder / "02_stack.ssf").read_text(encoding="utf-8") == siril.script_text(
        stack_settings.with_settings(core.LEGACY_STACK_SCRIPT, plan))
    assert real.options[job["job_dir"]]["02_stack"] == {"config": folder / "work" / stack_settings.CONFIG_NAME,
                                                         "stack_settings": True}
    assert all(not options for name, options in real.options[job["job_dir"]].items() if name != "02_stack")
    # Siril's own lines (already checked by the job): one normalization plan, T' workers
    assert stack_settings.log_lines(folder / "02_stack.log") == {"memory": [(10, plan["memory_gib"] * 1024)],
                                                                  "threads": [plan["threads"]]}
    assert "aester: stack settings contract: OMP_NUM_THREADS=1" in (folder / "02_stack.log").read_text(
        encoding="utf-8", errors="replace")
    # the stack's memory setting stayed in work/siril-stack.ini (removed): the job's configuration keeps ratio mode
    assert "mem_mode=0" in (folder / "siril.ini").read_text(encoding="utf-8").splitlines()
    assert not (folder / "work" / stack_settings.CONFIG_NAME).exists()
    assert job["admission_options"]["stack_settings"] == stack_settings.contract()
    # the export is the same as a job admitted before the contract; the masters may differ (one-time difference)
    old, _ = real.run(path)
    a, b = fingerprint(job), fingerprint(_succeeded(old))
    frames = [key for key in a if key.startswith("r_light_") and key.endswith(".fit")]
    assert len(frames) == selected and all(a[key] == b[key] for key in frames) and a["light_.seq"] == b["light_.seq"]
    # information only, not a criterion (the one-time difference may or may not show on this data)
    record_property("contract_master_equals_pre_contract_master", a["sky_master.fit"] == b["sky_master.fit"])


def test_real_siril_contract_stack_gives_one_master_for_any_settings(real):
    """KD-2 in small, real Siril: the contract stack repeated with other thread/memory settings (each worker one inner
    thread) gives one master; a pinned stack without the settings lines with the same T' and B is the same too."""
    folder = _script_job(real, "d4-settings")
    master = folder / "work" / "sky_master.fit"
    masters, plans = {}, {}
    for pin in (None, "cpu=3,mem=1", "cpu=1,mem=1", "cpu=7,mem=2"):
        master.unlink(missing_ok=True)
        with pytest.MonkeyPatch.context() as mp:
            if pin:
                mp.setenv(siril.SIRIL_PIN_ENV, pin)
            else:
                mp.delenv(siril.SIRIL_PIN_ENV, raising=False)
            with siril.job_stage_launches():
                plans[pin] = core._contract_stack(siril.run_script, str(STOCK_SIRIL), folder, core.LEGACY_STACK_SCRIPT,
                                                  lambda: None)
        masters[pin] = _fits(master)[0]
    assert [plans[p]["threads"] for p in plans] == [min(plans[None]["thread_cap"], FRAMES), 3, 1, 7]
    assert len(set(masters.values())) == 1, masters
    assert "mem_mode=1" not in (folder / "siril.ini").read_text(encoding="utf-8").splitlines()
    master.unlink()
    with pytest.MonkeyPatch.context() as mp:                     # the fixed-parameter control (pre-D4 script, pinned)
        mp.setenv(siril.SIRIL_PIN_ENV, "cpu=3,mem=1")
        siril.run_script(str(STOCK_SIRIL), folder, "02_stack", list(core.LEGACY_STACK_SCRIPT), lambda: None)
    assert _fits(master)[0] == masters[None]
