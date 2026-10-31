"""Run conditions and data for the P1c finishing site tests (docs/EFFICIENCY_P1C_PREREG_20261005.md §3.5, §6.2 A5).

- ``forced(case)``: one site run under a harness Case: its worker decision overridden through AESTER_FRAME_WORKERS
  (None keeps the site's own decision), the pool's in-flight bound through a wrapper of frame_pool.run_ordered (the
  product never passes one), the frame pool's test seams (state policy, jitter, the R6 audit), and optionally the
  delivered unit indices recorded (§6.2 (c)).
- ``notes()``: the pool notes a site wrote (``frame_pool.<site>`` timings notes) while the block ran.
- Synthetic scenes: a Milky Way render scene with subnormal mask weights (R21) and a signed layer stack (diffuse D,
  base B, source S, star residual R = B - D, trail residual T = S - B) in which a chosen row block can be made to fail.
- The F48 fixture's finishing inputs and outputs (read only; outputs go to the test's basetemp).
"""
from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image, ImageCms

import p1c_site_harness as h
from starstack import frame_pool, profiling

ENV = frame_pool.ENV_WORKERS


@contextlib.contextmanager
def forced(case, *, delivered=None, audit=True, head_last=False, head_seconds=0.05):
    saved_env = os.environ.get(ENV)
    saved_run = frame_pool.run_ordered
    try:
        if case.workers is None:
            os.environ.pop(ENV, None)
        else:
            os.environ[ENV] = str(case.workers)

        def run(fn, items, **kwargs):
            if case.inflight is not None:
                kwargs.setdefault("inflight", case.inflight)
            if delivered is not None:
                deliver = kwargs.get("on_result")

                def record(i, value):
                    delivered.append(i)
                    if deliver is not None:
                        deliver(i, value)
                kwargs["on_result"] = record
            return saved_run(fn, items, **kwargs)

        frame_pool.run_ordered = run
        with h.pool_conditions(case, audit=audit, head_last=head_last, head_seconds=head_seconds):
            yield case
    finally:
        frame_pool.run_ordered = saved_run
        if saved_env is None:
            os.environ.pop(ENV, None)
        else:
            os.environ[ENV] = saved_env


@contextlib.contextmanager
def notes():
    """``with notes() as seen:`` -- afterwards ``seen`` holds the timings notes of a StageTimer live during the block
    (``frame_pool.<site>`` lists, one entry per call)."""
    timer = profiling.StageTimer()
    seen = {}
    try:
        yield seen
    finally:
        seen.update(timer.finish().get("notes") or {})


def pool_entries(seen, site):
    return seen.get(f"frame_pool.{site}") or []


# ---------------------------------------------------------------------------------------------- synthetic scenes

def milkyway_scene(seed, dtype=np.uint16, shape=(96, 72)):
    """rgb, sky, protect, mask with exclusions, a protected line, subnormal mask weights and (float32) near-tiny RGB."""
    rng = np.random.default_rng(seed)
    hgt, wid = shape
    y, x = np.mgrid[:hgt, :wid].astype(np.float32)
    base = .18 + .14 * x / wid + .05 * y / hgt + .01 * np.sin(x * .7 + seed)
    rgb = np.stack((base * 1.05, base, base * .93), axis=-1).astype(np.float32)
    rgb += rng.normal(0, .01, rgb.shape).astype(np.float32)
    rgb = np.clip(rgb, .001, .999).astype(np.float32)
    rgb[5, 7] = (.99, .99, .99)                                    # a near-white locked pixel
    sky = np.ones(shape, np.float32)
    sky[-10:] = 0
    sky[-12:-10] = .5
    protect = np.zeros(shape, bool)
    protect[hgt // 2, wid // 4:3 * wid // 4] = True
    mask = np.clip(rng.random(shape).astype(np.float32) * 1.4 - .2, 0, 1).astype(np.float32)
    mask[:, :5] = 0
    tiny = np.nextafter(np.float32(0), np.float32(1))
    mask[10:30, 20:40] = tiny                                       # subnormal weights (R21, prereg §3.5)
    mask[31:33, 20:40] = np.float32(np.finfo(np.float32).tiny) / 3
    if dtype == np.uint16:
        rgb = np.rint(rgb * 65535).astype(np.uint16)
    else:
        rgb[40, 10:14] = np.float32(np.finfo(np.float32).tiny) * 2  # float32: the near-tiny lock
        # float32: unlocked values just above 4 tiny under the subnormal weights (R21): there base + delta * alpha
        # keeps the subnormal delta only while denormals are kept, so flush-to-zero changes these pixels' last bits
        rgb[31:33, 20:40] = np.float32(np.finfo(np.float32).tiny) * 6
    return rgb, sky, protect, mask


def layer_stack(seed, shape=(200, 40), *, paired=False, negative=True):
    """Diffuse D, base B, source S (uint16 RGB), R = B - D and T = S - B (int32), sky and protect masks, with stars only
    where the diffuse layer may change (pure sky outside protection; corridors too when paired)."""
    rng = np.random.default_rng(seed)
    hgt, wid = shape
    d = rng.integers(3000, 40000, shape + (3,), dtype=np.int64)
    sky = np.ones(shape, np.float32)
    sky[-9:] = 0
    sky[-11:-9] = .5
    protect = np.zeros(shape, bool)
    protect[:, wid // 3:wid // 3 + 2] = True
    allowed = sky == 1
    if not paired:
        allowed &= ~protect
    stars = np.zeros(shape + (3,), np.int64)
    spots = rng.random(shape) < .08
    stars[spots] = rng.integers(200, 24000, (int(spots.sum()), 3))
    if negative:
        dents = (rng.random(shape) < .02) & ~spots
        stars[dents] = -rng.integers(1, 1500, (int(dents.sum()), 3))
    stars[~allowed] = 0
    b = np.clip(d + stars, 1, 65534)
    trails = np.zeros_like(b)
    if paired:
        trails[:, wid // 3:wid // 3 + 2] = rng.integers(-800, 9000, (hgt, 2, 3))
    s = np.clip(b + trails, 1, 65534)
    d, b, s = (a.astype(np.uint16) for a in (d, b, s))
    r = b.astype(np.int32) - d.astype(np.int32)
    t = s.astype(np.int32) - b.astype(np.int32)
    return {"s": s, "b": b, "d": d, "r": r, "t": t if paired else None, "sky": sky, "protect": protect,
            "paired": paired, "shape": s.shape}


def reduction_controls(seed, shape, *, extended):
    """weight (float32, exact 0 and 1 included) and local strength (float64, exact 1 included) for the reduction."""
    rng = np.random.default_rng(seed + 1000)
    weight = rng.random(shape).astype(np.float32)
    weight[:, :3] = 1
    weight[:2] = 0
    local = None
    if extended:
        local = np.clip(rng.random(shape) * 1.3, 0, 1).astype(np.float64)
        local[:, :3] = 1
    return weight, local


# ---------------------------------------------------------------------------------------------- display TIFF files

ICC = None


def icc():
    global ICC
    if ICC is None:
        ICC = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    return ICC


DISPLAY = {"scene_linear": False, "display_encoding_applied": True, "roworder": "TOP-DOWN", "color_space": "encoded_sRGB"}


def write_display(path, array, extra=None):
    profile = icc()
    tifffile.imwrite(path, array, photometric="rgb", metadata={**DISPLAY, **(extra or {})},
                     extratags=[(34675, "B", len(profile), profile, False)])
    return Path(path)


def make_separation(folder, stack, *, sky_extra_rows=()):
    """A succeeded separation (star_separation.separate_stars, a provider returning ``stack``'s diffuse layer) on disk
    for reduce_stars / suggest_star_reduction / export_star_layers. Returns the result.json path."""
    from starstack import core
    from starstack.star_separation import separate_stars
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    source = write_display(folder / "source.tif", stack["s"])
    base = None
    if stack["paired"]:
        base = write_display(folder / "base.tif", stack["b"], {"transient_background_for_sha256": core._sha(source)})
    sky_path, protect_path = folder / "sky.tif", folder / "protect.png"
    tifffile.imwrite(sky_path, stack["sky"])
    Image.fromarray((stack["protect"] * 255).astype(np.uint8)).save(protect_path)
    diffuse = stack["d"]

    def provider(image, out, _stage):
        out[:] = diffuse
        return {"synthetic": True}

    record = separate_stars(source, folder / "separation", sky_path, protect_path, provider=provider,
                            provider_identity={"backend": "p1c-test"}, transient_background=base)
    assert record["state"] == "succeeded", record.get("error")
    return Path(record["artifacts"]["diffuse_srgb16.tif"]["path"]).parent / "result.json"


def milkyway_files(folder, *, shape=(640, 48), seed=3):
    """finish_milkyway inputs on disk: a display TIFF, PNG sky / protected / Milky Way masks and bound evidence (a
    synthetic declaration exercising the enabled branch, not a review). Returns a dict of paths."""
    from starstack import core
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    rgb, sky, protect, mask = milkyway_scene(seed, np.uint16, shape)
    paths = {"source": write_display(folder / "source.tif", rgb)}
    for name, values in (("sky", np.rint(sky * 255)), ("protected", protect * 255), ("milkyway", np.rint(mask * 255))):
        paths[name] = folder / (name + ".png")
        Image.fromarray(values.astype(np.uint8)).save(paths[name])
    evidence = folder / "evidence.json"
    evidence.write_text(json.dumps({"version": "milkyway-subject-evidence-1", "status": "present",
                                    "method": "human-review", "reviewer": "p1c synthetic fixture",
                                    "rationale": "Synthetic enabled-path fixture, not a sky review.",
                                    "source_sha256": core._sha(paths["source"]),
                                    "milkyway_mask_sha256": core._sha(paths["milkyway"])}), encoding="utf-8")
    paths["evidence"] = evidence
    return paths


# ---------------------------------------------------------------------------------------------- the F48 fixture

F48_WORKFLOW = "workflow-d7aab079dffc34757bcace4f"
F48_JOB = "job-eb54de1d6246d4c7ff7714fd"


class F48:
    """Paths of the F48 reference run's finishing inputs and outputs (skips when the fixture is missing)."""

    def __init__(self):
        run = h.f48_run()
        self.workflow = run / "c" / F48_WORKFLOW
        self.job = self.workflow / "stack" / F48_JOB
        if not self.job.is_dir():
            h.fixture_path("outputs", "perf-baseline-20260928", "runs", "F48", "c", F48_WORKFLOW, "stack", F48_JOB)
        self.base_photo = self.job / "products" / "565088f08515" / "natural_display_16bit.tif"
        self.trails_photo = self.job / "products" / "2051b01213ed4c918a2598f41b6ab1f5" / "transients" / "natural_display_16bit.tif"
        self.compose_weight = self.job / "products" / "565088f08515" / "sky_weight.tiff"
        self.sky = self.workflow / "masks-8b946b236f" / "sky_weight.tiff"
        self.protect = self.workflow / "masks-8b946b236f" / "protected.png"
        self.region = self.workflow / "milkyway-region-54deed431f.tif"
        finishing = self.workflow / "finishing"
        self.separation = finishing / "separation" / "separation-775f2350a457" / "result.json"
        self.milkyway = finishing / "milkyway" / "milkyway-0fc0d88053fe"
        self.reduction = finishing / "reduction" / "star-reduction-7290a7d288d4"
        self.layers = finishing / "layers" / "star-layers-28b9491bdce4"
        self.advice = finishing / "advice" / "star-advice-90a006f10936"

    def record(self, folder):
        return json.loads((folder / "result.json").read_text(encoding="utf-8"))

    def evidence(self, folder):
        keys = [k for k in self.record(folder)["input_hashes"] if "milkyway-evidence" in k]
        assert len(keys) == 1
        return Path(keys[0])

    def milkyway_inputs(self):
        from starstack import photographic_job as job
        image = job._read_display(self.base_photo)
        shape = image.shape[:2]
        return (image, job._mask(self.sky, shape), job._mask(self.protect, shape, binary=True),
                job._mask(self.region, shape))


def assert_record_like(expected, actual, *, ignore=("seconds", "render_seconds", "processing_directory")):
    """Records equal (values and key order) apart from timings and the staging / publication paths."""
    def strip(value):
        if isinstance(value, dict):
            return {k: (None if k in ignore else strip(v)) for k, v in value.items()}
        if isinstance(value, list):
            return [strip(v) for v in value]
        return value
    h.assert_same(strip(expected), strip(actual))
