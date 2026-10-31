"""Reusable parallel rendering (docs/CAMERA_LOOKS_AND_PARALLEL_PREREG_20260926.md A1–A4): the fast HueSatMap paths equal
the reference functions bit for bit, thread-safe looks render identically for any worker count with identical
counters, the camera→master conversion is identical in parallel, and the worker policy keeps its guards."""
import dataclasses
import threading

import numpy as np
import pytest

from starstack import camera_looks as cl, display_transform as dt, experimental_dcp as dcp, huesat_fast as hf
from starstack import master_export as me, performance, photo_render as pr
from starstack.experimental_huesat import lookup_huesat

LOOK_ID = "sony-ilce7m4-rt-cc0-d50"


def _rgb(n=200_000, seed=0):
    rng = np.random.default_rng(seed)
    rgb = np.abs(rng.normal(0.2, 0.3, (n, 3)))
    rgb[:500] = rgb[:500, :1]                       # grey
    rgb[500:1000] = 0.0                             # black
    rgb[1000:1500, 1:] = 0.0                        # one channel
    rgb[1500:2000, 0] = rgb[1500:2000, 1]           # argmax ties
    rgb[2000:2500] = rng.integers(0, 4, (500, 3)) / 3.0
    return rgb


def test_fast_hsv_equals_reference_bit_for_bit():
    rgb = _rgb()
    for a, b in zip(dcp.rgb_to_hsv(rgb), hf.rgb_to_hsv(rgb)):
        assert np.array_equal(a, b) and np.array_equal(np.signbit(a), np.signbit(b))
    h, s, v = dcp.rgb_to_hsv(rgb)
    rng = np.random.default_rng(1)
    s2 = s * rng.uniform(0.5, 2.0, s.size)
    h2 = h + rng.normal(0, 40, h.size)
    h2[:8] = [-1e-20, 360.0, 359.9999999999, 0.0, -0.0, 720.5, -360.0, 1e-300]
    a, b = dcp.hsv_to_rgb(h2, s2, v), hf.hsv_to_rgb(h2, s2, v)
    assert np.array_equal(a, b) and np.array_equal(np.signbit(a), np.signbit(b))


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_fast_lookup_equals_reference_bit_for_bit(dtype):
    rgb = _rgb(seed=2)
    h, s, _ = dcp.rgb_to_hsv(rgb)
    h = h + np.random.default_rng(3).normal(0, 50, h.size)
    sat = np.clip(s, 0.0, 1.0)
    sat[:4] = [0.0, 1.0, 1 - 1e-16, 0.5]
    for table in (cl.table(LOOK_ID), cl.scaled_table(LOOK_ID, 0.5)):
        t = table.astype(dtype)
        assert np.array_equal(lookup_huesat(t, h, sat), hf.lookup(hf.prepare(t), h, sat))
    with pytest.raises(ValueError):
        hf.prepare(np.full((90, 30, 3), np.nan))
    with pytest.raises(ValueError):
        hf.prepare(-np.ones((90, 30, 3)))


def _scene(h=700, w=96, seed=4):
    rng = np.random.default_rng(seed)
    data = np.abs(rng.normal(0.03, 0.02, (h, w, 3))).astype(np.float32)
    data[:40, :20] = [-0.01, 0.05, 0.02]            # negative in ProPhoto → bypass
    data[100:110, 50:60] = 2.0                      # over white
    saturated = np.zeros((h, w), bool)
    saturated[100:110, 50:60] = True
    region = (np.arange(h)[:, None] < h // 2).astype(np.uint8).repeat(w, axis=1)
    return data, saturated, region


def _params(**kw):
    return dt.Params(anchor=0.03, noise_sigma=0.004, shadow_smooth_radius=3, shadow_smooth_start=0.04,
                     shadow_smooth_end=0.16, **kw).validate()


def _render(data, saturated, region, params, look, workers):
    flags = np.zeros(data.shape[:2], np.uint8)
    out, stats = dt.render(data, params, camera_saturated=saturated, look=look, region=region, flags=flags, workers=workers)
    return out, flags, stats


@pytest.mark.parametrize("kind", ["camera_hsm", "hsm_table", "none"])
def test_thread_safe_looks_render_identically_for_any_worker_count(kind):
    data, saturated, region = _scene()
    results = []
    for workers in (1, 2, 4, 8):
        if kind == "camera_hsm":
            params, look = _params(look="camera_hsm", look_id=LOOK_ID, look_strength=0.5), None
            look, record = cl.make_look(LOOK_ID, 0.5, "rec2020")
            counters = record["counters"]
        elif kind == "hsm_table":
            params = _params(look="dcp_hsm")
            look, counters = pr.hsm_look(cl.table(LOOK_ID), "rec2020")
        else:
            params, look, counters = _params(), None, None
        if look is not None:
            assert look.thread_safe is True
        out, flags, stats = _render(data, saturated, region, params, look, workers)
        results.append((out, flags, stats, None if counters is None else dict(counters)))
    first = results[0]
    for out, flags, stats, counters in results[1:]:
        assert np.array_equal(out, first[0]) and np.array_equal(flags, first[1]) and stats == first[2] and counters == first[3]
    if first[3] is not None:
        assert first[3]["bypassed_negative"] > 0 and first[3]["processed"] > 0


def test_parallel_look_really_runs_on_several_threads():
    data, saturated, region = _scene(h=1024)
    base, _ = pr.hsm_look(cl.table(LOOK_ID), "rec2020")
    seen = set()

    def look(block):
        seen.add(threading.get_ident())
        return base(block)
    look.thread_safe = True
    dt.render(data, _params(look="dcp_hsm"), look=look, workers=4)
    assert len(seen) > 1


def test_unmarked_callbacks_stay_serial():
    seen = set()

    def look(block):
        seen.add(threading.get_ident())
        return block
    dt.render(np.full((600, 8, 3), .03, np.float32), dt.Params(), look=look, workers=8)
    assert seen == {threading.get_ident()}


def test_convert_camera_master_is_identical_in_parallel(monkeypatch):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import color_fixtures as cf
    cam, *_ = cf.synthetic_camera_master(h=900, w=120)
    basis = me.colour_basis_from_dng(cf.write_reference_dng(Path(__import__("tempfile").mkdtemp()) / "r.dng"))
    results = []
    for workers in (1, 3, 8):
        monkeypatch.setattr(performance, "render_workers", lambda *a, _w=workers, **k: _w)
        results.append(me.convert_camera_master(cam, basis, "rec2020", rows=128))
    for other in results[1:]:
        assert all(np.array_equal(a, b) for a, b in zip(results[0], other))
    bad = cam.copy()
    bad[850, 3, 1] = np.nan
    with pytest.raises(ValueError):
        me.convert_camera_master(bad, basis, "rec2020", rows=128)


@pytest.mark.parametrize("height,rows,workers", [(1, 128, 8), (127, 128, 4), (1000, 128, 3), (9566, 128, 8), (426, 128, 8)])
def test_row_ranges_are_block_aligned_and_cover(height, rows, workers):
    ranges = performance.row_ranges(height, rows, workers)
    assert ranges[0][0] == 0 and ranges[-1][1] == height and len(ranges) <= workers
    assert all(a[1] == b[0] for a, b in zip(ranges, ranges[1:]))
    assert all(start % rows == 0 for start, _ in ranges)


def test_worker_policy(monkeypatch):
    monkeypatch.delenv("AESTER_RENDER_WORKERS", raising=False)
    big = 128 * 1024**3
    assert performance.render_workers(9566, 6374, cpu_count=192, memory_bytes=big) == 8
    assert performance.render_workers(9566, 6374, cpu_count=8, memory_bytes=big) == 4
    # a look's larger temporaries shrink the memory-bound worker count
    small = 6 * 1024**3
    assert (performance.render_workers(9566, 6374, cpu_count=192, memory_bytes=small, bytes_per_pixel=1536)
            <= performance.render_workers(9566, 6374, cpu_count=192, memory_bytes=small))
    assert performance.render_workers(426, 640, cpu_count=192, memory_bytes=big) == 1
    assert performance.render_workers(426, 640, cpu_count=192, memory_bytes=big, min_pixels=performance.PREVIEW_MIN_PIXELS) > 1
    monkeypatch.setenv("AESTER_RENDER_WORKERS", "8")
    assert performance.render_workers(9566, 6374, cpu_count=192, memory_bytes=big) == 8
    monkeypatch.setenv("AESTER_RENDER_WORKERS", "9")
    with pytest.raises(ValueError):
        performance.render_workers(9566, 6374)
    with pytest.raises(ValueError):
        dt.render(np.ones((10, 10, 3), np.float32) * .03, dt.Params(), workers=9)


def test_display_workers_follow_the_look(monkeypatch):
    monkeypatch.delenv("AESTER_RENDER_WORKERS", raising=False)
    calls = []
    monkeypatch.setattr(performance, "render_workers", lambda h, w, **k: calls.append(k) or 1)
    pr.display_workers(dt.Params(), 400, 600, preview=True)
    pr.display_workers(dataclasses.replace(dt.Params(), look="camera_hsm", look_id=LOOK_ID, look_strength=0.5), 4000, 6000)
    assert calls[0] == {"bytes_per_pixel": 768, "min_pixels": performance.PREVIEW_MIN_PIXELS}
    assert calls[1] == {"bytes_per_pixel": pr.LOOK_BYTES_PER_PIXEL}
