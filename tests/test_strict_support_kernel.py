"""P1a strict-support kernels are bit-identical to the unchanged reference paths (the oracles).

F-1 registered mask kernel, F-5a SIP grid release, F-5b separable anchor grid, G-4 native quick decisions.
Synthetic only; the opt-in real-data check at the end reads a kept strict job without modifying it.
"""
import hashlib
import json
import os
import shutil
import warnings
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from starstack import hashing, registered_support as rs, siril_support_map as sm, support_kernel as sk
from starstack.interpolation_support import _cubic_weights
from starstack.registration_distortion import RadialDistortion
from starstack.support_anchor import anchor_grid_from_validity, build_anchor_grid
from starstack.support_bounds import invalid_integral

from test_registered_support_independent import _seq, _write_rgb  # noqa: E402  (shared tiny-FITS helpers)


class Cancelled(Exception):
    pass


def _outcome(fn):
    """(value, None) or (None, (type, message)); warnings are returned as a sorted list of messages."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            value, error = fn(), None
        except (ValueError, np.linalg.LinAlgError) as exc:
            value, error = None, (type(exc).__name__, str(exc))
    return value, error, sorted(str(w.message) for w in caught)


def _masks(rng, h, w):
    ground = np.ones((h, w), bool)
    ground[int(h * .7):, :] = False
    ground[rng.integers(0, h, 4), rng.integers(0, w, 4)] = False
    lines = np.ones((h, w), bool)
    lines[:, w // 3] = False
    lines[h // 2, :] = False
    return {"all": np.ones((h, w), bool), "none": np.zeros((h, w), bool),
            "holes": rng.random((h, w)) > .01, "noise": rng.random((h, w)) > .3,
            "ground": ground, "lines": lines}


def _layouts(v):
    strided = np.zeros((v.shape[0] * 2, v.shape[1] * 3), bool)
    strided[::2, ::3] = v
    read_only = v.copy()
    read_only.flags.writeable = False
    return {"C": v, "F": np.asfortranarray(v), "strided": strided[::2, ::3], "read_only": read_only}


# --------------------------------------------------------------------------- F-5b anchor grid
@pytest.mark.parametrize("shape", [(1, 1), (1, 30), (30, 1), (9, 40), (10, 40), (11, 40), (40, 9), (40, 10),
                                   (40, 11), (10, 10), (11, 11), (37, 53), (129, 17), (300, 260)])
def test_anchor_grid_matches_integral_version_with_the_same_check_calls(shape):
    rng = np.random.default_rng(sum(shape))
    for mask in _masks(rng, *shape).values():
        for v in _layouts(mask).values():
            for before, after in ((4, 5), (0, 0), (0, 3), (2, 0), (1, 1), (6, 2), (10**30, 1)):
                for rows in (1, 3, 128, 10**30):
                    old_calls, new_calls = [], []
                    expected = build_anchor_grid(invalid_integral(v), before=before, after=after, rows=rows,
                                                 check=lambda: old_calls.append(1))
                    actual = anchor_grid_from_validity(v, before=before, after=after, rows=rows,
                                                       check=lambda: new_calls.append(1))
                    assert np.array_equal(actual, expected) and len(new_calls) == len(old_calls)
                    assert actual.flags.c_contiguous and not actual.flags.writeable and actual.flags.owndata
                    assert set(np.unique(actual.view(np.uint8)).tolist()) <= {0, 1}


@pytest.mark.parametrize("bad", [
    dict(validity=[[True]]), dict(validity=np.ones((3, 3), np.uint8)), dict(validity=np.ones((2, 2, 2), bool)),
    dict(validity=np.ones((0, 5), bool)), dict(before=-1), dict(before=True), dict(after=2.0),
    dict(rows=0), dict(rows=np.int64(4)), dict(validity=np.ones((3, 3), np.uint8), rows=0)])
def test_anchor_grid_errors_match_in_order(bad):
    kwargs = {"validity": np.ones((12, 12), bool), "before": 4, "after": 5, "rows": 128} | bad
    validity = kwargs.pop("validity")
    _, old, _ = _outcome(lambda: build_anchor_grid(invalid_integral(validity), **kwargs))
    _, new, _ = _outcome(lambda: anchor_grid_from_validity(validity, **kwargs))
    assert old is not None and new == old


def test_anchor_grid_reads_noncanonical_bool_bytes_logically():
    rng = np.random.default_rng(1)
    raw = np.where(rng.random((40, 50)) < .97, rng.choice([1, 2, 255], size=(40, 50)), 0).astype(np.uint8)
    v = raw.view(bool)
    for before, after in ((4, 5), (0, 0)):
        expected = build_anchor_grid(invalid_integral(raw != 0), before=before, after=after)
        actual = anchor_grid_from_validity(v, before=before, after=after)
        assert np.array_equal(actual.view(np.uint8), expected.view(np.uint8))   # 0/1 bytes even when span == 0


# --------------------------------------------------------------------------- F-1 mask kernel
def _sip_header(rng, w, h, order, strength):
    header = {"CTYPE1": "RA---TAN-SIP", "CTYPE2": "DEC--TAN-SIP", "CRPIX1": float(rng.uniform(0, w)),
              "CRPIX2": float(rng.uniform(0, h)), "AP_ORDER": order, "BP_ORDER": order}
    for prefix in ("AP", "BP"):
        for total in range(order + 1):
            for p in range(total, -1, -1):
                if rng.random() < .7:
                    header[f"{prefix}_{p}_{total - p}"] = float(rng.normal() * strength * max(w, h) ** (1 - total))
    return header


def _homographies(rng, w, h):
    def about_centre(a):
        t = np.array([[1., 0., w / 2], [0., 1., h / 2], [0., 0., 1.]])
        return t @ a @ np.linalg.inv(t)
    affine = np.eye(3)
    affine[:2, :2] += rng.normal(0, .02, (2, 2))
    affine[:2, 2] = rng.normal(0, 1.5, 2)
    perspective = affine.copy()
    perspective[2, :2] = rng.normal(0, 2e-3, 2)
    horizon = np.eye(3)
    horizon[2, 1] = -2. / h                                   # den == 0 crosses the image
    zoom = np.diag([rng.choice([.3, 3.]), rng.choice([.3, 3.]), 1.])
    return {"identity": np.eye(3), "shift": np.array([[1., 0., 2.], [0., 1., -1.], [0., 0., 1.]]),
            "half": np.array([[1., 0., .5], [0., 1., -.5], [0., 0., 1.]]), "affine": affine,
            "perspective": perspective, "horizon": horizon, "rot180": about_centre(np.diag([-1., -1., 1.])),
            "far": np.array([[1., 0., 5. * w], [0., 1., 0.], [0., 0., 1.]]), "zoom": about_centre(zoom)}


def _mask_pair(v, h, reference, header, rows):
    """(reference registered_mask outcome, kernel outcome) for one case; the SIP cache is emptied first."""
    height, width = v.shape
    anchors = build_anchor_grid(invalid_integral(v))

    def old():
        sm.release_distortion_grid()
        mapping = sm.make_export_mapping(h, reference, width, height, header)
        return rs.registered_mask(v, h, None, mapping=mapping, anchors=anchors)

    def new():
        sm.release_distortion_grid()
        params = sm.make_export_mapping(h, reference, width, height, header).export_parameters
        kernel = sk.ExportMaskKernel(width, height, params.spec, rows=rows)
        return sk.registered_mask_kernel(v, h, params=params, anchors=anchor_grid_from_validity(v), kernel=kernel)

    return _outcome(old), _outcome(new)


@pytest.mark.parametrize("rows", [None, 1, 2, 5, 1000])
@pytest.mark.parametrize("sip", [None, (0, 1e-3), (1, 3e-2), (3, 1e-3), (5, 3e-2), (2, 0.)])
def test_registered_mask_kernel_matches_reference(sip, rows):
    rng = np.random.default_rng(17 + (rows or 0) + (0 if sip is None else 100 * sip[0]))
    mixed = 0
    for height, width in ((20, 23), (41, 97), (64, 90)):
        header = None if sip is None else _sip_header(rng, width, height, *sip)
        masks = list(_masks(rng, height, width).values())
        for number, (name, h) in enumerate(_homographies(rng, width, height).items()):
            reference = np.eye(3) * (1., 1e30, 1e-30, 1e38, 1e-38)[number % 5]
            if number % 3 == 1:
                reference = reference @ np.array([[1., .01, .3], [-.01, 1., -.2], [0., 0., 1.]])
            v = masks[number % len(masks)]
            (expected, old_error, old_warn), (actual, new_error, new_warn) = _mask_pair(v, h, reference, header, rows)
            assert (new_error, new_warn) == (old_error, old_warn), name
            if expected is not None:
                assert np.array_equal(actual, expected), name
                assert actual.flags.c_contiguous == expected.flags.c_contiguous
                mixed += bool(expected.any() and not expected.all())
    assert mixed >= 5


@pytest.mark.parametrize("layout", ["C", "F", "strided", "read_only"])
def test_kernel_output_layout_and_saved_bytes_match(layout, tmp_path):
    rng = np.random.default_rng(3)
    v = _layouts(_masks(rng, 37, 53)["ground"])[layout]
    h = np.array([[1., .01, 1.3], [-.01, 1., -.7], [1e-4, 0., 1.]])
    (expected, _, _), (actual, _, _) = _mask_pair(v, h, np.eye(3), _sip_header(rng, 53, 37, 3, 1e-3), None)
    np.save(tmp_path / "old.npy", expected)
    np.save(tmp_path / "new.npy", actual)
    assert (tmp_path / "old.npy").read_bytes() == (tmp_path / "new.npy").read_bytes()


def test_kernel_block_indices_match_the_closure():
    rng = np.random.default_rng(5)
    for header in (None, _sip_header(rng, 90, 64, 3, 1e-2)):
        for h in _homographies(rng, 90, 64).values():
            sm.release_distortion_grid()
            mapping = sm.make_export_mapping(h, np.eye(3), 90, 64, header)
            kernel = sk.ExportMaskKernel(90, 64, mapping.export_parameters.spec, rows=7)
            kernel.set_frame(mapping.export_parameters.inverse)
            ones = np.ones(90 * 64, bool)
            for y0 in range(0, 64, 7):
                y1 = min(y0 + 7, 64)
                out = np.empty((y1 - y0, 90), bool)
                kernel.block(y0, y1, ones, out)
                yy, xx = np.mgrid[y0:y1, :90]
                mapped = mapping(np.column_stack((xx.ravel(), yy.ravel())).astype(np.float64))
                inside = ((mapped[:, 0] >= 0) & (mapped[:, 0] < 90) & (mapped[:, 1] >= 0) & (mapped[:, 1] < 64))
                assert np.array_equal(out.ravel(), inside)
                r = slice(0, y1 - y0)
                assert np.array_equal(kernel.ix[r].ravel()[inside], np.floor(mapped[inside, 0]).astype(np.int64))
                assert np.array_equal(kernel.iy[r].ravel()[inside], np.floor(mapped[inside, 1]).astype(np.int64))


@pytest.mark.parametrize("bad", ["list", "int", "3d", "empty", "nan", "singular", "anchor_shape", "anchor_dtype"])
def test_kernel_errors_match_reference_order_and_text(bad):
    v, h = np.ones((20, 20), bool), np.eye(3)
    anchors = build_anchor_grid(invalid_integral(v))
    params = sm.make_export_mapping(h, np.eye(3), 20, 20).export_parameters
    kernel = sk.ExportMaskKernel(20, 20, None)
    if bad == "list":
        v = v.tolist()
    elif bad == "int":
        v = v.astype(np.uint8)
    elif bad == "3d":
        v = v[None]
    elif bad == "empty":
        v = np.ones((0, 20), bool)
    elif bad == "nan":
        h = np.full((3, 3), np.nan)
    elif bad == "singular":
        h = np.zeros((3, 3))
    elif bad == "anchor_shape":
        anchors = anchors[:10]
    else:
        anchors = anchors.astype(np.uint8)
    mapping = sm.make_export_mapping(np.eye(3), np.eye(3), 20, 20)
    _, old, _ = _outcome(lambda: rs.registered_mask(v, h, None, mapping=mapping, anchors=anchors))
    _, new, _ = _outcome(lambda: sk.registered_mask_kernel(v, h, params=params, anchors=anchors, kernel=kernel))
    assert old is not None and new == old


@pytest.mark.parametrize("translate", [0., 1e6])
def test_nonfinite_sip_grid_fails_like_the_closure_only_when_a_point_is_valid(translate):
    header = {"CTYPE1": "RA---TAN-SIP", "CTYPE2": "DEC--TAN-SIP", "CRPIX1": 10., "CRPIX2": 10.,
              "AP_ORDER": 5, "BP_ORDER": 0, "AP_5_0": 1e300}
    v = np.ones((24, 30), bool)
    h = np.array([[1., 0., translate], [0., 1., 0.], [0., 0., 1.]])
    (_, old, _), (_, new, _) = _mask_pair(v, h, np.eye(3), header, None)
    assert new == old
    assert (old is None) == bool(translate)
    assert sm._distortion_grid.cache_info().currsize == 0


def test_check_exception_propagates_from_the_kernel():
    v = np.ones((30, 40), bool)
    params = sm.make_export_mapping(np.eye(3), np.eye(3), 40, 30).export_parameters
    for k in (1, 2, 3):
        calls = []

        def check():
            calls.append(1)
            if len(calls) == k:
                raise Cancelled()
        with pytest.raises(Cancelled):
            sk.registered_mask_kernel(v, np.eye(3), params=params, anchors=anchor_grid_from_validity(v),
                                      kernel=sk.ExportMaskKernel(40, 30, None, rows=11), check=check)


# --------------------------------------------------------------------------- G-4 native quick decisions
def _scalar_code(valid, j, i):
    height, width = valid.shape
    taps = [valid[min(height - 1, max(0, j + oy)), min(width - 1, max(0, i + ox))]
            for oy in (-1, 0, 1, 2) for ox in (-1, 0, 1, 2)]
    return sk.TAPS_ALL if all(taps) else sk.TAPS_NONE if not any(taps) else sk.TAPS_MIXED


def test_tap_state_matches_scalar_clamped_taps_and_tiling():
    rng = np.random.default_rng(9)
    for shape in ((1, 1), (2, 3), (3, 2), (5, 7), (9, 4), (12, 16)):
        for density in (0., .3, .8, .97, 1.):
            valid = rng.random(shape) < density
            code = sk.tap_state(valid)
            assert code.shape == (shape[0] + 1, shape[1] + 1) and code.dtype == np.uint8
            for j in range(-1, shape[0]):
                for i in range(-1, shape[1]):
                    assert code[j + 1, i + 1] == _scalar_code(valid, j, i)
            for rows in (1, 2, 3, 256, 10**6):
                assert np.array_equal(sk.tap_state(valid, rows=rows), code)


class Shift:
    def __init__(self, dx, dy):
        self.dx, self.dy = dx, dy

    def apply(self, points):
        return points + [self.dx, self.dy]


class Table:
    """Returns fixed points (cycled), so edge coordinates reach every block and tap state."""
    def __init__(self, points, dtype=np.float64, columns=2):
        self.points, self.dtype, self.columns = np.asarray(points, dtype=np.float64), dtype, columns

    def apply(self, points):
        out = self.points[np.arange(len(points)) % len(self.points)]
        if self.columns != 2:
            out = np.column_stack([out] + [out[:, :1]] * (self.columns - 2))
        return out.astype(self.dtype)


def _edge_points(width, height):
    tiny = [-1e-20, -2.0**-60, -np.nextafter(0, 1), -0.0, 0.0, 2.0**-60, .5, np.nextafter(.5, 0), np.nextafter(.5, 1)]
    xs = [*tiny, *(t + width - 1 for t in tiny), width - 0.5, np.nextafter(width - 0.5, 0), -0.5,
          np.nextafter(-0.5, 0), 1 - 2.0**-53, width / 2 + .5, np.nan, np.inf, -np.inf]
    ys = [*tiny, *(t + height - 1 for t in tiny), height - 0.5, np.nextafter(height - 0.5, 0), -0.5,
          np.nextafter(-0.5, 0), height / 2 - .25, np.nan]
    return [(x, y) for x in xs for y in ys]


def test_fraction_can_round_to_exactly_one_and_the_tap_map_still_decides_it():
    x = np.array([-1e-20])
    assert (x - np.floor(x))[0] == 1.0            # sample_support's t == 1: offset 1 has weight 1
    valid = np.zeros((6, 7), bool)
    valid[:, 0] = True                           # the weight-1 tap (column 0) is valid, its neighbours not
    model = Table([(-1e-20, 2.), (-1e-20, -1e-20), (3 - 1e-20, 2.)])
    assert np.array_equal(rs.native_mask(valid, model, quick=True), rs._native_mask_reference(valid, model))


@pytest.mark.parametrize("rows", [1, 3, 64])
def test_native_quick_decisions_match_reference(rows):
    rng = np.random.default_rng(rows)
    count = 0
    for shape in ((1, 1), (2, 9), (5, 3), (12, 16), (31, 47)):
        height, width = shape
        models = [Shift(0, 0), Shift(.25, 0), Shift(.25, -.25), Shift(-.49, .49), Shift(1, -2), Shift(.5, .5),
                  Shift(2.0**-50, -2.0**-50), Shift(1e300, 0), Shift(-.5, -.5),
                  RadialDistortion(.04, -.01, width / 2, height / 2, max(shape), .98, "u_to_p"),
                  RadialDistortion(-.03, .01, width / 3, height / 2, max(shape), 1., "p_to_u"),
                  Table(_edge_points(width, height)), Table(_edge_points(width, height), dtype=np.float32),
                  Table([(1, 1), (0, 2)], dtype=np.int64)]
        for density in (0., .5, .95, 1.):
            valid = rng.random(shape) < density
            if density == .95 and min(shape) > 4:
                valid[height // 2:, :] = False    # large all-invalid and all-valid regions
            for model in models:
                expected, old_error, old_warn = _outcome(lambda: rs._native_mask_reference(valid, model, rows=rows))
                actual, new_error, new_warn = _outcome(lambda: rs.native_mask(valid, model, rows=rows, quick=True))
                assert (new_error, new_warn) == (old_error, old_warn)
                assert np.array_equal(actual, expected) and actual.flags.c_contiguous == expected.flags.c_contiguous
                count += 1
    assert count > 250


def test_native_wrong_model_output_shape_fails_like_reference():
    valid = np.ones((6, 6), bool)
    model = Table([(1., 1.)], columns=3)
    _, old, _ = _outcome(lambda: rs._native_mask_reference(valid, model))
    _, new, _ = _outcome(lambda: rs.native_mask(valid, model, quick=True))
    assert old is not None and new == old


def test_nearest_tap_weight_never_vanishes():
    t = np.concatenate([np.linspace(0., 1., 200001), [2.0**-52, np.nextafter(.5, 0), .5, np.nextafter(.5, 1),
                                                      np.nextafter(1., 0), 1.]])
    weights = _cubic_weights(t, (-1, 0, 1, 2))
    nearest = np.where(t <= .5, weights[:, 1], weights[:, 2])
    assert nearest.min() >= .5625 - 1e-15 and nearest.min() > 0
    assert weights[-1, 2] == 1.0                 # t == 1: offset 1, z == 0


def test_native_mask_switch_and_default():
    valid = np.ones((12, 16), bool)
    valid[5, 5] = False
    for quick in (None, True, False):
        assert np.array_equal(rs.native_mask(valid, Shift(.25, .25), quick=quick),
                              rs._native_mask_reference(valid, Shift(.25, .25)))


# --------------------------------------------------------------------------- prepare_exports / finish_stack
class Static:
    def __init__(self, v):
        self.v = v

    def for_frame(self, stem):
        return self.v


class PerFrame:
    def __init__(self, table):
        self.table = table

    def for_frame(self, stem):
        return self.table[stem]


def _work(path, rng, height, width, count, sip):
    path.mkdir(parents=True)
    matrices = [np.array([[1 + rng.normal(0, .01), rng.normal(0, .01), rng.normal(0, 2)],
                          [rng.normal(0, .01), 1 + rng.normal(0, .01), rng.normal(0, 2)],
                          [rng.normal(0, 1e-5), rng.normal(0, 1e-5), 1.]]) for _ in range(count)]
    _seq(path / rs.EXPORT_SEQUENCE, matrices)
    if sip:
        with (path / rs.EXPORT_SEQUENCE).open("a", encoding="utf-8") as stream:
            stream.write("D1 2 distortion.wcs\n")
        fits.PrimaryHDU(header=fits.Header(_sip_header(rng, width, height, 3, 1e-3))).writeto(path / "distortion.wcs")
    for number in range(1, count + 1):
        _write_rgb(path / f"r_light_{number:05d}.fit", rng.random((height, width)) + .1)


def _files(work):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(work.iterdir())
            if p.is_file() and p.name != rs.RECORD}


def _run(work, source, model, scope, **kwargs):
    if scope is None:
        return rs.prepare_exports(work, source, model, **kwargs)
    with hashing.run_scope("p1a-test", mode=scope):
        return rs.prepare_exports(work, source, model, **kwargs)


@pytest.mark.parametrize("sip,per_frame,scope", [(False, False, None), (True, False, None), (True, True, "on"),
                                                 (False, True, "shadow"), (True, False, "off"), (True, False, "on")])
def test_prepare_exports_twins_are_byte_identical(tmp_path, monkeypatch, sip, per_frame, scope):
    rng = np.random.default_rng(11)
    height, width, count = 41, 61, 3
    _work(tmp_path / "src", rng, height, width, count, sip)
    v = np.ones((height, width), bool)
    v[int(height * .7):, :] = False
    table = {f"light_{i:05d}": v if i % 2 else v & (rng.random((height, width)) > .02) for i in range(1, count + 1)}
    model = RadialDistortion(.01, -.002, width / 2, height / 2, max(width, height) / 2, .99, "u_to_p") if sip else None
    results = {}
    for label, env in (("reference", "reference"), ("buffered", None)):
        shutil.copytree(tmp_path / "src", tmp_path / label)
        if env:
            monkeypatch.setenv("AESTER_STRICT_KERNEL", env)
        else:
            monkeypatch.delenv("AESTER_STRICT_KERNEL", raising=False)
        record = _run(tmp_path / label, PerFrame(table) if per_frame else Static(v), model, scope,
                      check=lambda: None, progress=lambda *a: None, retain_frame_support=True)
        assert sm._distortion_grid.cache_info().currsize == 0            # F-5a: released
        results[label] = (record, _files(tmp_path / label))
    (old, old_files), (new, new_files) = results["reference"], results["buffered"]
    assert old["implementation"] == rs.REFERENCE_IMPLEMENTATION and new["implementation"] == rs.SUPPORT_IMPLEMENTATION
    strip = lambda record: {k: v for k, v in record.items() if k not in ("filter_seconds", "implementation")}
    assert strip(new) == strip(old) and new_files == old_files


def test_mapping_without_parameters_falls_back_and_is_labelled_reference(tmp_path, monkeypatch):
    rng = np.random.default_rng(12)
    _work(tmp_path / "src", rng, 30, 40, 2, False)
    shutil.copytree(tmp_path / "src", tmp_path / "plain")
    real = rs.make_export_mapping

    def stripped(*args):
        mapping = real(*args)

        def plain(points):                             # a caller's own mapping: no export_parameters
            return mapping(points)
        plain.contract = mapping.contract
        return plain
    monkeypatch.delenv("AESTER_STRICT_KERNEL", raising=False)
    monkeypatch.setattr(rs, "make_export_mapping", stripped)
    record = rs.prepare_exports(tmp_path / "src", Static(np.ones((30, 40), bool)), None)
    assert record["implementation"] == rs.REFERENCE_IMPLEMENTATION
    monkeypatch.setattr(rs, "make_export_mapping", real)
    again = rs.prepare_exports(tmp_path / "plain", Static(np.ones((30, 40), bool)), None)
    assert again["implementation"] == rs.SUPPORT_IMPLEMENTATION
    assert _files(tmp_path / "src") == _files(tmp_path / "plain")


@pytest.mark.parametrize("k", [1, 2, 3, 5])
def test_cancellation_publishes_no_record_and_releases_the_grid(tmp_path, monkeypatch, k):
    monkeypatch.delenv("AESTER_STRICT_KERNEL", raising=False)
    work = tmp_path / "work"
    _work(work, np.random.default_rng(k), 30, 40, 2, True)
    calls = []

    def check():
        calls.append(1)
        if len(calls) == k:
            raise Cancelled()
    model = RadialDistortion(.01, 0, 20, 15, 25, 1, "u_to_p")
    with pytest.raises(Cancelled):
        rs.prepare_exports(work, Static(np.ones((30, 40), bool)), model, check=check)
    assert not (work / rs.RECORD).exists()
    assert sm._distortion_grid.cache_info().currsize == 0


@pytest.mark.parametrize("label", ["buffered", "reference", "old_record"])
def test_finish_stack_native_path_follows_the_record_label(tmp_path, monkeypatch, label):
    monkeypatch.delenv("AESTER_STRICT_KERNEL", raising=False)
    rng = np.random.default_rng(21)
    _work(tmp_path / "work", rng, 30, 40, 2, False)
    v = np.ones((30, 40), bool)
    v[22:, :] = False
    rs.prepare_exports(tmp_path / "work", Static(v), None)
    record = json.loads((tmp_path / "work" / rs.RECORD).read_text(encoding="utf-8"))
    if label == "reference":
        record["implementation"] = rs.REFERENCE_IMPLEMENTATION
    elif label == "old_record":
        record.pop("implementation")
    (tmp_path / "work" / rs.RECORD).write_text(json.dumps(record), encoding="utf-8")
    _write_rgb(tmp_path / "work" / "sky_master.fit", rng.random((30, 40)) + .1)
    # The finishing process's switch does not matter: the record decides.
    monkeypatch.setenv("AESTER_STRICT_KERNEL", "reference" if label == "buffered" else "buffered")
    maps = []
    monkeypatch.setattr(sk, "tap_state", lambda valid, **kw: maps.append(1) or real_tap_state(valid, **kw))
    model = RadialDistortion(.02, 0, 20, 15, 25, 1, "u_to_p")
    done = rs.finish_stack(tmp_path / "work", tmp_path / "work" / "sky_master.fit", model)
    assert len(maps) == (1 if label == "buffered" else 0)
    stack = rs.load_array(tmp_path / "work", done["stack_valid"])
    assert np.array_equal(rs.load_array(tmp_path / "work", done["native_valid"]),
                          rs._native_mask_reference(stack, model))


real_tap_state = sk.tap_state


# --------------------------------------------------------------------------- real data (opt-in, read-only)
@pytest.mark.skipif(not os.environ.get("AESTER_P1A_REAL_JOB"), reason="set AESTER_P1A_REAL_JOB to a kept strict job")
def test_kept_job_frames_and_native_map_are_reproduced():
    from starstack import core
    job_dir = Path(os.environ["AESTER_P1A_REAL_JOB"])
    job, work = json.loads((job_dir / "job.json").read_text(encoding="utf-8")), job_dir / "work"
    record = json.loads((work / rs.RECORD).read_text(encoding="utf-8"))
    height, width = record["shape"]
    source = core._validity_source(job_dir, job, width, height, int(json.loads(
        (work / "sky_validity.json").read_text(encoding="utf-8"))["dilation_px"]))
    transforms = rs.export_transforms(work / rs.EXPORT_SEQUENCE)
    from starstack.siril_sequence import read_seq
    sequence = read_seq(work / rs.EXPORT_SEQUENCE)
    reference_h = sequence["regdata"][1][sequence["header"]["reference_image"]]["H"]
    header = fits.getheader(work / "distortion.wcs") if record.get("model") is not None else None
    kernel, old_valid, anchors, counts = None, None, None, np.zeros((height, width), np.uint16)
    for entry in record["frames"]:
        valid = source.for_frame(entry["stem"])
        if valid is not old_valid:
            anchors, old_valid = anchor_grid_from_validity(valid), valid
        params = sm.make_export_mapping(transforms[entry["stem"]], reference_h, width, height, header).export_parameters
        kernel = sk.kernel_for(kernel, params)
        supported = sk.registered_mask_kernel(valid, transforms[entry["stem"]], params=params, anchors=anchors, kernel=kernel)
        if "explicit_validity" in entry:
            assert np.array_equal(supported, rs.load_frame_support(work, entry["explicit_validity"]))
        assert int(supported.sum()) == entry["supported_pixels"]
        counts += supported
    sm.release_distortion_grid()
    assert np.array_equal(counts, rs.load_array(work, record["eligible"]))
    if record.get("state") == "complete" and record.get("model") is not None:
        native = rs.native_mask(rs.load_array(work, record["stack_valid"]), RadialDistortion(**record["model"]), quick=True)
        assert np.array_equal(native, rs.load_array(work, record["native_valid"]))
