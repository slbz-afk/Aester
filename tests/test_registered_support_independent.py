"""Independent support-stage acceptance with scalar footprints and tiny FITS.

No real engine: the actual Siril coordinate/kernel contract is validated in
the companion experiment. These tests check propagation, persistence, and
data-state boundaries after that contract has been declared.
"""
import hashlib
import json
import math

import numpy as np
import pytest
from astropy.io import fits

from starstack import registered_support as rs
from starstack import fits_crop
from starstack.records import atomic_json
from starstack.registration_distortion import RadialDistortion


def _write_rgb(path, data, *, bottom_up=True):
    a = np.asarray(data, dtype=np.float32)
    if a.ndim == 2:
        a = np.repeat(a[None], 3, axis=0)
    if bottom_up:
        a = a[:, ::-1]
    hdu = fits.PrimaryHDU(a)
    hdu.header["ROWORDER"] = "BOTTOM-UP" if bottom_up else "TOP-DOWN"
    hdu.writeto(path)


def _seq(path, matrices, selected=None):
    selected = selected or [True] * len(matrices)
    lines = [f"S 'light_' 1 {len(matrices)} {sum(selected)} 5 0 6 0 0 0", "L 3"]
    lines += [f"I {i} {int(take)}" for i, take in enumerate(selected, 1)]
    lines += ["R1 4 4 0.7 0 0.02 20 H " + " ".join(str(v) for v in np.ravel(h)) for h in matrices]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _source_oracle(mask, matrix, model=None):
    # Explicit projective coordinate formula, followed by the u->p polynomial.
    inv = np.linalg.inv(matrix)
    height, width = mask.shape
    out = np.zeros_like(mask)
    for y in range(height):
        for x in range(width):
            okay = True
            for dx, dy in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
                px, py = x + dx, y + dy
                if not (0 <= px < width and 0 <= py < height):
                    okay = False
                    break
                den = inv[2, 0] * px + inv[2, 1] * py + inv[2, 2]
                if den == 0:
                    okay = False
                    break
                sx = (inv[0, 0] * px + inv[0, 1] * py + inv[0, 2]) / den
                sy = (inv[1, 0] * px + inv[1, 1] * py + inv[1, 2]) / den
                if model is not None:
                    ux, uy = (sx - model.cx) / model.magnification, (sy - model.cy) / model.magnification
                    r2 = (ux * ux + uy * uy) / model.scale**2
                    factor = 1 + model.k1 * r2 + model.k2 * r2 * r2
                    sx, sy = model.cx + ux * factor, model.cy + uy * factor
                ix, iy = math.floor(sx), math.floor(sy)
                if ix < 4 or iy < 4 or ix + 5 >= width or iy + 5 >= height:
                    okay = False
                    break
                if not all(mask[yy, xx] for yy in range(iy - 4, iy + 6) for xx in range(ix - 4, ix + 6)):
                    okay = False
                    break
            out[y, x] = okay
    return out


@pytest.mark.parametrize("translation", [(0, 0), (.25, .75), (-.25, -.5), (2, -1)])
@pytest.mark.parametrize("with_model", [False, True])
def test_registered_mask_against_scalar_five_cross_footprint(translation, with_model):
    mask = np.ones((28, 36), dtype=bool)
    mask[11, 14] = False
    mask[19:, 29:] = False
    h = np.array([[1., .004, translation[0]], [-.002, 1., translation[1]], [.00001, .00002, 1.]])
    model = RadialDistortion(.04, -.002, 17.5, 13.5, 30, .98, "u_to_p") if with_model else None
    expected = _source_oracle(mask, h, model)
    mapping = None
    if with_model:
        # An explicit synthetic mapping tests nonlinear support propagation.
        # Actual WCS dispatch has its own independent tests and real-engine run.
        def mapping(points):
            homogeneous = np.column_stack((points, np.ones(len(points)))) @ np.linalg.inv(h).T
            return model.inverse(homogeneous[:, :2] / homogeneous[:, 2:])
    got = rs.registered_mask(mask, h, model, mapping=mapping, rows=5)
    np.testing.assert_array_equal(got, expected)
    np.testing.assert_array_equal(rs.registered_mask(mask, h, model, mapping=mapping, rows=100), expected)


def _keys(t):
    t = abs(t)
    if t < 1:
        return 1.5 * t**3 - 2.5 * t**2 + 1
    if t < 2:
        return -.5 * t**3 + 2.5 * t**2 - 4 * t + 2
    return 0.


class Shift:
    def __init__(self, dx, dy):
        self.dx, self.dy = dx, dy

    def apply(self, points):
        return points + [self.dx, self.dy]


def _native_oracle(mask, dx, dy):
    height, width = mask.shape
    result = np.zeros_like(mask)
    for y in range(height):
        for x in range(width):
            sx, sy = x + dx, y + dy
            if not (-.5 <= sx < width - .5 and -.5 <= sy < height - .5):
                continue
            ix, iy = math.floor(sx), math.floor(sy)
            taps = []
            for oy in (-1, 0, 1, 2):
                for ox in (-1, 0, 1, 2):
                    if _keys(sx - ix - ox) * _keys(sy - iy - oy) != 0:
                        taps.append(mask[min(height - 1, max(0, iy + oy)), min(width - 1, max(0, ix + ox))])
            result[y, x] = all(taps)
    return result


@pytest.mark.parametrize("shift", [(0, 0), (.25, 0), (.25, -.25), (-.49, .49), (1, -2)])
def test_native_stage_includes_negative_taps_but_not_exact_zero_taps(shift):
    valid = np.ones((12, 16), dtype=bool)
    valid[5, 5] = False
    valid[8:, 10] = False
    actual = rs.native_mask(valid, Shift(*shift), rows=3)
    np.testing.assert_array_equal(actual, _native_oracle(valid, *shift))
    if shift == (0, 0):
        np.testing.assert_array_equal(actual, valid)
        assert actual[5, 4]  # false neighbor has exact zero cubic weight


def test_stack_combines_complementary_frames_before_propagating_native_support():
    left = np.zeros((15, 20), dtype=bool)
    left[:, :10] = True
    right = ~left
    union_first = rs.native_mask(left | right, Shift(.25, .25))
    per_frame_union = rs.native_mask(left, Shift(.25, .25)) | rs.native_mask(right, Shift(.25, .25))
    assert union_first[7, 9]
    assert not per_frame_union[7, 9]


@pytest.mark.parametrize("bottom_up", [False, True])
def test_finish_stack_retains_eligibility_but_marks_zeros_and_nonfinite_unavailable(tmp_path, bottom_up):
    counts = np.ones((12, 18), dtype=np.uint16)
    counts[1, 2] = 0
    desc = rs.save_array(tmp_path / rs.COUNTS, counts)
    record = {"version": rs.SUPPORT_VERSION, "mode": rs.MODE, "state": "exports_filtered", "eligible": desc}
    atomic_json(tmp_path / rs.RECORD, record)
    data = np.full((3, 12, 18), .25, dtype=np.float32)
    data[1, 4, 6] = 0  # even one absent channel rejects the whole RGB pixel
    data[:, 5, 7] = -0.0
    data[2, 6, 8] = np.nan
    data[0, 7, 9] = np.inf
    data[:, 9, 3] = -.1  # retained finite nonzero, should not be called a measured effective exposure
    path = tmp_path / "sky_master.fit"
    _write_rgb(path, data, bottom_up=bottom_up)
    result = rs.finish_stack(tmp_path, path, None)
    valid = rs.load_array(tmp_path, result["stack_valid"])
    expected = np.ones((12, 18), dtype=bool)
    for y, x in ((1, 2), (4, 6), (5, 7), (6, 8), (7, 9)):
        expected[y, x] = False
    np.testing.assert_array_equal(valid, expected)
    np.testing.assert_array_equal(rs.load_array(tmp_path, desc), counts)
    np.testing.assert_array_equal(rs.load_array(tmp_path, result["native_valid"]), expected)
    assert result["ambiguous_zero_or_nonfinite_pixels"] == 4
    assert result["stack_supported_pixels"] == expected.sum()
    assert result["effective_exposure"].startswith("not_measured")


class Source:
    def __init__(self, values):
        self.values = values

    def for_frame(self, stem):
        return self.values[stem]


def test_export_filter_counts_selection_and_hashes_are_bound_to_filtered_pixels(tmp_path):
    identity = np.eye(3)
    _seq(tmp_path / rs.EXPORT_SEQUENCE, [identity, identity, identity], [True, False, True])
    mask1 = np.ones((28, 36), dtype=bool)
    mask1[12:, :9] = False
    mask3 = np.ones_like(mask1)
    mask3[:17, 28:] = False
    data = np.full(mask1.shape, .31, dtype=np.float32)
    data[14, 15] = 0.0  # eligibility is geometric, not a nonzero count
    values = {"light_00001": mask1, "light_00003": mask3}
    before = {}
    for stem in values:
        path = tmp_path / f"r_{stem}.fit"
        _write_rgb(path, data)
        before[stem] = rs.sha256(path)
    result = rs.prepare_exports(tmp_path, Source(values), None)
    assert [f["stem"] for f in result["frames"]] == ["light_00001", "light_00003"]
    expected_count = np.zeros(mask1.shape, dtype=np.uint16)
    for frame in result["frames"]:
        stem = frame["stem"]
        valid = _source_oracle(values[stem], identity)
        expected_count += valid
        expected = np.where(valid, data, 0)
        path = tmp_path / f"r_{stem}.fit"
        np.testing.assert_array_equal(fits_crop.crop_top_down(path, 1, 0, 36, 0, 28), expected)
        assert frame["source_valid_sha256"] == hashlib.sha256(values[stem].tobytes()).hexdigest()
        assert frame["export_before_sha256"] == before[stem]
        assert frame["export_after_sha256"] == rs.sha256(path)
    np.testing.assert_array_equal(rs.load_array(tmp_path, result["eligible"]), expected_count)
    assert result["counts_semantics"].endswith("not_effective_exposure")


def test_export_refuses_stale_extra_frame_and_does_not_publish_success(tmp_path):
    _seq(tmp_path / rs.EXPORT_SEQUENCE, [np.eye(3), np.eye(3)], [True, False])
    for i in (1, 2):
        _write_rgb(tmp_path / f"r_light_{i:05d}.fit", np.ones((20, 20)))
    with pytest.raises(ValueError, match="不一致"):
        rs.prepare_exports(tmp_path, Source({"light_00001": np.ones((20, 20), dtype=bool)}), None)
    assert not (tmp_path / rs.RECORD).exists()


@pytest.mark.parametrize("descriptor_change", ["hash", "shape", "dtype", "path"])
def test_sidecar_descriptor_checks(tmp_path, descriptor_change):
    desc = rs.save_array(tmp_path / rs.NATIVE_VALID, np.ones((10, 11), dtype=bool))
    if descriptor_change == "hash":
        desc["sha256"] = "0" * 64
    elif descriptor_change == "shape":
        desc["shape"] = [11, 10]
    elif descriptor_change == "dtype":
        desc["dtype"] = "float32"
    else:
        desc["file"] = "../" + rs.NATIVE_VALID
    with pytest.raises(ValueError):
        rs.load_array(tmp_path, desc)


def _ready_record(tmp_path):
    mask = np.ones((10, 11), dtype=bool)
    _write_rgb(tmp_path / "sky_master.fit", np.ones(mask.shape))
    desc = rs.save_array(tmp_path / rs.NATIVE_VALID, mask)
    record = {"version": rs.SUPPORT_VERSION, "mode": rs.MODE, "state": "complete", "native_valid": desc,
              "native_master_sha256": rs.sha256(tmp_path / "sky_master.fit")}
    atomic_json(tmp_path / rs.RECORD, record)
    return mask, record


@pytest.mark.parametrize("kind", ["master", "sidecar", "state", "mode", "version", "shape"])
def test_native_read_refuses_changed_or_incomplete_provenance(tmp_path, kind):
    mask, record = _ready_record(tmp_path)
    if kind == "master":
        with (tmp_path / "sky_master.fit").open("ab") as f:
            f.write(b"changed")
    elif kind == "sidecar":
        with (tmp_path / rs.NATIVE_VALID).open("ab") as f:
            f.write(b"changed")
    elif kind in ("state", "mode", "version"):
        record[kind] = "other"
        atomic_json(tmp_path / rs.RECORD, record)
    shape = (11, 10) if kind == "shape" else mask.shape
    with pytest.raises(ValueError):
        rs.read_native(tmp_path, shape)


def test_cancellation_never_publishes_complete_export_record(tmp_path):
    _seq(tmp_path / rs.EXPORT_SEQUENCE, [np.eye(3)])
    _write_rgb(tmp_path / "r_light_00001.fit", np.ones((20, 20)))
    class Cancelled(Exception):
        pass
    def check():
        raise Cancelled()
    with pytest.raises(Cancelled):
        rs.prepare_exports(tmp_path, Source({"light_00001": np.ones((20, 20), dtype=bool)}), None, check=check)
    assert not (tmp_path / rs.RECORD).exists()


def test_ideal_radial_model_without_explicit_export_mapping_is_refused():
    with pytest.raises(ValueError, match="actual-WCS"):
        rs.registered_mask(np.ones((20, 20), dtype=bool), np.eye(3),
                           RadialDistortion(.04, 0, 10, 10, 20, 1, "u_to_p"))


@pytest.mark.parametrize("rows", [0, -1, True, np.int64(4), 2.5])
def test_native_invalid_chunk_sizes_fail_instead_of_silent_empty_results(rows):
    with pytest.raises(ValueError):
        rs.native_mask(np.ones((10, 10), dtype=bool), Shift(.25, .25), rows=rows)


def test_core_split_export_snapshots_before_engine_rewrites_and_invalidates_stats(tmp_path, monkeypatch):
    from starstack import core
    work = tmp_path / "work"
    work.mkdir()
    sequence = work / core.SEQUENCE_FILE
    sequence.write_text("before export", encoding="utf-8")
    (work / "light_00001.fit").write_bytes(b"placeholder")
    events = []
    def run_script(exe, directory, stage, lines, check):
        events.append((stage, list(lines)))
        if stage == "02_export":
            assert (work / rs.EXPORT_SEQUENCE).read_text() == "before export"
            sequence.write_text("rounded after export", encoding="utf-8")
    def prepare_exports(*args, **kwargs):
        assert (work / rs.EXPORT_SEQUENCE).read_text() == "before export"
        assert sequence.read_text() == "rounded after export"
        events.append(("filter", None))
    monkeypatch.setattr(core, "run_script", run_script)
    monkeypatch.setattr(core, "_validate_frame", lambda *args: (3, 20, 30))
    monkeypatch.setattr(core, "_validity_source", lambda *args: object())
    monkeypatch.setattr(core, "_grid_model", lambda *args: None)
    monkeypatch.setattr(core, "_update", lambda *args, **kwargs: None)
    monkeypatch.setattr(rs, "prepare_exports", prepare_exports)
    core._supported_stack("siril", tmp_path, work, {}, {}, False, {"dilation_px": 3}, lambda: None)
    assert [event[0] for event in events] == ["02_export", "filter", "02_stack"]
    assert events[0][1] == ["cd work", rs.EXPORT_COMMAND]
    assert events[-1][1] == ["cd work", "seqclean r_light -stat", core.STACK_COMMAND]


def test_core_strict_compose_uses_exact_full_resolution_support_and_bottom_up_weight(tmp_path, monkeypatch):
    from PIL import Image
    from starstack import core, imaging
    work = tmp_path / "work"
    work.mkdir()
    h, w = 18, 23
    mask = np.ones((h, w), dtype=np.float32)
    mask[-3:] = 0
    mask[-4] = .4
    mask[1, 2] = .2
    Image.fromarray(np.rint(mask * 255).astype(np.uint8)).save(tmp_path / "sky_mask.png")
    supported = np.ones((h, w), dtype=bool)
    supported[4, 5] = False  # not aligned to the old 4-pixel sample grid
    supported[7, 9:14] = False
    _write_rgb(work / "sky_master.fit", np.full((h, w), .2))
    _write_rgb(work / "foreground_master.fit", np.full((h, w), .4))
    Image.new("L", (6, 5)).save(work / "sky_coverage.png")
    desc = rs.save_array(work / rs.NATIVE_VALID, supported)
    record = {"version": rs.SUPPORT_VERSION, "mode": rs.MODE, "state": "complete", "native_valid": desc,
              "native_master_sha256": rs.sha256(work / "sky_master.fit"),
              "mask_sha256": rs.sha256(tmp_path / "sky_mask.png"), "counts_semantics": "eligibility_not_exposure",
              "ambiguous_zero_or_nonfinite_pixels": 0}
    atomic_json(work / rs.RECORD, record)
    class MaskSource:
        pass
    source = MaskSource()
    source.mask = mask
    monkeypatch.setattr(core, "_validity_source", lambda *a: source)
    job = {"mask": {"path": str(tmp_path / "sky_mask.png")}, "support_mode": rs.MODE}
    result = core._compose_sky(tmp_path, job, work, None, False, {"dilation_px": 3},
                               imaging.read_rgb(work / "sky_master.fit"), [], {"NAXIS2": h, "NAXIS1": w}, lambda: None)
    expected_weight = mask * supported
    np.testing.assert_array_equal(result["weight"], expected_weight)
    expected = .2 * expected_weight[::-1] + .4 * (1 - expected_weight[::-1])
    np.testing.assert_allclose(result["output"], np.repeat(expected[..., None], 3, axis=-1), rtol=0, atol=3e-8)
    assert result["weight"][4, 5] == 0
    assert result["weight"][4, 4] == 1


def test_core_write_coverage_strict_real_body_publishes_boolean_semantics(tmp_path, monkeypatch):
    """Exercise the real branch that failed late in the first 48-frame run."""
    from PIL import Image
    from starstack import core
    work = tmp_path / "work"
    work.mkdir()
    supported, record = _ready_record(work)
    supported[4, 8] = False       # visible in the diagnostic 4-pixel sampling
    supported[3, 5] = False       # retained by full-resolution sidecar only
    record["native_valid"] = rs.save_array(work / rs.NATIVE_VALID, supported)
    record["frames"] = [{"stem": f"light_{i:05d}"} for i in range(1, 4)]
    atomic_json(work / rs.RECORD, record)
    class MaskSource:
        pass
    source = MaskSource()
    source.mask = np.ones_like(supported, dtype=np.float32)
    monkeypatch.setattr(core, "_validity_source", lambda *a: source)
    job = {"support_mode": rs.MODE}
    returned = core._write_coverage(work, tmp_path, None, False, {"dilation_px": 3}, lambda: None, job=job)
    assert returned == work / "sky_coverage.png"
    image = np.asarray(Image.open(returned))
    np.testing.assert_array_equal(image, supported[::core.SKY_COVERAGE_STEP, ::core.SKY_COVERAGE_STEP].astype(np.uint16))
    assert set(np.unique(image)) == {0, 1}
    summary = json.loads((work / "sky_coverage.json").read_text(encoding="utf-8"))
    assert summary["semantics"] == "native_full_dependency_support_boolean_not_frame_count"
    assert summary["interpolation_support"] == rs.MODE
    assert summary["input_frames"] == 3
    assert summary["frames"] == 1  # Boolean support is never styled as three effective exposures.
    assert summary["counts_file"] == rs.COUNTS
    assert summary["counts_grid"] == "registered_before_native_warp"
    full, _ = rs.read_native(work, supported.shape)
    np.testing.assert_array_equal(full, supported)
    assert not full[3, 5]
