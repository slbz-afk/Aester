"""Pure numeric contracts and controlled Siril I/O; no live engine claims."""
import hashlib
import json

import numpy as np
import pytest
from astropy.io import fits

from starstack import experimental_rcd as rcd


PATTERN = [[0, 1], [3, 2]]


@pytest.mark.parametrize("pattern", [[[0, 1], [3, 2]], [[1, 0], [2, 3]],
                                     [[2, 1], [3, 0]], [[1, 2], [0, 3]]])
@pytest.mark.parametrize("shape", [(2, 2), (7, 8), (8, 7)])
def test_even_reflection_preserves_cfa_phase_and_fixed_extrema(pattern, shape):
    p = np.where(np.asarray(pattern) == 3, 1, pattern)
    rows, cols = np.indices(shape)
    cfa = np.array([-100., 30., 250.], np.float32)[p[rows % 2, cols % 2]]
    original = cfa.copy()
    encoded, plan = rcd.prepare_fixed_range(cfa, pattern, lower_dn=-200, upper_dn=300)
    assert encoded.dtype == np.float32
    assert encoded.shape == (shape[0]+64, shape[1]+64)
    assert encoded.min() == 0 and encoded.max() == 1
    assert encoded[0, 0] == 0 and encoded[0, 1] == 1
    # Reflection maps indices without changing parity, including odd dimensions.
    yy, xx = np.indices(encoded.shape)
    expected = np.array([.2, .46, .9], np.float32)[p[yy % 2, xx % 2]]
    expected[0, :2] = [0, 1]
    np.testing.assert_array_equal(encoded, expected)
    np.testing.assert_array_equal(cfa, original)
    assert plan["pattern_rgb"] == p.tolist()
    assert plan["per_frame_range_estimation"] is False


def test_interval_is_fixed_for_constant_frames_and_unrelated_extrema():
    a = np.full((64, 64), -3., np.float32)
    b = a.copy()
    b[3, 3] = 19.
    x, first = rcd.prepare_fixed_range(a, PATTERN, lower_dn=-20, upper_dn=20)
    y, second = rcd.prepare_fixed_range(b, PATTERN, lower_dn=-20, upper_dn=20)
    np.testing.assert_array_equal(x[64:80, 64:80], y[64:80, 64:80])
    assert first["lower_dn"] == second["lower_dn"] == -20
    assert first["upper_dn"] == second["upper_dn"] == 20
    assert first["maximum_encoding_roundtrip_error_DN"] < 1e-6


@pytest.mark.parametrize("cfa,pattern,bounds", [
    (np.zeros((1, 10), np.float32), PATTERN, (-1, 1)),
    (np.zeros((2, 2), np.float64), PATTERN, (-1, 1)),
    (np.full((2, 2), np.nan, np.float32), PATTERN, (-1, 1)),
    (np.full((2, 2), np.inf, np.float32), PATTERN, (-1, 1)),
    (np.full((2, 2), 2, np.float32), PATTERN, (-1, 1)),
    (np.zeros((2, 2), np.float32), [[0, True], [1, 2]], (-1, 1)),
    (np.zeros((2, 2), np.float32), [[0, 1], [2, 1]], (-1, 1)),
    (np.zeros((2, 2), np.float32), PATTERN, (True, 10)),
    (np.zeros((2, 2), np.float32), PATTERN, ("-1", 1)),
    (np.zeros((2, 2), np.float32), PATTERN, (1, 1)),
    (np.zeros((2, 2), np.float32), PATTERN, (10, 1)),
    (np.zeros((2, 2), np.float32), PATTERN, (-np.inf, 1)),
    (np.zeros((2, 2), np.float32), PATTERN, (-1e308, 1e308)),
])
def test_invalid_input_never_guesses_a_range(cfa, pattern, bounds):
    with pytest.raises(ValueError):
        rcd.prepare_fixed_range(cfa, pattern, lower_dn=bounds[0], upper_dn=bounds[1])


def test_sample_restoration_keeps_negative_zero_and_records_error():
    cfa = np.array([[-0., 1.25, -2.5], [3.75, -4., .5]], np.float32)
    padded, _ = rcd.prepare_fixed_range(cfa, PATTERN, lower_dn=-5, upper_dn=5)
    encoded_rgb = np.repeat(padded[..., None], 3, axis=2)
    encoded_rgb[32:-32, 32:-32] += np.float32(.01)
    untouched = encoded_rgb.copy()
    result, report = rcd.unpack_fixed_range(encoded_rgb, cfa, PATTERN, lower_dn=-5, upper_dn=5)
    p = np.where(np.asarray(PATTERN) == 3, 1, PATTERN)
    for y in range(2):
        for x in range(2):
            np.testing.assert_array_equal(result[y::2, x::2, p[y, x]].view(np.uint32),
                                          cfa[y::2, x::2].view(np.uint32))
    assert result.shape == (*cfa.shape, 3)
    assert report["measured_sites_before_restoration"]["maximum_absolute_error_DN"] == pytest.approx(.1, abs=1e-6)
    assert report["measured_sites_before_restoration"]["bitwise_different_count"] == cfa.size
    assert report["measured_sites_restored_bitwise"] is True
    np.testing.assert_array_equal(encoded_rgb, untouched)
    assert not np.shares_memory(result, cfa)


def test_finite_reconstruction_overshoot_survives_and_limits_are_distinguished():
    cfa = np.zeros((4, 4), np.float32)
    rgb = np.full((68, 68, 3), .5, np.float32)
    rgb[32, 32, 1:] = [-.25, 1.25]  # G/B reconstructed at measured R.
    rgb[32, 33, [0, 2]] = [0, 1]  # R/B reconstructed at measured G.
    result, report = rcd.unpack_fixed_range(rgb, cfa, PATTERN, lower_dn=-1, upper_dn=1)
    assert result[0, 0, 1] == -1.5 and result[0, 0, 2] == 1.5
    assert result[0, 1, 0] == -1 and result[0, 1, 2] == 1
    counts = report["reconstructed_limits_before_restoration"]
    assert counts["count"] == 32
    assert counts["below_lower_dn"] == counts["above_upper_dn"] == 1
    assert counts["at_lower_dn"] == counts["at_upper_dn"] == 1
    assert counts["at_or_below_lower_dn"] == counts["at_or_above_upper_dn"] == 2
    assert counts["output_clipping_applied"] is False


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_engine_output_cannot_be_hidden_by_measured_restoration(value):
    cfa = np.zeros((4, 4), np.float32)
    rgb = np.full((68, 68, 3), .5, np.float32)
    rgb[32, 32, 0] = value  # A measured site, which would later be overwritten.
    with pytest.raises(ValueError):
        rcd.unpack_fixed_range(rgb, cfa, PATTERN, lower_dn=-1, upper_dn=1)


def test_inverse_float32_overflow_is_rejected_before_restoration():
    cfa = np.zeros((4, 4), np.float32)
    rgb = np.full((68, 68, 3), .5, np.float32)
    rgb[32, 32, 0] = np.finfo(np.float32).max
    with pytest.raises(ValueError, match="float32"):
        rcd.unpack_fixed_range(rgb, cfa, PATTERN, lower_dn=-1e10, upper_dn=1e10)


@pytest.fixture
def fake_engine(tmp_path, monkeypatch):
    exe = tmp_path/"controlled-engine.exe"
    exe.write_bytes(b"not an executable; only a mocked engine identity")
    monkeypatch.setattr(rcd.siril, "doctor", lambda path=None: {
        "available": True, "executable": str(exe), "version": "controlled-test-double"})

    def run(executable, directory, name, commands, check):
        assert "set debayer.interpolation=8" in commands
        (directory/f"{name}.ssf").write_text("\n".join(commands), encoding="utf-8")
        (directory/f"{name}.log").write_text("controlled test double\n", encoding="utf-8")
        (directory/"siril.ini").write_text("", encoding="utf-8")
        check()
        encoded = fits.getdata(directory/"stage/input.fit").astype(np.float32)
        rgb = np.repeat(encoded[None], 3, axis=0)[:, ::-1]  # Explicit BOTTOM-UP wire format.
        (directory/"rgb").mkdir()
        fits.writeto(directory/"rgb/rgb_00001.fit", rgb, fits.Header({"ROWORDER": "BOTTOM-UP"}))
        check()

    monkeypatch.setattr(rcd.siril, "run_script", run)
    return run


def test_adapter_retains_unique_hashed_evidence_and_normalizes_fits_orientation(tmp_path, fake_engine):
    cfa = np.arange(35, dtype=np.float32).reshape(5, 7)-17
    before = cfa.tobytes()
    result, record = rcd.rcd_fixed_range(cfa, PATTERN, tmp_path/"owned", lower_dn=-20, upper_dn=20)
    np.testing.assert_allclose(result, np.repeat(cfa[..., None], 3, axis=2), atol=2e-6)
    assert cfa.tobytes() == before
    assert record["state"] == "succeeded" and record["input_unchanged"]
    assert record["encoded_input_unchanged"]
    from pathlib import Path
    directory = Path(record["directory"])
    assert json.loads((directory/"experiment.json").read_text(encoding="utf-8")) == record
    for name in ("siril.ini", "rcd.ssf", "rcd.log", "stage/input.fit", "rgb/rgb_00001.fit"):
        assert record["artifacts"][name]["sha256"] == hashlib.sha256((directory/name).read_bytes()).hexdigest()
    _, repeated = rcd.rcd_fixed_range(cfa, PATTERN, tmp_path/"owned", lower_dn=-20, upper_dn=20)
    assert repeated["directory"] != record["directory"]
    assert (directory/"experiment.json").is_file()


def test_zero_exit_without_rgb_product_is_a_retained_failure(tmp_path, fake_engine, monkeypatch):
    monkeypatch.setattr(rcd.siril, "run_script", lambda *args: None)
    with pytest.raises(RuntimeError, match="实际 0"):
        rcd.rcd_fixed_range(np.zeros((4, 4), np.float32), PATTERN, tmp_path/"owned", lower_dn=-1, upper_dn=1)
    record = json.loads(next((tmp_path/"owned").glob("*/experiment.json")).read_text(encoding="utf-8"))
    assert record["state"] == "failed"
    assert "stage/input.fit" in record["artifacts"]


def test_timeout_is_forwarded_through_guard_callback_and_recorded(tmp_path, fake_engine, monkeypatch):
    clock = [0.]
    monkeypatch.setattr(rcd.time, "monotonic", lambda: clock[0])

    def slow(exe, directory, name, commands, check):
        clock[0] = 2.
        check()

    monkeypatch.setattr(rcd.siril, "run_script", slow)
    with pytest.raises(TimeoutError):
        rcd.rcd_fixed_range(np.zeros((4, 4), np.float32), PATTERN, tmp_path/"owned",
                            lower_dn=-1, upper_dn=1, timeout_seconds=1)
    record = json.loads(next((tmp_path/"owned").glob("*/experiment.json")).read_text(encoding="utf-8"))
    assert record["state"] == "failed" and record["error"]["type"] == "TimeoutError"


def test_cancel_callback_preserves_cancelled_state(tmp_path, fake_engine):
    calls = [0]

    def cancelled():
        calls[0] += 1
        if calls[0] == 2:
            raise rcd.siril.Cancelled("controlled cancel")

    with pytest.raises(rcd.siril.Cancelled):
        rcd.rcd_fixed_range(np.zeros((4, 4), np.float32), PATTERN, tmp_path/"owned",
                            lower_dn=-1, upper_dn=1, cancel_check=cancelled)
    record = json.loads(next((tmp_path/"owned").glob("*/experiment.json")).read_text(encoding="utf-8"))
    assert record["state"] == "cancelled"
    assert record["artifacts"]["rcd.log"]["size_bytes"] > 0
