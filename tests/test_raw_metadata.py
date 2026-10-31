"""Metadata contracts using controlled decoder objects, not live RAW validation."""

from collections import namedtuple
import hashlib
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from starstack.raw_metadata import read_raw_metadata
from starstack import raw_metadata


class UnsupportedRaw(Exception):
    pass


class ControlledRaw:
    def __init__(self, **overrides):
        Size = namedtuple("Size", "raw_width raw_height width height left_margin top_margin flip pixel_aspect")
        self.metadata = {
            "sizes": Size(12, 10, 8, 6, 2, 2, 0, 1.0),
            "raw_pattern": np.array([[0, 1], [3, 2]], dtype=np.uint8),
            "color_desc": b"RGBG", "num_colors": np.int64(4),
            "black_level_per_channel": np.array([64, 65, 66, 67]),
            "white_level": 4095,
            "camera_white_level_per_channel": [4090, 4091, 4092, 4093],
            "camera_whitebalance": [2.0, 1.0, 1.5, 1.1],
            "daylight_whitebalance": [2.2, 1.1, 1.4, 1.2],
            "color_matrix": np.column_stack((np.eye(3), np.array([.1, -.1, .2]))),
            "rgb_xyz_matrix": np.vstack((np.eye(3), np.array([.05, .1, .2]))),
        }
        self.metadata.update(overrides)
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.closed = True

    def __getattr__(self, name):
        value = self.metadata[name]
        if isinstance(value, Exception):
            raise value
        return value

    def postprocess(self, **_):
        pytest.fail("Metadata inspection must not run color processing")


@pytest.fixture
def original(tmp_path):
    source = tmp_path / "输入 photo.CR2"
    source.write_bytes(b"controlled RAW input, never rewritten")
    return source


def install_decoder(monkeypatch, raw=None, error=None):
    calls = []

    def imread(path):
        calls.append(path)
        if error:
            raise error
        return raw

    monkeypatch.setitem(sys.modules, "rawpy", SimpleNamespace(
        __version__="0.27.1", libraw_version=(0, 21, 4), imread=imread,
        LibRawFileUnsupportedError=UnsupportedRaw,
    ))
    return calls


def test_missing_optional_rawpy_is_reported(original, monkeypatch):
    monkeypatch.setitem(sys.modules, "rawpy", None)
    result = read_raw_metadata(original)
    assert result["available"] is False
    assert result["status"] == "rawpy_unavailable"
    assert result["versions"] == {"rawpy": None, "libraw": None}
    assert all(field["value"] is None for field in result["fields"].values())
    json.dumps(result, allow_nan=False)


def test_missing_file_does_not_invoke_decoder(tmp_path, monkeypatch):
    calls = install_decoder(monkeypatch, ControlledRaw())
    result = read_raw_metadata(tmp_path / "absent.CR2")
    assert result["status"] == "input_not_found"
    assert not result["available"]
    assert calls == []


def test_broken_native_dependency_does_not_break_siril_caller(original, monkeypatch):
    def broken_import(_):
        raise OSError("decoder DLL unavailable")

    monkeypatch.setattr(raw_metadata.importlib, "import_module", broken_import)
    result = read_raw_metadata(original)
    assert not result["available"] and result["status"] == "rawpy_import_error"
    assert result["errors"][0]["type"] == "OSError"


@pytest.mark.parametrize("error,status", [(UnsupportedRaw("unsupported"), "unsupported_format"),
                                         (OSError("unreadable"), "read_error")])
def test_read_failure_is_data_not_an_exception(original, monkeypatch, error, status):
    install_decoder(monkeypatch, error=error)
    result = read_raw_metadata(original)
    assert not result["available"]
    assert result["status"] == status
    assert result["errors"]


def test_valid_metadata_keeps_four_channels_and_original_unchanged(original, monkeypatch):
    raw = ControlledRaw()
    calls = install_decoder(monkeypatch, raw)
    digest = hashlib.sha256(original.read_bytes()).digest()
    result = read_raw_metadata(original)
    assert result["available"] and result["status"] == "ok"
    assert result["source"] == "libraw_reported"
    assert result["versions"] == {"rawpy": "0.27.1", "libraw": [0, 21, 4]}
    assert not result["white_balance_applied"] and not result["color_correction_applied"]
    fields = result["fields"]
    assert fields["camera_whitebalance"]["value"] == [2.0, 1.0, 1.5, 1.1]
    assert fields["black_level_per_channel"]["value"] == [64, 65, 66, 67]
    assert fields["raw_pattern"]["value"] == [[0, 1], [3, 2]]
    assert fields["rgb_xyz_matrix"]["value"][3] == [.05, .1, .2]
    assert fields["sizes"]["value"]["left_margin"] == 2
    assert calls == [str(original)] and raw.closed
    assert hashlib.sha256(original.read_bytes()).digest() == digest
    raw.metadata["raw_pattern"][:] = 0
    assert fields["raw_pattern"]["value"] == [[0, 1], [3, 2]]
    assert json.loads(json.dumps(result, allow_nan=False)) == result


@pytest.mark.parametrize("field,value,issue", [
    ("camera_whitebalance", [2, 1, 1.2], "unexpected_shape"),
    ("camera_whitebalance", [2, 1, 1.2, 0], "non_positive_coefficient"),
    ("daylight_whitebalance", [2, 1, float("inf"), 1], "non_finite"),
    ("black_level_per_channel", [64, -1, 64, 64], "negative_black_level"),
    ("white_level", 32, "white_not_above_black"),
    ("camera_white_level_per_channel", [4095, 65, 4095, 4095], "white_not_above_black"),
    ("rgb_xyz_matrix", np.zeros((4, 3)), "rank_deficient_matrix"),
    ("color_matrix", np.eye(3), "unexpected_shape"),
    ("raw_pattern", [[0, 1], [4, 2]], "invalid_color_index"),
    ("num_colors", 5, "invalid_color_count"),
])
def test_invalid_field_is_preserved_and_cannot_be_used(original, monkeypatch, field, value, issue):
    raw = ControlledRaw(**{field: value})
    install_decoder(monkeypatch, raw)
    result = read_raw_metadata(original)
    assert result["available"] and result["status"] == "partial"
    report = result["fields"][field]
    assert not report["valid"] and report["status"] == "invalid"
    assert issue in report["issues"]
    assert raw.closed
    json.dumps(result, allow_nan=False)


def test_nonfinite_positions_are_null_in_strict_json(original, monkeypatch):
    raw = ControlledRaw(camera_whitebalance=[float("nan"), 1, float("inf"), 1])
    install_decoder(monkeypatch, raw)
    field = read_raw_metadata(original)["fields"]["camera_whitebalance"]
    assert field["value"] == [None, 1.0, None, 1.0]
    assert field["finite"] is False and field["shape_valid"] is True
    json.dumps(field, allow_nan=False)


def test_optional_missing_value_and_property_error_do_not_discard_other_fields(original, monkeypatch):
    raw = ControlledRaw(camera_white_level_per_channel=None, color_matrix=RuntimeError("no matrix"))
    install_decoder(monkeypatch, raw)
    result = read_raw_metadata(original)
    assert result["available"] and result["status"] == "partial"
    assert result["fields"]["camera_white_level_per_channel"]["status"] == "missing"
    assert result["fields"]["camera_white_level_per_channel"]["value"] is None
    assert result["fields"]["color_matrix"]["status"] == "error"
    assert result["fields"]["camera_whitebalance"]["valid"]
    assert raw.closed


def test_bad_sizes_and_nonmatching_color_description_are_invalid(original, monkeypatch):
    raw = ControlledRaw(sizes={"raw_width": 8, "raw_height": 6, "width": 8,
                               "height": 6, "left_margin": 2, "top_margin": 0},
                        color_desc=b"RGB")
    install_decoder(monkeypatch, raw)
    result = read_raw_metadata(original)
    assert "width_outside_raw_area" in result["fields"]["sizes"]["issues"]
    assert "color_index_not_in_description" in result["fields"]["raw_pattern"]["issues"]


def test_zero_unused_matrix_row_is_retained_without_requiring_four_color_rank(original, monkeypatch):
    raw = ControlledRaw(num_colors=3, rgb_xyz_matrix=np.vstack((np.eye(3), np.zeros(3))))
    install_decoder(monkeypatch, raw)
    field = read_raw_metadata(original)["fields"]["rgb_xyz_matrix"]
    assert field["valid"] and field["value"][3] == [0.0, 0.0, 0.0]


@pytest.mark.parametrize("aspect", [0, float("nan"), "1.0"])
def test_invalid_pixel_aspect_does_not_claim_valid_reference_geometry(original, monkeypatch, aspect):
    raw = ControlledRaw(sizes={"raw_width": 12, "raw_height": 10, "width": 8,
                               "height": 6, "pixel_aspect": aspect})
    install_decoder(monkeypatch, raw)
    field = read_raw_metadata(original)["fields"]["sizes"]
    assert not field["valid"] and "invalid_pixel_aspect" in field["issues"]
    json.dumps(field, allow_nan=False)


def test_three_index_cfa_accepts_unused_zero_coefficients_without_filling(original, monkeypatch):
    raw = ControlledRaw(
        num_colors=3, raw_pattern=np.array([[0, 1], [1, 2]]),
        camera_whitebalance=[2., 1., 1.5, 0.],
        daylight_whitebalance=[2.2, 1., 1.4, 0.],
        camera_white_level_per_channel=[4090, 4091, 4092, 0],
        black_level_per_channel=[64, 65, 66, 0],
    )
    install_decoder(monkeypatch, raw)
    result = read_raw_metadata(original)
    assert result["status"] == "ok"
    assert result["channel_usage"]["used_slots"] == [0, 1, 2]
    assert result["channel_usage"]["unused_slots"] == [3]
    for name in ("camera_whitebalance", "daylight_whitebalance", "camera_white_level_per_channel"):
        field = result["fields"][name]
        assert field["valid"] and field["value"][3] == 0
        applicability = field["applicability"]
        assert applicability["status"] == "applicable"
        assert applicability["slot_checks"][3]["usage"] == "unused"
        assert applicability["slot_checks"][3]["range_valid"] is True
    semantics = result["fields"]["camera_whitebalance"]["white_balance_semantics"]
    assert semantics["status"] == "direct_reported_slots"
    assert semantics["cfa_to_coefficient_slots"] == [0, 1, 2, None]
    assert not semantics["mapping_applied"]
    assert raw.metadata["camera_whitebalance"] == [2., 1., 1.5, 0.]
    assert raw.closed
    json.dumps(result, allow_nan=False)


def test_used_channel_indices_need_not_be_the_first_num_colors_slots(original, monkeypatch):
    # Third color is B in slot 3; the equally named slot 2 is absent from CFA.
    raw = ControlledRaw(num_colors=3, color_desc=b"RGBB", raw_pattern=[[0, 1], [1, 3]],
                        camera_whitebalance=[2., 1., 0., 1.5])
    install_decoder(monkeypatch, raw)
    result = read_raw_metadata(original)
    assert result["channel_usage"]["used_slots"] == [0, 1, 3]
    assert result["channel_usage"]["unused_slots"] == [2]
    field = result["fields"]["camera_whitebalance"]
    assert field["valid"] and field["value"] == [2., 1., 0., 1.5]
    assert field["applicability"]["slot_checks"][2]["usage"] == "unused"
    assert field["applicability"]["slot_checks"][3]["usage"] == "used"


@pytest.mark.parametrize("field_name", ["camera_whitebalance", "daylight_whitebalance"])
def test_three_color_four_position_bayer_requires_explicit_green_mapping(original, monkeypatch, field_name):
    # Real Z8 DNG-shaped LibRaw output: four CFA pixel indices, three WB values.
    # This controlled test does not claim to inspect DNG's AsShotNeutral tag.
    reported = [1.697265625, 1., 1.550781846, 0.]
    raw = ControlledRaw(num_colors=3, **{field_name: reported})
    install_decoder(monkeypatch, raw)
    result = read_raw_metadata(original)
    assert result["channel_usage"]["used_slots"] == [0, 1, 2, 3]
    assert result["channel_usage"]["unused_slots"] == []
    field = result["fields"][field_name]
    assert field["value"] == reported
    assert field["valid"] is False  # Raw four-slot array cannot be applied directly.
    assert field["applicability"]["status"] == "not_applicable"
    assert field["applicability"]["slot_checks"][3]["usage"] == "used"
    semantics = field["white_balance_semantics"]
    assert semantics["status"] == "requires_explicit_green_mapping"
    assert semantics["message"] == "RGB白平衡可用，第二绿色需显式映射；尚未应用"
    assert semantics["rgb_coefficients"] == reported[:3]
    assert semantics["rgb_coefficients_valid"] is True
    assert semantics["cfa_to_coefficient_slots"] == [0, 1, 2, 1]
    assert "libraw" in semantics["mapping_basis"]
    assert "/0.22.1/" in semantics["mapping_reference"]
    assert semantics["mapping_applied"] is False
    assert semantics["file_white_balance_tags_verified"] is False
    assert raw.metadata[field_name] == reported
    assert not result["white_balance_applied"] and not result["color_correction_applied"]
    assert json.loads(json.dumps(result, allow_nan=False)) == result


def test_four_positive_reported_coefficients_keep_distinct_green_values(original, monkeypatch):
    raw = ControlledRaw(num_colors=3, camera_whitebalance=[2., 1., 1.5, 1.25])
    install_decoder(monkeypatch, raw)
    field = read_raw_metadata(original)["fields"]["camera_whitebalance"]
    assert field["valid"]
    assert field["value"][1] != field["value"][3]
    assert field["white_balance_semantics"]["status"] == "direct_reported_slots"
    assert field["white_balance_semantics"]["cfa_to_coefficient_slots"] == [0, 1, 2, 3]


@pytest.mark.parametrize("pattern", [[[0, 1], [1, 2]], [[0, 1], [3, 2]]])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.])
def test_invalid_fourth_slot_is_not_hidden_by_three_color_semantics(original, monkeypatch, pattern, bad):
    raw = ControlledRaw(num_colors=3, raw_pattern=pattern, camera_whitebalance=[2., 1., 1.5, bad])
    install_decoder(monkeypatch, raw)
    field = read_raw_metadata(original)["fields"]["camera_whitebalance"]
    assert not field["valid"]
    assert field["applicability"]["status"] == "not_applicable"
    assert field["white_balance_semantics"]["status"] == "not_applicable"
    assert field["white_balance_semantics"]["rgb_coefficients_valid"] is not True
    assert field["value"][3] == (bad if bad == -1 else None)
    json.dumps(field, allow_nan=False)


@pytest.mark.parametrize("overrides", [
    {"num_colors": 4},
    {"camera_whitebalance": [2., 0., 1.5, 0.]},
    {"camera_whitebalance": [0., 1., 1.5, 0.]},
    {"camera_whitebalance": [2., 1., 0., 0.]},
    {"camera_whitebalance": [2., 1., 1.5]},
    {"color_desc": b"RGBE", "num_colors": 4},
])
def test_no_green_mapping_for_missing_rgb_or_four_color_data(original, monkeypatch, overrides):
    metadata = {"num_colors": 3, "camera_whitebalance": [2., 1., 1.5, 0.]}
    metadata.update(overrides)
    install_decoder(monkeypatch, ControlledRaw(**metadata))
    field = read_raw_metadata(original)["fields"]["camera_whitebalance"]
    assert not field["valid"]
    assert field["white_balance_semantics"]["status"] == "not_applicable"
    assert field["white_balance_semantics"]["cfa_to_coefficient_slots"] is None


@pytest.mark.parametrize("overrides", [
    {"raw_pattern": None}, {"raw_pattern": [[0, 1], [4, 2]]},
    {"color_desc": None}, {"num_colors": None},
    {"num_colors": 2}, {"color_desc": b"RGBE"},
])
def test_unknown_or_conflicting_channel_metadata_does_not_relax_zero_wb(original, monkeypatch, overrides):
    metadata = {"num_colors": 3, "camera_whitebalance": [2., 1., 1.5, 0.]}
    metadata.update(overrides)
    install_decoder(monkeypatch, ControlledRaw(**metadata))
    result = read_raw_metadata(original)
    assert result["status"] == "partial"
    assert result["channel_usage"]["status"] == "unknown"
    assert result["channel_usage"]["used_slots"] is None
    field = result["fields"]["camera_whitebalance"]
    assert not field["valid"]
    assert field["applicability"]["status"] == "unknown"
    assert field["white_balance_semantics"]["status"] == "unknown"
    json.dumps(result, allow_nan=False)


def test_saturation_comparison_uses_actual_cfa_slots_and_preserves_unused_values(original, monkeypatch):
    raw = ControlledRaw(num_colors=3, raw_pattern=[[0, 1], [1, 2]],
                        black_level_per_channel=[64, 65, 66, 5000],
                        camera_white_level_per_channel=[4090, 4091, 4092, 0])
    install_decoder(monkeypatch, raw)
    result = read_raw_metadata(original)
    assert result["fields"]["black_level_per_channel"]["value"][3] == 5000
    assert result["fields"]["white_level"]["valid"]
    assert result["fields"]["camera_white_level_per_channel"]["valid"]
    assert result["fields"]["white_level"]["applicability"]["used_slots"] == [0, 1, 2]


@pytest.mark.parametrize("name,value,issue", [
    ("black_level_per_channel", [64, 65, 66, -1], "negative_black_level"),
    ("black_level_per_channel", [64, 65, 66, float("nan")], "non_finite"),
    ("camera_white_level_per_channel", [4090, 4091, 4092, float("inf")], "non_finite"),
    ("camera_white_level_per_channel", [4090, 65, 4092, 0], "white_not_above_black"),
])
def test_channel_levels_remain_conservative_for_bad_values(original, monkeypatch, name, value, issue):
    raw = ControlledRaw(num_colors=3, raw_pattern=[[0, 1], [1, 2]], **{name: value})
    install_decoder(monkeypatch, raw)
    result = read_raw_metadata(original)
    assert not result["fields"][name]["valid"]
    assert issue in result["fields"][name]["issues"]
    json.dumps(result, allow_nan=False)


def test_white_applicability_is_unknown_without_black_level(original, monkeypatch):
    install_decoder(monkeypatch, ControlledRaw(black_level_per_channel=None))
    result = read_raw_metadata(original)
    field = result["fields"]["white_level"]
    assert field["valid"]  # Existing scalar structure/numeric semantics retained.
    assert field["applicability"]["status"] == "unknown"
    assert field["applicability"]["reason"] == "black_level_comparison_unavailable"


def test_nonnumeric_slot_report_does_not_raise_or_infer_coefficients(original, monkeypatch):
    install_decoder(monkeypatch, ControlledRaw(camera_whitebalance=[2, 1, 1, 10 ** 400]))
    result = read_raw_metadata(original)
    field = result["fields"]["camera_whitebalance"]
    assert not field["valid"] and "not_numeric" in field["issues"]
    assert field["white_balance_semantics"]["status"] == "not_applicable"
    json.dumps(result, allow_nan=False)
