"""Sony α7R V adaptation, stage 2 (docs/SONY_ADAPTATION_PREREG_20260925.md C1–C4): the registry keeps α7R V at tier B
(user decision 2026-09-25) while recording a machine-checkable sub-variant; every other Sony body is unchanged and no
entry shares mutable objects; SonyRawFileType maps completely onto registry keys; colour export and the c1 acceptance
route a non-DNG reference the same way the job path does, with c1 thresholds unchanged."""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile

from starstack import color_products as cp, devices, devices_registry as reg, master_export as me

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import c1_accept_color_contract as c1  # noqa: E402

OTHER_SONY = ("ILCE-7M3", "ILCE-7M4", "ILCE-7CM2", "ILCE-7CR", "ILCE-7RM3", "ILCE-7RM3A")


def by_model():
    return {m: e for e in reg.REGISTRY for m in e["model_ids"]}


def test_sony_entries_do_not_share_mutable_objects():
    sony = [e for e in reg.REGISTRY if e["make"] == "SONY"]
    assert len(sony) == 8          # α7S III and α6400 added by the look batch (docs/CAMERA_LOOKS_AND_PARALLEL_PREREG_20260926.md)
    for i, a in enumerate(sony):
        for b in sony[i + 1:]:
            for key in ("raw_variants", "verified_frames", "evidence", "quirks"):
                assert a[key] is not b[key], key
            for name in a["raw_variants"]:
                if name in b["raw_variants"]:
                    assert a["raw_variants"][name] is not b["raw_variants"][name]
    assert reg.SONY_COMMON["verified_frames"] == [] and reg.SONY_COMMON["tier"] == "B"


def test_a7rv_stays_tier_b_with_one_checked_subvariant():
    entry = by_model()["ILCE-7RM5"]
    assert entry["tier"] == "B" and entry["verified_frames"] == []          # 用户 2026-09-25：无专用颜色配置前保持 B
    [sub] = entry["checked_subvariants"]
    assert sub["raw_file_type"] == 4 and sub["cfa"] is True and sub["raw_ifd_wh"] == [9728, 6656]
    assert sub["firmware"] == "ILCE-7RM5 v2.02" and len(sub["frames"]) == 6
    assert all(len(f["sha256"]) == 64 for f in sub["frames"])
    # 附录 E.2：变体 decoder 状态不变，核对只由精确匹配的子变体表达（类型 3 等不会被标成已核对）
    assert {k: v["decoder"] for k, v in entry["raw_variants"].items()} == {
        "uncompressed": "declared", "lossy_compressed": "declared", "lossless_compressed": "declared"}
    assert "本产品未测" in entry["quirks"]["star_eater"] and "cloudynights 902676" in entry["quirks"]["star_eater"]
    assert "专用颜色配置" in entry["next_note"] and "暗场不作门槛" in entry["next_note"]
    assert "不得打包" in entry["public_samples"]
    assert any("项目自验" in e for e in entry["evidence"])
    assert devices.registry_problems() == []


def test_other_sony_bodies_unchanged():
    models = by_model()
    for model in OTHER_SONY:
        e = models[model]
        assert e["tier"] == "B" and e["verified_frames"] == [] and "checked_subvariants" not in e
        assert set(e["raw_variants"]) == {"uncompressed", "lossy_compressed", "lossless_compressed"}
        assert all(v["decoder"] == "declared" for v in e["raw_variants"].values())
        assert "star_eater" in e["quirks"] and "本产品未测（用户" not in e["quirks"]["star_eater"]


def test_raw_file_type_mapping_is_complete():
    assert devices.REGISTRY_VARIANT_OF_TYPE == {0: "uncompressed", 1: "uncompressed", 2: "lossy_compressed",
                                                6: "lossy_compressed", 3: "lossless_compressed", 4: "lossless_compressed"}
    assert set(devices.SONY_RAW_FILE_TYPES) >= set(devices.REGISTRY_VARIANT_OF_TYPE)


def test_registry_self_check_rejects_b_plus_without_frames(monkeypatch):
    entry = copy.deepcopy(by_model()["ILCE-7RM5"])
    entry["tier"] = "B+"
    monkeypatch.setattr(reg, "REGISTRY", [entry])
    assert any("B+" in p for p in devices.registry_problems())


def _arw(path, *, raw_file_type=4, photometric="cfa", software="ILCE-7RM5 v2.02", shape=(6, 10)):
    tags = [(271, 2, None, "SONY", True), (272, 2, None, "ILCE-7RM5", True), (0x7000, 3, 1, raw_file_type, True)]
    tifffile.imwrite(path, np.zeros(shape, np.uint16), photometric=photometric, software=software, extratags=tags)
    return path


@pytest.fixture
def small_subvariant(monkeypatch):
    """The sub-variant logic is tested on a tiny synthetic raw IFD; the real constants are checked above."""
    entry = copy.deepcopy(by_model()["ILCE-7RM5"])
    entry["checked_subvariants"][0]["raw_ifd_wh"] = [10, 6]
    monkeypatch.setattr(reg, "REGISTRY", [e if "ILCE-7RM5" not in e["model_ids"] else entry for e in reg.REGISTRY])
    return entry


@pytest.mark.parametrize("kwargs,expected", [
    ({}, "checked"),
    ({"raw_file_type": 2}, "unchecked"),
    ({"raw_file_type": 0}, "unchecked"),
    ({"raw_file_type": 3}, "unchecked"),
    ({"photometric": "minisblack"}, "unchecked"),
    ({"software": "ILCE-7RM5 v3.00"}, "unchecked"),
    ({"shape": (8, 10)}, "unchecked"),
])
def test_subvariant_check_is_exact_and_never_changes_the_tier(tmp_path, small_subvariant, kwargs, expected):
    path = _arw(tmp_path / "DSC00001.ARW", **kwargs)
    ident = devices.identify_camera(path)
    assert ident["registry_variant"] == devices.REGISTRY_VARIANT_OF_TYPE[kwargs.get("raw_file_type", 4)]
    report = devices.inspect_device(path)          # synthetic file: metadata route fails softly
    assert report["tier"] == "B"
    assert report["subvariant_check"]["state"] == expected
    if expected == "unchecked":
        assert report["subvariant_check"]["reasons"]
        assert "未核对" in report["statement"]
    else:
        assert "已核对子变体" in report["statement"]
    json.dumps(report, allow_nan=False)


def test_other_body_reports_no_checked_subvariant(tmp_path):
    tags = [(271, 2, None, "SONY", True), (272, 2, None, "ILCE-7M4", True), (0x7000, 3, 1, 3, True)]
    path = tmp_path / "DSC00002.ARW"
    tifffile.imwrite(path, np.zeros((6, 10), np.uint16), photometric="cfa", extratags=tags)
    report = devices.inspect_device(path)
    assert report["tier"] == "B" and report["subvariant_check"]["state"] == "none_registered"


def test_export_master_routes_the_reference_like_the_job_path(tmp_path, monkeypatch):
    calls = []

    def fake_router(path, *, wb_mode="daylight", metadata=None):
        calls.append((Path(path).suffix.lower(), wb_mode))
        return {"router": True}

    monkeypatch.setattr(me, "colour_basis_for_reference", fake_router)
    monkeypatch.setattr(me, "colour_basis_from_dng", lambda *a, **k: pytest.fail("must route through colour_basis_for_reference"))
    monkeypatch.setattr(cp, "resolve_job_master", lambda source: tmp_path / "master_linear.fit")
    def fake_export(fits, basis, out, **k):
        Path(out).mkdir(parents=True, exist_ok=True)
        return {"basis": basis, "master_sha256": "x"}

    monkeypatch.setattr(me, "export_master", fake_export)
    for name in ("frame.ARW", "frame.dng"):
        reference = tmp_path / name
        reference.write_bytes(b"x")
        record = cp.export_master(tmp_path, tmp_path / "out", reference_dng=reference, wb_mode="as_shot")
        assert record["basis"] == {"router": True}
    assert calls == [(".arw", "as_shot"), (".dng", "as_shot")]
    from starstack import color_export
    import inspect
    assert "colour_basis_for_reference" in inspect.getsource(color_export.job_colour_basis)


THRESHOLDS_HEAD = {"M4_spread_codes": 2, "M4_near_white_luminance": 0.95, "M4_edge_min_fraction": 0.97,
                   "M6_pixel_median": 0.5, "M6_pixel_p99": 2.0, "M6_block_median": 0.5, "M6_block_p99": 2.0, "M6_row_stride": 2,
                   "M9c_mean_ab_delta": 0.5, "M9e_luminance_codes": 2}          # c1 at HEAD 1108e31 / e1d3df9


def test_c1_thresholds_unchanged():
    assert c1.THRESHOLDS == THRESHOLDS_HEAD


def _job(tmp_path, diagnostics):
    job = tmp_path / "job"
    job.mkdir(parents=True)
    (job / "job.json").write_text(json.dumps({"state": "succeeded", "diagnostics": diagnostics, "mask": None}), encoding="utf-8")
    return job


def test_c1_default_reference_prefers_the_colour_export_reference(tmp_path):
    job = _job(tmp_path, {"reference_input": "geo.ARW", "color_export": {"reference_input": "first.ARW"}})
    record, reference, _mask = c1._job_defaults(job)
    assert reference == "first.ARW" and c1.reference_source(record) == "color_export"
    legacy = _job(tmp_path / "b", {"reference_input": "geo.dng"})          # no color_export at all: older job
    record, reference, _mask = c1._job_defaults(legacy)
    assert reference == "geo.dng" and c1.reference_source(record) == "diagnostics_legacy"
    broken = _job(tmp_path / "c", {"reference_input": "geo.ARW", "color_export": {"state": "failed"}})
    with pytest.raises(ValueError):
        c1._job_defaults(broken)                                           # never fall back to the geometric frame
    empty = _job(tmp_path / "d", {})
    with pytest.raises(ValueError):
        c1._job_defaults(empty)


def test_c1_accepts_reference_alias():
    parser_args = c1.build_parser().parse_args(["--job", "j", "--output", "o", "--reference", "r.ARW"])
    assert parser_args.reference == "r.ARW"
    legacy = c1.build_parser().parse_args(["--job", "j", "--output", "o", "--reference-dng", "r.dng"])
    assert legacy.reference == "r.dng"


def test_c1_explicit_reference_still_requires_a_succeeded_job(tmp_path):
    job = tmp_path / "job"
    job.mkdir()
    (job / "job.json").write_text(json.dumps({"state": "failed", "diagnostics": {}}), encoding="utf-8")
    reference = tmp_path / "frame.ARW"
    reference.write_bytes(b"x")
    with pytest.raises(ValueError):
        c1.main(["--job", str(job), "--output", str(tmp_path / "out"), "--reference", str(reference)])
    assert not (tmp_path / "out").exists()                    # refused before any output is created


@pytest.mark.parametrize("frames", [[], ["not-a-dict"], [{"name": "a"}], [{"name": "a", "sha256": "Z" * 64}],
                                    [{"name": "a", "sha256": None}]])
def test_registry_self_check_reports_malformed_checked_frames(monkeypatch, frames):
    entry = copy.deepcopy(by_model()["ILCE-7RM5"])
    entry["checked_subvariants"][0]["frames"] = frames
    monkeypatch.setattr(reg, "REGISTRY", [entry])
    assert any("核对帧" in p for p in devices.registry_problems())      # reported, never raised


def test_a7rv_next_tier_uses_the_body_specific_rule(tmp_path):
    path = _arw(tmp_path / "DSC00003.ARW")
    report = devices.inspect_device(path)
    assert report["next_tier"] == by_model()["ILCE-7RM5"]["next_note"]
