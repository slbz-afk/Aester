"""The real-scene acceptance script runs end to end on a synthetic job and reports the pre-registered numbers."""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from starstack import imaging

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import c1_accept_color_contract as accept  # noqa: E402
import color_fixtures as cf  # noqa: E402


def _synthetic_job(tmp_path):
    dng = cf.write_reference_dng(tmp_path / "frame00001.dng")
    cam, sky, saturated, nodata = cf.synthetic_camera_master()
    job = tmp_path / "job-synthetic"
    products = job / "products" / "abcdef012345"
    products.mkdir(parents=True)
    imaging.write_linear(products / "master_linear.fit", np.flipud(cam), "synthetic camera master")
    mask = tmp_path / "sky.png"
    Image.fromarray((sky * 255).astype(np.uint8)).save(mask)
    (job / "job.json").write_text(json.dumps({
        "state": "succeeded", "mask": {"path": str(mask), "sha256": "0" * 64},      # the job's own {path, sha256} form
        "artifacts": [{"path": str(products / "master_linear.fit"), "kind": "fits"}],
        "diagnostics": {"reference_input": str(dng)}}), encoding="utf-8")
    return job


def _legacy_old_tiff(tmp_path):
    """A stand-in for the old v21 output: the legacy per-channel mapping of the same synthetic scene (uint16 sRGB)."""
    import tifffile
    from starstack import master_export as me
    dng = tmp_path / "frame00001.dng"
    cam, *_ = cf.synthetic_camera_master()
    linear, _, _ = me.convert_camera_master(cam, me.colour_basis_from_dng(dng, wb_mode="as_shot"), "srgb")
    legacy = np.flipud(imaging.display_base(np.flipud(linear)))
    path = tmp_path / "old_v21_like.tif"
    tifffile.imwrite(path, np.rint(legacy * 65535).astype(np.uint16), photometric="rgb")
    return path


def test_acceptance_script_runs_and_reports_all_criteria(tmp_path, capsys):
    job = _synthetic_job(tmp_path)
    old = _legacy_old_tiff(tmp_path)
    code = accept.main(["--job", str(job), "--output", str(tmp_path / "acc"), "--old-tiff", str(old)])
    assert code == 0
    printed = json.loads(capsys.readouterr().out)
    out = Path(printed["directory"])
    record = json.loads((out / "COLOR_ACCEPTANCE.json").read_text(encoding="utf-8"))
    assert record["state"] == "succeeded" and record["criteria"] == accept.CRITERIA_DOC
    assert record["sky_mask"] == str(tmp_path / "sky.png") and record["steps"]["render_photo"]["statistics"]["gamut_negative_mapped_by_region"]
    results = record["criteria_results"]
    assert set(results) == {"M1", "M4", "M5", "M6", "M7", "M8", "M9"}
    assert record["criteria_revision"] == accept.CRITERIA_REVISION and record["photo_wb"] == "as_shot"
    assert results["M1"]["pass"] and results["M1"]["replay_sha256_equal"] and results["M1"]["nodata_master"] == results["M1"]["nodata_source"]
    assert results["M4"]["pass"] and results["M4"]["a_camera_saturated_pixels"] > 0
    assert results["M5"]["pass"]
    assert results["M7"]["pass"]
    # M6: verdict by 修订 2 (block means); the pre-registered pixel-wise numbers are still reported
    m6 = results["M6"]
    assert m6["pass"] == m6["pass_revision2_block_means"] and "pixel_median" in m6
    assert record["verdict"]["M6_preregistered_pixelwise"] in ("PASS", "FAIL")
    # M5: verdict by 修订 3 (rounding excluded); both counts reported
    m5 = results["M5"]
    assert m5["pass"] == m5["pass_revision3"] and "pass_preregistered" in m5
    # M9: measured against the old TIFF with the same code; smoothing on vs off
    m9 = results["M9"]
    assert m9["old"]["a_noise"] > 0 and m9["new"]["a_noise"] <= m9["smoothing_off"]["a_noise"]
    assert m9["e_luminance_max_delta_codes"] <= 2 and m9["c_pass"] and m9["e_pass"]
    assert m9["b_ground_B_zero_fraction_new"] is not None and m9["b_ground_B_zero_fraction_old"] is not None
    assert record["verdict"]["M9"] in ("PASS", "FAIL") and results["M8"]["pass"] == all(results[k]["pass"] for k in ("M1", "M4", "M5", "M6", "M7", "M9"))
    assert "render_photo_smoothing_off" in record["steps"]
    assert not (out / "replay").exists()
    md = (out / "COLOR_ACCEPTANCE.md").read_text(encoding="utf-8")
    assert "M1 母版合同" in md and "修订 2" in md and "M9 阴影色度" in md and "与 F2 同口径" in md
    assert (Path(record["steps"]["render_photo"]["directory"]) / "candidates" / "CANDIDATES.md").is_file()
    # the job and its master were only read
    assert sorted(p.name for p in (job / "products" / "abcdef012345").iterdir()) == ["master_linear.fit"]


def test_thresholds_are_the_preregistered_numbers():
    t = accept.THRESHOLDS
    assert t["M4_spread_codes"] == 2 and t["M4_near_white_luminance"] == 0.95 and t["M4_edge_min_fraction"] == 0.97
    assert t["M6_pixel_median"] == 0.5 and t["M6_pixel_p99"] == 2.0 and t["M6_block_median"] == 0.5 and t["M6_block_p99"] == 2.0
    assert t["M9c_mean_ab_delta"] == 0.5 and t["M9e_luminance_codes"] == 2 and accept.PATCH == {"x0": 2880, "y0": 5000, "size": 256}


def test_acceptance_reports_s2_when_the_job_built_its_own_m_and_d(tmp_path, capsys):
    from starstack import color_export as ce, master_export as me
    job = _synthetic_job(tmp_path)
    old = _legacy_old_tiff(tmp_path)
    products = job / "products" / "abcdef012345"
    cam, sky, *_ = cf.synthetic_camera_master()
    basis = ce.job_colour_basis(tmp_path / "frame00001.dng")
    ce.export_job_products(np.flipud(cam).copy(), products, lambda: None, basis=basis, source_fits=products / "master_linear.fit",
                           sky_mask_path=tmp_path / "sky.png")
    record = json.loads((job / "job.json").read_text(encoding="utf-8"))
    # Real colour exports record the frame they used (core.py color_export.reference_input); since the Sony package C4
    # (docs/SONY_ADAPTATION_PREREG_20260925.md, appendix E.3) c1 defaults to it and never falls back to the geometric frame.
    record["diagnostics"]["color_export"] = {"state": "succeeded", "contract": "M+D photo-default-1",
                                             "reference_input": record["diagnostics"]["reference_input"]}
    (job / "job.json").write_text(json.dumps(record), encoding="utf-8")
    assert accept.main(["--job", str(job), "--output", str(tmp_path / "acc2"), "--old-tiff", str(old)]) == 0
    out = Path(json.loads(capsys.readouterr().out)["directory"])
    result = json.loads((out / "COLOR_ACCEPTANCE.json").read_text(encoding="utf-8"))
    s2 = result["criteria_results"]["S2"]
    assert s2["a_pass"] and s2["b_pass"] and s2["c_pass"] and s2["pass"] and result["verdict"]["S2"] == "PASS"
    assert "S2 作业内建 M+D" in (out / "COLOR_ACCEPTANCE.md").read_text(encoding="utf-8")
