"""Independent H5 file-binding, corruption and read-snapshot fault checks.

Tiny synthetic files only. Pixel-quality thresholds are not changed here.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import tifffile
from astropy.io import fits

from starstack import compose_evidence as evidence
from starstack import h5_job_validation as h5
from starstack import synthetic_starfield as sf


@pytest.fixture
def scene(tmp_path):
    root = tmp_path / "job"
    work, export = root / "work", root / "products" / "one"
    work.mkdir(parents=True)
    export.mkdir(parents=True)
    rows, cols = np.mgrid[:12, :11]
    sky = np.stack([.2 + rows*.005, .25 + cols*.002, .3 + rows*.004], axis=-1).astype(np.float32)
    ground = np.stack([.02 + cols*.001, .04 + rows*.001, .03 + cols*.002], axis=-1).astype(np.float32)
    mask = np.full((12, 11), 128, np.uint8)
    mask[:4] = 255
    mask[8:] = 0
    weight = mask.astype(np.float32) / 255
    weight[1, 2] = 0  # one nominal sky pixel really has no observed sky
    composite = weight[..., None]*sky + (1-weight[..., None])*ground
    sky_path, ground_path = work / "sky_master.fit", work / "foreground_master.fit"
    composite_path = export / "composite.fit"
    for path, data in [(sky_path, sky), (ground_path, ground), (composite_path, composite)]:
        sf.write_fits_rgb32(path, np.moveaxis(data, -1, 0))
    Image.fromarray(mask).save(root / "sky_mask.png")
    descriptor = evidence.write_weight(export, weight, composite_path)
    job = {"state": "succeeded", "mask": "sky_mask.png",
           "artifacts": [{"label": "合成线性母版", "path": str(composite_path)}],
           "diagnostics": {"sky_validity": {"compose_weight": descriptor}}}
    (root / "job.json").write_text(json.dumps(job), encoding="utf-8")
    return {"root": root, "job": job, "descriptor": descriptor, "weight": weight,
            "composite": composite, "composite_path": composite_path,
            "weight_path": Path(descriptor["path"])}


def test_complete_current_evidence_is_bound_and_top_down(scene):
    before = deepcopy(scene["job"])
    result = h5.evaluate_job(scene["root"])
    assert result["judgement"]["status"] == "passed"
    assert result["job_json_sha256"] == evidence.sha256(scene["root"] / "job.json")
    assert np.array_equal(h5.rgb_top_down(scene["composite_path"]), scene["composite"])
    assert scene["job"] == before


@pytest.mark.parametrize("edit", [
    lambda job: job.update(artifacts=[None]),
    lambda job: job.update(artifacts="not a list"),
    lambda job: job.update(diagnostics=[1]),
    lambda job: job["diagnostics"].update(sky_validity=[1]),
])
def test_invalid_metadata_returns_failed_verdict_instead_of_attribute_error(scene, edit):
    job = deepcopy(scene["job"])
    edit(job)
    assert h5.evaluate_job(scene["root"], job=job)["judgement"]["status"] == "failed"


def test_missing_evidence_or_incomplete_job_never_passes(scene):
    job = deepcopy(scene["job"])
    job["diagnostics"]["sky_validity"].pop("compose_weight")
    assert h5.evaluate_job(scene["root"], job=job)["judgement"]["status"] == "insufficient_data"
    job = {**scene["job"], "state": "running"}
    assert h5.evaluate_job(scene["root"], job=job)["judgement"]["status"] == "insufficient_data"
    scene["weight_path"].unlink()
    assert h5.evaluate_job(scene["root"])["judgement"]["status"] == "insufficient_data"


def test_no_mask_is_not_applicable_and_not_success(scene):
    result = h5.evaluate_job(scene["root"], job={**scene["job"], "mask": None})["judgement"]
    assert result["status"] == "not_applicable" and result["passed"] is False


@pytest.mark.parametrize("field,value", [("version", "future"), ("roworder", "BOTTOM-UP"),
                                         ("shape", [11, 12]), ("dtype", "float64"),
                                         ("sha256", "0"*64), ("composite_sha256", "0"*64)])
def test_descriptor_mismatch_is_failed(scene, field, value):
    job = deepcopy(scene["job"])
    job["diagnostics"]["sky_validity"]["compose_weight"][field] = value
    assert h5.evaluate_job(scene["root"], job=job)["judgement"]["status"] == "failed"


def test_paths_outside_job_and_wrong_composite_binding_rejected(scene, tmp_path):
    descriptor = deepcopy(scene["descriptor"])
    descriptor["path"] = str(tmp_path / "outside.tiff")
    with pytest.raises(ValueError):
        evidence.read_weight(scene["root"], descriptor, scene["composite_path"])
    descriptor = deepcopy(scene["descriptor"])
    descriptor["composite"] = str(scene["root"] / "another.fit")
    with pytest.raises(ValueError):
        evidence.read_weight(scene["root"], descriptor, scene["composite_path"])


def test_mutated_weight_file_fails_hash_before_decoding(scene):
    scene["weight_path"].write_bytes(b"replacement, not a TIFF")
    result = h5.evaluate_job(scene["root"])
    assert result["judgement"]["status"] == "failed"
    assert "hash mismatch" in " ".join(result["judgement"]["failing"])


def test_weight_decode_uses_exact_hashed_bytes_even_if_path_replaced(scene, monkeypatch):
    original_read = Path.read_bytes
    replaced = []
    def replace_after_read(path):
        raw = original_read(path)
        if path == scene["weight_path"] and not replaced:
            replaced.append(True)
            tifffile.imwrite(path, np.full_like(scene["weight"], .75))
        return raw
    monkeypatch.setattr(Path, "read_bytes", replace_after_read)
    returned = evidence.read_weight(scene["root"], scene["descriptor"], scene["composite_path"])
    assert replaced and np.array_equal(returned, scene["weight"])
    assert not np.array_equal(tifffile.imread(scene["weight_path"]), returned)


def test_composite_changed_after_precheck_is_rejected_when_decoded(scene, monkeypatch):
    original_read_weight = h5.read_weight
    def replace_after_precheck(*args, **kwargs):
        returned = original_read_weight(*args, **kwargs)
        sf.write_fits_rgb32(scene["composite_path"], np.moveaxis(scene["composite"] + .1, -1, 0))
        return returned
    monkeypatch.setattr(h5, "read_weight", replace_after_precheck)
    result = h5.evaluate_job(scene["root"])
    assert result["judgement"]["status"] == "failed"
    assert "snapshot hash mismatch" in " ".join(result["judgement"]["failing"])


def test_composite_decode_uses_one_validated_snapshot(scene, monkeypatch):
    original_read = Path.read_bytes
    calls = []
    def replace_after_snapshot(path):
        raw = original_read(path)
        if path == scene["composite_path"] and not calls:
            calls.append(1)
            sf.write_fits_rgb32(path, np.moveaxis(scene["composite"] + .1, -1, 0))
        return raw
    monkeypatch.setattr(Path, "read_bytes", replace_after_snapshot)
    actual = h5.rgb_top_down(scene["composite_path"], expected_sha256=scene["descriptor"]["composite_sha256"])
    assert calls == [1]
    assert np.array_equal(actual, scene["composite"])


def test_explicit_unsupported_row_order_is_rejected(scene):
    with fits.open(scene["composite_path"], mode="update") as hdus:
        hdus[0].header["ROWORDER"] = "TOP-DOWN"
    with pytest.raises(ValueError, match="BOTTOM-UP"):
        h5.rgb_top_down(scene["composite_path"])


def test_wrong_requested_mask_is_failed(scene):
    other = scene["root"] / "other-mask.png"
    Image.fromarray(np.zeros(scene["weight"].shape, np.uint8)).save(other)
    assert h5.evaluate_job(scene["root"], mask_path=other)["judgement"]["status"] == "failed"
