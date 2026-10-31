"""Independent output-boundary probes using genuine encoded master records."""
from pathlib import Path
import json
import os

import numpy as np
import pytest
import tifffile
from PIL import Image

from starstack import workflow as wf, master_export as me
from starstack.mcp_server import WorkspaceTools, AccessDenied
from test_workflow import setup
import color_fixtures as cf


def test_old_mcp_host_still_runs_unseparated_workflow(setup, tmp_path, monkeypatch):
    monkeypatch.delenv("STARSTACK_CACHE_ROOT", raising=False)
    service = WorkspaceTools(setup["inputs"], tmp_path / "legacy", setup["engine"])
    record = service.submit_workflow(str(setup["inputs"]))
    assert "delivery_root" not in record["request"]
    done = wf.run_workflow(record["workflow_dir"])
    checked = service.workflow_status(done["workflow_dir"])
    assert checked["state"] == "succeeded"
    assert Path(checked["delivery_dir"]).parent == Path(checked["workflow_dir"])


@pytest.mark.parametrize("path_style", ["absolute", "relative_parent"])
def test_delivered_master_cannot_name_an_external_saturation_mask(tmp_path, path_style):
    raw = tmp_path / "raw"
    raw.mkdir()
    works = tmp_path / "works"
    private = tmp_path / "outside-host"
    private.mkdir()
    reference = cf.write_reference_dng(raw / "reference.dng")
    basis = me.colour_basis_from_dng(reference)
    camera, *_ = cf.synthetic_camera_master(144, 160, ground_from=120, nodata_columns=0)
    record = me.export_master_array(camera, basis, works / "manual",
        source_path="synthetic-test.fit", source_sha256="0" * 64)
    secret_mask = private / "private.png"
    Image.new("L", (160, 144), 255).save(secret_mask)
    master = Path(record["master"])
    data, meta, profile = me.read_master(master)
    pixels = np.array(data, copy=True)
    del data
    mask_name = str(secret_mask) if path_style == "absolute" else os.path.relpath(secret_mask, master.parent)
    meta["camera_saturation"].update(mask_file=mask_name, mask_sha256=me.sha256_file(secret_mask), count=144 * 160)
    altered = master.with_name("altered-master.tif")
    tifffile.imwrite(altered, pixels, photometric="rgb", iccprofile=profile, metadata=meta)
    service = WorkspaceTools(raw, works, cache_root=tmp_path / "cache")
    with pytest.raises((AccessDenied, ValueError), match="蒙版|mask|范围|目录|文件名"):
        service.render_photo(str(altered))


def test_real_master_in_works_can_render_to_cache_without_changing_it(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    works, cache = tmp_path / "works", tmp_path / "cache"
    reference = cf.write_reference_dng(raw / "reference.dng")
    basis = me.colour_basis_from_dng(reference)
    camera, *_ = cf.synthetic_camera_master(144, 160, ground_from=120, nodata_columns=0)
    record = me.export_master_array(camera, basis, works / "manual",
        source_path="synthetic-test.fit", source_sha256="0" * 64)
    master = Path(record["master"])
    before = me.sha256_file(master)
    service = WorkspaceTools(raw, works, cache_root=cache)
    result = service.render_photo(str(master))
    assert Path(result["directory"]).is_relative_to(cache)
    assert me.sha256_file(master) == before
