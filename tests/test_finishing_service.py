"""Independent P2 service checks using real WorkspaceTools and tiny fake renders.

No GIMP process is started. XCF fixtures are only lifecycle/product-binding data.
"""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import threading
import time

import numpy as np
from PIL import Image, ImageCms
import pytest
import tifffile

from starstack import finishing_candidates as candidates, finishing_session as sessions
from starstack.finishing_assets import sha256_file
from starstack.finishing_backend import RenderCancelled, verify_products
from starstack.finishing_recipe import recipe_digest
from starstack.finishing_service import FinishingTools
from starstack.mcp_server import AccessDenied, WorkspaceTools


def _rgb(path, width=32, height=24):
    pixels = (np.arange(width * height * 3, dtype=np.uint32).reshape(height, width, 3) * 37 % 65536).astype(np.uint16)
    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    tifffile.imwrite(path, pixels, photometric="rgb", metadata=None,
                     extratags=[(34675, "B", len(icc), icc, False)])
    return pixels, icc


@pytest.fixture
def public(tmp_path, monkeypatch):
    monkeypatch.delenv("AESTER_GIMP", raising=False)
    roots = {key: tmp_path / key for key in ("input", "cache", "delivery", "outside")}
    for path in roots.values():
        path.mkdir()
    source = roots["input"] / "source.tif"
    _rgb(source)
    workspace = WorkspaceTools(input_root=roots["input"], output_root=roots["delivery"], cache_root=roots["cache"])
    service = FinishingTools(workspace, gimp_path="host-configured-fake-console.exe")
    record = service.create(str(source), "P2 independent tiny service checks")
    yield service, record, roots
    service.close()


def _save_record(record):
    sessions._save(Path(record["directory"]) / "session.json", record)


def _render_products(source, recipe, masks, work, **kwargs):
    if kwargs.get("cancel") and kwargs["cancel"]():
        raise RenderCancelled("mock cancelled before products")
    artifacts = work / "artifacts"
    artifacts.mkdir()
    shutil.copyfile(source["path"], artifacts / "result.tif")
    shutil.copyfile(source["path"], artifacts / "replay.tif")
    with tifffile.TiffFile(source["path"]) as tf:
        pixels = tf.pages[0].asarray()
        icc = bytes(tf.pages[0].tags[34675].value)
    Image.fromarray((pixels // 257).astype(np.uint8)).save(artifacts / "result.jpg", icc_profile=icc)
    (artifacts / "result.xcf").write_bytes(b"gimp xcf v013\x00mock service product")
    receipt = {"schema": "aester-finishing-gimp-receipt-1", "ok": True,
               "source_sha256": source["sha256"], "recipe_sha256": recipe_digest(recipe),
               "width": source["width"], "height": source["height"], "gimp_version": "mock-no-GIMP",
               "applied_operations": [op["id"] for op in recipe["operations"] if op["enabled"]]}
    (artifacts / "receipt.json").write_text(json.dumps(receipt), encoding="utf8")
    return {"status": "validated", "products": verify_products(artifacts, source, recipe)}


def _render_one(public, monkeypatch):
    service, record, _ = public
    monkeypatch.setattr(candidates, "render_recipe", _render_products)
    return service.render(record["directory"], expected_revision=0)


def test_host_intent_edit_preserves_render_recipe_and_initial_text(public):
    service, record, _ = public
    before = deepcopy(record["revisions"])
    edited = service.set_intent(record["directory"], "更克制的颜色", expected_intent_revision=0)
    assert edited["initial_intent"] == record["intent"]
    assert edited["intent_revision"] == 1
    assert edited["intent_history"][0]["origin"] == "user"
    assert edited["revisions"] == before and edited["revision"] == 0
    edited = service.set_intent(record["directory"], "保留绿色气辉", expected_intent_revision=1)
    assert edited["initial_intent"] == record["intent"]
    assert edited["intent_history"][1]["previous_intent"] == "更克制的颜色"
    assert edited["intent"] == "保留绿色气辉"
    assert service.get(record["directory"])["intent_revision"] == 2


@pytest.mark.parametrize("expected", [True, -1, 2, 0.0, None])
def test_host_intent_edit_rejects_stale_or_invalid_counter_without_write(public, expected):
    service, record, _ = public
    path = Path(record["directory"]) / "session.json"
    before = path.read_bytes()
    with pytest.raises(sessions.RevisionConflict):
        service.set_intent(record["directory"], "new intent", expected_intent_revision=expected)
    assert path.read_bytes() == before


def test_host_user_accept_and_agent_accept_are_distinct(public, monkeypatch):
    service, record, _ = public
    candidate = _render_one(public, monkeypatch)
    with pytest.raises(ValueError, match="Agent feedback"):
        service.feedback(record["directory"], candidate["id"], "agent claim", "accept")
    accepted = service.feedback_user(record["directory"], candidate["id"], "我选择这一版", "accept")
    assert accepted["feedback"][-1]["origin"] == "user"
    assert accepted["feedback"][-1]["candidate_id"] == candidate["id"]
    Path(candidate["execution"]["products"]["jpeg"]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        service.feedback_user(record["directory"], candidate["id"], "stale image", "accept")


def test_host_reference_preview_is_labeled_profiled_and_never_recipe_input(public):
    service, record, roots = public
    path = roots["input"] / "参考图.png"
    original = np.full((11, 19, 3), (35, 80, 140), np.uint8)
    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    Image.fromarray(original).save(path, icc_profile=icc)
    value = service.add_reference(record["directory"], str(path), "generated")
    reference = value["references"][0]
    preview = service.reference_preview(record["directory"], reference["id"])
    assert preview["kind"] == "generated" and preview["reference_pixels_used"] is False
    assert preview["source_sha256"] == reference["sha256"]
    with Image.open(preview["path"]) as image:
        np.testing.assert_array_equal(np.asarray(image), original)
        assert image.info["icc_profile"]
    assert service.get(record["directory"])["revisions"] == record["revisions"]
    assert sessions.renderer_masks(value, value["revisions"][0]["recipe"]) == {}


def test_host_reference_preview_rejects_registered_asset_changed(public):
    service, record, roots = public
    path = roots["input"] / "reference.png"
    Image.new("RGB", (20, 10), "red").save(path)
    value = service.add_reference(record["directory"], str(path), "external")
    reference = value["references"][0]
    Image.new("RGB", (20, 10), "blue").save(reference["path"])
    with pytest.raises(ValueError, match="changed"):
        service.reference_preview(record["directory"], reference["id"])
    with pytest.raises(ValueError, match="Unknown reference"):
        service.reference_preview(record["directory"], "missing")


def test_host_reference_orientation_transparency_and_assumed_color_are_explicit(public):
    service, record, roots = public
    path = roots["input"] / "transparent.png"
    exif = Image.Exif()
    exif[274] = 6
    Image.new("RGBA", (20, 10), (255, 0, 0, 0)).save(path, exif=exif)
    value = service.add_reference(record["directory"], str(path), "external")
    preview = service.reference_preview(record["directory"], value["references"][0]["id"])
    assert (preview["width"], preview["height"]) == (10, 20)
    assert "assumed sRGB" in preview["color_note"] and "transparency" in preview["color_note"]
    with Image.open(preview["path"]) as image:
        assert np.all(np.asarray(image) == 32)


def test_host_reference_invalid_icc_is_not_silently_ignored(public):
    service, record, roots = public
    path = roots["input"] / "bad-profile.png"
    Image.new("RGB", (20, 10), "red").save(path, icc_profile=b"invalid")
    value = service.add_reference(record["directory"], str(path), "external")
    with pytest.raises(ValueError, match="ICC"):
        service.reference_preview(record["directory"], value["references"][0]["id"])


def test_real_workspace_cache_session_and_host_delivery_roots(public, monkeypatch):
    service, record, roots = public
    assert Path(record["directory"]).parent == roots["cache"] / "finishing"
    assert service.get(record["directory"]) == record
    candidate = _render_one(public, monkeypatch)
    exported = service.export(record["directory"], candidate["id"])
    assert Path(exported["directory"]).parent == roots["delivery"]
    assert {p.name for p in Path(exported["directory"]).iterdir()} == {"photo.tif", "photo.jpg", "photo.xcf", "manifest.json"}
    for key, product in exported["manifest"]["products"].items():
        assert sha256_file(Path(exported["directory"]) / product["file"]) == product["sha256"]


@pytest.mark.parametrize("root_name", ["input", "cache", "delivery"])
def test_source_can_use_each_explicit_host_root(public, root_name):
    service, _, roots = public
    source = roots[root_name] / "host-approved.tif"
    _rgb(source)
    assert service.create(str(source), "allowed source")["source"]["path"] == str(source)


@pytest.mark.parametrize("operation", ["create", "mask", "reference"])
def test_external_asset_paths_are_denied_by_real_workspace(public, operation):
    service, record, roots = public
    outside = roots["outside"] / "unapproved.tif"
    if operation == "mask":
        tifffile.imwrite(outside, np.zeros((24, 32), np.uint8), metadata=None)
    else:
        _rgb(outside)
    before = (Path(record["directory"]) / "session.json").read_bytes()
    with pytest.raises(AccessDenied):
        if operation == "create":
            service.create(str(outside), "deny")
        elif operation == "mask":
            service.add_mask(record["directory"], str(outside), "outside", record["source"]["sha256"])
        else:
            service.add_reference(record["directory"], str(outside), "external")
    assert (Path(record["directory"]) / "session.json").read_bytes() == before


@pytest.mark.parametrize("bad_path", ["../outside/source.tif", "input/../../outside/source.tif", "C:source.tif"])
def test_traversal_and_drive_relative_inputs_are_denied(public, bad_path):
    service, _, _ = public
    with pytest.raises(AccessDenied):
        service.create(bad_path, "deny")


def test_valid_session_outside_dedicated_finishing_root_is_denied(public):
    service, record, roots = public
    foreign = sessions.create_session(record["source"]["path"], roots["outside"], intent="foreign")
    with pytest.raises(AccessDenied):
        service.get(foreign["directory"])


@pytest.mark.parametrize("field", ["source", "mask", "reference", "candidate"])
def test_persisted_path_tampering_is_denied_before_service_work(public, field):
    service, record, roots = public
    outside = roots["outside"] / "outside.tif"
    _rgb(outside)
    forged = deepcopy(record)
    if field == "source":
        forged["source"]["path"] = str(outside)
    elif field == "mask":
        forged["masks"]["forged"] = {"path": str(outside)}
    elif field == "reference":
        forged["references"] = [{"path": str(outside)}]
    else:
        forged["candidates"] = [{"directory": str(roots["outside"]), "status": "running"}]
    _save_record(forged)
    with pytest.raises(AccessDenied):
        service.get(record["directory"])


def test_linked_source_is_rejected_when_platform_allows_link(public):
    service, record, roots = public
    link = roots["input"] / "linked.tif"
    try:
        link.symlink_to(record["source"]["path"])
    except OSError as exc:
        pytest.skip(f"Platform cannot create test symlink: {exc}")
    with pytest.raises(AccessDenied):
        service.create(str(link), "deny linked source")


def test_registered_mask_origin_is_agent_and_snapshot_stays_internal(public):
    service, record, roots = public
    path = roots["input"] / "mask.png"
    Image.fromarray(np.full((24, 32), 128, np.uint8)).save(path)
    result = service.add_mask(record["directory"], str(path), "coverage", record["source"]["sha256"])
    mask = result["masks"]["coverage"]
    assert mask["origin"] == "agent"
    assert Path(mask["path"]).parent == Path(record["directory"]) / "assets"
    before = Path(mask["path"]).read_bytes()
    Image.fromarray(np.zeros((24, 32), np.uint8)).save(path)
    assert Path(mask["path"]).read_bytes() == before


def test_wrong_source_mask_and_generated_reference_role(public):
    service, record, roots = public
    path = roots["input"] / "mask.png"
    Image.fromarray(np.full((24, 32), 128, np.uint8)).save(path)
    with pytest.raises(ValueError, match="different source"):
        service.add_mask(record["directory"], str(path), "coverage", "0" * 64)
    result = service.add_reference(record["directory"], str(path), "generated", "style only")
    assert result["references"][0]["kind"] == "generated"
    assert Path(result["references"][0]["path"]).parent == Path(record["directory"]) / "assets"
    assert sessions.renderer_masks(result, result["revisions"][0]["recipe"]) == {}


def test_polygon_union_inversion_feather_and_agent_provenance(public):
    service, record, _ = public
    directory = record["directory"]
    polygons = [[[3, 3], [10, 3], [10, 12], [3, 12]], [[20, 8], [27, 8], [27, 16], [20, 16]]]
    plain = service.draw_mask(directory, "plain", polygons)
    inverted = service.draw_mask(directory, "inverse", polygons, invert=True)
    feather = service.draw_mask(directory, "soft", polygons, feather_px=2)
    a = tifffile.imread(plain["session"]["masks"]["plain"]["path"])
    b = tifffile.imread(inverted["session"]["masks"]["inverse"]["path"])
    c = tifffile.imread(feather["session"]["masks"]["soft"]["path"])
    assert a.shape == (24, 32) and set(np.unique(a)) == {0, 255}
    assert a[5, 5] == a[10, 23] == 255 and a[0, 0] == a[8, 15] == 0
    assert np.array_equal(b, 255 - a)
    assert np.any((c > 0) & (c < 255)) and not np.array_equal(c, a)
    assert plain["session"]["masks"]["plain"]["origin"] == "agent"
    assert plain["mask_preview"]["kind"] == "mask_overlay"
    assert plain["mask_preview"]["numerical_validation"] is False
    before = Path(plain["session"]["masks"]["plain"]["path"]).read_bytes()
    with pytest.raises(ValueError, match="replaced"):
        service.draw_mask(directory, "plain", polygons)
    assert Path(plain["session"]["masks"]["plain"]["path"]).read_bytes() == before


@pytest.mark.parametrize("change", [
    {"polygons": []}, {"polygons": [[[0, 0], [1, 1]]]},
    {"polygons": [[[0, 0], [-1, 1], [2, 2]]]},
    {"polygons": [[[0, 0], [33, 1], [2, 2]]]},
    {"polygons": [[[0, 0], [float("nan"), 1], [2, 2]]]},
    {"polygons": [[[0, 0], [True, 1], [2, 2]]]},
    {"feather_px": -1}, {"feather_px": float("inf")}, {"feather_px": True}, {"invert": 1},
])
def test_invalid_polygon_inputs_do_not_create_registered_assets(public, change):
    service, record, _ = public
    args = {"polygons": [[[1, 1], [3, 1], [3, 3]]], **change}
    with pytest.raises(ValueError):
        service.draw_mask(record["directory"], "bad", **args)
    assert service.get(record["directory"])["masks"] == {}


def test_polygon_edge_coordinates_match_declared_raster_geometry(public):
    service, record, _ = public
    result = service.draw_mask(record["directory"], "rectangle", [[[2, 2], [6, 2], [6, 6], [2, 6]]])
    pixels = tifffile.imread(result["session"]["masks"]["rectangle"]["path"])
    assert result["geometry"]["coordinates"] == "source-pixel edges"
    expected = np.zeros((24, 32), np.uint8)
    expected[2:6, 2:6] = 255
    assert np.array_equal(pixels, expected), "edge-coordinate 4x4 rectangle must not paint a fifth row/column"


@pytest.mark.parametrize("bounds,rows,columns", [
    ([2.25, 2.75, 6.25, 6.75], (3, 7), (2, 6)),
    ([2.5, 2.5, 6.5, 6.5], (2, 6), (2, 6)),
    ([0, 0, 32, 24], (0, 24), (0, 32)),
    ([31.49, 23.49, 32, 24], (23, 24), (31, 32)),
    ([31.51, 23.51, 32, 24], (24, 24), (32, 32)),
])
@pytest.mark.parametrize("reverse", [False, True])
def test_fractional_and_frame_boundary_polygons_use_pixel_centres(public, bounds, rows, columns, reverse):
    service, record, _ = public
    x0, y0, x1, y1 = bounds
    polygon = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    if reverse:
        polygon.reverse()
    result = service.draw_mask(record["directory"], "fractional", [polygon])
    pixels = tifffile.imread(result["session"]["masks"]["fractional"]["path"])
    expected = np.zeros((24, 32), np.uint8)
    expected[slice(*rows), slice(*columns)] = 255
    assert np.array_equal(pixels, expected)
    reopened = service.get(record["directory"])["masks"]["fractional"]
    assert reopened["origin"] == "agent"
    assert reopened["creation_geometry"] == result["geometry"]
    assert reopened["creation_geometry"]["polygons"] == [polygon]


def test_diagonal_and_overlapping_polygons_are_geometric_union(public):
    service, record, _ = public
    polygons = [[[0, 0], [8, 0], [0, 8]], [[2, 2], [6, 2], [6, 6], [2, 6]]]
    result = service.draw_mask(record["directory"], "union", polygons)
    pixels = tifffile.imread(result["session"]["masks"]["union"]["path"])
    yy, xx = np.indices((24, 32))
    expected = ((xx + yy < 7) | ((xx >= 2) & (xx < 6) & (yy >= 2) & (yy < 6))).astype(np.uint8) * 255
    assert np.array_equal(pixels, expected)


def test_feather_then_invert_and_geometry_survive_get(public):
    service, record, _ = public
    polygons = [[[7.25, 6.75], [20.25, 6.75], [20.25, 17.75], [7.25, 17.75]]]
    soft = service.draw_mask(record["directory"], "soft", polygons, feather_px=1.5)
    inverted = service.draw_mask(record["directory"], "inverted", polygons, feather_px=1.5, invert=True)
    a = tifffile.imread(soft["session"]["masks"]["soft"]["path"])
    b = tifffile.imread(inverted["session"]["masks"]["inverted"]["path"])
    assert np.array_equal(b, 255 - a) and np.any((a > 0) & (a < 255))
    geometry = service.get(record["directory"])["masks"]["inverted"]["creation_geometry"]
    assert geometry["polygons"] == polygons and geometry["feather_px"] == 1.5 and geometry["invert"] is True


@pytest.mark.parametrize("crop", [[-1, 0, 4, 4], [0, 0, 0, 4], [30, 0, 4, 4], [0, 0, 3], [0, 0, 2.0, 3], [True, 0, 2, 3]])
def test_bad_preview_crops_are_rejected(public, crop):
    service, record, _ = public
    with pytest.raises(ValueError, match="Crop"):
        service.preview(record["directory"], crop=crop)


def test_preview_xywh_native_crop_and_source_binding(public):
    service, record, _ = public
    crop = [7, 5, 13, 9]
    preview = service.preview(record["directory"], crop=crop)
    assert preview["source_crop"] == crop and (preview["width"], preview["height"]) == (13, 9)
    assert preview["scale"] == 1 and preview["native_scale"] is True
    assert preview["source_sha256"] == record["source"]["sha256"]
    assert Path(preview["path"]).parent == Path(record["directory"]) / "previews"
    with Image.open(preview["path"]) as image:
        assert image.mode == "RGB" and image.size == (13, 9) and image.info["icc_profile"]


def test_resized_preview_reports_independent_xy_scales(public):
    service, _, roots = public
    source = roots["input"] / "long-thin.tif"
    _rgb(source, width=2010, height=40)
    record = service.create(str(source), "mapping of rounded thumbnail dimensions")
    preview = service.preview(record["directory"], crop=[3, 2, 2001, 31])
    sx, sy = preview["width"] / 2001, preview["height"] / 31
    assert sx != sy  # Integer dimensions make the two actual scales different.
    assert preview.get("scale_x") == sx and preview.get("scale_y") == sy, "both axes must map back to source XYWH"


def test_agent_feedback_never_accepts_and_observe_keeps_engineering_status(public, monkeypatch):
    service, record, _ = public
    candidate = _render_one(public, monkeypatch)
    with pytest.raises(ValueError, match="Agent feedback"):
        service.feedback(record["directory"], candidate["id"], "user likes this (agent report)", decision="accept")
    result = service.feedback(record["directory"], candidate["id"], "consider weaker color", decision="revise")
    assert result["feedback"][-1]["origin"] == "agent" and result["feedback"][-1]["decision"] == "revise"
    assert result["candidates"][0]["status"] == "validated"


def test_unknown_job_and_stale_revision_do_not_launch(public, monkeypatch):
    service, record, _ = public
    monkeypatch.setattr(candidates, "render_recipe", lambda *a, **kw: pytest.fail("must not launch"))
    with pytest.raises(sessions.RevisionConflict):
        service.submit_render(record["directory"], expected_revision=1)
    for func in (service.render_status, service.cancel_render):
        with pytest.raises(ValueError, match="Unknown"):
            func("render-" + "f" * 32)
    assert service.get(record["directory"])["candidates"] == []


def test_async_running_duplicate_guard_and_success(public, monkeypatch):
    service, record, _ = public
    started, release = threading.Event(), threading.Event()
    def blocked(*args, **kwargs):
        started.set()
        assert release.wait(5), "test release timed out"
        return _render_products(*args, **kwargs)
    monkeypatch.setattr(candidates, "render_recipe", blocked)
    job = service.submit_render(record["directory"], 0)
    try:
        assert started.wait(5)
        assert service.render_status(job["job_id"])["status"] == "running"
        assert service.get(record["directory"])["candidates"][-1]["status"] == "running"
        with pytest.raises(ValueError, match="active render"):
            service.submit_render(record["directory"], 0)
        with pytest.raises(sessions.RevisionConflict):
            service.recover(record["directory"])
    finally:
        release.set()
    service._jobs[job["job_id"]]["future"].result(timeout=5)
    assert service.render_status(job["job_id"])["status"] == "validated"
    assert service.get(record["directory"])["candidates"][-1]["status"] == "validated"


@pytest.mark.parametrize("error,status", [(RuntimeError("plugin mock failure"), "failed"), (RenderCancelled("mock stopped"), "cancelled")])
def test_async_failure_and_cancelled_states_are_persisted(public, monkeypatch, error, status):
    service, record, _ = public
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr(candidates, "render_recipe", fail)
    job = service.submit_render(record["directory"], 0)
    with pytest.raises(type(error)):
        service._jobs[job["job_id"]]["future"].result(timeout=5)
    assert service.render_status(job["job_id"])["status"] == status
    persisted = service.get(record["directory"])["candidates"][-1]
    assert persisted["status"] == status and "execution" not in persisted
    with pytest.raises(ValueError, match="not a validated"):
        service.export(record["directory"], persisted["id"])


def test_cancel_running_event_reaches_renderer_and_cannot_publish(public, monkeypatch):
    service, record, _ = public
    started = threading.Event()
    def until_cancelled(*args, **kwargs):
        started.set()
        deadline = time.monotonic() + 5
        while not kwargs["cancel"]() and time.monotonic() < deadline:
            threading.Event().wait(.01)
        assert kwargs["cancel"](), "test cancel timed out"
        raise RenderCancelled("received cancellation")
    monkeypatch.setattr(candidates, "render_recipe", until_cancelled)
    job = service.submit_render(record["directory"], 0)
    assert started.wait(5)
    assert service.cancel_render(job["job_id"])["status"] in {"cancelling", "cancelled"}
    with pytest.raises(RenderCancelled):
        service._jobs[job["job_id"]]["future"].result(timeout=5)
    assert service.render_status(job["job_id"])["status"] == "cancelled"
    assert service.get(record["directory"])["candidates"][-1]["status"] == "cancelled"


def test_close_reports_queued_future_as_cancelled_not_failed(public, monkeypatch):
    service, record, _ = public
    second = service.create(record["source"]["path"], "second queued session")
    started = threading.Event()
    def until_cancelled(*args, **kwargs):
        started.set()
        deadline = time.monotonic() + 5
        while not kwargs["cancel"]() and time.monotonic() < deadline:
            threading.Event().wait(.01)
        assert kwargs["cancel"](), "test close timed out"
        raise RenderCancelled("service shutdown")
    monkeypatch.setattr(candidates, "render_recipe", until_cancelled)
    first_job = service.submit_render(record["directory"], 0)
    assert started.wait(5)
    second_job = service.submit_render(second["directory"], 0)
    assert not service._jobs[second_job["job_id"]]["future"].running()
    service.close()
    assert service.render_status(first_job["job_id"])["status"] == "cancelled"
    assert service.render_status(second_job["job_id"])["status"] == "cancelled"
    assert service.get(second["directory"])["candidates"] == []


def test_restart_uses_persisted_recovery_without_republishing(public):
    service, record, _ = public
    render_dir = Path(record["directory"]) / "renders" / ("candidate-" + "a" * 32)
    render_dir.mkdir(parents=True)
    record["candidates"] = [{"id": render_dir.name, "directory": str(render_dir), "status": "running",
                              "revision": 0, "recipe_sha256": record["revisions"][0]["recipe_sha256"]}]
    _save_record(record)
    restarted = FinishingTools(service.workspace, gimp_path="unused")
    try:
        with pytest.raises(ValueError, match="Unknown render"):
            restarted.render_status("render-" + "0" * 32)
        with sessions._locked(record["directory"]):
            with pytest.raises(sessions.RevisionConflict):
                restarted.recover(record["directory"])
        recovered = restarted.recover(record["directory"])
        assert recovered["candidates"][0]["status"] == "interrupted"
        assert "execution" not in recovered["candidates"][0]
        assert restarted.recover(record["directory"]) == recovered
    finally:
        restarted.close()
