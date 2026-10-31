"""Offscreen public-service UI checks; never launch GIMP or process user photos."""
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
from PIL import Image, ImageCms
import pytest
import tifffile
from PySide6.QtCore import QPoint, QThread, QTimer, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from starstack import finishing_candidates as candidates
from starstack.finishing_backend import verify_products, RenderCancelled
from starstack.finishing_recipe import recipe_digest
from starstack.finishing_service import FinishingTools
from starstack.mcp_server import WorkspaceTools
from aester_desktop.finishing_gui import (FinishingWindow, HostConfig, mcp_configuration,
                                    preview_point_to_source, reference_color_text)


@pytest.fixture(scope="module")
def app():
    value = QApplication.instance() or QApplication([])
    yield value


def drain(app, condition, timeout=10):
    deadline = time.monotonic() + timeout
    while not condition() and time.monotonic() < deadline:
        app.processEvents()
        # Let Python service workers acquire the GIL as well as pumping Qt.
        time.sleep(.002)
    assert condition(), "Qt operation did not finish"
    app.processEvents()


def idle(app, window):
    drain(app, lambda: window._task is None)


@pytest.fixture
def host(tmp_path, monkeypatch, app):
    monkeypatch.delenv("AESTER_GIMP", raising=False)
    monkeypatch.setattr(candidates, "render_recipe", lambda *args, **kwargs: pytest.fail("Real renderer is forbidden"))
    roots = [tmp_path / name for name in ("输入 照片", "导出 作品", "缓存 会话")]
    for root in roots:
        root.mkdir()
    pixels = (np.arange(32 * 24 * 3, dtype=np.uint32).reshape(24, 32, 3) * 17).astype(np.uint16)
    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    source = roots[0] / "星野 原图.tif"
    tifffile.imwrite(source, pixels, photometric="rgb", metadata=None,
                     extratags=[(34675, "B", len(icc), icc, False)])
    reference = roots[0] / "生成 参考.png"
    Image.fromarray((pixels // 257).astype(np.uint8)).save(reference)
    service = FinishingTools(WorkspaceTools(input_root=roots[0], output_root=roots[1], cache_root=roots[2]),
                             gimp_path="host fake GIMP.exe")
    config = HostConfig(*(str(path) for path in roots), gimp=service.gimp_path)
    def checker(**kwargs):
        pytest.fail("Component execution/discovery requires an explicit test stub")
    window = FinishingWindow(service, config, component_checker=checker)
    window.source_path.setText(str(source))
    window.intent.setPlainText("保留星空颜色，稍微提亮地景")
    yield window, service, roots, source, reference
    if window.active_job:
        window.cancel()
        drain(app, lambda: window.active_job is None)
    idle(app, window)
    window.close()
    drain(app, lambda: window._closed_ready)
    service.close()


def create(app, host):
    window = host[0]
    window.create_button.click()
    idle(app, window)
    assert window.session is not None
    return window.session


def fake_renderer(source, recipe, masks, work, **kwargs):
    """Synthetic product-binding fixture, not a GIMP/photography assertion."""
    if kwargs.get("cancel") and kwargs["cancel"]():
        raise RenderCancelled("synthetic cancellation")
    folder = work / "artifacts"
    folder.mkdir()
    shutil.copyfile(source["path"], folder / "result.tif")
    shutil.copyfile(source["path"], folder / "replay.tif")
    with tifffile.TiffFile(source["path"]) as tf:
        pixels, icc = tf.pages[0].asarray(), bytes(tf.pages[0].tags[34675].value)
    Image.fromarray((pixels // 257).astype(np.uint8)).save(folder / "result.jpg", icc_profile=icc)
    (folder / "result.xcf").write_bytes(b"gimp xcf v013\x00synthetic UI fixture")
    receipt = {"schema": "aester-finishing-gimp-receipt-1", "ok": True,
        "source_sha256": source["sha256"], "recipe_sha256": recipe_digest(recipe),
        "width": source["width"], "height": source["height"], "gimp_version": "mock-no-GIMP",
        "applied_operations": [op["id"] for op in recipe["operations"] if op["enabled"]]}
    (folder / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    return {"status": "validated", "products": verify_products(folder, source, recipe)}


def candidate(app, host, monkeypatch):
    window, service = host[:2]
    create(app, host)
    monkeypatch.setattr(candidates, "render_recipe", fake_renderer)
    value = service.render(window.session["directory"], expected_revision=0)
    window.refresh_button.click()
    idle(app, window)
    return value


def test_create_real_service_chinese_space_paths_and_reopen(app, host):
    window, service, roots, source, _ = host
    record = create(app, host)
    assert Path(record["directory"]).parent == roots[2] / "finishing"
    assert record["source"]["path"] == str(source)
    assert window.views["source"]._image.width() == 32
    assert window.views["candidate"]._image is None
    window.open_session(record["directory"])
    idle(app, window)
    assert window.session["directory"] == record["directory"]
    assert not window.feedback_button.isEnabled()


def test_offscreen_window_and_all_tabs_render_without_native_application(app, host):
    window = host[0]
    window.show()
    app.processEvents()
    assert window.isVisible()
    for index in range(window.tabs.count()):
        window.tabs.setCurrentIndex(index)
        app.processEvents()
        picture = window.grab().toImage()
        assert not picture.isNull()
        assert picture.width() > 500 and picture.height() > 400
    window.tabs.setCurrentIndex(0)
    app.processEvents()
    tops = [view.mapTo(window, QPoint(0, 0)).y() for view in window.views.values()]
    assert len(set(tops)) == 1


def test_create_outside_host_input_is_denied(app, host, tmp_path):
    window = host[0]
    forbidden = tmp_path / "outside.tif"
    shutil.copyfile(host[3], forbidden)
    window.source_path.setText(str(forbidden))
    window.create_button.click()
    idle(app, window)
    assert window.session is None
    assert "未完成" in window.status.text()


def test_open_outside_session_root_does_not_rebind(app, host, tmp_path):
    record = create(app, host)
    window = host[0]
    window.open_session(str(tmp_path))
    idle(app, window)
    assert window.session["directory"] == record["directory"]
    assert "未完成" in window.status.text()


def test_intent_click_uses_host_revision_and_preserves_recipe(app, host):
    record = create(app, host)
    window, service = host[:2]
    original = deepcopy(record["revisions"])
    window.intent.setPlainText("新的意图，不抹去气辉")
    window.intent_button.click()
    idle(app, window)
    actual = service.get(record["directory"])
    assert actual["intent"] == "新的意图，不抹去气辉"
    assert actual["intent_revision"] == 1
    assert actual["revisions"] == original


def test_intent_conflict_refreshes_without_overwriting_other_writer(app, host):
    record = create(app, host)
    window, service = host[:2]
    service.set_intent(record["directory"], "另一个窗口的新目标", expected_intent_revision=0)
    window.intent.setPlainText("过期目标")
    window.intent_button.click()
    idle(app, window)
    assert window.intent.toPlainText() == "过期目标"
    assert "另一个窗口的新目标" in window.intent_conflict.text()
    assert "保留草稿" in window.status.text()
    assert service.get(record["directory"])["intent_revision"] == 1
    window.intent.setPlainText("比较后确认的新目标")
    window.intent_button.click()
    idle(app, window)
    assert service.get(record["directory"])["intent"] == "比较后确认的新目标"
    assert window.intent_conflict.text() == ""


def test_reference_label_color_note_and_recipe_isolation(app, host):
    record = create(app, host)
    window, service, _, _, reference = host
    window.reference_path.setText(str(reference))
    window.reference_kind.setCurrentIndex(window.reference_kind.findData("generated"))
    window.reference_note.setText("仅参考氛围")
    window.add_reference()
    idle(app, window)
    actual = service.get(record["directory"])
    assert actual["references"][0]["kind"] == "generated"
    assert actual["revisions"] == record["revisions"]
    assert "生成参考" in window.references.currentText()
    assert window.views["reference"]._image is not None
    assert "sRGB" in window.view_note.text()
    assert "参考未带色彩配置" in window.view_note.text()
    assert "untagged reference assumed sRGB" in window.view_note.toolTip()


@pytest.mark.parametrize("original, expected", [
    ("embedded ICC converted to sRGB for display", "参考已按内嵌色彩配置转换为 sRGB 预览"),
    ("untagged reference assumed sRGB for display; original color space unverified", "参考未带色彩配置，预览暂按 sRGB 显示"),
    ("embedded ICC converted to sRGB for display; transparency displayed over dark gray", "参考已按内嵌色彩配置转换为 sRGB 预览；透明区域显示为深灰背景"),
    ("future diagnostic", "future diagnostic")])
def test_reference_notice_translation_retains_unknown_detail(original, expected):
    assert reference_color_text(original) == expected


def test_native_xywh_uses_public_preview_for_source_and_candidate(app, host, monkeypatch):
    value = candidate(app, host, monkeypatch)
    window, service = host[:2]
    calls = []
    original = service.preview
    def preview(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)
    monkeypatch.setattr(service, "preview", preview)
    for spin, coordinate in zip(window.crop, [5, 3, 11, 7]):
        spin.setValue(coordinate)
    window.crop_button.click()
    idle(app, window)
    assert [call["crop"] for call in calls] == [[5, 3, 11, 7]] * 2
    assert calls[1]["candidate_id"] == value["id"]
    for name in ("source", "candidate"):
        view = window.views[name]
        assert view.native
        assert (view._image.width(), view._image.height()) == (11, 7)
        assert (view.label.pixmap().width(), view.label.pixmap().height()) == (11, 7)


def test_double_click_fit_view_syncs_bounded_native_crop_and_ignores_reference(app, host, monkeypatch):
    candidate(app, host, monkeypatch)
    window, service = host[:2]
    window.show()
    app.processEvents()
    calls = []
    original = service.preview
    def preview(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)
    monkeypatch.setattr(service, "preview", preview)
    window.crop[2].setValue(10)
    window.crop[3].setValue(8)
    label = window.views["candidate"].label
    QTest.mouseDClick(label, Qt.MouseButton.LeftButton, pos=QPoint(label.width() - 1, label.height() - 1))
    idle(app, window)
    assert [item["crop"] for item in calls] == [[22, 16, 10, 8]] * 2
    assert window.views["source"].native and window.views["candidate"].native
    calls.clear()
    QTest.mouseDClick(window.views["reference"].label, Qt.MouseButton.LeftButton, pos=QPoint(2, 2))
    app.processEvents()
    assert calls == []


def test_real_preview_mapping_uses_both_axes_and_nonzero_crop_origin(app, host):
    window, service, roots = host[:3]
    path = roots[0] / "双轴 缩图.tif"
    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    tifffile.imwrite(path, np.zeros((31, 2001, 3), dtype=np.uint16), photometric="rgb", metadata=None,
                     extratags=[(34675, "B", len(icc), icc, False)])
    record = service.create(str(path), "坐标映射检查")
    view = service.preview(record["directory"], crop=[7, 2, 1801, 29])
    assert view["scale_x"] != view["scale_y"]
    mapped = preview_point_to_source(view, 150, 2.5, 200, 5)
    assert mapped == pytest.approx((7 + 1801 * .75, 2 + 29 * .5))
    assert preview_point_to_source(view, 0, 0, 200, 5) == (7, 2)
    assert preview_point_to_source(view, 200, 5, 200, 5) == pytest.approx((1808, 31))


def test_scaled_crop_is_never_labelled_100_percent(app, host, monkeypatch):
    create(app, host)
    window, service = host[:2]
    original = service.preview
    def preview(**kwargs):
        return {**original(**kwargs), "native_scale": False}
    monkeypatch.setattr(service, "preview", preview)
    window.crop_button.click()
    idle(app, window)
    assert "不能显示为 100%" in window.status.text()
    assert window.views["source"]._image is None


@pytest.mark.parametrize("decision", ["observe", "revise", "reject", "accept"])
def test_explicit_click_records_user_feedback_bound_to_valid_candidate(app, host, monkeypatch, decision):
    value = candidate(app, host, monkeypatch)
    window, service = host[:2]
    assert not service.get(window.session["directory"])["feedback"]
    window.feedback_text.setText("用户看图后的真实意见")
    window.decision.setCurrentIndex(window.decision.findData(decision))
    window.feedback_button.click()
    idle(app, window)
    feedback = service.get(window.session["directory"])["feedback"][-1]
    assert (feedback["origin"], feedback["decision"], feedback["candidate_id"]) == ("user", decision, value["id"])


def test_missing_host_feedback_disables_button_no_agent_fallback(app, host, monkeypatch):
    candidate(app, host, monkeypatch)
    window, service = host[:2]
    monkeypatch.setattr(service, "feedback_user", None)
    window._sync_controls()
    assert not window.feedback_button.isEnabled()
    window.feedback()
    assert not service.get(window.session["directory"])["feedback"]


def test_tampered_product_cannot_be_accepted_or_exported(app, host, monkeypatch):
    value = candidate(app, host, monkeypatch)
    window, service = host[:2]
    Path(value["execution"]["products"]["jpeg"]["path"]).write_bytes(b"tampered")
    window.decision.setCurrentIndex(window.decision.findData("accept"))
    window.feedback_button.click()
    idle(app, window)
    assert "未完成" in window.status.text()
    assert not service.get(window.session["directory"])["feedback"]
    window.export_button.click()
    idle(app, window)
    assert "未完成" in window.status.text()
    assert not list(host[2][1].glob("aester-photo-*"))


def test_export_uses_host_delivery_root_and_complete_manifest(app, host, monkeypatch):
    value = candidate(app, host, monkeypatch)
    window, _, roots = host[:3]
    window.export_button.click()
    idle(app, window)
    exports = list(roots[1].glob("aester-photo-*"))
    assert len(exports) == 1
    manifest = json.loads((exports[0] / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["candidate_id"] == value["id"]
    assert set(manifest["products"]) == {"jpeg", "tiff", "xcf"}
    assert all((exports[0] / record["file"]).exists() for record in manifest["products"].values())


def test_worker_io_does_not_block_qt_and_updates_on_gui_thread(app, host, monkeypatch):
    window, service = host[:2]
    started, release = threading.Event(), threading.Event()
    original = service.create
    def blocked(**kwargs):
        assert QThread.currentThread() != app.thread()
        started.set()
        assert release.wait(3)
        return original(**kwargs)
    monkeypatch.setattr(service, "create", blocked)
    loaded = window._apply_session
    def checked(value):
        assert QThread.currentThread() == app.thread()
        loaded(value)
    monkeypatch.setattr(window, "_apply_session", checked)
    window.create_button.click()
    drain(app, started.is_set)
    tick = []
    QTimer.singleShot(0, lambda: tick.append(True))
    drain(app, lambda: bool(tick))
    assert window._task is not None
    assert not window.create_button.isEnabled()
    release.set()
    idle(app, window)
    assert window.session is not None


def test_render_failure_does_not_add_selectable_candidate(app, host, monkeypatch):
    create(app, host)
    window = host[0]
    def fail(*args, **kwargs):
        raise ValueError("synthetic backend failure")
    monkeypatch.setattr(candidates, "render_recipe", fail)
    window.render_button.click()
    drain(app, lambda: window.active_job is not None or "失败" in window.status.text())
    drain(app, lambda: window.active_job is None and window._task is None)
    assert "失败" in window.status.text()
    assert window.candidates.count() == 1
    assert not window.export_button.isEnabled()


def test_submit_then_cancel_only_owned_job_and_keep_shared_service(app, host, monkeypatch):
    create(app, host)
    window, service = host[:2]
    started = threading.Event()
    def blocked(source, recipe, masks, work, **kwargs):
        started.set()
        deadline = time.monotonic() + 3
        while not kwargs["cancel"]() and time.monotonic() < deadline:
            time.sleep(.01)
        raise RenderCancelled("synthetic cancelled")
    monkeypatch.setattr(candidates, "render_recipe", blocked)
    cancellations = []
    cancel = service.cancel_render
    def tracked(job_id):
        cancellations.append(job_id)
        return cancel(job_id)
    monkeypatch.setattr(service, "cancel_render", tracked)
    window.render_button.click()
    drain(app, lambda: started.is_set() and window.active_job is not None)
    owned = window.active_job
    assert window.cancel_button.isEnabled()
    assert not window.open_button.isEnabled()
    window.close()
    drain(app, lambda: window._closed_ready)
    assert cancellations == [owned]
    assert service.render_status(owned)["status"] == "cancelled"
    assert not service._pool._shutdown


def test_owned_service_closes_asynchronously_after_idle(app, host, monkeypatch):
    window, service = host[:2]
    threads = []
    close = service.close
    def tracked():
        threads.append(QThread.currentThread())
        close()
    monkeypatch.setattr(service, "close", tracked)
    window.owns_service = True
    window.close()
    drain(app, lambda: window._closed_ready)
    assert len(threads) == 1 and threads[0] != app.thread()


def test_async_success_refreshes_valid_candidate_without_user_acceptance(app, host, monkeypatch):
    create(app, host)
    window, service = host[:2]
    monkeypatch.setattr(candidates, "render_recipe", fake_renderer)
    window.render_button.click()
    drain(app, lambda: window.active_job is not None)
    drain(app, lambda: window.active_job is None and window._task is None)
    assert window.candidates.currentData() is not None
    assert window._viewed_candidate == window.candidates.currentData()
    assert window.feedback_button.isEnabled()
    assert not service.get(window.session["directory"])["feedback"]


def test_unseen_candidate_preview_failure_blocks_feedback_and_clears_pixels(app, host, monkeypatch):
    candidate(app, host, monkeypatch)
    window, service = host[:2]
    def fail(**kwargs):
        raise ValueError("synthetic image read failure")
    monkeypatch.setattr(service, "preview", fail)
    window.load_views()
    idle(app, window)
    assert window._viewed_candidate is None
    assert not window.feedback_button.isEnabled()
    assert window.views["candidate"]._image is None
    window.feedback()
    assert not service.get(window.session["directory"])["feedback"]


@pytest.mark.parametrize("state, checked, version", [
    ("available", False, None), ("available", True, "3.2.6"),
    ("not_found", False, None), ("selected_missing", False, None),
    ("incompatible_version", True, "3.2.8"), ("check_failed", False, None)])
def test_component_status_never_claims_operator_validation(app, host, state, checked, version):
    window = host[0]
    calls = []
    def check(**kwargs):
        calls.append(kwargs)
        return {"status": state, "version_checked": checked, "version": version, "diagnostics": []}
    window.component_checker = check
    window.check_gimp(checked)
    idle(app, window)
    assert calls == [{"selected_path": "host fake GIMP.exe", "probe": checked}]
    assert "尚未验证实际算子" in window.component_note.text()


def test_discovery_suggestion_requires_apply_and_preserves_existing_config(app, host):
    window, service = host[:2]
    old = service.gimp_path
    window.gimp_path.clear()
    found = str(host[2][0] / "GIMP 候选.exe")
    window.component_checker = lambda **kw: {"status": "available", "version_checked": False,
        "selected_path": found, "diagnostics": []}
    window.check_gimp(False)
    idle(app, window)
    assert window.gimp_path.text() == found and service.gimp_path == old
    window.apply_gimp()
    idle(app, window)
    assert service.gimp_path == found
    config = json.loads(window.agent_config.toPlainText())["mcpServers"]["aester-retouch"]
    assert config["args"][-2:] == ["--gimp", found]


def test_empty_apply_does_not_discard_user_choice(app, host):
    window, service = host[:2]
    previous = service.gimp_path
    window.gimp_path.clear()
    window.apply_gimp()
    assert service.gimp_path == previous
    assert window._task is None


def test_mcp_config_contains_argument_array_explicit_roots_and_no_automatic_action(host):
    window, service, roots = host[:3]
    actual = mcp_configuration(window.config)["mcpServers"]["aester-retouch"]
    assert actual["args"][:4] == ["-X", "utf8", "-m", "starstack.finishing_mcp"]
    for flag, root in zip(("--input-root", "--output-root", "--cache-root"), roots):
        assert actual["args"][actual["args"].index(flag) + 1] == str(root)
    assert "--session" not in actual["args"]
    assert "PYTHONPATH" in actual["env"]
    assert "不自动运行 Agent" in window.agent_instructions.toPlainText()
    assert "SKILL.md" in window.agent_instructions.toPlainText()
    assert "尚无安装发行入口" in window.agent_instructions.toPlainText()


def test_standalone_module_help_runs_without_service_or_gui():
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src"),
           "QT_QPA_PLATFORM": "offscreen"}
    result = subprocess.run([sys.executable, "-B", "-X", "utf8", "-m", "aester_desktop.finishing_gui", "--help"],
                            env=env, capture_output=True, text=True, encoding="utf-8", timeout=15,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    assert result.returncode == 0
    assert "--session" in result.stdout and "--input-root" in result.stdout
