from pathlib import Path
import copy
import os

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import numpy as np
import pytest
from astropy.io import fits
from PIL import Image
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication
from PySide6.QtTest import QTest
from starstack import regions, editing_project
from aester_desktop.region_canvas import RegionCanvas
from aester_desktop.region_dialog import RegionDialog


@pytest.fixture
def sample(tmp_path):
    source = tmp_path / 'raw'
    source.mkdir()
    files = []
    for i in range(3):
        path = source / f'{i}.nef'
        path.write_bytes(f'original {i}'.encode())
        files.append(str(path))
    cache = tmp_path / 'cache'
    cache.mkdir()
    linear = cache / 'first.fit'
    fits.writeto(linear, np.zeros((3, 80, 120), dtype=np.float32))
    preview = cache / 'first.png'
    image = np.full((80, 120, 3), 40, np.uint8)
    image[50:] = 5
    Image.fromarray(image).save(preview)
    reference = {'state': 'succeeded', 'source_path': files[0], 'source_sha256': regions._sha(files[0]),
                 'linear_path': str(linear), 'linear_sha256': regions._sha(linear),
                 'preview_path': str(preview), 'preview_sha256': regions._sha(preview), 'width': 120, 'height': 80}
    return reference, files, regions.create_project(reference, files), tmp_path / 'out'


def horizon(project):
    return regions.revise(project, scene_mode='landscape', operations=[{'kind': 'horizon', 'points': [[.1, .6], [.9, .6]]}])


def test_horizon_extrapolates_to_image_edges_and_paint_has_local_scope(sample):
    _, _, project, _ = sample
    project = horizon(project)
    sky, _ = regions.rasterize(project)
    assert np.all(sky[:46] == 255) and np.all(sky[49:] == 0)
    painted = regions.revise(project, operations=project['operations'] + [
        {'kind': 'paint', 'target': 'ground', 'points': [[.5, .25]], 'radius': .05}])
    new, _ = regions.rasterize(painted)
    assert new[20, 60] == 0
    yy, xx = np.where(new != sky)
    assert np.max(np.abs(xx - 60)) <= 5 and np.max(np.abs(yy - 20)) <= 5


def test_pure_sky_explicitly_ignores_old_paint(sample):
    project = regions.revise(horizon(sample[2]), scene_mode='sky')
    assert np.all(regions.rasterize(project)[0] == 255)


def test_rotated_mw_has_soft_top_down_footprint(sample):
    project = regions.revise(sample[2], scene_mode='sky')
    project = regions.revise(project, milkyway={**project['milkyway'], 'ellipse': [.5, .5, .1, .3], 'angle_degrees': 90.})
    _, mask = regions.rasterize(project)
    y, x = np.where(mask > .1)
    assert np.ptp(x) > np.ptp(y)
    assert mask[40, 60] > .99 and mask[0, 0] == 0


def test_mw_confirmation_and_authority_gate(sample):
    _, files, project, out = sample
    project = horizon(project)
    mw = {**project['milkyway'], 'ellipse': [.5, .4, .2, .4]}
    unconfirmed = regions.revise(project, milkyway=mw)
    assert regions.build_masks(unconfirmed, out, input_files=files)['milkyway_mask_path'] is None
    with pytest.raises(ValueError):
        regions.revise(project, milkyway={**mw, 'status': 'present'})
    mw.update(status='present', confirmed=True, reviewer='test human', rationale='visible structure')
    human = regions.revise(project, milkyway=mw)
    result = regions.build_masks(human, out)
    import tifffile
    weight = tifffile.imread(result['milkyway_mask_path'])
    sky = np.asarray(Image.open(result['sky_mask_path']))
    assert weight.dtype == np.float32 and weight.max() == 1
    assert np.all(weight[sky == 0] == 0)
    assert np.any((weight > 0) & (weight < 1))
    agent = regions.revise(human, milkyway={**mw, 'authority': 'agent'})
    assert regions.build_masks(agent, out)['milkyway_mask_path'] is None


def test_immutable_revisions_and_exact_input_order(sample):
    _, files, project, out = sample
    project = horizon(project)
    saved = regions.save_project(project, out)
    again = regions.save_project(project, out)
    assert saved['project_path'] == again['project_path']
    loaded = regions.load_project(saved['project_path'], input_files=files)
    assert loaded['operations'] == project['operations']
    with pytest.raises(ValueError, match='顺序'):
        regions.load_project(saved['project_path'], input_files=list(reversed(files)))
    Path(files[1]).write_bytes(b'changed')
    with pytest.raises(ValueError, match='素材已改变'):
        regions.load_project(saved['project_path'])


def test_modified_geometry_and_coordinate_system_rejected(sample):
    project = copy.deepcopy(sample[2])
    project['source_binding']['reference']['width'] = 121
    with pytest.raises(ValueError, match='尺寸'):
        regions.validate_project(regions._seal(project))
    project = copy.deepcopy(sample[2])
    project['coordinate_system'] = 'bottom-up'
    with pytest.raises(ValueError, match='坐标'):
        regions.validate_project(regions._seal(project))


@pytest.mark.parametrize('operation', [None, [], 2, {'kind': 'paint', 'points': [[float('nan'), 1]], 'target': 'sky', 'radius': .1},
                                      {'kind': 'horizon', 'points': [[.8, .5], [.2, .5]]}])
def test_malformed_operation_is_value_error(sample, operation):
    with pytest.raises(ValueError):
        regions.revise(sample[2], operations=[operation])


def test_generated_mask_tampering_and_raw_output_rejected(sample):
    _, files, project, out = sample
    project = horizon(project)
    result = regions.build_masks(project, out)
    Image.new('L', (120, 80), 255).save(result['sky_mask_path'])
    with pytest.raises(ValueError, match='修改'):
        regions.build_masks(project, out)
    with pytest.raises(ValueError, match='原片'):
        regions.build_masks(project, Path(files[0]).parent / 'output')


def test_failure_does_not_publish_partial_assets(sample, monkeypatch):
    project, out = horizon(sample[2]), sample[3]
    def broken(*args, **kwargs):
        raise OSError('disk full')
    monkeypatch.setattr(Image.Image, 'save', broken)
    with pytest.raises(OSError):
        regions.build_masks(project, out)
    assert not list(out.glob('region-assets-*'))
    assert not list(out.glob('.partial-regions-*'))
    from starstack.locking import is_locked
    assert all(not is_locked(path) for path in out.glob('*.lock'))


def test_local_assistant_never_escapes_band():
    image = np.zeros((100, 160, 3), np.uint8)
    image[:54] = 100
    image[:15] = 255  # Strong unrelated edge is outside band.
    found = regions.suggest_horizon(image, [[0, .5], [1, .5]], band=.07)
    assert all(.43 <= y <= .58 for _, y in found)
    assert abs(np.median([y for _, y in found]) - .54) < .025
    with pytest.raises(ValueError):
        regions.suggest_horizon(np.zeros((1, 5, 3)), [[0, .5], [1, .5]])


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


def test_canvas_letterboxing_and_undo_redo(sample, app):
    reference, _, project, _ = sample
    canvas = RegionCanvas(QImage(reference['preview_path']), project)
    canvas.resize(900, 300)
    rect = canvas.image_rect()
    assert rect.left() > 200
    assert canvas.normalized_point(QPointF(10, 150)) is None
    assert canvas.normalized_point(rect.center()) == pytest.approx([.5, .5])
    canvas.zoom = 3
    canvas.pan = QPointF(30, 20)
    assert canvas.normalized_point(canvas.screen_point([.65, .43])) == pytest.approx([.65, .43])
    canvas.fit_view()
    assert canvas.zoom == 1 and canvas.pan == QPointF()
    canvas.commit(horizon(project))
    signature = canvas.project['revision']
    canvas.undo()
    assert canvas.project['revision'] == project['revision']
    canvas.redo()
    assert canvas.project['revision'] == signature
    canvas.close()


def test_mouse_events_draw_ground_and_horizon_assist_preserves_it(sample, app):
    reference, _, project, _ = sample
    canvas = RegionCanvas(QImage(reference['preview_path']), horizon(project))
    canvas.resize(800, 600)
    canvas.show()
    app.processEvents()
    canvas.set_mode('ground')
    point = canvas.screen_point([.4, .2]).toPoint()
    QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=point)
    assert canvas.project['operations'][-1]['kind'] == 'paint'
    assert regions.rasterize(canvas.project)[0][16, 48] == 0
    canvas.suggestion = [[0., .7], [1., .7]]
    canvas.accept_suggestion()
    assert regions.rasterize(canvas.project)[0][16, 48] == 0
    assert canvas.project['operations'][0]['points'] == [[0., .7], [1., .7]]
    canvas.close()


def test_drawing_is_explicit_instruction_with_undo(sample, app):
    reference, _, project, _ = sample
    dialog = RegionDialog(reference, horizon(project))
    mw = {**dialog.canvas.project['milkyway'], 'ellipse': [.5, .3, .3, .3]}
    dialog.canvas.commit(regions.revise(dialog.canvas.project, milkyway=mw))
    assert not regions.milkyway_active(dialog.canvas.project)
    dialog.canvas.commit(regions.draw_milkyway(dialog.canvas.project, mw['ellipse']))
    assert dialog.canvas.project['milkyway']['status'] == 'present'
    dialog.canvas.undo()
    assert not regions.milkyway_active(dialog.canvas.project)
    dialog.canvas.redo()
    assert regions.milkyway_active(dialog.canvas.project)
    dialog.close()


def test_session_roundtrip_preserves_colors_inputs_and_geometry(sample):
    reference, files, project, out = sample
    project = horizon(project)
    state = {'input_files': files, 'color_adjustments': {'warmth': 12., 'tint': -8., 'exposure_ev': .3},
             'scene_mode': 'landscape', 'trails_enabled': True, 'stars_enabled': False,
             'milkyway_enabled': False, 'milkyway_strength': .6, 'background_strength': .8,
             'auto_strength': True, 'stack_range': False, 'sky_mask_path': '',
             'backend_options': {'channel_order': 'rgb', 'value_range': '0..1', 'layout': 'nhwc', 'stitch': 'blend'}}
    path = out / 'evening.aester.json'
    editing_project.save_project(path, state, reference, project)
    saved = editing_project.load_project(path)
    assert saved['state'] == state
    assert saved['regions']['revision'] == project['revision']
    assert saved['reference'] == reference
    Path(reference['preview_path']).write_bytes(b'changed')
    with pytest.raises(ValueError, match='预览'):
        editing_project.load_project(path)


def test_main_gui_project_save_restore_and_input_invalidation(sample, app, monkeypatch):
    from aester_desktop import gui
    reference, files, project, out = sample
    monkeypatch.setattr(gui.MainWindow, 'check_backend', lambda self: None)
    window = gui.MainWindow(persist_preferences=False)
    calls = []
    monkeypatch.setattr(window, '_run_async', lambda *args: calls.append(args))
    try:
        window.output_dir.setText(str(out))
        window.input_files = files
        window._selection_changed(preview=False)
        window._reference_data = reference
        window._reference_scene = tuple(files)
        window._regions_project = horizon(project)
        window.scene_mode.setCurrentIndex(2)
        window.warmth_slider.setValue(17)
        window.tint_slider.setValue(-9)
        window.exposure_slider.setValue(4)
        window.trails_enabled.setChecked(True)
        window._color_timer.stop()
        path = out / 'session.aester.json'
        window.save_editing_project(str(path))
        name, operation, callback = calls.pop()
        callback(operation())
        window.input_files = []
        window._selection_changed(preview=False)
        window.reset_color_adjustments()
        window._restore_editing_project(editing_project.load_project(path))
        window._color_timer.stop()
        assert window.input_files == files
        assert window.color_adjustments() == {'warmth': 17., 'tint': -9., 'exposure_ev': .4}
        assert window.trails_enabled.isChecked()
        assert window._regions_project['revision'] == horizon(project)['revision']
        window._selection_changed(preview=True)
        window._color_timer.stop()
        assert window._regions_project is None and window._regions_path is None
        assert not window.milkyway_enabled.isChecked()
    finally:
        window._color_timer.stop()
        window.close()
