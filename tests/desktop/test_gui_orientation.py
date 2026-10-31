"""Sequence orientation admission, state restoration and Qt routing contracts."""
from pathlib import Path

import pytest
from PIL import Image

from starstack import references, workflow
from aester_desktop import gui
from test_gui_aester_p0 import app, window
from test_gui_language_routing import make_window


def setup_scene(window, tmp_path, monkeypatch, *, rotation=90, capture=90):
    source = tmp_path / 'photos'
    source.mkdir()
    paths = []
    for i in range(3):
        path = source / f'{i}.dng'
        path.write_bytes(b'original unchanged')
        paths.append(str(path))
    window.input_files = paths
    window._selection_changed()
    window._color_timer.stop()
    window._reference_pending = False
    window.cache_dir.setText(str(tmp_path / 'cache'))
    record = make_record(tmp_path, rotation, capture)
    window._reference_data = record
    window._reference_scene = tuple(paths)
    window._image_rotation = rotation
    pending = []
    monkeypatch.setattr(window, '_run_async', lambda *args: pending.append(args))
    monkeypatch.setattr(window, '_load_preview_image', lambda *args, **kw: None)
    window._update_controls()
    return paths, record, pending


def make_record(tmp_path, rotation, capture=90):
    directory = tmp_path / f'ref-{rotation}'
    directory.mkdir(exist_ok=True)
    size = (24, 32) if rotation % 180 else (32, 24)
    preview, mask = directory / 'preview.png', directory / 'sky.png'
    Image.new('RGB', size).save(preview)
    Image.new('L', size, 255).save(mask)
    return {'reference_dir': str(directory), 'preview_path': str(preview), 'mask_template_path': str(mask),
            'width': size[0], 'height': size[1], 'preview_width': size[0], 'preview_height': size[1],
            'image_rotation': rotation, 'capture_rotation': capture}


def finish(request):
    request[2](request[1]())


def test_new_first_frame_uses_auto_orientation_and_adopts_reported_portrait(window, tmp_path, monkeypatch):
    _, record, pending = setup_scene(window, tmp_path, monkeypatch)
    window._image_rotation = None
    window._reference_data = None
    window._reference_scene = ()
    calls = []
    monkeypatch.setattr(references, 'prepare_reference', lambda *a, **kw: calls.append(kw) or record)
    window.prepare_reference()
    finish(pending.pop())
    assert calls[0].get('image_rotation') is None
    assert window._image_rotation == 90 and window._orientation_ready()
    assert window.rotate_left_button.isEnabled()


@pytest.mark.parametrize('delta,expected', [(-90, 0), (90, 180)])
def test_rotate_buttons_clear_all_masks_and_route_then_render(window, tmp_path, monkeypatch, delta, expected):
    paths, _, pending = setup_scene(window, tmp_path, monkeypatch)
    window._regions_project = {'scene_mode': 'landscape', 'operations': [{'kind': 'horizon'}],
                               'milkyway': {'status': 'unknown'}}
    window._regions_path = 'old-regions.json'
    window.sky_mask.setText('old-external.png')
    calls = []
    rotated = make_record(tmp_path, expected)
    monkeypatch.setattr(references, 'rotate_reference', lambda path, **kw:
                        calls.append(('rotate', path, kw)) or rotated, raising=False)
    monkeypatch.setattr(references, 'render_reference', lambda path, **kw:
                        calls.append(('render', path, kw)) or rotated)
    (window.rotate_left_button if delta < 0 else window.rotate_right_button).click()
    assert window._image_rotation == expected
    assert not window._regions_project and not window._regions_path and not window.sky_mask.text()
    assert not window.milkyway_enabled.isChecked()
    assert not window.start_button.isEnabled() and not window.regions_button.isEnabled()
    assert not window.rotate_right_button.isEnabled()
    window.start_processing()
    assert len(pending) == 1
    finish(pending.pop())
    assert calls[0][2] == {'image_rotation': expected}
    assert calls[1][1] == rotated['reference_dir']
    assert window._reference_data['image_rotation'] == expected
    assert window.start_button.isEnabled() and window.regions_button.isEnabled()
    assert '重新标记' in window.message_label.text()
    assert all(Path(path).read_bytes() == b'original unchanged' for path in paths)


def test_reset_returns_capture_orientation_without_redecoding(window, tmp_path, monkeypatch):
    _, _, pending = setup_scene(window, tmp_path, monkeypatch, rotation=180, capture=90)
    calls = []
    record = make_record(tmp_path, 90)
    monkeypatch.setattr(references, 'rotate_reference', lambda path, **kw: calls.append(kw) or record, raising=False)
    monkeypatch.setattr(references, 'render_reference', lambda *a, **kw: record)
    monkeypatch.setattr(references, 'prepare_reference', lambda *a, **kw: pytest.fail('Unexpected RAW decode'))
    window.restore_orientation_button.click()
    finish(pending.pop())
    assert calls == [{'image_rotation': 90}]
    window.restore_orientation_button.click()
    assert not pending


def test_legacy_reset_rereads_capture_metadata(window, tmp_path, monkeypatch):
    _, record, pending = setup_scene(window, tmp_path, monkeypatch, rotation=0)
    record.pop('capture_rotation')
    new = make_record(tmp_path, 270, 270)
    calls = []
    monkeypatch.setattr(references, 'prepare_reference', lambda *a, **kw: calls.append(kw) or new)
    window.restore_capture_orientation()
    finish(pending.pop())
    assert calls[0].get('image_rotation') is None and window._image_rotation == 270


def test_append_or_remove_later_frame_keeps_rotation_but_new_first_resets(window, tmp_path, monkeypatch):
    paths, _, _ = setup_scene(window, tmp_path, monkeypatch, rotation=270)
    additional = tmp_path / 'photos' / 'extra.dng'
    additional.write_bytes(b'new')
    window.input_files.append(str(additional))
    window._selection_changed()
    window._color_timer.stop()
    assert window._image_rotation == 270
    window.input_files.pop()
    window._selection_changed()
    window._color_timer.stop()
    assert window._image_rotation == 270
    window.input_files = paths[1:]
    window._selection_changed()
    window._color_timer.stop()
    assert window._image_rotation is None


def test_rotating_late_callback_cannot_resurrect_cleared_selection(window, tmp_path, monkeypatch):
    _, _, pending = setup_scene(window, tmp_path, monkeypatch)
    rotated = make_record(tmp_path, 180)
    monkeypatch.setattr(references, 'rotate_reference', lambda *a, **kw: rotated, raising=False)
    monkeypatch.setattr(references, 'render_reference', lambda *a, **kw: rotated)
    shown = []
    monkeypatch.setattr(window, 'show_reference', lambda: shown.append(True))
    window.rotate_input_images(90)
    request = pending.pop()
    window.clear_input_files()
    finish(request)
    assert not shown and window._reference_data is None and window._image_rotation is None


def test_failed_rotation_keeps_previous_direction_visible_and_retryable(window, tmp_path, monkeypatch):
    _, _, pending = setup_scene(window, tmp_path, monkeypatch)
    window.rotate_input_images(90)
    assert not window.reference_view_button.isEnabled()
    window._handle_error(pending[-1][0], 'disk full fixture')
    assert window._image_rotation == 90 and window._reference_data['image_rotation'] == 90
    assert window.rotate_right_button.isEnabled() and window._orientation_ready()
    assert '之前的方向' in window.message_label.text()


def test_submission_uses_same_explicit_rotation_as_canvas(window, tmp_path, monkeypatch):
    _, _, pending = setup_scene(window, tmp_path, monkeypatch, rotation=270)
    calls = []
    monkeypatch.setattr(workflow, 'submit_workflow', lambda *a, **kw: calls.append(kw) or {})
    window.start_processing()
    pending.pop()[1]()
    assert calls[0]['image_rotation'] == 270
    assert calls[0]['sky_mask_path'] == window._reference_data['mask_template_path']
    window._busy = False


@pytest.mark.parametrize('old', [False, True])
def test_editing_project_reopens_with_saved_rotation_or_legacy_zero(window, tmp_path, monkeypatch, old):
    _, record, _ = setup_scene(window, tmp_path, monkeypatch, rotation=270)
    state = window._editing_state()
    assert state['image_rotation'] == 270
    if old:
        state.pop('image_rotation')
        record.pop('image_rotation')
        record.pop('capture_rotation')
    window._restore_editing_project({'state': state, 'reference': record, 'regions': None, 'regions_path': None})
    window._color_timer.stop()
    window._reference_pending = False
    assert window._image_rotation == (0 if old else 270)
    assert window._orientation_ready()


@pytest.mark.parametrize('active', ['queued', 'running', 'cancel_requested'])
def test_active_workflow_blocks_orientation_mutation(window, tmp_path, monkeypatch, active):
    _, _, pending = setup_scene(window, tmp_path, monkeypatch)
    window.job_state = active
    window.rotate_input_images(90)
    assert window._image_rotation == 90 and not pending
    window.job_state = ''


def test_workflow_resume_recovers_rotation(window, tmp_path, monkeypatch):
    paths, _, _ = setup_scene(window, tmp_path, monkeypatch)
    window._workflow_submitted({'workflow_dir': str(tmp_path), 'state': 'queued',
                               'config': {'input_files': paths, 'image_rotation': 270}})
    assert window._image_rotation == 270
    window.timer.stop()
    window.job_state = ''


@pytest.mark.parametrize('language', ['zh_CN', 'en'])
@pytest.mark.parametrize('theme', ['dark', 'light'])
def test_orientation_controls_fit_minimum_window_without_overlap(make_window, app, language, theme):
    window, _ = make_window(language)
    window.set_theme(theme)
    window.resize(960, 680)
    app.processEvents()
    buttons = [window.rotate_left_button, window.rotate_right_button, window.restore_orientation_button]
    assert all(button.width() >= button.sizeHint().width() for button in buttons)
    assert buttons[0].geometry().right() < buttons[1].geometry().left()
    assert buttons[1].geometry().right() < buttons[2].geometry().left()
    assert buttons[-1].geometry().right() <= window.preview.geometry().right()
