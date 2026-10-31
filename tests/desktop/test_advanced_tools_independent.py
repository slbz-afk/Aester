"""Independent advanced-form dispatch guards; offscreen, no image model runs."""
import os
import json
from pathlib import Path

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import pytest
from PySide6.QtWidgets import QApplication

from aester_desktop import gui


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def panel(app, monkeypatch, tmp_path):
    monkeypatch.setattr(gui.MainWindow, 'check_backend', lambda self: None)
    window = gui.MainWindow(persist_preferences=False)
    window.timer.stop(); window.output_dir.setText(str(tmp_path / 'output'))
    window._show_tool('milkyway')
    tools = window._extra_tools
    pending = []
    monkeypatch.setattr(window, '_run_async', lambda name, op, done: pending.append((name, op, done)))
    yield window, tools, pending
    window._busy = False; window.job_state = ''; window.close(); app.processEvents()


def set_milkyway_inputs(tools, folder):
    fields = tools.fields['milkyway']; paths = {}
    for key in ('source', 'sky', 'protect', 'milkyway'):
        path = folder / (key + '.tif'); path.write_bytes(('frozen ' + key).encode())
        fields[key].setText(str(path)); paths[key] = path
    fields['confirm'].setChecked(True)
    return fields, paths


@pytest.mark.parametrize('changed', ['source', 'milkyway'])
def test_human_confirmation_does_not_authorize_replacement_at_same_path(panel, monkeypatch, tmp_path, changed):
    from starstack import photographic_job
    _, tools, pending = panel
    fields, paths = set_milkyway_inputs(tools, tmp_path)
    called = []
    monkeypatch.setattr(photographic_job, 'finish_milkyway', lambda *a, **kw: called.append((a, kw)) or {'state': 'succeeded'})
    paths[changed].write_bytes(b'different image after human confirmation')
    tools.run_milkyway()
    if pending:
        with pytest.raises(ValueError): pending.pop()[1]()
    assert called == []


def test_queued_milkyway_call_freezes_values_and_rechecks_confirmed_files(panel, monkeypatch, tmp_path):
    from starstack import photographic_job
    _, tools, pending = panel
    fields, paths = set_milkyway_inputs(tools, tmp_path)
    called = []
    monkeypatch.setattr(photographic_job, 'finish_milkyway', lambda *a, **kw: called.append((a, kw)) or {'state': 'succeeded'})
    fields['strength'].setValue(37)
    window = panel[0]
    window.jpeg_quality_slider.setValue(81)              # the main window's JPEG quality, read at the click
    tools.run_milkyway(); assert len(pending) == 1
    fields['strength'].setValue(91)  # simulate programmatic changes while disabled
    window.jpeg_quality_slider.setValue(64)
    result = pending.pop()[1]()
    assert result['state'] == 'succeeded' and called[0][1]['strength'] == .37
    assert called[0][1]['jpeg_quality'] == 81


def test_file_replacement_after_dispatch_before_worker_is_rejected(panel, monkeypatch, tmp_path):
    from starstack import photographic_job
    _, tools, pending = panel
    _, paths = set_milkyway_inputs(tools, tmp_path)
    called = []
    monkeypatch.setattr(photographic_job, 'finish_milkyway', lambda *a, **kw: called.append((a, kw)))
    tools.run_milkyway(); assert len(pending) == 1
    paths['milkyway'].write_bytes(b'different queued mask')
    with pytest.raises(ValueError): pending.pop()[1]()
    assert not called


def test_switching_job_clears_old_candidate_before_review(panel, monkeypatch, tmp_path):
    from starstack import core
    _, tools, pending = panel
    a, b = tmp_path / 'job-a', tmp_path / 'job-b'; a.mkdir(); b.mkdir()
    fields = tools.fields['trails']; fields['job'].setText(str(a))
    tools._trails_loaded({'state': 'succeeded', 'job_dir': str(a),
                          'candidates': [{'id': '09-8', 'frame': 'frame-a', 'effective_label': 'unknown'}]})
    fields['job'].setText(str(b))
    assert not tools.candidates and tools.trail_candidates.count() == 0
    fields['label'].setCurrentIndex(1)
    called = []
    monkeypatch.setattr(core, 'recompose_transients', lambda *a, **kw: called.append((a, kw)))
    tools.run_trail_review()
    assert not called and not pending


@pytest.mark.parametrize('state', ['cancelled', 'cancel_requested', 'running', 'queued'])
def test_non_success_backend_state_is_not_shown_as_completed(panel, state):
    window, tools, pending = panel
    tools._run('layers', lambda: {'state': state, 'processing_directory': 'unfinished'})
    name, op, callback = pending.pop()
    try:
        callback(op())
    except ValueError as exc:
        window._handle_error(name, str(exc))
    assert 'layers' not in tools.outputs
    assert '处理完成' not in tools.statuses['layers'].text()
    assert '高级操作已完成' not in window.message_label.text()
    assert not window._busy


def test_layer_dispatch_keeps_selected_record_and_output_frozen(panel, monkeypatch, tmp_path):
    from starstack import star_layer_export
    window, tools, pending = panel
    source = tmp_path / 'result.json'; source.write_text('{}', encoding='utf8')
    tools.fields['layers']['separation'].setText(str(source))
    output = window.output_dir.text()
    calls = []
    monkeypatch.setattr(star_layer_export, 'export_star_layers', lambda *a, **kw: calls.append((a, kw)) or {'state': 'succeeded'})
    tools.run_layers(); assert len(pending) == 1
    tools.fields['layers']['separation'].setText('different/result.json')
    window.output_dir.setText(str(tmp_path / 'different-output'))
    pending.pop()[1]()
    assert calls == [((str(source), output), {})]


def test_background_feedback_previews_after_not_before(panel, monkeypatch, tmp_path):
    window, tools, pending = panel
    before, after = tmp_path / 'before.jpg', tmp_path / 'after.jpg'
    before.write_bytes(b'before fixture'); after.write_bytes(b'after fixture')
    previews = []
    monkeypatch.setattr(window, '_load_preview_image', lambda p, **kw: previews.append(Path(p)))
    result = {'state': 'succeeded', 'directory': str(tmp_path),
              'artifacts': [{'path': str(before), 'kind': 'jpeg', 'label': '协调前（共同显示）'},
                            {'path': str(after), 'kind': 'jpeg', 'label': '协调后（共同显示）'}]}
    tools._run('background', lambda: result)
    _, op, callback = pending.pop(); callback(op())
    assert previews == [after]


def test_background_dispatch_freezes_task_samples_and_strength(panel, monkeypatch, tmp_path):
    from starstack import background_job
    _, tools, pending = panel
    fields = tools.fields['background']
    fields['job'].setText('job-a'); fields['samples'].setText('samples-a.png'); fields['strength'].setValue(28)
    calls = []
    monkeypatch.setattr(background_job, 'render_background', lambda *a, **kw: calls.append((a, kw)) or {'state': 'succeeded'})
    tools.run_background()
    fields['job'].setText('job-b'); fields['samples'].setText('samples-b.png'); fields['strength'].setValue(82)
    pending.pop()[1]()
    assert calls == [(('job-a', 'samples-a.png'), {'mode': 'both', 'strength': .28})]


@pytest.mark.parametrize('state,snapshot', [('running', True), ('cancel_requested', True), ('succeeded', False)])
def test_trail_review_refuses_live_or_old_frozen_workflow_before_mutation(panel, monkeypatch, tmp_path, state, snapshot):
    from starstack import core
    _, tools, pending = panel
    parent = tmp_path / 'workflow-parent'; child = parent / 'stack' / 'job-child'; child.mkdir(parents=True)
    delivery = parent / 'delivery-existing'; delivery.mkdir(); final = delivery / 'final.jpg'; final.write_bytes(b'original workflow photo')
    frozen = parent / ('stack_snapshot-fixture.json' if snapshot else 'job.json'); frozen.write_text('{}')
    record = {'state': state, 'checkpoints': {'stack': {'data': {}, 'files': [{'path': str(frozen)}]}}}
    workflow_path = parent / 'workflow.json'; workflow_path.write_text(json.dumps(record), encoding='utf8')
    before = workflow_path.read_bytes()
    fields = tools.fields['trails']; fields['job'].setText(str(child))
    tools._trails_loaded({'state': 'succeeded', 'job_dir': str(child), 'candidates': [{'id': '09-8', 'frame': 'native-09'}]})
    fields['label'].setCurrentIndex(1); fields['decision'].setCurrentIndex(1)
    calls = []
    monkeypatch.setattr(core, 'recompose_transients', lambda *a, **kw: calls.append((a, kw)))
    tools.run_trail_review(); assert len(pending) == 1
    with pytest.raises(ValueError): pending.pop()[1]()
    assert calls == []
    assert final.read_bytes() == b'original workflow photo' and workflow_path.read_bytes() == before
