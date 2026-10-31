"""Explicit welcome consent, cancellation and adapter propagation through the real Qt slots."""
import sys
from unittest.mock import Mock

from PySide6.QtWidgets import QPushButton

from starstack import components
from test_gui_workflow import app, window


def queue_workers(window, monkeypatch):
    queue = []
    monkeypatch.setattr(window, '_run_async', lambda *args: queue.append(args))
    return queue


def components_ready(window, monkeypatch):
    queue = queue_workers(window, monkeypatch)
    experience = window.experience
    experience.show_components()
    queue.pop()[2]({'ready': True, 'components': [{'id': 'base', 'status': 'ready'}]})
    return experience, queue


def test_welcome_explains_optional_source_and_later_does_nothing(window, monkeypatch):
    queue = queue_workers(window, monkeypatch)
    install = Mock(side_effect=AssertionError('no consent'))
    monkeypatch.setattr(components, 'install_nox_support', install)
    experience = window.experience
    experience.show_welcome()
    text = experience._nox_description()
    assert 'MIT' in text and '训练数据' in text and '尚未提供' in text
    assert not queue and not window.stars_enabled.isChecked()
    dialog = window._advanced_dialogs['welcome']
    dialog.findChild(QPushButton, 'welcomeNoxLater').click()
    assert not dialog.isVisible() and not queue
    install.assert_not_called()


def test_welcome_consent_waits_for_status_then_starts_one_install(window, monkeypatch):
    queue = queue_workers(window, monkeypatch)
    install = Mock(return_value={'ready': False, 'error': 'download address not published', 'components': []})
    monkeypatch.setattr(components, 'install_nox_support', install)
    experience = window.experience
    experience.show_welcome()
    window._advanced_dialogs['welcome'].findChild(QPushButton, 'welcomeNoxConsent').click()
    assert len(queue) == 1 and queue[0][0] == 'component-status'
    install.assert_not_called()
    queue[0][2]({'ready': True, 'components': []})
    assert len(queue) == 2 and queue[-1][0] == 'component-install'
    result = queue[-1][1]()
    queue[-1][2](result)
    assert install.call_args.kwargs['consent'] is True
    assert experience.component_install.isEnabled() and not window._busy
    assert '重试' in experience.component_message.text()
    assert not window.stars_enabled.isChecked()


def test_success_updates_complete_adapter_without_enabling_effects(window, monkeypatch):
    experience, queue = components_ready(window, monkeypatch)
    experience.star_support.setChecked(True)
    adapter = {'channel_order': 'bgr', 'value_range': '-1..1', 'layout': 'nhwc', 'stitch': 'crop'}
    result = {'ready': True, 'components': [], 'python': sys.executable,
              'siril': 'prepared/siril-cli.exe', 'model_path': 'prepared/nox_color.onnx',
              'star_backend_options': adapter}
    monkeypatch.setattr(components, 'install_nox_support', lambda **kwargs: result)
    experience._install_components()
    finished = queue[-1][1]()
    queue[-1][2](finished)
    assert window.star_model_path.text() == result['model_path']
    assert window.siril_path.text() == result['siril']
    assert [control.currentData() for control in [window.model_channel_order, window.model_range,
                                                 window.model_layout, window.model_stitch]] == ['bgr', '-1..1', 'nhwc', 'crop']
    assert not window.stars_enabled.isChecked() and not window.trails_enabled.isChecked()
    assert not experience._restart_required and not window._busy


def test_cancel_and_failure_leave_previous_model_adapter_and_runtime_untouched(window, monkeypatch):
    experience, queue = components_ready(window, monkeypatch)
    experience.star_support.setChecked(True)
    window.star_model_path.setText('old/model.onnx')
    before = (window.star_model_path.text(), window.model_channel_order.currentData(),
              window.model_range.currentData(), window.model_layout.currentData(), window.model_stitch.currentData())
    def install(**kwargs):
        assert kwargs['cancelled']()
        raise RuntimeError('cancelled before activation')
    monkeypatch.setattr(components, 'install_nox_support', install)
    experience._install_components()
    assert experience.component_cancel.isEnabled()
    experience.component_cancel.click()
    result = queue[-1][1]()
    queue[-1][2](result)
    after = (window.star_model_path.text(), window.model_channel_order.currentData(),
             window.model_range.currentData(), window.model_layout.currentData(), window.model_stitch.currentData())
    assert after == before and experience._next_python is None
    assert not window._busy and experience.component_install.isEnabled()
    assert not experience.component_cancel.isEnabled() and '取消' in experience.component_message.text()


def test_late_cancel_after_successful_commit_does_not_misreport_cancelled(window, monkeypatch):
    experience, queue = components_ready(window, monkeypatch)
    experience.star_support.setChecked(True)
    committed = {'ready': True, 'components': [], 'python': sys.executable,
                 'model_path': 'committed/nox_color.onnx',
                 'star_backend_options': {'channel_order': 'bgr', 'value_range': '-1..1',
                                          'layout': 'nhwc', 'stitch': 'crop'}}
    monkeypatch.setattr(components, 'install_nox_support', lambda **kwargs: committed)
    experience._install_components()
    result = queue[-1][1]()  # worker already committed; its GUI completion is still queued
    experience.component_cancel.click()
    assert experience._component_cancel.is_set()
    queue[-1][2](result)
    assert window.star_model_path.text() == committed['model_path']
    assert '已就绪' in experience.component_message.text() and '取消' not in experience.component_message.text()
    assert not window._busy and not experience.component_cancel.isEnabled()


def test_chosen_existing_model_is_passed_only_after_explicit_prepare(window, monkeypatch):
    from aester_desktop import experience as module
    experience, queue = components_ready(window, monkeypatch)
    monkeypatch.setattr(module.QFileDialog, 'getOpenFileName', lambda *a: ('existing/model.onnx', ''))
    install = Mock(return_value={'ready': False, 'components': [], 'error': 'fixture'})
    monkeypatch.setattr(components, 'install_nox_support', install)
    experience._pick_model()
    assert not queue and experience.star_support.isChecked()
    install.assert_not_called()
    experience._install_components()
    queue[-1][2](queue[-1][1]())
    assert install.call_args.kwargs['model_source'] == 'existing/model.onnx'
    assert install.call_args.kwargs['consent'] is True
