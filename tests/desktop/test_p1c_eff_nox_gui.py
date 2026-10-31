"""EFF-NOX (P1c K1), KN-1: a GUI submission with star tidying freezes the explicit 8 ORT threads.

The GUI names no thread count; the real workflow.submit_workflow admission (not replaced here) supplies it.
"""
import pytest

from starstack import starless_runtime as rt
from test_gui_workflow import app, window  # noqa: F401  (Qt fixtures)
from test_p1_gui_routing_independent import scene


@pytest.mark.parametrize("stars", [True, False])
def test_gui_submission_gets_8_only_with_nox(window, monkeypatch, tmp_path, stars):
    scene(window, monkeypatch, tmp_path)
    window.stars_enabled.setChecked(stars)
    captured = []
    monkeypatch.setattr(window, "_run_async", lambda *a: captured.append(a))
    window.start_processing()
    try:
        assert len(captured) == 1
        admitted = captured[0][1]()              # real workflow.submit_workflow
        assert admitted["request"]["stars_enabled"] is stars
        assert admitted["request"]["backend_options"]["intra_op_threads"] == (rt.NOX_INTRA_OP_THREADS if stars else None)
        assert admitted["config"]["backend_options"] == admitted["request"]["backend_options"]
    finally:
        window._busy = False
