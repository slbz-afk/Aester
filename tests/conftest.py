"""Keep GUI default caches inside each test's temporary filesystem, and guard every Siril launch of the session."""
import pytest

import siril_guard


@pytest.fixture(autouse=True)
def isolated_gui_cache(monkeypatch, tmp_path):
    monkeypatch.setenv('AESTER_CACHE_ROOT', str(tmp_path / 'aester-cache'))


@pytest.fixture
def pre_contract_jobs(monkeypatch):
    """Jobs submitted in the test are admitted as before the EFF-D4 stack-settings contract (efficiency P1c §4.2):
    no ``admission_options.stack_settings``, so ``02_stack`` and R's trial stack keep the earlier scripts byte for byte.

    For modules whose controlled engines model Siril's file contract before EFF-D4 (no settings lines, no
    'Memory per image' / 'allocating data' lines, often no STACKCNT) while they test something else.  Contract jobs
    are covered by tests/test_stack_settings.py and the real-Siril tests."""
    from starstack import stack_settings
    monkeypatch.setattr(stack_settings, "admission_field", lambda *args, **kwargs: None)


@pytest.fixture(autouse=True, scope="session")
def siril_environment_guard():
    """Efficiency P1b W1b: no Siril process started during the tests may use the user's data folder or session bus
    (rules in tests/siril_guard.py)."""
    patch = pytest.MonkeyPatch()
    siril_guard.install(patch)
    try:
        yield
    finally:
        patch.undo()


def pytest_sessionfinish(session, exitstatus):
    if siril_guard.VIOLATIONS:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
        print("\nSIRIL ENVIRONMENT VIOLATIONS (tests/siril_guard.py):\n" + "\n".join(siril_guard.VIOLATIONS))
