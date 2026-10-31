from pathlib import Path
import importlib.util
import sys

import pytest
from starstack import components,install_ownership,install_lifecycle


@pytest.fixture
def setup(tmp_path,monkeypatch):
    monkeypatch.setitem(sys.modules,'components',components)
    monkeypatch.setitem(sys.modules,'install_ownership',install_ownership)
    monkeypatch.setitem(sys.modules,'install_lifecycle',install_lifecycle)
    spec=importlib.util.spec_from_file_location('installer_preflight_test',Path(__file__).parents[1]/'packaging/aester_p2/installer_worker.py')
    worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
    root=tmp_path/'app';version=root/'versions/v1';version.mkdir(parents=True)
    output=tmp_path/'prepared.json'
    base=['worker','--app-root',str(version),'--component-root',str(tmp_path/'components'),'--cache',str(tmp_path/'downloads'),'--result',str(output)]
    def run(*options):
        monkeypatch.setattr(sys,'argv',base+list(options));return worker.main()
    return worker,root,run,output


def test_prepare_marker_blocks_new_runtime_until_success_or_rollback(setup):
    worker,root,run,output=setup
    assert run('--preflight')==0
    with pytest.raises(install_lifecycle.InstallationBusy):
        with install_lifecycle.runtime_lease(root):pass
    assert run('--rollback')==0
    with install_lifecycle.runtime_lease(root):pass


def test_existing_running_app_blocks_preflight_before_components_created(setup):
    worker,root,run,output=setup
    install_lifecycle.initialize_install_root(root)
    with install_lifecycle.runtime_lease(root):
        with pytest.raises(install_lifecycle.InstallationBusy):run('--preflight')
    assert not (root/'uninstall.pending').exists()
    assert not (root.parent/'components').exists()


def test_retry_prepare_does_not_mistake_own_pending_for_previous_failed_uninstall(setup):
    worker,root,run,output=setup
    run('--preflight');run('--preflight');run('--rollback')
    assert not (root/'uninstall.pending').exists()


def test_rollback_never_clears_original_failed_uninstall_marker(setup):
    worker,root,run,output=setup
    install_lifecycle.initialize_install_root(root)
    with install_lifecycle.exclusive_maintenance(root):pass
    run('--preflight');run('--rollback')
    assert (root/'uninstall.pending').exists()


def test_rollback_refuses_to_clear_marker_if_active_state_changed(setup):
    worker,root,run,output=setup
    (root/'install-state.json').write_text('old')
    run('--preflight')
    (root/'install-state.json').write_text('changed')
    run('--rollback')
    assert (root/'uninstall.pending').exists()


def test_preflight_receipt_write_failure_does_not_leave_new_marker(setup,monkeypatch):
    worker,root,run,output=setup
    def fail(*args,**kwargs):raise PermissionError('receipt locked')
    monkeypatch.setattr(worker.components,'_atomic',fail)
    with pytest.raises(PermissionError):run('--preflight')
    assert not (root/'uninstall.pending').exists()


def test_failed_prepare_then_cancel_restores_old_launch_permission(setup,monkeypatch):
    worker,root,run,output=setup
    def fail(*args,**kwargs):raise RuntimeError('simulated failed component prepare')
    monkeypatch.setattr(worker.components,'install_components',fail)
    run('--preflight')
    assert run()==1
    run('--rollback')
    with install_lifecycle.runtime_lease(root):pass

@pytest.mark.parametrize('concurrent_reader',[False,True])
def test_failed_activation_refences_pending_and_cancel_cannot_reopen(setup,monkeypatch,concurrent_reader):
    import json,subprocess
    worker,root,run,output=setup
    run('--preflight')
    component=root.parent/'components';component.mkdir()
    receipt=component/'receipt.json'
    receipt.write_text(json.dumps({'app_root':str(root/'versions/v1'),'python':str(root/'not-executed.exe'),'version':'v1'}))
    output.write_text(json.dumps({'receipt_path':str(receipt),'receipt_sha256':worker.components._digest(receipt)}))
    monkeypatch.setattr(worker.ownership,'record_application',lambda app:None)
    held=[]
    def fail(*args,**kwargs):
        assert '-B' in args[0]
        if concurrent_reader:
            context=install_lifecycle.runtime_lease(root);context.__enter__();held.append(context)
        raise subprocess.CalledProcessError(1,args[0],stderr=b'failed import')
    monkeypatch.setattr(worker.subprocess,'run',fail)
    try:
        assert run('--activate',str(root/'install-state.json'))==1
        assert (root/'uninstall.pending').exists()
        assert not (root/'install-state.json').exists()
        assert not (component/'current.json').exists()
        run('--rollback')
        assert (root/'uninstall.pending').exists()
    finally:
        for context in held:context.__exit__(None,None,None)
    with pytest.raises(install_lifecycle.InstallationBusy):
        with install_lifecycle.runtime_lease(root):pass
