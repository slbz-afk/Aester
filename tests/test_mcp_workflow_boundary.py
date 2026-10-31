"""Independent adapter checks: host paths cannot be replaced through a job record."""
import json
from pathlib import Path
import pytest
from starstack.mcp_server import WorkspaceTools, AccessDenied
from starstack.user_guide import guide_text


@pytest.fixture
def setup(tmp_path):
    inp, out = tmp_path/'input', tmp_path/'output'
    inp.mkdir(); out.mkdir()
    raw = inp/'light.NEF'; raw.write_bytes(b'raw')
    exe = tmp_path/'siril.exe'; exe.write_bytes(b'engine')
    tools = WorkspaceTools(inp, out, exe)
    directory = out/'workflow-example'; directory.mkdir()
    config = {'input_dir':str(inp), 'output_root':str(out), 'siril_path':str(exe),
              'stack_engine_path':None, 'star_model_path':None, 'sky_mask_path':None}
    request = {'input_dir':str(inp), 'input_files':[str(raw)], 'config':dict(config),
               'assets':{'siril_path':{'path':str(exe)}}, 'files':[{'path':str(raw)}, {'path':str(exe)}]}
    record = {'workflow_dir':str(directory), 'config':config, 'request':request, 'artifacts':[], 'checkpoints':{}}
    return tools, directory, record, tmp_path


def save(directory, record):
    (directory/'workflow.json').write_text(json.dumps(record), encoding='utf-8')


def test_host_binding_accepts_exact_config(setup):
    service, directory, record, _ = setup
    save(directory, record)
    assert service._workflow(str(directory))[0] == directory


@pytest.mark.parametrize('field', ['siril_path', 'stack_engine_path', 'star_model_path'])
def test_engine_model_substitution_refused(setup, field):
    service, directory, record, root = setup
    unwanted = root/'unrelated'; unwanted.write_bytes(b'private')
    record['config'][field] = str(unwanted)
    record['request']['config'][field] = str(unwanted)
    save(directory, record)
    with pytest.raises(AccessDenied):
        service._workflow(str(directory))


def test_display_config_cannot_hide_executed_request(setup):
    service, directory, record, root = setup
    record['request']['config']['input_dir'] = str(root)
    save(directory, record)
    with pytest.raises(AccessDenied, match='冻结请求'):
        service._workflow(str(directory))


def test_unlisted_external_frozen_source_refused(setup):
    service, directory, record, root = setup
    record['request']['files'].append({'path':str(root/'secret')})
    save(directory, record)
    with pytest.raises(AccessDenied, match='未授权输入'):
        service._workflow(str(directory))


def test_checkpoint_cannot_read_outside_output(setup):
    service, directory, record, root = setup
    record['checkpoints']={'separation':{'files':[{'path':str(root/'private')}]}}
    save(directory, record)
    with pytest.raises(AccessDenied):
        service._workflow(str(directory))


def test_reference_mask_in_output_accepted_and_escape_rejected(setup):
    service, directory, record, root = setup
    mask = service.output_root/'reference.png'; mask.write_bytes(b'PNG')
    assert service._workflow_mask(str(mask)) == mask
    with pytest.raises(AccessDenied):
        service._workflow_mask('../outside.png')


def test_star_reduction_requires_host_model_before_submission(setup):
    service, *_ = setup
    with pytest.raises(ValueError, match='尚未配置'):
        service.submit_workflow('.', star_reduction=True)


def test_shared_help_matches_packaged_source_and_local_document():
    root = Path(__file__).resolve().parents[1]
    assert (root/'docs/USER_GUIDE.md').read_text(encoding='utf-8') == guide_text('zh_CN')
    assert 'submit_workflow' in guide_text('zh_CN') and 'prepare_reference' in guide_text('zh_CN')
    assert '尚未全部验收' in guide_text('zh_CN')
