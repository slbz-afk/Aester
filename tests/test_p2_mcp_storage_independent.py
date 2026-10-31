"""Host-level storage separation: real admission/publication, synthetic pixels."""
import copy
import json
from pathlib import Path

import pytest
from starstack import workflow as wf
from starstack.mcp_server import WorkspaceTools, AccessDenied
from test_workflow import setup


@pytest.fixture(autouse=True)
def legacy_delivery_fixture(monkeypatch):
    # These original admission tests use uncalibrated placeholder TIFFs. Keep
    # the v1 fixture contract; real new-format masters have separate tests.
    original = wf.submit_workflow
    monkeypatch.setattr(wf, 'submit_workflow', lambda *a, **kw:
        original(*a, delivery_format='aester-delivery-1', **kw))


def submit(setup, tmp_path):
    cache=tmp_path/'cache';works=tmp_path/'photographs'
    service=WorkspaceTools(setup['inputs'],works,setup['engine'],cache_root=cache)
    job=service.submit_workflow(str(setup['inputs']))
    assert Path(job['workflow_dir']).parent==cache
    wf.run_workflow(job['workflow_dir'])
    result=service.workflow_status(job['workflow_dir'])
    assert result['state']=='succeeded',result.get('error')
    return service,result,cache,works


def test_shared_mcp_new_outputs_use_host_roots(setup,tmp_path):
    service,result,cache,works=submit(setup,tmp_path)
    assert all(Path(item['path']).is_relative_to(works) for item in result['artifacts'])
    assert Path(result['stack_job_dir']).is_relative_to(cache)
    assert result['request']['delivery_root']==str(works)
    assert result['config']['delivery_root']==str(works)
    assert service._product_file(result['artifacts'][0]['path']).is_relative_to(works)


@pytest.mark.parametrize('field',['root','artifact','checkpoint'])
def test_same_host_folder_does_not_authorize_another_workflow_delivery(setup,tmp_path,field):
    service,result,cache,works=submit(setup,tmp_path)
    path=Path(result['workflow_dir'])/'workflow.json'
    unrelated=works/'aester-other'/'delivery-private';unrelated.mkdir(parents=True)
    foreign=unrelated/'secret.jpg';foreign.write_bytes(b'not this work')
    changed=copy.deepcopy(result)
    if field=='root':
        changed['config']['delivery_root']=str(works/'nested')
        changed['request']['config']=copy.deepcopy(changed['config'])
        changed['request']['delivery_root']=str(works/'nested')
    elif field=='artifact':changed['artifacts'][0]['path']=str(foreign)
    else:changed['checkpoints']['stack']['files'][0]['path']=str(foreign)
    path.write_text(json.dumps(changed),encoding='utf8')
    with pytest.raises((ValueError,AccessDenied)):
        service.workflow_status(str(path.parent))


def test_cache_is_not_a_tool_argument(setup,tmp_path):
    import asyncio
    from starstack.mcp_server import create_server
    service=WorkspaceTools(setup['inputs'],tmp_path/'photos',setup['engine'],cache_root=tmp_path/'cache')
    tools=asyncio.run(create_server(service).list_tools())
    submit=next(t for t in tools if t.name=='submit_workflow')
    assert not {'cache_root','delivery_root','output_root'} & submit.inputSchema['properties'].keys()
