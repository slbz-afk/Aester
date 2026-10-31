import json
from pathlib import Path
import sys

import pytest
from starstack.onboarding import connection_config, probe_connection, setup_prompt
from starstack.mcp_server import WorkspaceTools, AccessDenied


def locations(tmp_path):
    inp=tmp_path/'素材 中文';inp.mkdir()
    return inp,tmp_path/'作品',tmp_path/'缓存'


def test_config_explicit_paths_no_credentials_no_client_mutation(tmp_path,monkeypatch):
    paths=locations(tmp_path);monkeypatch.setenv('SECRET_API_KEY','never-copy-this')
    before=set(tmp_path.rglob('*'))
    config=connection_config(*paths)
    spec=config['mcpServers']['aester']
    assert spec['args']==['-X','utf8','-m','starstack.mcp_server']
    assert spec['env']['STARSTACK_CACHE_ROOT']==str(paths[2])
    assert spec['env']['STARSTACK_OUTPUT_ROOT']==str(paths[1])
    assert 'never-copy-this' not in json.dumps(config)
    assert set(tmp_path.rglob('*'))==before
    assert '不要扩大素材目录权限' in setup_prompt(config)


@pytest.mark.parametrize('slot',[0,1,2])
def test_blank_scope_not_implicit_working_directory(tmp_path,slot):
    paths=list(locations(tmp_path));paths[slot]=' '
    with pytest.raises(ValueError,match='不能为空'):connection_config(*paths)


def test_overlapping_cache_and_works_rejected(tmp_path):
    inp,out,cache=locations(tmp_path)
    with pytest.raises(ValueError):connection_config(inp,out,out/'cache')
    with pytest.raises(AccessDenied):WorkspaceTools(inp,out,cache_root=out/'cache')


def test_host_optional_cache_preserves_old_mode(tmp_path):
    inp,out,cache=locations(tmp_path)
    old=WorkspaceTools(inp,out)
    new=WorkspaceTools(inp,out,cache_root=cache)
    assert old.output_root==out and old.delivery_root is None
    assert new.output_root==cache and new.delivery_root==out


def test_actual_local_mcp_handshake_no_external_client_claim(tmp_path):
    config=connection_config(*locations(tmp_path))
    result=probe_connection(config,timeout=30)
    assert result['state']=='ready'
    assert result['checked']==['initialize','list_tools','get_help']
    assert result['external_client_connected'] is False
    assert result['tool_count']>=8
