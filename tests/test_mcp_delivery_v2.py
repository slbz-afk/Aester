"""MCP receipt boundaries using real calibrated synthetic image deliveries."""
from pathlib import Path
import copy
import json
import shutil

import pytest
from starstack import workflow as wf
from starstack.mcp_server import WorkspaceTools, AccessDenied
from test_workflow import setup as base_setup
from test_delivery_images import completed


@pytest.fixture
def setup(base_setup):
    # The original pure-workflow fixture puts its mask beside both host roots;
    # an actual MCP client may only use a mask in an admitted root.
    mask = base_setup['inputs'] / 'sky.png'
    shutil.copyfile(base_setup['mask'], mask)
    base_setup['mask'] = mask
    original = base_setup['submit']
    def scoped_submit(*args, **kwargs):
        record = original(*args, **kwargs)
        record['mask'] = {'path': str(mask), 'sha256': wf.core._sha(mask)} if kwargs.get('sky_mask_path') else None
        return record
    base_setup['submit'] = scoped_submit
    return base_setup


def service_for(setup, done):
    return WorkspaceTools(setup['inputs'], Path(done['request']['delivery_root']),
                          setup['engine'], cache_root=setup['output'],
                          star_model_path=setup['model'])


def test_mcp_v2_artifacts_are_only_user_images(setup, completed):
    done, _, _ = completed
    result = service_for(setup, done).workflow_status(done['workflow_dir'])
    assert result['state'] == 'succeeded'
    assert {a['name'] for a in result['artifacts']} == {'final.jpg', 'final_srgb16.tif', 'master_scene_linear.tif'}


@pytest.mark.parametrize('change', ['missing', 'foreign', 'altered', 'relative', 'format'])
def test_mcp_receipts_cannot_bypass_ownership(setup, completed, change):
    done, _, _ = completed
    service = service_for(setup, done)
    assert service.workflow_status(done['workflow_dir'])['state'] == 'succeeded'
    record = copy.deepcopy(done)
    descriptor = record['publication_receipts'][record['delivery_dir']]
    if change == 'missing':
        record['publication_receipts'] = {}
    elif change == 'foreign':
        foreign = Path(record['workflow_dir']).parent / 'other-private.json'
        foreign.write_bytes(Path(descriptor['path']).read_bytes())
        descriptor.update(wf._file(foreign))
    elif change == 'altered':
        Path(descriptor['path']).write_text('{}', encoding='utf8')
    elif change == 'relative':
        descriptor['path'] = 'publication-receipts/something.json'
    else:
        record['request']['delivery_format'] = 'aester-delivery-1'
    (Path(record['workflow_dir']) / 'workflow.json').write_text(json.dumps(record), encoding='utf8')
    with pytest.raises((ValueError, AccessDenied)):
        service.workflow_status(record['workflow_dir'])
