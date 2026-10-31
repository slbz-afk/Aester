"""RAW import MCP bridge: real tiny DNGs, host scope, no external executable."""
import asyncio
from contextlib import contextmanager
import json
from pathlib import Path
import shutil

import numpy as np
from astropy.io import fits
from PIL import Image
import pytest

from starstack import core, raw_import, references, workflow, regions
from starstack.mcp_server import WorkspaceTools, AccessDenied, create_server
from test_raw_import_independent import write_cfa, digest


@pytest.fixture
def sample(tmp_path, monkeypatch):
    inp, products, cache = tmp_path/'photos',tmp_path/'works',tmp_path/'cache'
    inp.mkdir(); products.mkdir(); cache.mkdir()
    sources=[]
    for i in range(3):
        p=inp/f'{i}.nef';write_cfa(p);sources.append(str(p))
    exe=tmp_path/'host-adobe.exe';exe.write_bytes(b'never execute')
    service=WorkspaceTools(inp,products,cache_root=cache,dng_converter_path=exe)
    @contextmanager
    def fake(argv,**kw):
        target=Path(argv[argv.index('-d')+1]);source=Path(argv[-1])
        shutil.copyfile(source,target/(source.stem+'.dng'))
        class Child:
            returncode=0
            def poll(self):return 0
        yield Child()
    monkeypatch.setattr(raw_import,'guarded_process',fake)
    result=service.import_photos(sources,force_dng=True)
    return service,sources,result


def make_reference(service, paths):
    directory=service.output_root/'reference-import-test';directory.mkdir(exist_ok=True)
    linear=directory/'linear.fit';fits.writeto(linear,np.full((3,48,64),.1,np.float32),overwrite=True)
    preview=directory/'preview.jpg';Image.new('RGB',(64,48),(32,32,32)).save(preview)
    mask=directory/'mask.png';Image.new('L',(64,48),255).save(mask)
    record={'reference_dir':str(directory),'state':'succeeded','source_path':paths[0],
            'source_sha256':digest(Path(paths[0])),'linear_path':str(linear),'linear_sha256':digest(linear),
            'preview_path':str(preview),'preview_sha256':digest(preview),
            'mask_template_path':str(mask),'width':64,'height':48}
    (directory/'reference.json').write_text(json.dumps(record),encoding='utf8')
    return record


def test_import_uses_host_cache_and_keeps_originals(sample):
    service,sources,result=sample
    assert result['converted']
    assert Path(result['manifest_path']).is_relative_to(service.output_root/'raw-imports')
    assert not list(service.delivery_root.iterdir())
    assert result['originals']==sources
    assert service.inspect_sources(input_files=result['input_files'])['files']==result['input_files']


def test_batch_validates_one_manifest_once_and_rechecks_next_call(sample,monkeypatch):
    service,_,result=sample
    actual=raw_import.validate_import;calls=[]
    def validate(path,**kw):
        calls.append((path,kw));return actual(path,**kw)
    monkeypatch.setattr(raw_import,'validate_import',validate)
    service.inspect_sources(input_files=result['input_files'])
    assert len(calls)==1
    assert calls[0][1]=={'original_root':service.input_root,'cache_root':service.output_root}
    service.inspect_sources(input_files=result['input_files'])
    assert len(calls)==2


def test_native_without_converter_does_not_fall_back_to_discovery(sample,monkeypatch):
    service,sources,_=sample;service.dng_converter_path=None
    monkeypatch.setattr(raw_import,'find_converter',lambda *a,**kw:pytest.fail('No converter lookup for native sources'))
    result=service.import_photos(sources)
    assert not result['converted'] and result['input_files']==sources


def test_unsupported_without_host_converter_refuses_auto_discovery(sample,monkeypatch):
    service,sources,_=sample;service.dng_converter_path=None
    monkeypatch.setattr(raw_import,'_probe',lambda p:False)
    monkeypatch.setattr(raw_import,'find_converter',lambda *a,**kw:pytest.fail('No fallback to a non-host executable'))
    with pytest.raises(raw_import.ConverterRequired):service.import_photos(sources,force_dng=True)


@pytest.mark.parametrize('kind',['unlisted','arbitrary-cache','wrong-cache','original-outside'])
def test_untrusted_paths_rejected_before_imaging_core(sample,tmp_path,monkeypatch,kind):
    service,_,result=sample
    selected=list(result['input_files'])
    if kind=='unlisted':target=Path(selected[0]).with_name('extra.dng')
    elif kind=='arbitrary-cache':target=service.output_root/'other.dng'
    else:target=tmp_path/('outside.nef' if kind=='original-outside' else 'outside.dng')
    write_cfa(target);selected[0]=str(target)
    monkeypatch.setattr(core,'inspect_sources',lambda *a,**kw:pytest.fail('Unsafe input reached core'))
    with pytest.raises(AccessDenied):service.inspect_sources(input_files=selected)


def test_import_tool_does_not_accept_executable_parameter(sample):
    service,sources,_=sample
    with pytest.raises(TypeError):service.import_photos(sources,converter_path='agent-choice.exe')


def test_import_record_outside_original_scope_fails_before_foreign_hash(sample,tmp_path,monkeypatch):
    service,_,result=sample
    foreign=tmp_path/'private.nef';write_cfa(foreign)
    manifest=Path(result['manifest_path']);record=json.loads(manifest.read_text())
    record['sources'][0]['path']=str(foreign);record['sources'][0]['sha256']=digest(foreign)
    record['originals'][0]=str(foreign);manifest.write_text(json.dumps(record))
    monkeypatch.setattr(raw_import,'_sha',lambda p:pytest.fail('Boundary checks precede all image hashes'))
    with pytest.raises(ValueError):service.inspect_sources(input_files=result['input_files'])


def test_cache_folder_is_never_scanned_as_originals(sample):
    service,_,result=sample
    with pytest.raises(AccessDenied):service.inspect_sources(str(Path(result['input_files'][0]).parent))


def test_reference_region_edit_and_reopen_accept_exact_imports(sample,monkeypatch):
    service,_,result=sample;paths=result['input_files']
    reference=make_reference(service,paths)
    seen=[]
    def prepare(directory,output,siril,**kw):
        seen.append(kw['input_files']);return dict(reference)
    monkeypatch.setattr(references,'prepare_reference',prepare)
    actual=service.prepare_reference(input_files=paths)
    assert seen==[paths] and actual['source_path']==paths[0]
    project=service.create_regions(actual['reference_dir'],paths,scene_mode='sky')
    assert service.get_regions(project['project_path'])['revision']==project['revision']
    drawing=service.draw_regions(actual['reference_dir'],regions_path=project['project_path'],
                                 horizon=[[0,.7],[1,.7]])
    assert service.get_regions(drawing['project_path'])['scene_mode']=='landscape'
    assert Path(drawing['preview']['preview_path']).is_relative_to(service.output_root)


def test_region_record_cannot_substitute_unlisted_cache_file(sample,monkeypatch):
    service,_,result=sample
    ref=make_reference(service,result['input_files'])
    project=service.create_regions(ref['reference_dir'],result['input_files'],scene_mode='sky')
    extra=service.output_root/'private.dng';write_cfa(extra)
    record=json.loads(Path(project['project_path']).read_text())
    record['source_binding']['input_files'][1]['path']=str(extra)
    Path(project['project_path']).write_text(json.dumps(record))
    monkeypatch.setattr(regions,'_sha',lambda p:pytest.fail('Unsafe project reached region hashes'))
    for batch in ('_sha_many','_first_changed'):
        monkeypatch.setattr(regions,batch,lambda paths,*rest:pytest.fail('Unsafe project reached region hashes'))
    with pytest.raises(AccessDenied):service.get_regions(project['project_path'])


def test_workflow_submit_and_saved_status_bridge_imported_inputs(sample,monkeypatch):
    service,_,result=sample;paths=result['input_files']
    host=service.input_root.parent/'host-siril.exe';host.write_bytes(b'never execute');service.siril_path=str(host)
    saved=[]
    def submit(input_dir,output_root,**kwargs):
        assert kwargs['input_files']==paths
        directory=Path(output_root)/'workflow-import-test';directory.mkdir()
        config={'input_dir':input_dir,'output_root':output_root,'siril_path':str(host),
                'stack_engine_path':None,'star_model_path':None,'sky_mask_path':None}
        record={'workflow_dir':str(directory),'state':'queued','config':config,
                'request':{'input_dir':input_dir,'config':dict(config),'input_files':paths,
                           'files':[{'path':p} for p in paths],'assets':{}},
                'artifacts':[],'checkpoints':{}}
        (directory/'workflow.json').write_text(json.dumps(record));saved.append(record)
        return record
    monkeypatch.setattr(workflow,'submit_workflow',submit)
    monkeypatch.setattr(workflow,'workflow_status',lambda directory:dict(saved[0]))
    posted=service.submit_workflow(input_files=paths)
    assert posted['state']=='queued'
    assert service.workflow_status(posted['workflow_dir'])['request']['input_files']==paths
    Path(paths[1]).write_bytes(b'changed cached pixels')
    with pytest.raises(ValueError):service.workflow_status(posted['workflow_dir'])


def test_sdk_import_schema_has_no_executable_or_output_paths(sample):
    server=create_server(sample[0])
    async def inspect():
        tools=await server.list_tools()
        tool=next(t for t in tools if t.name=='import_photos')
        assert set(tool.inputSchema['properties'])=={'input_files','force_dng'}
    asyncio.run(inspect())
