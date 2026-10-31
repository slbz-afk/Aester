from pathlib import Path
import hashlib
import importlib.util
import json
from unittest.mock import patch

import pytest
from starstack import install_ownership as own


def prepared(tmp_path, name='install'):
    root=tmp_path/name; app=root/'versions/v1'; app.mkdir(parents=True)
    comp=tmp_path/'components'; cache=tmp_path/'downloads'
    comp.mkdir(exist_ok=True);cache.mkdir(exist_ok=True)
    own.register_install(root,comp,cache)
    (app/'a.py').write_text('answer=42')
    own.atomic(app/'PACKAGE_CONTENTS.json',{'version_id':'v1','sha256':{'a.py':own.digest(app/'a.py')}})
    own.record_application(app)
    return root,app,comp,cache


def version(app,comp,name='runtime-v1',files=None):
    stage=comp/('stage-'+name);stage.mkdir()
    for relative,body in (files or {'runtime/python.exe':b'fake runtime','runtime/mod.py':b'x=1','siril/bin/siril-cli.exe':b'fake siril'}).items():
        p=stage/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(body)
    own.seal_version(stage,name,app)
    target=comp/'versions'/name;target.parent.mkdir(exist_ok=True);stage.rename(target)
    own.claim_version(comp,name,app)
    own.atomic(comp/'current.json',{'version':name})
    return target


def test_owned_components_and_generated_bytecode_removed_unknown_files_survive(tmp_path):
    root,app,comp,cache=prepared(tmp_path); target=version(app,comp)
    unknown=target/'user-photo.tif';unknown.write_bytes(b'private')
    bytecode=target/'runtime/__pycache__/mod.cpython-313.pyc';bytecode.parent.mkdir();bytecode.write_bytes(importlib.util.MAGIC_NUMBER+b'cache')
    arbitrary=bytecode.with_name('stranger.cpython-313.pyc');arbitrary.write_bytes(importlib.util.MAGIC_NUMBER+b'user')
    appcache=app/'__pycache__/a.cpython-313.pyc';appcache.parent.mkdir();appcache.write_bytes(importlib.util.MAGIC_NUMBER+b'app-cache')
    result=own.cleanup(root)
    assert result['state']=='complete' and not (target/'runtime/python.exe').exists()
    assert unknown.read_bytes()==b'private' and arbitrary.exists()
    assert not bytecode.exists() and not appcache.exists()
    assert not (app/'a.py').exists()  # Failures must abort before Inno removes retry entry.
    assert not (comp/'current.json').exists()


def test_old_unowned_versions_and_existing_download_are_never_adopted(tmp_path):
    root,app,comp,cache=prepared(tmp_path)
    legacy=comp/'versions/legacy';legacy.mkdir(parents=True);(legacy/'python.exe').write_bytes(b'old')
    archive=cache/'dependency.zip';archive.write_bytes(b'old cache')
    item={'filename':archive.name,'size':archive.stat().st_size,'sha256':own.digest(archive)}
    own.claim_download(cache,item,app,existed_before=True)
    assert not (cache/own.DOWNLOAD_OWNERS).exists()
    result=own.cleanup(root)
    assert (legacy/'python.exe').read_bytes()==b'old' and archive.exists()
    assert any('旧版' in x['reason'] for x in result['preserved'])


def test_registered_download_shared_by_two_installs_survives_first_uninstall(tmp_path):
    one,app,comp,cache=prepared(tmp_path,'one');two,other,_,_=prepared(tmp_path,'two')
    path=cache/'dependency.zip';path.write_bytes(b'cached')
    item={'filename':path.name,'size':path.stat().st_size,'sha256':own.digest(path)}
    own.claim_download(cache,item,app,existed_before=False)
    own.claim_download(cache,item,other,existed_before=True)
    own.cleanup(one);assert path.exists()
    own.cleanup(two);assert not path.exists()


def test_unknown_or_changed_download_survives(tmp_path):
    root,app,comp,cache=prepared(tmp_path);path=cache/'dependency.zip';path.write_bytes(b'cached')
    own.claim_download(cache,{'filename':path.name,'size':6,'sha256':own.digest(path)},app,existed_before=False)
    path.write_bytes(b'user replacement')
    result=own.cleanup(root)
    assert path.read_bytes()==b'user replacement'
    assert any(x['reason']=='下载文件已修改' for x in result['preserved'])


def test_late_delete_failure_keeps_manifests_and_owner_for_retry(tmp_path):
    root,app,comp,cache=prepared(tmp_path);target=version(app,comp)
    original=Path.unlink
    def deny(path,*args,**kwargs):
        if path.name=='siril-cli.exe':raise PermissionError('simulated lock')
        return original(path,*args,**kwargs)
    with patch.object(Path,'unlink',deny),pytest.raises(PermissionError):own.cleanup(root)
    assert (target/own.VERSION_FILES).exists()
    assert list((comp/'ownership').glob('*.json'))
    assert own._read(root/'uninstall-result.json')['state']=='failed'
    assert own.cleanup(root)['state']=='complete'
    assert not (target/'siril/bin/siril-cli.exe').exists()


@pytest.mark.parametrize('relative',['../outside','C:/outside','a\\b','/root','a/../b','a./x','x:ads'])
def test_manifest_relative_paths_reject_escape(tmp_path,relative):
    with pytest.raises(ValueError):own.child(tmp_path,relative)


def test_preflight_rejects_tampered_manifest_before_any_deletion(tmp_path):
    root,app,comp,cache=prepared(tmp_path);target=version(app,comp)
    manifest=own._read(target/own.VERSION_FILES);manifest['files']['../../outside']={'bytes':1,'sha256':'0'*64}
    own.atomic(target/own.VERSION_FILES,manifest)
    with pytest.raises(ValueError):own.cleanup(root)
    assert (target/'runtime/python.exe').read_bytes()==b'fake runtime'


def test_historical_versions_and_repair_all_registered(tmp_path):
    root,app,comp,cache=prepared(tmp_path)
    first=version(app,comp,'first');repair=version(app,comp,'repair-v1')
    nextapp=root/'versions/v2';nextapp.mkdir();(nextapp/'a.py').write_text('new')
    own.atomic(nextapp/'PACKAGE_CONTENTS.json',{'version_id':'v2','sha256':{'a.py':own.digest(nextapp/'a.py')}})
    own.record_application(nextapp);nextversion=version(nextapp,comp,'second')
    own.cleanup(root)
    assert not first.exists() and not repair.exists() and not nextversion.exists()


def test_shared_component_owner_keeps_runtime_and_current(tmp_path):
    root,app,comp,cache=prepared(tmp_path);target=version(app,comp)
    otherid='a'*32
    own.atomic(comp/'ownership'/(otherid+'.json'),{'schema_version':1,'installation_id':otherid,'install_root':str(tmp_path/'other'),
        'versions':{'runtime-v1':{'manifest_sha256':own.digest(target/own.VERSION_FILES)}}})
    own.cleanup(root)
    assert (target/'runtime/python.exe').exists() and (comp/'current.json').exists()


def test_corrupt_shared_owner_blocks_cleanup(tmp_path):
    root,app,comp,cache=prepared(tmp_path);target=version(app,comp)
    (comp/'ownership/bad.json').write_text('{}')
    with pytest.raises(ValueError):own.cleanup(root)
    assert (target/'runtime/python.exe').exists()


def test_missing_owned_file_is_idempotent(tmp_path):
    root,app,comp,cache=prepared(tmp_path);target=version(app,comp)
    (target/'runtime/python.exe').unlink()
    assert own.cleanup(root)['state']=='complete'


def test_metadata_delete_failure_retains_installation_bound_retry_manifest(tmp_path):
    root,app,comp,cache=prepared(tmp_path);target=version(app,comp)
    record=next((comp/'ownership').glob('*.json')); original=Path.unlink
    def deny(path,*args,**kwargs):
        if path==record:raise PermissionError('owner record locked')
        return original(path,*args,**kwargs)
    with patch.object(Path,'unlink',deny),pytest.raises(PermissionError):own.cleanup(root)
    assert not (target/own.VERSION_FILES).exists() and record.exists()
    assert own.cleanup(root)['state']=='complete'
    assert not target.exists()
    assert own.cleanup(root)['state']=='complete'


def test_junction_or_symlink_never_traversed(tmp_path):
    root,app,comp,cache=prepared(tmp_path);target=version(app,comp)
    outside=tmp_path/'outside';outside.mkdir();(outside/'photo.tif').write_bytes(b'keep')
    try:(target/'link').symlink_to(outside,target_is_directory=True)
    except OSError:pytest.skip('symlink privilege not available')
    with pytest.raises(ValueError):own.cleanup(root)
    assert (outside/'photo.tif').read_bytes()==b'keep'
    assert (target/'runtime/python.exe').exists()
