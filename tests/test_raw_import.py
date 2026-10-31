from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import pytest
import rawpy
from test_raw_audit import dng_fixture
from starstack import raw_import as ri


@pytest.fixture
def sources(tmp_path):
    folder = tmp_path/'originals';folder.mkdir()
    paths = [folder/f'frame{i}.nef' for i in range(2)]
    for p in paths:
        dng_fixture(p)
    return paths


@pytest.fixture
def converter(tmp_path, monkeypatch):
    executable = tmp_path/'Adobe DNG Converter.exe'
    executable.write_bytes(b'fake executable, never launched')
    calls = []
    @contextmanager
    def fake(args, **kw):
        assert args[1:4] == ['-c','-dng1.4','-p0']
        assert kw['stdin'] is not None
        target, original = Path(args[-2]), Path(args[-1])
        calls.append(list(args))
        shutil.copyfile(original, target/(original.stem+'.dng'))
        class Process:
            returncode = 0
            def poll(self): return 0
        yield Process()
    monkeypatch.setattr(ri, 'guarded_process', fake)
    monkeypatch.setattr(ri, '_probe', lambda p: False)
    return executable, calls


def test_probe_forces_lazy_unpack(monkeypatch,tmp_path):
    class HeaderOnly:
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def unpack(self): raise rawpy.LibRawFileUnsupportedError('unsupported compression')
    monkeypatch.setattr(rawpy,'imread',lambda p:HeaderOnly())
    assert ri._probe(tmp_path/'camera.nef') is False


def test_probe_does_not_convert_corruption_or_permission_errors(monkeypatch,tmp_path):
    monkeypatch.setattr(rawpy,'imread',lambda p: (_ for _ in ()).throw(PermissionError('locked')))
    with pytest.raises(ValueError,match='locked'):
        ri._probe(tmp_path/'camera.nef')


def test_existing_dng_does_not_decode_or_create_cache(tmp_path,monkeypatch):
    source=tmp_path/'frame.dng';source.write_bytes(b'header checking belongs to preview/job')
    monkeypatch.setattr(ri,'_probe',lambda p:pytest.fail('DNG should not be re-decoded at import'))
    result=ri.prepare_import([str(source)],tmp_path/'cache')
    assert result['input_files']==[str(source)] and not result['converted']
    assert not (tmp_path/'cache').exists()


def test_native_supported_stays_in_original_location(sources,tmp_path):
    result=ri.prepare_import([str(p) for p in sources],tmp_path/'cache')
    assert not result['converted'] and result['input_files']==list(map(str,sources))


def test_conversion_and_reuse_keep_originals_and_mapping(sources,converter,tmp_path):
    exe,calls=converter
    original=[p.read_bytes() for p in sources]
    result=ri.prepare_import(list(map(str,sources)),tmp_path/'cache',converter_path=exe)
    assert result['converted'] and not result['cache_hit'] and len(calls)==2
    assert result['originals']==list(map(str,sources))
    assert [p.read_bytes() for p in sources]==original
    verified=ri.validate_import(result['manifest_path'],original_root=sources[0].parent)
    assert verified['input_files']==result['input_files']
    second=ri.prepare_import(list(map(str,sources)),tmp_path/'cache',converter_path=exe)
    assert second['cache_hit'] and len(calls)==2


def test_same_basename_in_different_folders_cannot_overwrite(sources,converter,tmp_path):
    other=tmp_path/'other';other.mkdir()
    second=other/sources[0].name;shutil.copyfile(sources[1],second)
    result=ri.prepare_import([str(sources[0]),str(second)],tmp_path/'cache',converter_path=converter[0])
    assert len(set(result['input_files']))==2
    assert all(Path(p).is_file() for p in result['input_files'])


@pytest.mark.parametrize('mutate', ['source','dng','escape','scope'])
def test_tampered_import_and_outside_sources_are_rejected(sources,converter,tmp_path,mutate):
    result=ri.prepare_import(list(map(str,sources)),tmp_path/'cache',converter_path=converter[0])
    path=Path(result['manifest_path']);record=json.loads(path.read_text('utf-8'))
    scope=sources[0].parent
    if mutate=='source': sources[0].write_bytes(b'changed')
    elif mutate=='dng': Path(result['input_files'][0]).write_bytes(b'changed')
    elif mutate=='escape':
        record['outputs'][0]['relative_path']=str(sources[0])
        path.write_text(json.dumps(record),'utf-8')
    else: scope=tmp_path/'elsewhere'
    with pytest.raises(ValueError): ri.validate_import(path,original_root=scope)


def test_cache_inside_originals_is_rejected(sources,tmp_path):
    with pytest.raises(ValueError,match='outside'):
        ri.prepare_import(list(map(str,sources)),sources[0].parent/'subdir')
    assert not (sources[0].parent/'subdir').exists()


def test_cancel_mid_conversion_never_publishes(sources,converter,tmp_path):
    checks=[]
    def progress(event): checks.append(event)
    def cancel(): return len(converter[1])>0
    with pytest.raises(InterruptedError):
        ri.prepare_import(list(map(str,sources)),tmp_path/'cache',converter_path=converter[0],
                          progress=progress,cancelled=cancel)
    assert not list((tmp_path/'cache/raw-imports').glob('*/import.json'))
    assert all(p.exists() for p in sources)


def test_missing_converter_has_actionable_error(sources,tmp_path,monkeypatch):
    monkeypatch.setattr(ri,'_probe',lambda p:False)
    with pytest.raises(ri.ConverterRequired,match='Adobe DNG Converter'):
        ri.prepare_import(list(map(str,sources)),tmp_path/'cache',converter_path=tmp_path/'missing.exe')
