"""Independent import contracts: synthetic Bayer files, no real converter/RAW."""
from contextlib import contextmanager
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess

import numpy as np
import pytest
import rawpy
import tifffile

from starstack import raw_import as ri


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_cfa(path, shape=(48, 64)):
    """Independent nonsquare Bayer test image with distinct pixel information."""
    yy, xx = np.indices(shape)
    pixels = ((yy * 97 + xx * 13) % 3900 + 64).astype(np.uint16)
    tifffile.imwrite(path, pixels, photometric=32803, metadata=None, extratags=[
        (50706, 'B', 4, (1, 4, 0, 0), False),
        (33421, 'H', 2, (2, 2), False),
        (33422, 'B', 4, (2, 1, 1, 0), False),
        (50710, 'B', 3, (0, 1, 2), False),
        (50714, 'H', 1, 64, False),
        (50717, 'I', 1, 4095, False),
        (50829, 'I', 4, (0, 0, *shape), False),
    ])
    return pixels


@pytest.fixture
def scene(tmp_path):
    folder = tmp_path / 'read-only originals'
    folder.mkdir()
    inputs = [folder / f'frame {i}.nef' for i in range(2)]
    for p in inputs:
        write_cfa(p)
    exe = tmp_path / 'converter with spaces.exe'
    exe.write_bytes(b'stub; never launched')
    return inputs, exe, tmp_path / 'cache'


def fake_runner(monkeypatch, *, write=None, running=False, exitcode=0):
    state = {'calls': [], 'closed': 0, 'started': 0}
    @contextmanager
    def runner(argv, **options):
        state['calls'].append((argv, options))
        state['started'] += 1
        frame, source = Path(argv[argv.index('-d') + 1]), Path(argv[-1])
        if write:
            write(frame, source)
        elif not running:
            shutil.copyfile(source, frame / (source.stem + '.dng'))
        class Child:
            returncode = exitcode
            def poll(self): return None if running else exitcode
        try:
            yield Child()
        finally:
            state['closed'] += 1
    monkeypatch.setattr(ri, 'guarded_process', runner)
    return state


def conversion(scene, **kw):
    files, exe, cache = scene
    return ri.prepare_import([str(p) for p in files], cache, converter_path=exe, force_dng=True, **kw)


def test_independent_full_resolution_cfa_and_atomic_argv(scene, monkeypatch):
    state = fake_runner(monkeypatch)
    before = [p.read_bytes() for p in scene[0]]
    result = conversion(scene)
    assert result['converted'] and len(state['calls']) == 2
    for original, output in zip(scene[0], result['input_files']):
        with rawpy.imread(str(output)) as image:
            np.testing.assert_array_equal(image.raw_image_visible, tifffile.imread(original))
            assert image.raw_image_visible.shape == (48, 64)
            # LibRaw may name the second green channel slot 3. Compare the
            # declared colors, not its internal channel numbering.
            assert [[chr(image.color_desc[i]) for i in row] for row in image.raw_pattern] == [['B','G'],['G','R']]
    for argv, options in state['calls']:
        assert isinstance(argv, list)
        assert argv[0] == str(scene[1])
        assert argv[1:4] == ['-c', '-dng1.4', '-p0']
        assert '-l' not in argv and '-lossy' not in argv
        assert options.get('shell', False) is False
        assert Path(argv[5]).is_relative_to(scene[2])
        assert argv[-1] in list(map(str, scene[0]))
    assert [p.read_bytes() for p in scene[0]] == before
    assert state['closed'] == 2


def test_lazy_pixel_failure_not_header_success_controls_fallback(scene, monkeypatch):
    real = rawpy.imread
    class Header:
        def __init__(self, path): self.inner = real(path)
        def __enter__(self): return self
        def __exit__(self, *unused): self.inner.close()
        @property
        def sizes(self): return self.inner.sizes
        def unpack(self): raise rawpy.LibRawNotImplementedError('lazy codec unsupported')
    monkeypatch.setattr(rawpy, 'imread', lambda path: Header(path) if str(path).endswith('.nef') else real(path))
    state = fake_runner(monkeypatch)
    result = ri.prepare_import(list(map(str, scene[0])), scene[2], converter_path=scene[1])
    assert result['reason'] == 'unsupported_native_compression'
    assert len(state['calls']) == 2


def test_cancel_running_child_exits_guard_and_does_not_publish(scene, monkeypatch):
    state = fake_runner(monkeypatch, running=True)
    before = [p.read_bytes() for p in scene[0]]
    with pytest.raises(InterruptedError):
        conversion(scene, cancelled=lambda: state['started'] > 0)
    assert state['closed'] == 1
    assert not list(scene[2].rglob('import.json'))
    assert [p.read_bytes() for p in scene[0]] == before


def test_timeout_exits_guard_and_does_not_publish(scene, monkeypatch):
    state = fake_runner(monkeypatch, running=True)
    ticks = iter((0, 301))
    monkeypatch.setattr(ri.time, 'monotonic', lambda: next(ticks))
    with pytest.raises(TimeoutError): conversion(scene)
    assert state['closed'] == 1 and not list(scene[2].rglob('import.json'))


@pytest.mark.parametrize('kind', ['smaller', 'rgb', 'multiple', 'exit'])
def test_invalid_conversion_never_publishes(scene, monkeypatch, kind):
    def write(frame, source):
        destination = frame / 'converted.dng'
        if kind == 'smaller': write_cfa(destination, (24, 32))
        elif kind == 'rgb': tifffile.imwrite(destination, np.zeros((48,64,3), np.uint16), photometric='rgb')
        else:
            shutil.copyfile(source, destination)
            if kind == 'multiple': shutil.copyfile(source, frame/'another.dng')
    fake_runner(monkeypatch, write=write, exitcode=9 if kind == 'exit' else 0)
    with pytest.raises((ValueError, rawpy.LibRawError)):
        conversion(scene)
    assert not list(scene[2].rglob('import.json'))


def test_source_changed_by_external_converter_is_detected_not_published(scene, monkeypatch):
    def write(frame, source):
        shutil.copyfile(source, frame/'converted.dng')
        source.write_bytes(source.read_bytes() + b'changed externally')
    fake_runner(monkeypatch, write=write)
    with pytest.raises(ValueError, match='changed'):
        conversion(scene)
    assert not list(scene[2].rglob('import.json'))


def make_manifest(tmp_path):
    original_root = tmp_path/'sources';original_root.mkdir()
    cache = tmp_path/'cache';target=cache/'raw-imports/test';target.mkdir(parents=True)
    sources, outputs = [], []
    for i in range(2):
        original=original_root/f'{i}.nef';write_cfa(original)
        dng=target/f'{i}.dng';shutil.copyfile(original,dng)
        sources.append({'path':str(original),'sha256':digest(original),'bytes':original.stat().st_size})
        outputs.append({'relative_path':dng.name,'sha256':digest(dng),'bytes':dng.stat().st_size})
    value={'schema_version':1,'recipe':'cfa-lossless-dng14-v1','state':'ready','converted':True,
           'sources':sources,'outputs':outputs,'originals':[x['path'] for x in sources],
           'input_files':[str(target/x['relative_path']) for x in outputs]}
    manifest=target/'import.json';manifest.write_text(json.dumps(value))
    return manifest,original_root,cache,value


def forbid_hash(monkeypatch):
    def forbidden(path): raise AssertionError('An image was read before every path passed admission: '+str(path))
    monkeypatch.setattr(ri,'_sha',forbidden)


@pytest.mark.parametrize('which', ['source', 'output'])
def test_all_paths_are_preflighted_before_any_image_hash(tmp_path, monkeypatch, which):
    manifest,scope,cache,value=make_manifest(tmp_path)
    outside=tmp_path/'outside.nef';write_cfa(outside)
    if which=='source': value['sources'][1]['path']=str(outside)
    else: value['outputs'][1]['relative_path']=str(outside)
    manifest.write_text(json.dumps(value));forbid_hash(monkeypatch)
    with pytest.raises(ValueError): ri.validate_import(manifest, original_root=scope)


def test_host_cache_scope_is_checked_before_manifest_read(tmp_path, monkeypatch):
    manifest,scope,cache,value=make_manifest(tmp_path)
    original_read=Path.read_text
    def protected(path,*args,**kw):
        if path==manifest: raise AssertionError('Untrusted manifest was read outside host cache')
        return original_read(path,*args,**kw)
    monkeypatch.setattr(Path,'read_text',protected)
    with pytest.raises(ValueError):
        ri.validate_import(manifest, original_root=scope, cache_root=tmp_path/'other-cache')


@pytest.mark.parametrize('which', ['manifest', 'source'])
def test_outside_paths_rejected_before_metadata_lookup(tmp_path, monkeypatch, which):
    manifest,scope,cache,value=make_manifest(tmp_path)
    outside=tmp_path/'out-of-scope.nef';write_cfa(outside)
    if which=='source':
        value['sources'][0]['path']=str(outside)
        manifest.write_text(json.dumps(value))
        blocked=outside
        allowed_cache=cache
    else:
        blocked=manifest
        allowed_cache=tmp_path/'another-cache'
    old_lstat=Path.lstat
    def lstat(path,*args,**kwargs):
        if path==blocked:
            raise AssertionError('An out-of-scope path reached a filesystem metadata lookup')
        return old_lstat(path,*args,**kwargs)
    monkeypatch.setattr(Path,'lstat',lstat)
    with pytest.raises(ValueError):
        ri.validate_import(manifest, original_root=scope, cache_root=allowed_cache)


def test_reuse_cannot_hash_unselected_source_from_changed_manifest(scene, monkeypatch, tmp_path):
    fake_runner(monkeypatch)
    result=conversion(scene)
    manifest=Path(result['manifest_path'])
    value=json.loads(manifest.read_text())
    outside=tmp_path/'not-selected.nef';write_cfa(outside)
    value['sources'][0]['path']=str(outside)
    value['sources'][0]['sha256']=digest(outside)
    value['originals'][0]=str(outside)
    manifest.write_text(json.dumps(value))
    original_sha=ri._sha
    def sha(path):
        if path==outside:
            raise AssertionError('Cached provenance allowed access to an unselected source')
        return original_sha(path)
    monkeypatch.setattr(ri,'_sha',sha)
    try:
        retry=conversion(scene)
    except ValueError:
        pass  # Explicitly rejecting a tampered record is also acceptable.
    else:
        assert retry['cache_hit'] is False


@pytest.mark.parametrize('change', ['duplicate-source','duplicate-output','source-type','output-mapping','source-mapping'])
def test_manifest_mapping_rejected_before_hash(tmp_path, monkeypatch, change):
    manifest,scope,cache,value=make_manifest(tmp_path)
    if change=='duplicate-source': value['sources'][1]=dict(value['sources'][0])
    elif change=='duplicate-output': value['outputs'][1]=dict(value['outputs'][0])
    elif change=='source-type':
        old=Path(value['sources'][0]['path']);new=old.with_suffix('.txt');shutil.copyfile(old,new)
        value['sources'][0]['path']=str(new)
    elif change=='output-mapping': value['input_files'].reverse()
    else: value['originals'].reverse()
    manifest.write_text(json.dumps(value));forbid_hash(monkeypatch)
    with pytest.raises(ValueError): ri.validate_import(manifest, original_root=scope)


def test_probe_cache_cannot_write_through_directory_link(scene, tmp_path, monkeypatch):
    outside=tmp_path/'outside-cache';outside.mkdir()
    parent=scene[2]/'raw-imports';parent.mkdir(parents=True)
    link=parent/'probes'
    try:
        link.symlink_to(outside,target_is_directory=True)
    except OSError:
        if os.name!='nt': raise
        status=subprocess.run(['cmd.exe','/d','/c','mklink','/J',str(link),str(outside)],capture_output=True)
        if status.returncode: pytest.skip('No permission to create an isolated test junction')
    monkeypatch.setattr(ri,'_probe',lambda path:True)
    try:
        with pytest.raises(ValueError): ri.prepare_import(list(map(str,scene[0])),scene[2])
        assert list(outside.iterdir())==[]
    finally:
        # Remove only our test link; never recursively remove the target.
        if link.is_symlink(): link.unlink()
        elif link.exists(): link.rmdir()


@pytest.mark.parametrize('nested', [False, True])
def test_import_write_tree_cannot_contain_originals(tmp_path, monkeypatch, nested):
    cache=tmp_path/'cache'
    original_folder=cache/'raw-imports'
    if nested: original_folder=original_folder/'photographs'
    original_folder.mkdir(parents=True)
    source=original_folder/'original.nef';write_cfa(source)
    before=source.read_bytes()
    monkeypatch.setattr(ri,'_probe',lambda p:True)
    with pytest.raises(ValueError): ri.prepare_import([str(source)],cache)
    assert source.read_bytes()==before
    assert not (cache/'raw-imports/probes').exists()


def test_cache_ancestor_is_safe_when_actual_import_tree_is_disjoint(tmp_path, monkeypatch):
    original_folder=tmp_path/'originals';original_folder.mkdir()
    source=original_folder/'original.nef';write_cfa(source)
    before=source.read_bytes()
    monkeypatch.setattr(ri,'_probe',lambda p:True)
    result=ri.prepare_import([str(source)],tmp_path)
    assert result['state']=='ready' and not result['converted']
    assert source.read_bytes()==before and list(original_folder.iterdir())==[source]


@pytest.mark.parametrize('damage', ['dng-modified','dng-deleted','malformed-json'])
def test_damaged_cache_rebuilds_elsewhere_without_overwriting(scene, monkeypatch, damage):
    state=fake_runner(monkeypatch)
    before=[p.read_bytes() for p in scene[0]]
    first=conversion(scene)
    manifest=Path(first['manifest_path']);dng=Path(first['input_files'][0])
    if damage=='dng-modified': dng.write_bytes(b'damaged output')
    elif damage=='dng-deleted': dng.unlink()
    else: manifest.write_text('{ damaged json')
    preserved_manifest=manifest.read_bytes()
    retry=conversion(scene)
    assert retry['cache_hit'] is False and Path(retry['manifest_path']).parent!=manifest.parent
    assert manifest.read_bytes()==preserved_manifest
    if damage=='dng-modified': assert dng.read_bytes()==b'damaged output'
    if damage=='dng-deleted': assert not dng.exists()
    assert len(state['calls'])==4 and [p.read_bytes() for p in scene[0]]==before
