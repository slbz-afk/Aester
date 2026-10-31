"""Optional, local RAW compatibility import. Originals are never overwritten.

Adobe DNG Converter is user installed, not bundled. Only an actual unsupported
decoder result triggers automatic conversion; damaged/missing files remain errors.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import time
import uuid

from .core import RAW, FITS, atomic_json
from .process_guard import guarded_process

ADOBE_DOWNLOAD_URL = 'https://helpx.adobe.com/camera-raw/using/adobe-dng-converter.html'


class ConverterRequired(ValueError):
    pass


def _sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def find_converter(selected=None, *, environ=None) -> dict:
    env = os.environ if environ is None else environ
    candidates = []
    if selected:
        candidates.append(Path(selected).expanduser())
    elif env.get('AESTER_DNG_CONVERTER'):
        candidates.append(Path(env['AESTER_DNG_CONVERTER']).expanduser())
    else:
        for base in (env.get('ProgramFiles'), env.get('ProgramFiles(x86)')):
            if base:
                candidates.append(Path(base)/'Adobe/Adobe DNG Converter/Adobe DNG Converter.exe')
        # Development-only fixture; never embed a developer absolute path in a release.
        root = Path(__file__).resolve().parents[2]
        if not env.get('AESTER_INSTALL_ROOT') and (root/'pyproject.toml').is_file():
            candidates.append(root/'.cache/z8-test/dng-portable/{app}/Adobe DNG Converter.exe')
    path = next((p.resolve() for p in candidates if p.is_file()), None)
    return {'available': path is not None, 'path': str(path) if path else None,
            'name': 'Adobe DNG Converter', 'download_url': ADOBE_DOWNLOAD_URL,
            'bundled': False}


def _cancel(check):
    if check and check():
        raise InterruptedError('RAW import cancelled; original photographs are unchanged')


def _no_links(path: Path) -> None:
    for part in (path, *path.parents):
        if part.is_symlink() or getattr(part, 'is_junction', lambda: False)():
            raise ValueError('Import paths must not contain symbolic links or directory junctions')


def _child(root: Path, *parts) -> Path:
    path = root.joinpath(*parts).absolute()
    _no_links(path)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('An import path escapes its cache directory')
    return path


def _probe(path: Path) -> bool:
    import rawpy
    unsupported = tuple(getattr(rawpy, n) for n in ('LibRawFileUnsupportedError', 'LibRawNotImplementedError')
                        if hasattr(rawpy, n))
    try:
        with rawpy.imread(str(path)) as raw:
            # rawpy.imread only parses the header; unsupported HE compression
            # can fail lazily on the first pixel access.
            raw.unpack()
            return True
    except unsupported:
        return False
    except Exception as error:
        raise ValueError(f'RAW cannot be read ({path.name}): {error}') from error


def _safe_cache(root, sources):
    raw = Path(root).expanduser().absolute()
    for part in (raw, *raw.parents):
        if part.is_symlink() or getattr(part, 'is_junction', lambda: False)():
            raise ValueError('The import cache must be a real directory')
    root = raw.resolve()
    if any(root == p.parent or root.is_relative_to(p.parent) for p in sources):
        raise ValueError('The import cache must be outside the original photograph folders')
    imports = root/'raw-imports'
    if any(p.parent == imports or p.parent.is_relative_to(imports) for p in sources):
        raise ValueError('The import cache must not contain original photograph folders')
    if imports.is_symlink() or getattr(imports, 'is_junction', lambda: False)():
        raise ValueError('The import cache must not redirect to another directory')
    return imports


def prepare_import(input_files, cache_root, *, converter_path=None, force_dng=False,
                   cancelled=None, progress=None, allow_converter_discovery=True) -> dict:
    if type(input_files) is not list or not 1 <= len(input_files) <= 300:
        raise ValueError('Select 1 to 300 RAW or FITS photographs')
    if type(force_dng) is not bool:
        raise ValueError('force_dng must be boolean')
    sources = [Path(p).expanduser().resolve(strict=True) for p in input_files]
    if len({os.path.normcase(str(p)) for p in sources}) != len(sources):
        raise ValueError('The import selection contains duplicate photographs')
    if any(not p.is_file() or p.suffix.lower() not in RAW | FITS for p in sources):
        raise ValueError('Only RAW or FITS photographs can be imported')
    extensions = {p.suffix.lower() for p in sources}
    if len(extensions) != 1:
        raise ValueError('Import one camera format at a time')
    originals = [str(p) for p in sources]
    if extensions <= {'.dng'} | FITS:
        return {'state':'ready', 'input_files':originals, 'originals':originals,
                'converted':False, 'probe':'existing_dng_or_fits'}
    root = _safe_cache(cache_root, sources)
    root.mkdir(parents=True, exist_ok=True)
    import rawpy
    decoder = f'unpack-v1/{rawpy.__version__}/{rawpy.libraw_version}'
    identity = []
    unsupported = []
    for index, p in enumerate(sources):
        _cancel(cancelled)
        if progress:
            progress({'phase':'inspect', 'index':index+1, 'total':len(sources), 'name':p.name})
        digest = _sha(p)
        entry = {'path':str(p), 'sha256':digest, 'bytes':p.stat().st_size}
        try:
            with rawpy.imread(str(p)) as header:
                entry['visible_shape'] = [header.sizes.height, header.sizes.width]
        except (rawpy.LibRawFileUnsupportedError, rawpy.LibRawNotImplementedError):
            entry['visible_shape'] = None
        identity.append(entry)
        probe_path = _child(root, 'probes', f'{digest}.json')
        supported = None
        if not force_dng:
            try:
                old = json.loads(probe_path.read_text('utf-8'))
                if old.get('decoder') == decoder and old.get('sha256') == digest and type(old.get('supported')) is bool:
                    supported = old['supported']
            except (OSError, ValueError):
                pass
            if supported is None:
                supported = _probe(p)
                probe_path.parent.mkdir(exist_ok=True)
                atomic_json(probe_path, {'decoder':decoder, 'sha256':digest, 'supported':supported})
        if force_dng or supported is False:
            unsupported.append(str(p))
    if not unsupported:
        return {'state':'ready', 'input_files':originals, 'originals':originals,
                'converted':False, 'probe':'native_decode'}
    converter = (find_converter(converter_path) if converter_path or allow_converter_discovery
                 else {'available': False})
    if not converter['available']:
        raise ConverterRequired('This RAW compression needs Adobe DNG Converter. Install it from Adobe, '
                                'or choose its executable in Settings → Processing backends, then import again. '
                                + ADOBE_DOWNLOAD_URL)
    exe = Path(converter['path'])
    converter_hash = _sha(exe)
    key = hashlib.sha256(json.dumps({'sources':identity,'converter':converter_hash,'recipe':'cfa-lossless-dng14-v1'},
                                   sort_keys=True).encode()).hexdigest()[:24]
    target = _child(root, key)
    manifest = _child(root, key, 'import.json')
    if manifest.is_file():
        try:
            result = validate_import(manifest, cache_root=cache_root, expected_source_paths=originals)
        except (OSError, ValueError):
            result = None  # Rebuild from this explicit selection; leave the old cache intact.
        if result and result['sources'] == identity and result.get('converter_sha256') == converter_hash:
            return {**result, 'state':'ready', 'manifest_path':str(manifest), 'cache_hit':True}
    partial = _child(root, '.partial-'+key+'-'+uuid.uuid4().hex)
    partial.mkdir()
    outputs = []
    for index, (p, source) in enumerate(zip(sources, identity)):
        _cancel(cancelled)
        if progress:
            progress({'phase':'convert', 'index':index+1, 'total':len(sources), 'name':p.name})
        frame = _child(partial, f'{index+1:04d}')
        frame.mkdir()
        log = _child(frame, 'converter.log')
        options = ['-c', '-dng1.4', '-p0', '-d', str(frame), str(p)]
        with log.open('wb') as stream, guarded_process([str(exe), *options], stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0) as child:
            deadline = time.monotonic() + 300
            while child.poll() is None:
                _cancel(cancelled)
                if time.monotonic() > deadline:
                    raise TimeoutError(f'DNG conversion exceeded five minutes: {p.name}')
                time.sleep(.05)
            if child.returncode != 0:
                raise ValueError(f'DNG conversion failed for {p.name} (exit {child.returncode}); see {log}')
        generated = list(frame.glob('*.dng')) + list(frame.glob('*.DNG')) if os.name != 'nt' else list(frame.glob('*.dng'))
        if len(generated) != 1:
            raise ValueError(f'The converter did not produce exactly one DNG for {p.name}')
        dng = generated[0]
        _no_links(dng)
        with rawpy.imread(str(dng)) as decoded:
            if decoded.raw_pattern is None or decoded.raw_image_visible.ndim != 2:
                raise ValueError('Compatibility import requires a mosaic RAW DNG, not rendered RGB')
            if source['visible_shape'] is not None and list(decoded.raw_image_visible.shape) != source['visible_shape']:
                raise ValueError('The converted DNG changed the visible RAW geometry')
        if _sha(p) != source['sha256']:
            raise ValueError('An original photograph changed during conversion')
        outputs.append({'relative_path':str(dng.relative_to(partial)), 'sha256':_sha(dng), 'bytes':dng.stat().st_size})
    _cancel(cancelled)
    if _sha(exe) != converter_hash or any(_sha(p) != item['sha256'] for p,item in zip(sources, identity)):
        raise ValueError('The converter or original photographs changed during import')
    # Never overwrite an earlier cache directory, including a damaged one.
    if target.exists():
        target = _child(root, key+'-'+uuid.uuid4().hex[:8])
    result = {'schema_version':1, 'state':'ready', 'converted':True, 'sources':identity,
              'originals':originals, 'converter':str(exe), 'converter_sha256':converter_hash,
              'recipe':'cfa-lossless-dng14-v1', 'outputs':outputs,
              'input_files':[str(target/item['relative_path']) for item in outputs],
              'reason':'explicit_conversion' if force_dng else 'unsupported_native_compression',
              'unsupported_sources':unsupported, 'cache_hit':False,
              'caution':'A decoded DNG is a compatibility copy, not proof of equality to camera ADC values'}
    atomic_json(partial/'import.json', result)
    partial.rename(target)
    return {**result, 'manifest_path':str(target/'import.json')}


def validate_import(manifest_path, *, original_root=None, cache_root=None, expected_source_paths=None) -> dict:
    """Verify an existing product import; a sidecar is provenance, not a signature."""
    manifest = Path(os.path.abspath(Path(manifest_path).expanduser()))
    if cache_root is not None:
        allowed = Path(os.path.abspath(Path(cache_root).expanduser()))/'raw-imports'
        if manifest.name != 'import.json' or manifest.parent.parent != allowed:
            raise ValueError('The import record is outside the host cache')
        _no_links(manifest)
        _no_links(allowed)
        if manifest.name != 'import.json' or manifest.parent.parent.resolve() != allowed.resolve():
            raise ValueError('The import record is outside the host cache')
    else:
        _no_links(manifest)
    manifest = manifest.resolve(strict=True)
    if not manifest.is_file() or manifest.stat().st_size > 2*1024**2:
        raise ValueError('The import record is invalid or too large')
    result = json.loads(manifest.read_text('utf-8'))
    if not isinstance(result,dict) or result.get('schema_version') != 1 or result.get('recipe') != 'cfa-lossless-dng14-v1':
        raise ValueError('Unknown import record')
    sources, outputs = result.get('sources'), result.get('outputs')
    if not isinstance(sources,list) or not isinstance(outputs,list) or not 1 <= len(sources) == len(outputs) <= 300:
        raise ValueError('Invalid import records')
    checked, originals, pairs = [], [], []
    scope = Path(os.path.abspath(Path(original_root).expanduser())) if original_root else None
    expected = {os.path.normcase(os.path.abspath(Path(p).expanduser())) for p in expected_source_paths} if expected_source_paths is not None else None
    if scope is not None:
        _no_links(scope)
        scope = scope.resolve()
    for source, output in zip(sources, outputs):
        if not isinstance(source,dict) or not isinstance(output,dict):
            raise ValueError('Invalid import records')
        for record in (source,output):
            if not isinstance(record.get('sha256'),str) or not re.fullmatch('[0-9a-f]{64}',record['sha256']):
                raise ValueError('Invalid import digest')
        if not isinstance(source.get('path'),str) or not Path(source['path']).is_absolute():
            raise ValueError('Original paths must be absolute')
        p = Path(os.path.abspath(Path(source['path']).expanduser()))
        if scope is not None and not p.is_relative_to(scope):
            raise ValueError('An imported photograph is outside the input scope')
        if expected is not None and os.path.normcase(str(p)) not in expected:
            raise ValueError('The import record references an unselected original photograph')
        _no_links(p)
        if p.suffix.lower() not in RAW:
            raise ValueError('An import source must be RAW')
        p = p.resolve()
        if scope is not None and not p.is_relative_to(scope):
            raise ValueError('An imported photograph is outside the input scope')
        relative = output.get('relative_path')
        if not isinstance(relative,str) or Path(relative).is_absolute() or '..' in Path(relative).parts:
            raise ValueError('An imported DNG must use a contained relative path')
        dng = _child(manifest.parent, relative).resolve()
        if dng.suffix.lower() != '.dng':
            raise ValueError('An imported DNG escapes its cache directory')
        checked.append(str(dng))
        originals.append(str(p))
        pairs.append((p,dng,source['sha256'],output['sha256']))
    if len(set(map(os.path.normcase, checked))) != len(checked) or len(set(map(os.path.normcase, originals))) != len(originals):
        raise ValueError('Duplicate entries in import record')
    if checked != result.get('input_files') or originals != result.get('originals'):
        raise ValueError('Import paths do not match the recorded source mapping')
    if expected is not None and set(map(os.path.normcase, originals)) != expected:
        raise ValueError('The import record does not match the selected original photographs')
    # All path permissions are checked before opening any referenced source.
    for p,dng,source_hash,output_hash in pairs:
        if not p.is_file() or not dng.is_file() or _sha(p) != source_hash or _sha(dng) != output_hash:
            raise ValueError('An original or its DNG compatibility copy changed')
    return result
