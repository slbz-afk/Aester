"""Freeze the reviewed retouch skill only; no photographs, runtime or local state."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import zipfile


def build(root, output):
    source = root / 'skills/aester-retouch'
    files = sorted(p for p in source.rglob('*') if p.is_file())
    if not (source/'SKILL.md').is_file() or not files:
        raise ValueError('Missing skill source')
    if any(p.is_symlink() or p.suffix not in {'.md', '.yaml', '.yml', '.py', '.json'} for p in files):
        raise ValueError('Unexpected file type or link in skill source')
    contents = {p.relative_to(source).as_posix(): p.read_bytes() for p in files}
    for name, body in contents.items():
        if name.endswith('.md'):
            for target in re.findall(r'\]\(([^)]+)\)', body.decode('utf-8')):
                target = target.strip('<>').split('#')[0]
                if not target or '://' in target:
                    continue
                actual = (source/name).parent / target
                if not actual.resolve().is_relative_to(source.resolve()) or not actual.is_file():
                    raise ValueError(f'Broken or external local link: {name}: {target}')
    inventory = {name: hashlib.sha256(body).hexdigest() for name,body in contents.items()}
    digest = hashlib.sha256(json.dumps(inventory,sort_keys=True).encode()).hexdigest()
    output.mkdir(parents=True,exist_ok=True)
    archive = output/f'aester-retouch-skill-{digest[:12]}.zip'
    if not archive.exists():
        with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
            for name,body in contents.items():
                z.writestr('aester-retouch/'+name,body)
    with zipfile.ZipFile(archive) as z:
        if set(z.namelist()) != {'aester-retouch/'+n for n in contents}:
            raise ValueError('Archive contents differ from skill source')
        for name,body in contents.items():
            if z.read('aester-retouch/'+name) != body:
                raise ValueError('Archive skill differs from frozen source')
    receipt = {'scope':'skill only; runtime and teaching photographs excluded',
        'archive':str(archive.resolve()),'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
        'skill_source_digest':digest,'files':inventory,'archive_readback_verified':True,
        'photography_accepted':False,'global_install_performed':False}
    archive.with_suffix('.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),'utf-8')
    return receipt


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-root',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(build(Path(__file__).resolve().parents[1],args.output_root),ensure_ascii=False,indent=2))
