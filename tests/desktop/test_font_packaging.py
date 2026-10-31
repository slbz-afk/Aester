"""Bundled OFL fonts reach the installer payload and the wheel, and only unmodified."""
import hashlib
import json
from pathlib import Path
import shutil
import tomllib

import pytest

from starstack import components
from test_finite_installer_distribution import builder_case

ROOT = Path(__file__).resolve().parents[2]
FONTS = ROOT / 'src/aester_desktop/assets/fonts'
REAL_RECORD = json.loads((FONTS / 'FONTS.json').read_text(encoding='utf-8'))
FONT_NAMES = ('NotoSansSC-Regular.otf', 'NotoSansSC-Medium.otf', 'IBMPlexSans-Regular.ttf', 'IBMPlexSans-Medium.ttf',
              'IBMPlexSans-SemiBold.ttf', 'IBMPlexMono-Regular.ttf', 'IBMPlexMono-Medium.ttf',
              'LXGWWenKaiLite-Regular.ttf')


def _synthetic_fonts(source, names=('Test-Regular.ttf', 'OFL-Test.txt')):
    """A tiny recorded font directory, so refusal tests need not copy 30 MB."""
    folder = source / 'src/aester_desktop/assets/fonts'
    folder.mkdir(parents=True)
    entries = []
    for name in names:
        data = ('synthetic ' + name).encode()
        (folder / name).write_bytes(data)
        entries.append({'file': name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(),
                        'source': 'test', 'licence': 'SIL OFL 1.1'})
    (folder / 'FONTS.json').write_text(json.dumps({'files': entries}), encoding='utf-8')
    return folder


def test_repository_fonts_match_their_record_and_carry_licences():
    recorded = {entry['file'] for entry in REAL_RECORD['files']}
    assert set(FONT_NAMES) <= recorded
    assert {'OFL-IBMPlex.txt', 'OFL-LXGWWenKai.txt', 'OFL-NotoSansCJK.txt'} <= recorded
    assert {entry['licence'] for entry in REAL_RECORD['files']} == {'SIL OFL 1.1'}
    from aester_desktop import fonts
    assert set(fonts.FONT_FILES) <= recorded


def test_real_font_directory_passes_the_builder_check(builder_case):
    files = builder_case.builder._font_files(ROOT)
    relative = {rel.as_posix() for _, rel in files}
    assert relative == {'src/aester_desktop/assets/fonts/' + entry['file'] for entry in REAL_RECORD['files']} | {
        'src/aester_desktop/assets/fonts/FONTS.json'}


def test_staged_application_and_package_contents_include_the_real_fonts(builder_case):
    c = builder_case
    shutil.copytree(FONTS, c.builder.ROOT / 'src/aester_desktop/assets/fonts')
    receipt = c.builder.build(c.output, c.compiler, c.model)
    app = Path(receipt['stage']) / 'application'
    contents = json.loads((app / 'PACKAGE_CONTENTS.json').read_text())['sha256']
    for entry in REAL_RECORD['files']:
        key = 'src/aester_desktop/assets/fonts/' + entry['file']
        staged = app / key
        assert staged.is_file() and staged.stat().st_size == entry['bytes']
        assert contents[key] == entry['sha256'] == components._digest(staged)
    assert contents['src/aester_desktop/assets/fonts/FONTS.json'] == components._digest(FONTS / 'FONTS.json')


def test_fixture_without_fonts_still_builds(builder_case):
    c = builder_case
    receipt = c.builder.build(c.output, c.compiler, c.model)
    contents = json.loads((Path(receipt['stage']) / 'application/PACKAGE_CONTENTS.json').read_text())['sha256']
    assert not [key for key in contents if '/fonts/' in key]


def test_font_module_without_font_directory_is_refused(builder_case):
    c = builder_case
    (c.builder.ROOT / 'src/aester_desktop/fonts.py').write_text('# needs bundled fonts\n')
    with pytest.raises(ValueError, match='FONTS.json'):
        c.builder.build(c.output, c.compiler, c.model)
    assert not c.compiled and not list(c.output.glob('build-*'))


@pytest.mark.parametrize('damage', ['missing', 'altered', 'resized', 'unrecorded', 'not_ofl', 'no_licence_text'])
def test_missing_or_altered_font_is_refused_before_staging(builder_case, damage):
    c = builder_case
    names = ('Test-Regular.ttf',) if damage == 'no_licence_text' else ('Test-Regular.ttf', 'OFL-Test.txt')
    folder = _synthetic_fonts(c.builder.ROOT, names)
    font = folder / 'Test-Regular.ttf'
    if damage == 'missing':
        font.unlink()
    elif damage == 'altered':
        font.write_bytes(b'synthetic Test-Regular.ttX')  # same size, different hash
    elif damage == 'resized':
        font.write_bytes(font.read_bytes() + b'!')
    elif damage == 'unrecorded':
        (folder / 'Extra-Bold.ttf').write_bytes(b'unreviewed')
    elif damage == 'not_ofl':
        record = json.loads((folder / 'FONTS.json').read_text())
        record['files'][0]['licence'] = 'Proprietary'
        (folder / 'FONTS.json').write_text(json.dumps(record))
    with pytest.raises(ValueError):
        c.builder.build(c.output, c.compiler, c.model)
    assert not c.compiled and not list(c.output.glob('build-*'))


def test_intact_synthetic_fonts_are_staged(builder_case):
    c = builder_case
    _synthetic_fonts(c.builder.ROOT)
    receipt = c.builder.build(c.output, c.compiler, c.model)
    app = Path(receipt['stage']) / 'application'
    for name in ('Test-Regular.ttf', 'OFL-Test.txt', 'FONTS.json'):
        assert (app / 'src/aester_desktop/assets/fonts' / name).is_file()


def test_fonts_ship_with_the_desktop_app_not_the_open_engine_distribution(builder_case):
    """The fonts belong to the desktop package: the installer build carries every one of them,
    while the open engine's pyproject ships no assets at all."""
    data = tomllib.loads((ROOT / 'pyproject.toml').read_text(encoding='utf-8'))
    assert data['tool']['setuptools']['package-data']['starstack'] == []
    shipped = {relative.as_posix() for _, relative in builder_case.builder._font_files(ROOT)}
    for path in FONTS.iterdir():
        assert 'src/aester_desktop/assets/fonts/' + path.name in shipped


def test_licence_documents_list_the_bundled_fonts():
    third_party = (ROOT / 'THIRD_PARTY.md').read_text(encoding='utf-8')
    pypi, rest = third_party.split('## 非 PyPI 组件', 1)
    for family in ('Noto Sans SC', 'IBM Plex Sans', 'IBM Plex Mono', 'LXGW WenKai Lite'):
        assert family not in pypi and family in rest
    assert rest.count('SIL OFL 1.1') >= 4 and rest.count('bundled, unmodified') >= 4
    assert '“Plex”' in rest
    notice = (ROOT / 'NOTICE.md').read_text(encoding='utf-8')
    assert 'SIL OFL 1.1' in notice and 'src/aester_desktop/assets/fonts/' in notice
    installer = (ROOT / 'packaging/aester_p2/THIRD_PARTY_INSTALLER.md').read_text(encoding='utf-8')
    assert 'Open Font License 1.1' in installer and 'FONTS.json' in installer
