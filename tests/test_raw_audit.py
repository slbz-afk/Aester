from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pytest
import tifffile

from starstack import raw_audit


def test_raw_audit_output_cannot_be_a_raw_folder_descendant(tmp_path):
    source = tmp_path/'raw'/'source.nef'
    source.parent.mkdir()
    source.write_bytes(b'source does not need decoding for this guard')
    for root in (source.parent, source.parent/'outputs', source.parent/'nested'/'outputs'):
        with pytest.raises(ValueError, match='含子目录'):
            raw_audit.inspect_raw(source, root)
    assert list(source.parent.iterdir()) == [source]


def test_partial_tag_errors_remain_visible_in_black_model():
    errors = [{'tag': 'BlackLevel', 'error': 'Invalid rational file tag'}]
    model = raw_audit.black_model({'status': 'partial', 'errors': errors}, (2, 2))
    assert model['status'] == 'unsupported'
    assert model['reasons'] == ['complete_dng_metadata_required']
    assert model['metadata_errors'] == errors


def dng_fixture(path, *, levels=(41, 4), active=(0, 0, 32, 32), opcode=None):
    data = np.full((32, 32), 10, dtype=np.uint16)
    data[0, :4] = [0, 10, 1000, 1023]
    tags = [(50706, 'B', 4, (1, 4, 0, 0), False),
            (33421, 'H', 2, (2, 2), False), (33422, 'B', 4, (0, 1, 1, 2), False),
            (50710, 'B', 3, (0, 1, 2), False), (50713, 'H', 2, (1, 1), False),
            (50714, '2I', 1, levels, False), (50717, 'I', 1, 1000, False),
            (50829, 'I', 4, active, False)]
    if opcode:
        tags.append((opcode, 'B', 4, (0, 0, 0, 0), False))
    tifffile.imwrite(path, data, photometric=32803, metadata=None, extratags=tags)
    return data


def test_rational_black_kept_and_negative_overwhite_retained(tmp_path):
    p = tmp_path/'in.dng'
    codes = dng_fixture(p)
    dng = raw_audit.read_dng_tags(p)
    black = dng['tags']['BlackLevel']
    assert black['value'] == [41, 4]
    assert black['numeric'] == [10.25]
    model = raw_audit.black_model(dng, codes.shape)
    result = raw_audit.subtract_declared_black(codes, model)
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result[0, :4], [-10.25, -.25, 989.75, 1012.75])
    assert codes[0, 0] == 0
    # Reject already subtracted float data; don't silently double subtract.
    with pytest.raises(ValueError):
        raw_audit.subtract_declared_black(result, model)


def test_four_cfa_positions_have_independent_black_levels():
    codes = np.full((4, 6), 100, np.uint16)
    model = {'status':'available', 'repeat_shape':[2,2], 'black_values':[100.25, 99.5, 101.75, 98]}
    actual = raw_audit.subtract_declared_black(codes, model)
    np.testing.assert_array_equal(actual, np.tile([[-.25, .5], [-1.75, 2]], (2,3)))


def test_version_tags_come_from_ifd0_and_black_from_cfa_subifd(tmp_path):
    path = tmp_path/'subifd.dng'
    with tifffile.TiffWriter(path) as writer:
        writer.write(np.zeros((8,8,3),np.uint8), photometric='rgb', subifds=1,
                     metadata=None, extratags=[(50706,'B',4,(1,4,0,0),False),
                     (50707,'B',4,(1,3,0,0),False), (50714,'2I',1,(999,1),False)])
        writer.write(np.zeros((32,32),np.uint16),photometric=32803,metadata=None,
                     extratags=[(50714,'2I',1,(41,4),False)])
    tags = raw_audit.read_dng_tags(path)
    assert tags['status']=='ok'
    assert tags['tags']['DNGVersion']['value']==[1,4,0,0]
    assert tags['tags']['DNGBackwardVersion']['value']==[1,3,0,0]
    assert tags['tags']['DNGVersion']['ifd_offset']!=tags['raw_ifd_offset']
    assert tags['tags']['BlackLevel']['ifd_offset']==tags['raw_ifd_offset']
    assert tags['tags']['BlackLevel']['numeric']==[10.25]


def test_cfa_tiff_with_dng_suffix_does_not_establish_dng_identity(tmp_path):
    path=tmp_path/'pretend.dng'
    tifffile.imwrite(path,np.zeros((32,32),np.uint16),photometric=32803,metadata=None)
    tags=raw_audit.read_dng_tags(path)
    assert tags['status']=='partial'
    assert raw_audit.black_model(tags,(32,32))['status']=='unsupported'


@pytest.mark.parametrize('opcode', [51008, 51009])
def test_unverified_early_opcodes_block_float_derivative(tmp_path, opcode):
    path = tmp_path/'test.dng'
    dng_fixture(path, opcode=opcode)
    tags = raw_audit.read_dng_tags(path)
    assert raw_audit.black_model(tags, (32,32))['status'] == 'unsupported'


def test_late_geometric_opcode_is_reported_without_application(tmp_path):
    path = tmp_path/'test.dng'
    dng_fixture(path, opcode=51022)
    tags = raw_audit.read_dng_tags(path)
    assert tags['tags']['OpcodeList3']['present']
    assert tags['tags']['OpcodeList3']['payload_read'] is False
    assert raw_audit.black_model(tags, (32,32))['status'] == 'available'


def test_cropped_active_area_and_mismatched_shape_not_silently_rebased(tmp_path):
    p = tmp_path/'in.dng'
    dng_fixture(p, active=(2,2,30,30))
    dng = raw_audit.read_dng_tags(p)
    assert raw_audit.black_model(dng,(32,32))['status'] == 'unsupported'
    assert 'decoder_and_ifd_shape_mismatch' in raw_audit.black_model(dng,(30,30))['reasons']


def fake_rawpy(monkeypatch, codes, *, fail=False, margins=(0, 0)):
    class Raw:
        def __enter__(self): return self
        def __exit__(self,*_): pass
        raw_image = codes
        raw_pattern = np.array([[0,1],[3,2]], np.uint8)
        sizes = SimpleNamespace(top_margin=margins[0],left_margin=margins[1],
                                height=codes.shape[0]-margins[0],width=codes.shape[1]-margins[1])
        def postprocess(self, **_): pytest.fail('must never render RAW')
    def imread(_):
        if fail: raise RuntimeError('unsupported HE-star test')
        return Raw()
    module = SimpleNamespace(__version__='test',libraw_version=(0,0,0),imread=imread)
    real_import = raw_audit.importlib.import_module
    monkeypatch.setattr(raw_audit.importlib,'import_module',lambda name: module if name=='rawpy' else real_import(name))
    monkeypatch.setattr(raw_audit,'read_raw_metadata',lambda _: {'source':'libraw_reported','status':'test'})


def test_audit_publishes_hashed_separate_artifacts_and_preserves_original(tmp_path, monkeypatch):
    p=tmp_path/'source'/'test.dng';p.parent.mkdir()
    codes=dng_fixture(p); before=p.read_bytes()
    fake_rawpy(monkeypatch,codes)
    result=raw_audit.inspect_raw(p,tmp_path/'output')
    assert result['state']=='succeeded'
    assert p.read_bytes()==before
    assert result['source_unchanged']
    assert result['implementation']['module_sha256']==raw_audit.core._sha(Path(raw_audit.__file__))
    stored=json.loads((Path(result['audit_dir'])/'audit.json').read_text(encoding='utf-8'))
    assert stored==result
    assert result['camera_pipeline']['status']=='unknown'
    np.testing.assert_array_equal(np.load(result['artifacts']['raw_codes']['path'],allow_pickle=False),codes)
    assert result['artifacts']['black_subtracted_cfa']['black_subtractions']==1
    assert result['numeric_summary']['negative_after_declared_black']>0
    assert result['numeric_summary']['at_or_above_declared_white']==2
    assert result['capture_time']['status']=='unknown'


def test_missing_file_black_is_not_filled_from_libraw(tmp_path, monkeypatch):
    p=tmp_path/'source'/'test.nef';p.parent.mkdir();p.write_bytes(b'fake-nef')
    fake_rawpy(monkeypatch,np.zeros((10,10),dtype=np.uint16))
    result=raw_audit.inspect_raw(p,tmp_path/'output')
    assert result['state']=='succeeded'
    assert result['black_model']['status']=='unsupported'
    assert set(result['artifacts'])=={'raw_codes'}


def test_full_cfa_phase_stays_fixed_when_visible_origin_has_odd_margins(tmp_path, monkeypatch):
    p=tmp_path/'source'/'test.nef';p.parent.mkdir();p.write_bytes(b'fake-nef')
    fake_rawpy(monkeypatch,np.zeros((10,10),dtype=np.uint16),margins=(1,0))
    result=raw_audit.inspect_raw(p,tmp_path/'output')
    assert result['geometry']['full_pattern']==[[0,1],[3,2]]
    assert result['geometry']['visible_pattern']==[[3,2],[0,1]]


def test_failed_decode_does_not_claim_success_or_make_image(tmp_path, monkeypatch):
    p=tmp_path/'source'/'test.nef';p.parent.mkdir();p.write_bytes(b'fake-nef')
    fake_rawpy(monkeypatch,np.zeros((10,10),dtype=np.uint16),fail=True)
    result=raw_audit.inspect_raw(p,tmp_path/'output')
    assert result['state']=='failed'
    assert not result['artifacts']
    assert json.loads((Path(result['audit_dir'])/'audit.json').read_text(encoding='utf-8'))['state']=='failed'


def test_source_change_during_audit_is_not_published_as_success(tmp_path, monkeypatch):
    p=tmp_path/'source'/'test.nef';p.parent.mkdir();p.write_bytes(b'fake-nef')
    fake_rawpy(monkeypatch,np.zeros((10,10),dtype=np.uint16))
    def changing_metadata(_):
        p.write_bytes(b'changed-source')
        return {'status':'test'}
    monkeypatch.setattr(raw_audit,'read_raw_metadata',changing_metadata)
    result=raw_audit.inspect_raw(p,tmp_path/'output')
    assert result['state']=='failed'
    assert not result.get('source_unchanged')
    assert '原片发生变化' in result['error']
