from pathlib import Path
import json
import shutil

import numpy as np
import pytest

from starstack import core, transients


def test_json_digest_is_bound_to_the_exact_parsed_bytes(tmp_path, monkeypatch):
    import hashlib
    path = tmp_path/'audit.json'
    original_bytes = b'{"state": "old"}'
    path.write_bytes(original_bytes)
    original_loads = json.loads
    def replace_during_parse(value, **kwargs):
        path.write_bytes(b'{"state": "new"}')
        return original_loads(value, **kwargs)
    monkeypatch.setattr(transients.json, 'loads', replace_during_parse)
    record, digest = transients._json_with_hash(path)
    assert record == {'state': 'old'}
    assert digest == hashlib.sha256(original_bytes).hexdigest()
    with pytest.raises(ValueError, match='哈希'):
        transients._sha_checked(path, digest)


def test_evidence_json_size_limit_is_enforced_while_reading(tmp_path, monkeypatch):
    path = tmp_path/'record.json'
    path.write_text('{"test": "data beyond limit"}', encoding='utf-8')
    monkeypatch.setattr(transients, 'MAX_RECORD_BYTES', 8)
    with pytest.raises(ValueError, match='大小'):
        transients._json_with_hash(path)


def test_candidate_output_cannot_be_in_raw_folder(evidence):
    source, _, directory, _, _ = evidence
    for out in (source.parent, source.parent/'child'):
        with pytest.raises(ValueError, match='含子目录'):
            transients.record_candidate(directory, out, rectangle=[0, 0, 2, 2])
    assert not (source.parent/'child').exists()


@pytest.fixture
def evidence(tmp_path):
    source=tmp_path/'raw'/'source.nef';source.parent.mkdir();source.write_bytes(b'controlled original')
    output=tmp_path/'outputs';directory=output/'raw-audit-test';directory.mkdir(parents=True)
    codes=np.arange(120,dtype=np.uint16).reshape(10,12)
    path=directory/'raw_codes.npy';np.save(path,codes,allow_pickle=False)
    audit={'schema_version':1,'state':'succeeded','audit_dir':str(directory),
           'source':{'path':str(source),'sha256':core._sha(source),'size_bytes':source.stat().st_size},
           'geometry':{'width':12,'height':10,'row_order':'TOP-DOWN',
                       'coordinate_basis':'decoder_full_raw_mosaic','full_pattern':[[0,1],[3,2]]},
           'capture_time':{'value':None,'timezone':None,'status':'unknown'},
           'artifacts':{'raw_codes':{'path':str(path),'sha256':core._sha(path),'dtype':'uint16','shape':[10,12]}}}
    core.atomic_json(directory/'audit.json',audit)
    return source,output,directory,codes,audit


def test_crop_is_exact_independent_and_preserves_odd_origin_phase(evidence):
    source,output,directory,codes,audit=evidence
    result=transients.record_candidate(directory,output,rectangle=[1,3,5,4],note='未知亮迹')
    assert result['state']=='succeeded'
    np.testing.assert_array_equal(np.load(result['artifacts']['crop']['path'],allow_pickle=False),codes[3:7,1:6])
    assert result['crop_cfa_pattern']==[[2,3],[1,0]]
    assert result['classification']=='unknown' and result['classification_source']=='manual'
    assert not any(result['processing'].values())
    assert core._sha(source)==audit['source']['sha256']
    assert json.loads((directory/'audit.json').read_text(encoding='utf-8'))==audit
    # Deleting the disposable inspection cache cannot remove retained evidence.
    shutil.rmtree(directory)
    source.unlink()
    verified=transients.read_candidate(result['candidate_dir'])
    assert verified['integrity']['owned_evidence_verified']
    assert not verified['integrity']['external_source_rechecked']


@pytest.mark.parametrize('rectangle', [[True,0,2,2],[0,0,0,2],[-1,0,2,2],
    [11,0,2,2],[0,9,2,2],[0,0,1.5,2],[0,0,float('nan'),2],[0,0,2],None])
def test_bad_rectangle_does_not_create_candidate(evidence,rectangle):
    _,output,directory,_,_=evidence
    with pytest.raises(ValueError):
        transients.record_candidate(directory,output,rectangle=rectangle)
    assert not list(output.glob('candidate-*'))


@pytest.mark.parametrize('changed',['source','codes','geometry','ownership','failed'])
def test_changed_evidence_rejected_before_creating_candidate(evidence,changed):
    source,output,directory,_,audit=evidence
    if changed=='source': source.write_bytes(b'modified original')
    elif changed=='codes': (directory/'raw_codes.npy').write_bytes(b'corrupt')
    else:
        if changed=='geometry': audit['geometry']['row_order']='BOTTOM-UP'
        elif changed=='ownership': audit['audit_dir']=str(output)
        else: audit['state']='failed'
        core.atomic_json(directory/'audit.json',audit)
    with pytest.raises(ValueError):
        transients.record_candidate(directory,output,rectangle=[0,0,2,2])
    assert not list(output.glob('candidate-*'))


def test_external_array_reference_is_rejected_even_with_matching_hash(evidence,tmp_path):
    _,output,directory,codes,audit=evidence
    external=tmp_path/'outside.npy';np.save(external,codes)
    audit['artifacts']['raw_codes'].update(path=str(external),sha256=core._sha(external))
    core.atomic_json(directory/'audit.json',audit)
    with pytest.raises(ValueError,match='raw_codes'):
        transients.record_candidate(directory,output,rectangle=[0,0,2,2])


@pytest.mark.parametrize('corrupt',['crop','snapshot','rectangle','shifted_origin','phase','source','outside_path'])
def test_read_candidate_checks_its_owned_evidence(evidence,corrupt):
    _,output,directory,_,_=evidence
    result=transients.record_candidate(directory,output,rectangle=[0,0,2,2])
    candidate=Path(result['candidate_dir'])
    if corrupt=='crop': (candidate/'crop.npy').write_bytes(b'broken')
    elif corrupt=='snapshot': (candidate/'audit_snapshot.json').write_bytes(b'broken')
    else:
        if corrupt=='rectangle': result['rectangle']=[0,0,3,2]
        elif corrupt=='shifted_origin': result['rectangle']=[2,2,2,2]
        elif corrupt=='phase': result['crop_cfa_pattern']=[[2,3],[1,0]]
        elif corrupt=='source': result['source']['sha256']='0'*64
        else: result['artifacts']['crop']['path']=str(directory/'raw_codes.npy')
        core.atomic_json(candidate/'candidate.json',result)
    with pytest.raises(ValueError): transients.read_candidate(candidate)


def test_failure_during_write_leaves_explicit_failed_candidate(evidence,monkeypatch):
    _,output,directory,_,_=evidence
    monkeypatch.setattr(transients.np,'save',lambda *_args,**_kwargs: (_ for _ in ()).throw(OSError('disk error')))
    result=transients.record_candidate(directory,output,rectangle=[0,0,2,2])
    assert result['state']=='failed' and 'disk error' in result['error']
    assert (Path(result['candidate_dir'])/'candidate.json').is_file()
    with pytest.raises(ValueError): transients.read_candidate(result['candidate_dir'])


def test_repeated_marks_get_separate_candidates(evidence):
    _,output,directory,_,_=evidence
    first=transients.record_candidate(directory,output,rectangle=[0,0,2,2])
    second=transients.record_candidate(directory,output,rectangle=[0,0,2,2])
    assert first['candidate_dir']!=second['candidate_dir']
    assert first['artifacts']['crop']['sha256']==second['artifacts']['crop']['sha256']
