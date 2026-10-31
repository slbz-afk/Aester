"""Shared reference direction, proxy re-use and mask geometry regressions."""
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from starstack import core, references, orientation, regions, region_preview
from starstack.imaging import write_linear, read_rgb
import color_fixtures as cf


@pytest.fixture
def reference(tmp_path, monkeypatch):
    src = tmp_path/'input'; src.mkdir()
    source = cf.write_reference_dng(src/'frame.dng')
    camera = np.random.default_rng(28).uniform(.018,.023,(256,512,3)).astype(np.float32)
    camera[3,7] = (.8,.1,.2)
    monkeypatch.setattr(core,'doctor',lambda *_:{'available':True,'executable':'fixture','version':'1.4.4'})
    monkeypatch.setattr(orientation,'source_orientation',lambda _: {'rotation':270,'status':'known','source':'fixture'})
    def convert(exe, directory, *args):
        write_linear(directory/'work/ref_00001.fit',camera[::-1],'controlled unrotated decode')
    monkeypatch.setattr(references.siril,'run_script',convert)
    record = references.prepare_reference(None,str(tmp_path/'output'),input_files=[str(source)],color_adjustments={})
    return source,camera,record


def test_capture_applies_before_full_frame_proxy_and_mask(reference):
    source,camera,record = reference
    assert record['image_rotation'] == record['capture_rotation'] == 270
    assert (record['width'],record['height']) == (256,512)
    # Explicit transpose/row index expectation, not the implementation helper.
    np.testing.assert_array_equal(read_rgb(record['linear_path'])[::-1],camera[:,::-1].transpose(1,0,2))
    assert Image.open(record['preview_path']).size == Image.open(record['mask_template_path']).size == (256,512)
    project = regions.create_project(record,[str(source)])
    project = regions.revise(project,operations=[{'kind':'horizon','points':[[0,.75],[1,.75]]}])
    mask,_ = regions.rasterize(project)
    assert mask.shape == (512,256) and not mask[-1].any() and mask[0].all()


def test_cached_quarter_turns_return_exact_samples_without_raw_decode(reference,monkeypatch):
    source,_,first = reference
    before = {str(p):core._sha(p) for p in Path(first['reference_dir']).rglob('*') if p.is_file()}
    source_hash = core._sha(source)
    monkeypatch.setattr(references.siril,'run_script',lambda *a,**k:pytest.fail('must not decode again'))
    current = first
    for angle in (0,90,180,270):
        current = references.rotate_reference(current['reference_dir'],image_rotation=angle)
        rendered = references.render_reference(current['reference_dir'],color_adjustments={'exposure_ev':.2})
        assert (rendered['width'],rendered['height']) == Image.open(current['mask_template_path']).size
        assert current['capture_rotation'] == 270 and current['image_rotation'] == angle
        assert current['state']=='succeeded'
    np.testing.assert_array_equal(read_rgb(first['linear_path']).view(np.uint32),read_rgb(current['linear_path']).view(np.uint32))
    for key in ('linear','saturated'):
        np.testing.assert_array_equal(np.load(first['color_cache'][key]),np.load(current['color_cache'][key]))
    np.testing.assert_array_equal(np.asarray(Image.open(first['preview_path'])),np.asarray(Image.open(current['preview_path'])))
    assert {p:core._sha(Path(p)) for p in before} == before
    assert core._sha(source)==source_hash


@pytest.mark.parametrize('key',['linear_path','preview_path','mask_template_path'])
def test_rotating_tampered_reference_fails(reference,key):
    _,_,record = reference
    with Path(record[key]).open('ab') as stream: stream.write(b'changed')
    with pytest.raises(ValueError,match='改变'):
        references.rotate_reference(record['reference_dir'],image_rotation=90)


def test_region_preview_rejects_direction_only_mismatch(reference):
    source,_,ref = reference
    project = regions.create_project(ref,[str(source)])
    changed={**ref,'image_rotation':90}
    with pytest.raises(ValueError,match='方向'):
        region_preview.render_preview(project,changed,Path(ref['reference_dir']).parent)


def test_explicit_zero_overrides_capture(reference):
    source,camera,ref=reference
    record=references.prepare_reference(None,str(Path(ref['reference_dir']).parent),
        input_files=[str(source)],image_rotation=0)
    assert record['image_rotation']==0 and record['capture_rotation']==270
    np.testing.assert_array_equal(read_rgb(record['linear_path'])[::-1],camera)


def test_old_reference_without_orientation_rotates_from_zero(reference):
    _,_,ref=reference
    path=Path(ref['reference_dir'])/'reference.json'
    record=json.loads(path.read_text(encoding='utf-8'))
    record.pop('image_rotation');record.pop('capture_rotation')
    core.atomic_json(path,record)
    result=references.rotate_reference(ref['reference_dir'],image_rotation=90)
    assert result['image_rotation']==90 and result['width']==record['height']
