"""Regression cases for independently reported provider and publication failures."""
from pathlib import Path
import importlib.util,json,os
import numpy as np
import pytest,tifffile
from PIL import Image,ImageCms
from starstack import core

from starstack import star_separation as ss


def fixture(root,compressed=False):
    a=np.full((32,40,3),17000,np.uint16);a[15,20]=24000
    icc=ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
    source=root/'source.tif';sky=root/'sky.png';protect=root/'protect.png'
    tifffile.imwrite(source,a,photometric='rgb',compression='zlib' if compressed else None,
      metadata={'scene_linear':False,'display_encoding_applied':True,'color_space':'encoded_sRGB','roworder':'TOP-DOWN'},extratags=[(34675,'B',len(icc),icc,False)])
    Image.fromarray(np.full(a.shape[:2],255,np.uint8)).save(sky)
    Image.fromarray(np.zeros(a.shape[:2],np.uint8)).save(protect)
    return source,sky,protect,a


def assert_failed(root):
    assert not list(root.glob('separation-*'))
    partial=list(root.glob('.partial-*'));assert len(partial)==1
    result=json.loads((partial[0]/'result.json').read_text('utf-8'))
    assert result['state']=='failed' and result['artifacts']=={}
    return result


@pytest.mark.parametrize('field',['source','sky','protect'])
def test_final_manifest_write_cannot_hide_source_change(tmp_path,monkeypatch,field):
    s,k,p,a=fixture(tmp_path);target={'source':s,'sky':k,'protect':p}[field]
    original=core.atomic_json;changed=False
    def save(path,value,**kwargs):
        nonlocal changed
        out=original(path,value,**kwargs)
        if value.get('state')=='succeeded' and not changed:
            with target.open('ab') as f:f.write(b'last-write-tamper')
            changed=True
        return out
    monkeypatch.setattr(core,'atomic_json',save)
    with pytest.raises(ValueError,match='input changed'):
        ss.separate_stars(s,tmp_path/'jobs',k,p)
    assert changed;assert_failed(tmp_path/'jobs')


@pytest.mark.parametrize('bypass',[False,True])
def test_compressed_provider_array_is_readonly_and_mutation_detected(tmp_path,bypass):
    s,k,p,a=fixture(tmp_path,True)
    def bad(image,out,stage):
        assert not image.flags.writeable
        if bypass:image.setflags(write=True)
        image[:]=1234;out[:]=image;return {'test':True}
    with pytest.raises(ValueError,match='changed source samples' if bypass else 'read-only'):
        ss.separate_stars(s,tmp_path/'jobs',k,p,provider=bad,provider_identity={'backend':'synthetic'})
    result=assert_failed(tmp_path/'jobs')
    np.testing.assert_array_equal(tifffile.imread(s),a)


@pytest.mark.parametrize('receipt',[{'bad':float('nan')},{'bad':object()},{'bad':np.array([1])}])
def test_invalid_provider_receipt_leaves_failed_serializable_record(tmp_path,receipt):
    s,k,p,a=fixture(tmp_path)
    def provider(image,out,stage):out[:]=image;return receipt
    with pytest.raises((TypeError,ValueError)):
        ss.separate_stars(s,tmp_path/'jobs',k,p,provider=provider,provider_identity={'backend':'synthetic'})
    result=assert_failed(tmp_path/'jobs');assert 'provider_receipt' not in result


def test_valid_receipt_is_detached_from_provider_owned_mutable_data(tmp_path):
    s,k,p,a=fixture(tmp_path,True);receipt={'list':[1]}
    def provider(image,out,stage):
        assert not image.flags.writeable;out[:]=image;return receipt
    result=ss.separate_stars(s,tmp_path/'jobs',k,p,provider=provider,provider_identity={'backend':'synthetic'})
    receipt['list'].append(2)
    assert result['provider_receipt']=={'list':[1]}
    d=tifffile.imread(result['artifacts']['diffuse_srgb16.tif']['path'])
    stars=tifffile.imread(result['artifacts']['stars_signed_codes.tif']['path'])
    np.testing.assert_array_equal(d.astype(np.int32)+stars,a.astype(np.int32))


def test_two_paths_to_same_file_are_rejected_without_source_mutation(tmp_path):
    original=np.full((32,40,3),12000,np.uint16);original[15,20]=35000
    source=tmp_path/'source.bin';alias=tmp_path/'alias.bin'
    writable=np.memmap(source,mode='w+',dtype=np.uint16,shape=original.shape)
    writable[:]=original;writable.flush();writable._mmap.close();os.link(source,alias)
    a=np.memmap(source,mode='r',dtype=np.uint16,shape=original.shape)
    b=np.memmap(alias,mode='r+',dtype=np.uint16,shape=original.shape)
    try:
        with pytest.raises(ValueError,match='aliases'):ss.compact_estimate(a,b)
        np.testing.assert_array_equal(a,original)
    finally:a._mmap.close();b._mmap.close()


def test_uncertain_backing_identity_fails_closed(tmp_path,monkeypatch):
    a=np.memmap(tmp_path/'a.bin',mode='w+',dtype=np.uint16,shape=(8,9,3));a[:]=1000
    b=np.memmap(tmp_path/'b.bin',mode='w+',dtype=np.uint16,shape=a.shape);b[:]=5678
    def unavailable(*args):raise PermissionError('synthetic failed stat')
    monkeypatch.setattr(ss.os.path,'samefile',unavailable)
    try:
        with pytest.raises(ValueError,match='cannot verify'):ss.compact_estimate(a,b)
        assert np.all(a==1000) and np.all(b==5678)
    finally:a._mmap.close();b._mmap.close()


