import numpy as np
import pytest
from starstack.workflow_edit import add_signed_trails


def test_signed_trails_preserve_final_outside_layer_and_exact_pair():
    rng=np.random.default_rng(7825)
    b=rng.integers(500,65000,(32,40,3),dtype=np.uint16)
    s=b.copy();s[8:19,10:26]=np.clip(s[8:19,10:26].astype(np.int32)+[312,-275,71],0,65535).astype(np.uint16)
    final=np.clip(b.astype(np.int32)+5000,0,65535).astype(np.uint16)
    out=np.empty_like(b);paired=np.empty_like(b)
    stats=add_signed_trails(s,b,final,out,paired)
    delta=s.astype(np.int32)-b.astype(np.int32)
    np.testing.assert_array_equal(out.astype(np.int32)-paired.astype(np.int32),delta)
    mask=np.any(delta!=0,axis=-1)
    np.testing.assert_array_equal(out[~mask],final[~mask])
    assert stats["headroom_limited_pixels"]>0
    assert stats["residual_mismatched_channels"]==0


def test_zero_trails_is_identity_even_finished_has_endpoint_changes():
    b=np.full((8,8,3),1200,np.uint16);s=b.copy()
    final=np.zeros_like(b);final[3:]=65535
    output=np.empty_like(b);paired=np.empty_like(b)
    stat=add_signed_trails(s,b,final,output,paired)
    np.testing.assert_array_equal(output,final)
    np.testing.assert_array_equal(paired,final)
    assert stat["headroom_limited_pixels"]==0


@pytest.mark.parametrize("trail,final",[(65535,62000),(0,500),(32000,65535),(500,0)])
def test_extreme_trail_has_exact_signed_pair_without_wrap(trail,final):
    b=np.full((8,8,3),600,np.uint16);s=b.copy();s[3,4]=trail
    f=np.full_like(b,final);out=np.empty_like(b);paired=np.empty_like(b)
    stats=add_signed_trails(s,b,f,out,paired)
    np.testing.assert_array_equal(out.astype(np.int32)-paired.astype(np.int32),s.astype(np.int32)-b.astype(np.int32))
    assert np.array_equal(out[0],f[0])
    assert stats["residual_pixels"]==1


def test_aliasing_rejected():
    a=np.ones((8,8,3),np.uint16)
    with pytest.raises(ValueError,match="共享内存"):
        add_signed_trails(a,a.copy(),a.copy(),a,np.empty_like(a))


def test_review_render_reuses_frozen_color_and_anchor(tmp_path):
    """A bright added patch must not re-estimate WB, exposure or anchor."""
    import json
    from pathlib import Path
    import tifffile
    from starstack import workflow_edit as edit, color_export as ce, imaging, master_export as me, photo_render as pr, display_transform as dt
    import color_fixtures as cf
    root=tmp_path/"workflow";root.mkdir()
    source=cf.write_reference_dng(root/"reference.dng")
    basis=me.colour_basis_from_dng(source)
    camera,_,_,_=cf.synthetic_camera_master(h=96,w=128,ground_from=76,nodata_columns=0)
    first=root/"base";first.mkdir();f=first/"base.fit"
    imaging.write_linear(f,np.flipud(camera),"base synthetic")
    _,base=ce.export_job_products(np.flipud(camera),first,lambda:None,basis=basis,source_fits=f,
                                 color_adjustments={"warmth":17,"tint":-8,"exposure_ev":.45})
    second=root/"reviewed";second.mkdir();f2=second/"transients.fit"
    camera[18:24,30:65]+=.075
    imaging.write_linear(f2,np.flipud(camera),"synthetic reviewed trail")
    _,changed=ce.export_job_products(np.flipud(camera),second,lambda:None,basis=basis,source_fits=f2,
                                    color_adjustments={"warmth":17,"tint":-8,"exposure_ev":.45})
    stage=root/"new";stage.mkdir()
    job={"job_dir":str(root),"diagnostics":{"transients":{"composite":str(f2),"color_export":changed}}}
    result=edit._render_fixed(stage,{"colour":base},job)
    original=json.loads(Path(base["photo"]["record"]).read_text())
    assert result["metadata"]["display_transform"]==original["metadata"]["display_transform"]
    frozen=pr.render_photo(changed["master"]["path"],root/"expected",
                           dt.Params.from_json(original["metadata"]["display_transform"]))
    np.testing.assert_array_equal(tifffile.imread(result["tiff"]),tifffile.imread(frozen["tiff"]))
    # Exact copy of the base leaves colour untouched outside the local input
    # change and the documented shadow-chroma smoothing neighbourhood.
    actual=tifffile.imread(result["tiff"])
    old=tifffile.imread(base["photo"]["tiff"])
    assert np.any(actual!=old)
    assert np.array_equal(actual[40:],old[40:])


def test_write_pair_real_tiff_domains_and_receipt(tmp_path):
    from starstack import workflow_edit as edit, photo_render as pr
    import tifffile
    import struct
    from PIL import Image
    meta={"roworder":"TOP-DOWN","scene_linear":False,"display_encoding_applied":True,"color_space":"encoded_sRGB"}
    b=np.full((16,24,3),3500,np.uint16);s=b.copy();s[4:10,6:9]+=np.array([300,200,100],np.uint16)
    final=b+1000
    paths=[]
    profiles=[]
    profile=pr.srgb_icc()
    for index,(name,array) in enumerate((("base",b),("source",s),("finished",final))):
        # LittleCMS stamps newly generated profiles with the current time.
        # Use distinct, valid creation dates to exercise which input profile
        # is preserved, without depending on a test staying in one second.
        stamped=bytearray(profile)
        stamped[24:36]=struct.pack(">6H",2026,1,1,0,0,index+10)
        profiles.append(bytes(stamped))
        p=tmp_path/(name+".tif")
        tifffile.imwrite(p,array,photometric="rgb",iccprofile=profiles[-1],metadata=meta)
        paths.append(str(p))
    stage=tmp_path/"edit";stage.mkdir()
    stat=edit._write_pair(stage,{"colour":{"photo":{"tiff":paths[0]}}},{"tiff":paths[1]},{"tiff":paths[2]})
    out=tifffile.imread(stage/"final_srgb16.tif");paired=tifffile.imread(stage/"finished_background_srgb16.tif")
    np.testing.assert_array_equal(out.astype(np.int32)-paired.astype(np.int32),s.astype(np.int32)-b.astype(np.int32))
    assert stat["headroom_limited_pixels"]==0 and stat["residual_pixels"]==18
    for name in ("final_srgb16.tif","finished_background_srgb16.tif"):
        with tifffile.TiffFile(stage/name) as tf:
            assert tf.pages[0].tags[34675].value==profiles[2]
    with Image.open(stage/"final.jpg") as image:
        assert image.info["icc_profile"]==profiles[2]
