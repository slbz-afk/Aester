"""Experimental separated finishing: identities, binding, scope, and interfaces."""
import asyncio
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile
from PIL import Image, ImageCms

from starstack import cli, core, milkyway, photographic_job as job
from starstack.mcp_server import AccessDenied, WorkspaceTools, create_server


def write_rgb(path, array, metadata, icc):
    tifffile.imwrite(path, array, photometric='rgb', metadata=metadata,
                     extratags=[(34675, 'B', len(icc), icc, False)])


@pytest.fixture
def scene(tmp_path):
    y, x = np.mgrid[:160, :192]
    diffuse = np.rint((.18+.02*np.sin(x/24)+.01*np.cos(y/17))[..., None]
                     *np.array([.95, 1., .87])*65535).astype(np.uint16)
    base = diffuse.copy()
    base[55:85, 112:115] += 3500
    base[55:85, 115:117] -= 600  # Preserve signed, not nonnegative-only, residual.
    source = base.copy()
    source[24:145, 86:88] += 1200
    source[24:145, 88:89] -= 400
    protected = np.zeros((160, 192), np.uint8); protected[20:150, 84:91] = 255
    sky = np.full((160, 192), 255, np.uint8); sky[150:] = 0; sky[149] = 128
    galaxy = np.zeros_like(sky); galaxy[:, 35:155] = 255
    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
    md = dict(scene_linear=False, display_encoding_applied=True,
              roworder='TOP-DOWN', color_space='encoded_sRGB')
    files = {name:tmp_path/(name+'.tif') for name in ('source','base','diffuse')}
    for name, values in [('sky',sky),('protected',protected),('galaxy',galaxy)]:
        files[name]=tmp_path/(name+'.png');Image.fromarray(values).save(files[name])
    s = SimpleNamespace(**files, output=tmp_path/'out', source_data=source,
                        base_data=base, diffuse_data=diffuse, sky_data=sky,
                        protected_data=protected, galaxy_data=galaxy, md=md, icc=icc)
    rewrite(s, paired=True)
    return s


def rewrite(s, paired):
    write_rgb(s.source, s.source_data if paired else s.base_data, s.md, s.icc)
    write_rgb(s.base, s.base_data, dict(s.md, transient_background_for_sha256=core._sha(s.source)), s.icc)
    s.diffuse_md = dict(s.md, star_separation_for_sha256=core._sha(s.base if paired else s.source),
                       sky_mask_sha256=core._sha(s.sky), protected_mask_sha256=core._sha(s.protected))
    write_rgb(s.diffuse, s.diffuse_data, s.diffuse_md, s.icc)


def finish(s, paired=True, **kw):
    if 'milkyway_evidence' not in kw:
        path = s.source.parent/'fixture-evidence.json'
        path.write_text(json.dumps(dict(version='milkyway-subject-evidence-1',
            status='present', method='human-review', reviewer='synthetic test fixture',
            rationale='Synthetic enabled-path fixture, not an actual human or sky review.',
            source_sha256=core._sha(s.source), milkyway_mask_sha256=core._sha(s.galaxy))), encoding='utf-8')
        kw['milkyway_evidence'] = path
    return job.finish_milkyway(s.source, s.output, s.sky, s.protected, s.galaxy,
        starless_background=s.diffuse, **({'transient_background':s.base} if paired else {}), **kw)


def read(result, name):
    return tifffile.imread(result['artifacts'][name]['path'])


def check_layers(s, result, paired):
    assert result['state']=='succeeded',result
    final=read(result,'milkyway_srgb16.tif').astype(np.int32)
    diffuse=read(result,'milkyway_diffuse_srgb16.tif').astype(np.int32)
    base=s.base_data.astype(np.int32); original=s.diffuse_data.astype(np.int32)
    source=(s.source_data if paired else s.base_data).astype(np.int32)
    np.testing.assert_array_equal(final-diffuse,source-original)
    np.testing.assert_array_equal(diffuse+(base-original)+(source-base),final)
    if paired:
        background=read(result,'milkyway_background_srgb16.tif').astype(np.int32)
        np.testing.assert_array_equal(background-diffuse,base-original)
        np.testing.assert_array_equal(final-background,source-base)
    held=(s.sky_data!=255)|(s.galaxy_data==0)
    if not paired: held|=s.protected_data>0
    np.testing.assert_array_equal(final[held],source[held])
    np.testing.assert_array_equal(diffuse[held],original[held])
    assert not np.any(((final==0)&(source>0))|((final==65535)&(source<65535)))
    assert not np.any(((diffuse==0)&(original>0))|((diffuse==65535)&(original<65535)))
    assert result['experimental_separation'] is True and result['star_layer_physical_truth'] is False
    assert result['physical_flux_preserved'] is False and result['quality_status']=='aesthetics_not_accepted'
    assert result['source_hashes'][str(s.diffuse)]==core._sha(s.diffuse)
    for item in result['artifacts'].values(): assert core._sha(Path(item['path']))==item['sha256']
    return final,diffuse


@pytest.mark.parametrize('paired',[False,True])
@pytest.mark.parametrize('strength',[0.,.6])
def test_signed_three_layer_identity_and_scope(scene,paired,strength):
    rewrite(scene,paired)
    result=finish(scene,paired,strength=strength)
    final,diffuse=check_layers(scene,result,paired)
    source=scene.source_data if paired else scene.base_data
    if strength==0:
        np.testing.assert_array_equal(final,source)
        np.testing.assert_array_equal(diffuse,scene.diffuse_data)
    else: assert np.any(final!=source)
    assert result['separation_validation']['residual_min_code']==-600
    assert result['separation_validation']['residual_max_code']==3500
    for name in ('milkyway_srgb16.tif','milkyway_diffuse_srgb16.tif'):
        with tifffile.TiffFile(result['artifacts'][name]['path']) as tf:
            md=json.loads(tf.pages[0].description)
            assert md['experimental_separation'] is True and md['star_layer_physical_truth'] is False
            assert 'star_separation_for_sha256' not in md, 'a derivative is not reusable with an old binding'


@pytest.mark.parametrize('paired',[False,True])
def test_actual_processing_uses_diffuse_and_original_protect_is_still_checked(scene,paired,monkeypatch):
    rewrite(scene,paired); original=milkyway.prepare;seen=[]
    def prepare(image,sky,protect,*args,**kw):
        np.testing.assert_array_equal(image,scene.diffuse_data)
        np.testing.assert_array_equal(protect,np.zeros_like(protect) if paired else scene.protected_data>0)
        seen.append(True);return original(image,sky,protect,*args,**kw)
    monkeypatch.setattr(milkyway,'prepare',prepare)
    check_layers(scene,finish(scene,paired),paired)
    assert seen==[True]


@pytest.mark.parametrize('field',['star_separation_for_sha256','sky_mask_sha256','protected_mask_sha256'])
@pytest.mark.parametrize('value',[None,'0'*64,[]])
def test_bad_layer_binding_rejected_before_output(scene,field,value,monkeypatch):
    md=scene.diffuse_md.copy();md[field]=value
    write_rgb(scene.diffuse,scene.diffuse_data,md,scene.icc)
    monkeypatch.setattr(milkyway,'prepare',lambda *a,**k:pytest.fail('processing reached'))
    with pytest.raises(ValueError,match='未绑定'): finish(scene)
    assert not scene.output.exists()


def test_paired_layer_cannot_bind_composite_instead_of_processing_base(scene):
    md=dict(scene.diffuse_md,star_separation_for_sha256=core._sha(scene.source))
    write_rgb(scene.diffuse,scene.diffuse_data,md,scene.icc)
    with pytest.raises(ValueError,match='未绑定'):finish(scene)


@pytest.mark.parametrize('kind',['shape','encoding','linear','space','roworder','dtype'])
def test_layer_must_pass_display_source_contract(scene,kind):
    data=scene.diffuse_data;md=scene.diffuse_md.copy()
    if kind=='shape':data=data[:,:-1]
    elif kind=='encoding':md['display_encoding_applied']=False
    elif kind=='linear':md['scene_linear']=True
    elif kind=='space':md['color_space']='linear-sRGB'
    elif kind=='roworder':md['roworder']='BOTTOM-UP'
    elif kind=='dtype':data=data.astype(np.float32)/65535
    write_rgb(scene.diffuse,data,md,scene.icc)
    with pytest.raises(ValueError):finish(scene)
    assert not scene.output.exists()


@pytest.mark.parametrize('point',[(155,110),(149,110),(70,87)])
@pytest.mark.parametrize('paired',[False,True])
def test_starless_cannot_change_ground_mixed_or_original_protected_scope(scene,point,paired,monkeypatch):
    rewrite(scene,paired);bad=scene.diffuse_data.copy();bad[point][0]+=1
    write_rgb(scene.diffuse,bad,scene.diffuse_md,scene.icc)
    maps=[];real=job._read_display
    def read_map(path):
        value=real(path);maps.append(value);return value
    monkeypatch.setattr(job,'_read_display',read_map)
    with pytest.raises(ValueError,match='纯天空分层范围'):finish(scene,paired)
    assert not scene.output.exists() and all(m._mmap.closed for m in maps)


@pytest.mark.parametrize('phase',['validation','render','preserve','manifest'])
def test_starless_tampering_rejected(scene,phase,monkeypatch):
    changed=False
    def tamper():
        nonlocal changed
        if not changed:
            with scene.diffuse.open('ab') as f:f.write(b'changed')
            changed=True
    if phase=='manifest':
        real=core.atomic_json
        def save(path,value,**kw):
            result=real(path,value,**kw)
            if value['state']=='succeeded':tamper()
            return result
        monkeypatch.setattr(core,'atomic_json',save)
    else:
        module=job if phase=='validation' else milkyway
        name={'validation':'_starless_scope','render':'render','preserve':'preserve_transient_residual'}[phase]
        real=getattr(module,name)
        def call(*args,**kw):
            result=real(*args,**kw);tamper();return result
        monkeypatch.setattr(module,name,call)
    if phase=='validation':
        with pytest.raises(ValueError,match='验证期间改变'):finish(scene)
        assert not scene.output.exists()
    else:
        result=finish(scene)
        assert result['state']=='failed' and result['artifacts']=={}
        assert '输入在处理期间改变' in result['error']
        assert not list(scene.output.glob('milkyway-*'))
    assert changed


@pytest.mark.parametrize('paired',[False,True])
def test_shared_headroom_and_existing_endpoints_are_not_clipped(scene,paired,monkeypatch):
    # Star and transient constraints occupy different pixels because the input
    # starless layer must already hold the original protected corridor.
    scene.base_data[65,112]=[65520,30000,40000]
    scene.diffuse_data[65,112]=[62020,30600,36500]
    scene.source_data[65,112]=scene.base_data[65,112]
    scene.base_data[66,115]=[5,30000,40000]
    scene.diffuse_data[66,115]=[605,26500,40600]
    scene.source_data[66,115]=scene.base_data[66,115]
    for y,value in [(67,0),(68,65535)]:
        scene.base_data[y,112]=[value,20000,20000]
        scene.diffuse_data[y,112]=scene.base_data[y,112]
        scene.source_data[y,112]=scene.base_data[y,112]
    scene.base_data[65,87]=[60000,20000,20000]
    scene.diffuse_data[65,87]=scene.base_data[65,87]
    scene.source_data[65,87]=[65528,20000,20000]
    rewrite(scene,paired)
    def render(image,sky,protect,galaxy,prepared,output,**kw):
        output[:]=image
        for pos,delta in [((65,112),1000),((66,115),-500),((67,112),1000),((68,112),-1000)]:
            proposal=image[pos].astype(np.int32)+delta
            assert np.all((proposal>=0)&(proposal<=65535))
            output[pos]=proposal.astype(np.uint16)
        if paired:output[65,87]=(image[65,87].astype(np.int32)+1000).astype(np.uint16)
        return {'test_proposal':True}
    monkeypatch.setattr(milkyway,'render',render)
    result=finish(scene,paired);final,diffuse=check_layers(scene,result,paired)
    assert final[65,112,0]==65534 and final[66,115,0]==1
    np.testing.assert_array_equal(final[67:69,112],scene.base_data[67:69,112])
    if paired:assert final[65,87,0]==65534
    assert result['star_statistics']['gamut_limited_pixels']>=4


def test_diffuse_range_failures_are_not_silently_clipped(scene):
    source=np.full((8,8,3),5000,np.uint16)
    diffuse=np.full_like(source,65530);final=source+100;out=np.empty_like(source)
    with pytest.raises(ValueError,match='超出16位'):job._finalize_diffuse(source,diffuse,final,out)
    diffuse[:]=65435
    with pytest.raises(ValueError,match='新端点'):job._finalize_diffuse(source,diffuse,final,out)


@pytest.mark.parametrize('phase',['diffuse_create','second_preserve','finalize'])
def test_separated_failure_closes_all_maps_and_publishes_nothing(scene,phase,monkeypatch):
    maps=[];real_map=tifffile.memmap
    def mapped(*args,**kw):
        if phase=='diffuse_create' and Path(args[0]).name=='milkyway_diffuse_srgb16.tif':
            raise OSError('injected diffuse creation failure')
        result=real_map(*args,**kw);maps.append(result);return result
    monkeypatch.setattr(tifffile,'memmap',mapped)
    if phase=='finalize':
        def fail(*a,**k):raise ValueError('injected finalize failure')
        monkeypatch.setattr(job,'_finalize_diffuse',fail)
    elif phase=='second_preserve':
        original=milkyway.preserve_transient_residual;count=0
        def fail(*a,**k):
            nonlocal count
            count+=1
            if count==2:raise ValueError('injected second preserve failure')
            return original(*a,**k)
        monkeypatch.setattr(milkyway,'preserve_transient_residual',fail)
    record=finish(scene)
    assert record['state']=='failed' and record['artifacts']=={}
    assert 'injected' in record['error']
    assert maps and all(value._mmap.closed for value in maps)
    assert not list(scene.output.glob('milkyway-*'))
    assert json.loads((Path(record['processing_directory'])/'result.json').read_text('utf-8'))['state']=='failed'


def test_compressed_starless_read_fallback(scene):
    tifffile.imwrite(scene.diffuse,scene.diffuse_data,photometric='rgb',compression='zlib',
        metadata=scene.diffuse_md,extratags=[(34675,'B',len(scene.icc),scene.icc,False)])
    with pytest.raises(ValueError):tifffile.memmap(scene.diffuse,mode='r')
    result=finish(scene,strength=0.)
    check_layers(scene,result,True)


def test_cli_routes_starless_only_when_requested(monkeypatch,capsys):
    calls=[]
    monkeypatch.setattr(job,'finish_milkyway',lambda *a,**kw:calls.append((a,kw)) or {'state':'succeeded'})
    monkeypatch.setattr(sys,'argv',['starstack','finish-milkyway','--source','s','--output','o',
        '--sky-mask','k','--protected-mask','p','--milkyway-mask','m','--starless-background','d'])
    assert cli.main()==0;capsys.readouterr()
    assert calls[0][1]['starless_background']=='d' and 'transient_background' not in calls[0][1]
    assert calls[0][1]['strength']==.60


def test_mcp_bounds_starless_and_routes_valid_path(tmp_path,monkeypatch):
    root=tmp_path/'out';root.mkdir();(root/'job').mkdir()
    paths=[root/f'{i}.tif' for i in range(5)]
    for p in paths:p.touch()
    outside=tmp_path/'outside.tif';outside.touch();calls=[]
    service=WorkspaceTools(tmp_path/'raw',root)
    result=dict(state='succeeded',processing_directory=str(root/'job'),artifacts={})
    monkeypatch.setattr(job,'finish_milkyway',lambda *a,**kw:calls.append((a,kw)) or result)
    with pytest.raises(AccessDenied):service.finish_milkyway(*map(str,paths[:4]),starless_background=str(outside))
    assert not calls
    assert service.finish_milkyway(*map(str,paths[:4]),starless_background=str(paths[4]))==result
    assert calls[0][1]['starless_background']==str(paths[4])


def test_async_mcp_starless_schema_and_routing(tmp_path):
    pytest.importorskip('mcp');calls=[]
    class Service(WorkspaceTools):
        def finish_milkyway(self,*args,**kw):calls.append((args,kw));return {'state':'succeeded'}
    server=create_server(Service(tmp_path/'raw',tmp_path/'out'))
    async def run():
        tool=next(t for t in await server.list_tools() if t.name=='finish_milkyway')
        assert 'starless_background' in tool.inputSchema['properties']
        assert 'starless_background' not in tool.inputSchema['required']
        params=dict(source='s',sky_mask='k',protected_mask='p',milkyway_mask='m',starless_background='d')
        await server.call_tool('finish_milkyway',params)
        assert calls[0][1]=={'starless_background':'d'}
        for bad in (True,1,['d'],{}):
            with pytest.raises(Exception):await server.call_tool('finish_milkyway',params|{'starless_background':bad})
        assert len(calls)==1
    asyncio.run(run())
