"""Independent contracts: no production edits or imported production oracles."""
from pathlib import Path
import hashlib, importlib.util, json, math, sys
import numpy as np
import pytest
import tifffile
from PIL import Image, ImageCms

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
sys.path.insert(0,str(ROOT/'src'))
from starstack.star_controls_curve import full_range_delta
from starstack.star_reduction import reduce_stars, suggest_star_reduction
from starstack.star_density import analyse_density, recommend_controls
from starstack.star_separation import separate_stars
from starstack import core

spec=importlib.util.spec_from_file_location('fixture_only',ROOT/'tests/test_star_reduction.py')
fixture_only=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture_only)
make_separation=fixture_only.make_separation

def scalar(B,S,D,k,w,s,lo=.1,hi=.45):
 r=[int(b)-int(d) for b,d in zip(B,D)]
 u=min(1,max(0,(int(k)/65535-lo)/(hi-lo)));h=u*u*(3-2*u)
 q=min(1,max(0,(s-.75)/.25));q=q*q*(3-2*q)
 desired=[float(w)*((1-q)*s*(1-h)*max(z,0)+q*z) for z in r]
 caps=[1.]
 for b,so,v in zip(B,S,desired):
  if v>0:caps.append(min(int(b),int(so))/v)
  elif v<0:caps.append(min(65535-int(b),65535-int(so))/-v)
 cap=min(caps)
 return [min(min(int(b),int(so)),max(max(int(b)-65535,int(so)-65535),round(v*cap))) for b,so,v in zip(B,S,desired)]

@pytest.mark.parametrize('strength',[0,.25,.55,.75,.751,.875,.999,1.])
def test_scalar_random_signed_and_headroom(strength):
 rng=np.random.default_rng(88213)
 B=rng.integers(0,65536,(11,17,3),dtype=np.uint16)
 S=rng.integers(0,65536,B.shape,dtype=np.uint16)
 D=rng.integers(0,65536,B.shape,dtype=np.uint16)
 W=rng.uniform(0,1,B.shape[:2]).astype(np.float64);W[0]=[0]+[1]*16
 K=np.maximum(B.astype('i4')-D,0).max(2).astype('u2')
 sm=np.full(B.shape[:2],strength,np.float64)
 delta=full_range_delta(B,S,D,K,W,sm)
 ref=np.array([[scalar(B[y,x],S[y,x],D[y,x],K[y,x],W[y,x],strength) for x in range(17)] for y in range(11)],np.int32)
 np.testing.assert_array_equal(delta,ref)
 bo=B.astype('i4')-delta;so=S.astype('i4')-delta
 assert np.all((bo>=0)&(bo<=65535)&(so>=0)&(so<=65535))
 np.testing.assert_array_equal(so-bo,S.astype('i4')-B)
 assert not np.any(delta[W==0])

def test_exact_full_endpoint_signed_unpaired_and_partial_weight():
 rng=np.random.default_rng(662)
 B=rng.integers(0,65536,(16,19,3),dtype='u2');D=rng.integers(0,65536,B.shape,dtype='u2')
 B[0,0]=[0,65535,123];D[0,0]=[65535,0,0]
 K=np.maximum(B.astype('i4')-D,0).max(2).astype('u2');W=np.ones(B.shape[:2],np.float64)
 delta=full_range_delta(B,B,D,K,W,W)
 np.testing.assert_array_equal(delta,B.astype('i4')-D)
 np.testing.assert_array_equal(B.astype('i4')-delta,D)
 half=full_range_delta(B,B,D,K,W*.5,W)
 np.testing.assert_array_equal(half,np.rint((B.astype('i4')-D)*.5).astype('i4'))

def test_unreachable_endpoint_common_cap_and_signed_T():
 B=np.array([[[100,1000,1000],[65000,65000,65000],[100,65000,1000]]],np.uint16)
 S=np.array([[[0,1000,1000],[65535,65000,65000],[50,65530,1000]]],np.uint16)
 D=np.array([[[20,900,1200],[65530,65530,65000],[0,65530,1000]]],np.uint16)
 K=np.maximum(B.astype('i4')-D,0).max(2).astype('u2');W=np.ones((1,3))
 delta=full_range_delta(B,S,D,K,W,W)
 np.testing.assert_array_equal(delta,np.array([[scalar(b,s,d,k,1,1) for b,s,d,k in zip(B[0],S[0],D[0],K[0])]],np.int32))
 assert np.any((B.astype('i4')-delta)!=D)
 np.testing.assert_array_equal((S.astype('i4')-delta)-(B.astype('i4')-delta),S.astype('i4')-B)

@pytest.mark.parametrize('trans',[lambda a:a,lambda a:a[::-1,::2],lambda a:a.astype(a.dtype.newbyteorder('>'))])
def test_views_endian_readonly_nonmutation(trans):
 rng=np.random.default_rng(198)
 B=rng.integers(5000,55000,(12,18,3),dtype='u2');S=B.copy();D=rng.integers(6000,54000,B.shape,dtype='u2')
 K=np.maximum(B.astype('i4')-D,0).max(2).astype('u2');W=np.ones(B.shape[:2]);sm=np.full(W.shape,.95)
 arrays=[trans(a) for a in (B,S,D,K,W,sm)];saved=[a.copy() for a in arrays]
 for a in arrays:a.flags.writeable=False
 delta=full_range_delta(*arrays)
 assert delta.dtype==np.int32 and delta.flags.c_contiguous
 for a,saved_a in zip(arrays,saved):np.testing.assert_array_equal(a,saved_a);assert not np.shares_memory(a,delta)

@pytest.mark.parametrize('bad',[np.nan,np.inf,-.1,1.01])
def test_invalid_map_even_when_weight_zero(bad):
 B=np.full((3,4,3),1000,np.uint16);K=np.full((3,4),1000,np.uint16);W=np.zeros((3,4))
 with pytest.raises(ValueError):full_range_delta(B,B,np.zeros_like(B),K,W,np.full(W.shape,bad))

@pytest.mark.parametrize('strength',[0,.55,.75,.875,1])
def test_per_pixel_brightness_order_without_conflicting_T(strength):
 k=np.arange(0,65536,131,dtype=np.uint16)[None,:]
 B=np.repeat(k[...,None],3,2);D=np.zeros_like(B);W=np.ones(k.shape)
 delta=full_range_delta(B,B,D,k,W,np.full(k.shape,strength,dtype=np.float64))
 out=B.astype('i4')-delta
 assert np.all(np.diff(out[0,:,0])>=0)

def test_strength_monotone_toward_D_without_conflicting_T():
 rng=np.random.default_rng(339);B=rng.integers(1000,64000,(7,12,3),dtype='u2');D=rng.integers(1000,64000,B.shape,dtype='u2')
 K=np.maximum(B.astype('i4')-D,0).max(2).astype('u2');W=np.ones(K.shape)
 old=np.abs(B.astype('i4')-D)
 for s in np.linspace(0,1,101):
  out=B.astype('i4')-full_range_delta(B,B,D,K,W,np.full(K.shape,s))
  distance=np.abs(out-D);assert np.all(distance<=old);old=distance
 assert not np.any(old)

def artifact(record,name):return tifffile.imread(record['artifacts'][name]['path'])

@pytest.mark.parametrize('paired',[False,True])
def test_real_tiff_job_endpoints_scope_and_T(tmp_path,paired):
 rp,sep,S,B,sky,protect=make_separation(tmp_path,paired)
 hashes={p:core._sha(Path(p)) for p in sep['source_hashes']}
 zero=reduce_stars(rp,tmp_path/'zero',background_strength=0,milkyway_offset=0,bright_radius=0)
 np.testing.assert_array_equal(artifact(zero,'star_reduced_srgb16.tif'),S)
 out=reduce_stars(rp,tmp_path/'full',background_strength=1,milkyway_offset=0,bright_radius=0)
 F=artifact(out,'star_reduced_srgb16.tif').astype('i4');R=artifact(out,'stars_reduced_signed_codes.tif')
 D=artifact(sep,'diffuse_srgb16.tif').astype('i4')
 BO=artifact(out,'background_srgb16.tif').astype('i4') if paired else F
 np.testing.assert_array_equal(D+R,BO)
 if paired:np.testing.assert_array_equal(F-BO,S.astype('i4')-B)
 else:np.testing.assert_array_equal(F[protect],S[protect])
 np.testing.assert_array_equal(F[sky!=1],S[sky!=1])
 assert not np.any(R[20:50,65:80])
 assert out['full_removal_achieved_on_requested_interior']
 assert all(core._sha(Path(p))==h for p,h in hashes.items())

def evidence(tmp,sep,mask,status='present'):
 ep=tmp/f'{status}.json';ep.write_text(json.dumps({'version':'milkyway-subject-evidence-1','status':status,'method':'human-review','reviewer':'independent-test','rationale':'synthetic-region','source_sha256':core._sha(Path(sep['source'])),'milkyway_mask_sha256':core._sha(mask)}));return ep

@pytest.mark.parametrize('offset',[-1,-.35,0,.25,1])
def test_mw_local_offset_endpoint_clamp_and_binding(tmp_path,offset):
 rp,sep,S,B,sky,protect=make_separation(tmp_path)
 mask=tmp_path/'mw.tif';MW=np.zeros(S.shape[:2],np.float32);MW[:,60:80]=.5;MW[:,80:]=1;tifffile.imwrite(mask,MW)
 ep=evidence(tmp_path,sep,mask)
 out=reduce_stars(rp,tmp_path/'result',background_strength=.60,milkyway_offset=offset,milkyway_mask=mask,milkyway_evidence=ep,bright_radius=0)
 F=artifact(out,'star_reduced_srgb16.tif');D=artifact(sep,'diffuse_srgb16.tif')
 for x in [20,70,90]:
  y=30;strength=min(1,max(0,.6+offset*float(MW[y,x])));k=max((B[y,x].astype('i4')-D[y,x]).max(),0)
  expected=S[y,x].astype('i4')-scalar(B[y,x],S[y,x],D[y,x],k,1,strength)
  np.testing.assert_array_equal(F[y,x],expected)

@pytest.mark.parametrize('status',[None,'unknown','absent'])
def test_unknown_mw_never_applies_offset(tmp_path,status):
 rp,sep,*_=make_separation(tmp_path);mask=tmp_path/'mw.png';Image.fromarray(np.full((96,120),255,np.uint8)).save(mask)
 ep=evidence(tmp_path,sep,mask,status) if status else None
 common=dict(background_strength=.6,milkyway_offset=1,bright_radius=0)
 a=reduce_stars(rp,tmp_path/'a',**common)
 b=reduce_stars(rp,tmp_path/'b',**common,milkyway_mask=mask,milkyway_evidence=ep)
 np.testing.assert_array_equal(artifact(a,'star_reduced_srgb16.tif'),artifact(b,'star_reduced_srgb16.tif'))

def scene(n=512,step=40,sigma=1.5,phase=0.,color=(1.,.8,.65)):
 y,x=np.mgrid[:n,:n];z=np.zeros((n,n),np.float32)
 for cy in range(20,n-20,step):
  for cx in range(20,n-20,step):z+=.12*np.exp(-((x-cx-phase)**2+(y-cy-phase)**2)/(2*sigma*sigma))
 return (z[...,None]*np.array(color,dtype=np.float32)).astype(np.float32)

@pytest.mark.parametrize('kind',['flat','rgb_noise','mono_noise','hotspots','mono_gaussian','lines','cloud'])
def test_density_false_positives_no_advice(kind):
 rng=np.random.default_rng(624);n=512;y,x=np.mgrid[:n,:n]
 if kind=='flat':a=np.full((n,n,3),.03,np.float32)
 elif kind=='rgb_noise':a=np.clip(rng.normal(.03,.01,(n,n,3)),0,1).astype('f4')
 elif kind=='mono_noise':a=np.repeat(np.clip(rng.normal(.03,.01,(n,n,1)),0,1).astype('f4'),3,2)
 elif kind=='hotspots':
  a=np.zeros((n,n,3),'f4');a[20:-20:15,20:-20:15]=.5
 elif kind=='mono_gaussian':a=scene(color=(1,0,0))
 elif kind=='lines':
  a=np.zeros((n,n,3),'f4');a[20:-20:15,:,:]=.3
 else:a=np.repeat((.2*np.exp(-((x-249)**2+(y-231)**2)/(2*80**2)))[...,None],3,2).astype('f4')
 metrics=analyse_density(a,np.ones((n,n),bool));advice=recommend_controls(metrics)
 assert metrics['background']['candidate_stars']<8
 assert not advice['available']

def test_density_monotone_with_distinct_same_shape_stars():
 values=[]
 for step in [80,40,20]:
  a=scene(step=step);m=analyse_density(a,np.ones(a.shape[:2],bool));q=recommend_controls(m)
  assert q['available'];values.append((m['background']['candidate_stars'],q['background_strength']))
 assert values[0][0]<values[1][0]<values[2][0]
 assert values[0][1]<values[1][1]<values[2][1]

def test_density_compact_halfpixel_plateau_not_empty():
 a=scene(phase=.5);m=analyse_density(a,np.ones(a.shape[:2],bool))
 assert m['background']['candidate_stars']>=120
 assert recommend_controls(m)['available']

def test_density_insufficient_eligible_area_no_advice():
 a=scene();mask=np.zeros(a.shape[:2],bool);mask[:90,:90]=True
 m=analyse_density(a,mask);q=recommend_controls(m)
 assert not q['available'] and q['background_strength'] is None

def test_density_replicated_compact_stars_preserve_count_and_bounded_advice():
 a=scene();b=np.repeat(np.repeat(a,2,0),2,1)
 m1=analyse_density(a,np.ones(a.shape[:2],bool));m2=analyse_density(b,np.ones(b.shape[:2],bool))
 q1=recommend_controls(m1);q2=recommend_controls(m2)
 assert m1['background']['candidate_stars']==m2['background']['candidate_stars']==144
 assert abs(m1['background']['density_per_megapixel']/m2['background']['density_per_megapixel']-1)<.02
 assert q1['available'] and q2['available']
 # Small native images are not upscaled. Prominence is not claimed to be
 # scale-invariant; retain an explicit bounded observed advice difference.
 assert abs(q1['background_strength']-q2['background_strength'])<=.03

def test_density_mask_regions_no_cross_region_sample_reuse():
 a=scene();mw=np.zeros(a.shape[:2],np.float32);mw[:,256:]=1
 m=analyse_density(a,np.ones(a.shape[:2],bool),mw)
 assert m['background']['candidate_stars']+m['milkyway']['candidate_stars']==m['all_sky']['candidate_stars']
 assert m['background']['area_pixels']+m['milkyway']['area_pixels']==m['all_sky']['area_pixels']
 assert recommend_controls(m,galaxy_enabled=True)['milkyway_offset']<=-.05

def test_recommendation_monotonic_bounded_and_galaxy_lighter():
 previous=0
 for d in [40,100,500,1000,5000,50000]:
  m={'background':{'area_pixels':1000000,'candidate_stars':d,'density_per_megapixel':d,'visible_density_per_megapixel':d,'visible_star_units':d},'milkyway':{'area_pixels':1000000,'candidate_stars':2*d,'density_per_megapixel':2*d,'visible_density_per_megapixel':2*d,'visible_star_units':2*d}}
  q=recommend_controls(m,galaxy_enabled=True)
  assert .35<=q['background_strength']<=.8 and q['background_strength']>=previous
  assert -.25<=q['milkyway_offset']<=0;assert q['milkyway_strength']<=q['background_strength'];previous=q['background_strength']

def test_advice_no_pixel_outputs_and_tamper_refusal(tmp_path,monkeypatch):
 rp,sep,*_=make_separation(tmp_path,True)
 result=suggest_star_reduction(rp,tmp_path/'advice')
 assert result['advice_only'] and not result['recommendation']['available']
 assert [p.name for p in Path(result['processing_directory']).iterdir()]==['result.json']
 orig=core.atomic_json
 def hook(path,record):
  orig(path,record)
  if record.get('version')=='star-advice-job-1' and record.get('state')=='succeeded':
   with Path(sep['source']).open('ab') as f:f.write(b'independent mutation')
 monkeypatch.setattr(core,'atomic_json',hook)
 with pytest.raises(ValueError,match='before publication'):suggest_star_reduction(rp,tmp_path/'changed')
 stages=list((tmp_path/'changed').iterdir());assert len(stages)==1 and stages[0].name.startswith('.partial-')
 assert json.loads((stages[0]/'result.json').read_text(encoding='utf-8'))['state']=='failed'

def test_full_job_reports_unreachable_signed_endpoint(tmp_path):
 B=np.full((96,120,3),1000,np.uint16);D=B.copy();S=B.copy()
 B[20,20]=[100,1000,1000];S[20,20]=[0,1000,1000];D[20,20]=[20,900,1200]
 B[20,21]=[65000,65000,65000];S[20,21]=[65535,65000,65000];D[20,21]=[65530,65530,65000]
 B[20,22]=[100,200,300];S[20,22]=B[20,22];D[20,22]=0
 icc=ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
 md={'scene_linear':False,'display_encoding_applied':True,'roworder':'TOP-DOWN','color_space':'encoded_sRGB'}
 def write(path,array,more=None):tifffile.imwrite(path,array,photometric='rgb',metadata={**md,**(more or {})},extratags=[(34675,'B',len(icc),icc,False)])
 source=tmp_path/'source.tif';base=tmp_path/'base.tif';write(source,S);write(base,B,{'transient_background_for_sha256':core._sha(source)})
 sky=tmp_path/'sky.tif';tifffile.imwrite(sky,np.ones((96,120),np.float32));pm=np.zeros((96,120),np.uint8);pm[20,20:22]=255
 protect=tmp_path/'protect.png';Image.fromarray(pm).save(protect)
 def provider(image,out,_):out[:]=D;return {'synthetic':'unreachable-range-fixture'}
 sep=separate_stars(source,tmp_path/'sep',sky,protect,provider=provider,provider_identity={'backend':'independent-fixture'},transient_background=base)
 rp=Path(sep['artifacts']['diffuse_srgb16.tif']['path']).parent/'result.json'
 result=reduce_stars(rp,tmp_path/'full',background_strength=1,milkyway_offset=0)
 F=artifact(result,'star_reduced_srgb16.tif').astype('i4');BO=artifact(result,'background_srgb16.tif').astype('i4');R=artifact(result,'stars_reduced_signed_codes.tif')
 np.testing.assert_array_equal(F-BO,S.astype('i4')-B)
 np.testing.assert_array_equal(BO,D.astype('i4')+R)
 assert result['full_removal_limited_pixels']==2
 assert result['full_removal_max_remaining_code']==530
 assert not result['full_removal_achieved_on_requested_interior']
 assert result['new_zero_channels']==3
 assert not np.any(F[20,22])

@pytest.mark.parametrize('offset',[True,-1.0001,1.0001,float('nan'),float('inf'),np.float32(.1)])
def test_invalid_offset_refused_before_io(tmp_path,offset):
 with pytest.raises(ValueError,match='milkyway_offset'):reduce_stars(tmp_path/'absent.json',tmp_path/'output',background_strength=.6,milkyway_offset=offset)
 assert not (tmp_path/'output').exists()
