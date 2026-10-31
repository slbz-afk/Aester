"""Bounded public Z8 DCP experiment: fixed D65 and signed float extension.

Only the pinned profile and the audited three-colour Z8 family are accepted.
The DNG ForwardMatrix/working-space row normalization follows the documented
Adobe SDK mathematics. HueSatMap uses a stated float extension inspired by
RawTherapee: negative ProPhoto pixels bypass the table; S/V are not clipped.
This is neither automatic dual-illuminant DCP nor exact Adobe/RT rendering.
All outputs are new derivatives, never a replacement for camera RGB masters.
The user-approved Z8 photographic default includes HueSatMap. Callers producing
a matrix-only linear derivative must explicitly pass apply_huesat=False.
"""
from pathlib import Path
import hashlib
import struct
import numpy as np
from .raw_color import validate_recipe
from .experimental_huesat import lookup_huesat

VERSION = 'experimental-z8-dcp-d65-float-1'
DEFAULT_PHOTOGRAPHIC_MODE = 'dcp_hsm'
PROFILE_SHA = 'eef2ad2709fcaeb3830a25f63bd9079d68166d4bcac5419bcefe21753cc3b228'
INPUT_STATE = 'black_subtracted_unbalanced_camera_RGB_DN'
D50 = np.array([.3457/.3585,1.,(1-.3457-.3585)/.3585],np.float64)
# Adobe SDK 1.7.1 dng_color_space.cpp named-space constructors. Each row is
# normalized to the common PCS D50, matching SetMatrixToPCS's convention.
PROPHOTO_BASE = np.array([[.7977,.1352,.0313],[.2880,.7119,.0001],[0.,0.,.8249]])
SRGB_D50_BASE = np.array([[.4361,.3851,.1431],[.2225,.7169,.0606],[.0139,.0971,.7141]])


def normalize_to_pcs(matrix):
    a=np.asarray(matrix,dtype=np.float64)
    if a.shape!=(3,3) or not np.isfinite(a).all() or np.any(a.sum(axis=1)<=0):
        raise ValueError('Invalid forward/working matrix')
    return a*(D50/a.sum(axis=1))[:,None]


PROPHOTO_TO_XYZ = normalize_to_pcs(PROPHOTO_BASE)
XYZ_TO_PROPHOTO = np.linalg.inv(PROPHOTO_TO_XYZ)
XYZ_TO_SRGB = np.linalg.inv(normalize_to_pcs(SRGB_D50_BASE))
for _a in [D50,PROPHOTO_BASE,SRGB_D50_BASE,PROPHOTO_TO_XYZ,XYZ_TO_PROPHOTO,XYZ_TO_SRGB]:
    _a.setflags(write=False)


def load_z8_profile(path):
    """Read one byte snapshot; reject anything but the audited profile asset."""
    if Path(path).stat().st_size!=65350:
        raise ValueError('Unexpected profile size')
    data=Path(path).read_bytes()
    if len(data)!=65350 or hashlib.sha256(data).hexdigest()!=PROFILE_SHA:
        raise ValueError('Only the pinned public Z8 profile is supported')
    endian='<' if data[:2]==b'II' else '>'
    magic,offset=struct.unpack_from(endian+'HI',data,2)
    if magic!=0x4352:raise ValueError('Unexpected DCP header')
    count=struct.unpack_from(endian+'H',data,offset)[0]
    tags={}
    sizes={2:1,3:2,4:4,5:8,10:8,11:4}
    for i in range(count):
        pos=offset+2+12*i
        tag,kind,n,ptr=struct.unpack_from(endian+'HHII',data,pos)
        if tag in tags or kind not in sizes:raise ValueError('Unexpected profile tags')
        length=sizes[kind]*n;start=pos+8 if length<=4 else ptr
        block=data[start:start+length]
        if len(block)!=length:raise ValueError('Truncated profile tag')
        if kind==2:value=block.rstrip(b'\0').decode('utf-8')
        elif kind in (5,10):
            pairs=np.asarray(struct.unpack(endian+('II' if kind==5 else 'ii')*n,block),dtype=np.float64).reshape(-1,2)
            if np.any(pairs[:,1]==0):raise ValueError('Invalid profile rational')
            value=(pairs[:,0]/pairs[:,1]).tolist()
        else:value=list(struct.unpack(endian+{3:'H',4:'I',11:'f'}[kind]*n,block))
        tags[tag]=value
    if tags[50779]!=[21] or tags[50937]!=[90,30,1] or tags[51110]!=[1]:
        raise ValueError('Unexpected pinned illuminant/map/black-render contract')
    fm=np.array(tags[50965],dtype=np.float64).reshape(3,3)
    table=np.array(tags[50939],dtype=np.float64).reshape(90,30,3)
    fm=normalize_to_pcs(fm)
    fm.setflags(write=False);table.setflags(write=False)
    return {'sha256':PROFILE_SHA,'name':tags[50936],'copyright':tags[50942],
            'forward_matrix':fm,'huesat':table,'default_black_render':'None (51110=1)',
            'huesat_encoding_tag_present':51107 in tags,'look_table_present':50982 in tags}


def rgb_to_hsv(rgb):
    """Float64, nonnegative linear RGB; hue in degrees and no V upper bound."""
    v=rgb.max(axis=-1);lo=rgb.min(axis=-1);d=v-lo
    s=np.divide(d,v,out=np.zeros_like(v),where=v!=0)
    h=np.zeros_like(v);channel=rgb.argmax(axis=-1)
    ratios=[rgb[...,1]-rgb[...,2],rgb[...,2]-rgb[...,0],rgb[...,0]-rgb[...,1]]
    for k,offset in [(0,0.),(1,2.),(2,4.)]:
        q=np.divide(ratios[k],d,out=np.zeros_like(v),where=d!=0)
        m=(channel==k)&(d!=0);h[m]=(q[m]+offset)*60.
    return np.remainder(h,360),s,v


def hsv_to_rgb(h,s,v):
    """Float extension deliberately allows saturation above one (negative RGB)."""
    sector=np.remainder(h,360.)/60.
    i=np.floor(sector).astype(np.intp)%6;f=sector-np.floor(sector)
    p=v*(1-s);q=v*(1-s*f);t=v*(1-s*(1-f))
    out=np.empty(h.shape+(3,),dtype=np.float64)
    for k,triplet in enumerate([(v,t,p),(q,v,p),(p,v,t),(p,q,v),(t,p,v),(v,p,q)]):
        m=i==k
        for c,a in enumerate(triplet):out[...,c][m]=a[m]
    return out


def render_camera(camera,raw_recipe,profile_path,*,apply_huesat=True,input_state=INPUT_STATE):
    """New float32 sRGB-coordinate derivative plus exact, replayable parameters.

HueSatMap output is a nonlinear colour derivative, not a scene-linear master.
No file-side metadata is guessed here; source AB/CC must have been audited by
the orchestration. Neither DNG BaselineExposure nor a display curve is applied.
"""
    raw=validate_recipe(raw_recipe)
    if type(apply_huesat) is not bool or input_state!=INPUT_STATE:raise ValueError('Explicit DCP input contract required')
    if not isinstance(camera,np.ndarray) or camera.dtype!=np.float32 or camera.ndim!=3 or camera.shape[-1]!=3 or min(camera.shape[:2])<1:
        raise ValueError('Need float32 HWC camera RGB')
    z8=np.array([[1.1423,-.4564,-.1123],[-.4816,1.2895,.2119],[-.021,.1061,.7282],[0,0,0]],np.float32)
    if not np.array_equal(np.asarray(raw['camera']['model']['cam_xyz'],np.float32),z8):
        raise ValueError('Only the audited Z8 matrix family is supported')
    profile=load_z8_profile(profile_path)
    gains=np.array(raw['white_balance']['gains_rgb'],np.float64)
    scale=raw['scale']['common_denominator_DN']
    fm=profile['forward_matrix'];table=profile['huesat']
    camera_to_xyz=fm@np.diag(gains/scale)
    to_pro=XYZ_TO_PROPHOTO@camera_to_xyz
    to_srgb=XYZ_TO_SRGB@camera_to_xyz
    pro_to_srgb=XYZ_TO_SRGB@PROPHOTO_TO_XYZ
    out=np.empty(camera.shape,np.float32)
    skipped=0;applied=0;over_sat=0
    try:
        with np.errstate(over='raise',invalid='raise',divide='raise'):
            for y in range(0,len(camera),128):
                tile=np.asarray(camera[y:y+128],np.float64)
                if not np.isfinite(tile).all():raise ValueError('Nonfinite camera RGB')
                if not apply_huesat:
                    rendered=tile@to_srgb.T
                else:
                    pro=tile@to_pro.T
                    good=(pro>=0).all(axis=-1)
                    skipped+=int((~good).sum());applied+=int(good.sum())
                    selected=pro[good]
                    if selected.size:
                        h,s,v=rgb_to_hsv(selected)
                        mods=lookup_huesat(table,h,s)
                        s=s*mods[:,1];v=v*mods[:,2];h=h+mods[:,0]
                        over_sat+=int((s>1).sum())
                        pro[good]=hsv_to_rgb(h,s,v)
                    rendered=pro@pro_to_srgb.T
                    # Preserve the matrix branch exactly on bypassed pixels,
                    # avoiding a needless matrix round trip in the signed domain.
                    rendered[~good]=tile[~good]@to_srgb.T
                out[y:y+128]=rendered.astype(np.float32)
    except FloatingPointError as exc:raise ValueError('DCP float overflow') from exc
    if not np.isfinite(out).all():raise ValueError('Nonfinite DCP output')
    record={'version':VERSION,'profile_sha256':profile['sha256'],'profile_name':profile['name'],
            'illuminant_selection':'fixed_D65_endpoint_2_not_automatic_interpolation',
            'camera_neutral':(1/gains).tolist(),'gains_rgb':gains.tolist(),'scale_DN':scale,
            'camera_calibration':'caller_verified_identity_AB_and_CC; mismatched_signature_ignored',
            'forward_matrix_to_XYZ_D50':fm.tolist(),'D50_PCS':D50.tolist(),
            'XYZ_D50_to_sRGB':XYZ_TO_SRGB.tolist(),'XYZ_D50_to_ProPhoto':XYZ_TO_PROPHOTO.tolist(),
            'ProPhoto_to_XYZ_D50':PROPHOTO_TO_XYZ.tolist(),'camera_DN_to_sRGB_matrix':to_srgb.tolist(),
            'huesat_applied':apply_huesat,'huesat_dimensions':[90,30,1],
            'range_policy':'negative_ProPhoto_bypasses_HSM; no_S_or_V_clipping; not_Adobe_SDR_equivalence',
            'huesat_bypassed_negative_pixels':skipped,'huesat_processed_pixels':applied,
            'huesat_saturation_above_one_pixels':over_sat,'baseline_exposure_applied':False,
            'tone_curve_applied':False,'look_table_applied':False,'automatic_black_applied':False,
            'output':'sRGB_coordinates_no_display_encoding; nonlinear_colour_derivative_if_HSM'}
    return out,record
