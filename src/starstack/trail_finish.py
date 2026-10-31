"""Optional photographic trail rendering; never claims complete flux recovery.

The original signed residual remains separately available. This derivative
subtracts a locally estimated residual background, lightly averages along the
trail, and tapers its measured endpoints. It cannot extend an unobserved trail.
"""
import math
import numpy as np

VERSION = 'trail-photographic-v3'


def _shift_sample(rgb, valid, dx, dy):
    """Bilinear samples at p+(dx,dy), with every nonzero dependency required."""
    height, width = valid.shape
    ix, iy = math.floor(dx), math.floor(dy)
    fx, fy = dx-ix, dy-iy
    xx = np.arange(width)[None, :] + ix
    yy = np.arange(height)[:, None] + iy
    result = np.zeros_like(rgb, dtype=np.float64)
    complete = np.ones(valid.shape, bool)
    for ox, oy, weight in ((0,0,(1-fx)*(1-fy)),(1,0,fx*(1-fy)),(0,1,(1-fx)*fy),(1,1,fx*fy)):
        if weight == 0:
            continue
        sx, sy = xx+ox, yy+oy
        inside = (sx>=0)&(sx<width)&(sy>=0)&(sy<height)
        sx, sy = np.clip(sx,0,width-1), np.clip(sy,0,height-1)
        complete &= inside & valid[sy,sx]
        result += rgb[sy,sx].astype(np.float64)*weight
    return result, complete


def finish_residual(residual, valid, along, across, *, length, half, direction,
                    noise_rgb, feather=2., extension=4., strength=.65):
    """Return weighted CHW signed layer, support, footprint and audit.

    All inputs describe a bounded registered-grid crop. `across` is signed.
    At zero strength this function is not the legacy renderer; callers select
    that renderer explicitly instead of silently mixing two layer contracts.
    """
    if type(residual) is not np.ndarray or residual.dtype != np.float32 or residual.ndim!=3 or residual.shape[2]!=3:
        raise ValueError('Residual must be HWC float32')
    h,w,_=residual.shape
    if min(h,w)<1 or type(valid) is not np.ndarray or valid.dtype!=bool or valid.shape!=(h,w):
        raise ValueError('Explicit valid pixels required')
    if (type(along) is not np.ndarray or type(across) is not np.ndarray or
            along.shape!=(h,w) or across.shape!=(h,w) or not np.isfinite(along).all() or not np.isfinite(across).all()):
        raise ValueError('Finite matching coordinates required')
    if not isinstance(direction,(tuple,list,np.ndarray)) or len(direction)!=2 or not isinstance(noise_rgb,(tuple,list,np.ndarray)) or len(noise_rgb)!=3:
        raise ValueError('Direction and RGB noise vectors required')
    values=(length,half,feather,extension,strength,*direction,*noise_rgb)
    if not all(np.isscalar(v) and not isinstance(v,(bool,np.bool_)) and np.isfinite(v) for v in values):
        raise ValueError('Finite numerical parameters required')
    if not (length>0 and half>0 and feather>0 and extension>0 and 0<strength<=1):
        raise ValueError('Invalid trail dimensions/strength')
    if len(direction)!=2 or not math.isclose(math.hypot(*direction),1.,abs_tol=1e-6):
        raise ValueError('Unit direction required')
    noise=np.asarray(noise_rgb,np.float64)
    if (noise.shape!=(3,) or np.any(noise<1e-12) or np.any(noise>1) or not np.isfinite(residual[valid]).all()
            or np.any(np.abs(residual[valid])>=8)):
        raise ValueError('Supported residual abs<8 and noise 1e-12..1 required')
    # Invalid observations never enter sideband statistics or interpolation.
    clean=np.where(valid[...,None],residual,0).astype(np.float32)
    centres=np.arange(-extension,length+extension+16,16.)
    levels=np.empty((len(centres),3),np.float64)
    usable=np.zeros(len(centres),bool)
    side=(np.abs(across)>=half+8)&(np.abs(across)<=half+24)&valid
    for i,t in enumerate(centres):
        select=side&(np.abs(along-t)<=16)
        if np.count_nonzero(select)>=12:
            levels[i]=np.median(clean[select].astype(np.float64),axis=0)
            usable[i]=True
    if np.count_nonzero(usable)<2:
        raise ValueError('Insufficient local sidebands for photographic trail')
    background=np.empty_like(clean)
    for c in range(3):
        background[...,c]=np.interp(along,centres[usable],levels[usable,c])
    corrected=(clean.astype(np.float64)-strength*background).astype(np.float32)
    corrected[~valid]=0
    smoothed=np.zeros_like(corrected,dtype=np.float64)
    supported=valid.copy()
    for offset,k in zip((-2,-1,0,1,2),(.0625,.25,.375,.25,.0625)):
        sample,ok=_shift_sample(corrected,valid,offset*direction[0],offset*direction[1])
        smoothed+=k*sample
        supported&=ok
    smoothed=(1-strength)*corrected.astype(np.float64)+strength*smoothed
    # One scalar shrinkage for all three signed channels, never RGB clipping.
    luminance=smoothed@np.array([.2126,.7152,.0722])
    sigma=float(np.sqrt(np.sum((noise*np.array([.2126,.7152,.0722]))**2)))
    confidence=luminance*luminance/(luminance*luminance+(1.5*sigma)**2)
    smoothed*=((1-strength)+strength*confidence)[...,None]
    cross=np.clip((half+feather-np.abs(across))/feather,0,1)
    t=np.minimum((along+extension)/(2*extension),(length+extension-along)/(2*extension))
    t=np.clip(t,0,1)
    ends=t*t*(3-2*t)
    weight=cross*ends
    footprint=weight>0
    data=np.moveaxis((smoothed*weight[...,None]).astype(np.float32),-1,0)
    # Keep the intended footprint even where dependencies are missing. Zeroing
    # weight here would turn missing data into a known exterior zero on warp.
    data[:,~supported]=0
    if not np.isfinite(data).all():
        raise ValueError('Nonfinite photographic trail output')
    return data,supported,weight,{
        'version':VERSION,'photographic_derivative':True,'complete_flux_recovery':False,
        'local_background_bins':int(usable.sum()),'strength':float(strength),
        'source_footprint_pixels':int(footprint.sum()),
        'removed_missing_dependencies':int(np.count_nonzero(footprint&~supported)),
        'outside_footprint_unchanged':True,'signed_channels_clipped':False,
        'geometry':'measured extent only; no extension of measured physical identity',
    }


def layer_box(frame_path,master_path,candidate,scale,frame_median,ref_median,*,params=None):
    from starstack import transient_trails as tt
    from starstack.transient_support import attach_source_support
    lp=dict(tt.DEFAULT_LAYER,**(params or {}))
    strength=float(lp.get('photographic_strength',.65))
    if strength==0:
        raw_params=dict(params or {},style='source_residual')
        return tt.layer_box(frame_path,master_path,candidate,scale,frame_median,ref_median,params=raw_params)
    m=candidate['measurement']
    p0=np.array([m['x0'],m['y0']],np.float64);p1=np.array([m['x1'],m['y1']],np.float64)
    length=float(np.linalg.norm(p1-p0));direction=(p1-p0)/length
    half=float(m['half_width_px'])+lp['footprint_margin_px']
    ext=float(lp['end_extension_px']);reach=half+27
    frame,master=tt._planes(frame_path),tt._planes(master_path)
    height,width=frame.shape
    x0=max(0,math.floor(min(p0[0],p1[0])-reach-ext))
    y0=max(0,math.floor(min(p0[1],p1[1])-reach-ext))
    x1=min(width,math.ceil(max(p0[0],p1[0])+reach+ext)+1)
    y1=min(height,math.ceil(max(p0[1],p1[1])+reach+ext)+1)
    yy,xx=np.mgrid[y0:y1,x0:x1]
    dx,dy=xx-p0[0],yy-p0[1]
    along=dx*direction[0]+dy*direction[1]
    across=-dx*direction[1]+dy*direction[0]
    residual=np.empty((y1-y0,x1-x0,3),np.float32)
    valid=np.ones((y1-y0,x1-x0),bool)
    for c in range(3):
        source=frame.crop(c,x0,x1,y0,y1);base=master.crop(c,x0,x1,y0,y1)
        valid&=np.isfinite(source)&np.isfinite(base)&(source!=0)&(base!=0)
        residual[...,c]=(scale[c]*(source-frame_median[c])+ref_median[c]-base).astype(np.float32)
    # Measurement noise is in pre-normalized residual units.
    noise=np.maximum(np.asarray(m['noise'],np.float64)*np.abs(scale),1e-12)
    data,support,weight,audit=finish_residual(residual,valid,along,across,length=length,half=half,
        direction=direction,noise_rgb=noise,feather=float(lp['feather_px']),extension=ext,strength=strength)
    footprint=weight>0
    box={'x0':x0,'y0':y0,'data':data,'weight':weight,'grid_shape':(height,width),
         'candidate_id':candidate.get('id',repr(tuple(p0)+tuple(p1))),
         'source_key':(str(frame.path.resolve()),str(master.path.resolve()),tuple(scale),tuple(frame_median),tuple(ref_median)),
         'footprint_pixels':int(footprint.sum()),'photographic_render':audit,
         'application_evidence':{'grid_footprint_pixels':audit['source_footprint_pixels'],
          'grid_finite_nonzero_pixels':int((footprint&support).sum()),
          'grid_zero_or_nonfinite_pixels':audit['removed_missing_dependencies'],
          'grid_nonzero_residual_pixels':int(np.count_nonzero(np.any(data!=0,axis=0))),
          'support_evidence':'legacy_finite_nonzero_plus_all_local_filter_dependencies',
          'photographic_render':audit}}
    return attach_source_support(box,support)


def union_photographic_boxes(boxes):
    """Same-exposure estimates: signed weighted mean times maximum coverage.

    Different candidate geometries have different background/filter estimates;
    the raw-residual max-feather selection assumption no longer applies. Exact
    duplicate IDs are idempotent; conflicting duplicates are rejected.
    """
    unique={}
    for box in boxes:
        source=box.get('source_key');grid=box.get('grid_shape')
        if (type(source) is not tuple or len(source)!=5 or
                not all(isinstance(v,str) and bool(v) for v in source[:2]) or
                any(type(v) is not tuple or len(v)!=3 or not all(np.isscalar(x) and np.isfinite(x) for x in v) for v in source[2:])):
            raise ValueError('Explicit same-source normalization identity required')
        if (type(grid) is not tuple or len(grid)!=2 or any(type(v) is not int or v<=0 for v in grid)):
            raise ValueError('Positive explicit grid dimensions required')
        if any(type(box.get(k)) is not int or box[k]<0 for k in ('x0','y0')):
            raise ValueError('Nonnegative integer box origin required')
        data=box.get('data');weight=box.get('weight');valid=box.get('validity')
        if (not isinstance(data,np.ndarray) or data.ndim!=3 or data.shape[0]!=3 or data.dtype!=np.float32 or
                not isinstance(weight,np.ndarray) or weight.shape!=data.shape[1:] or
                not isinstance(valid,np.ndarray) or valid.dtype!=bool or valid.shape!=weight.shape or
                not np.isfinite(data).all() or not np.isfinite(weight).all() or np.any(weight<0) or np.any(weight>1)):
            raise ValueError('Finite matching photographic arrays required')
        if (box['y0']+weight.shape[0]>grid[0] or box['x0']+weight.shape[1]>grid[1] or
                np.any(data[:,weight==0]!=0) or np.any(data[:,~valid]!=0)):
            raise ValueError('Box outside grid or data outside footprint/support')
        key=box.get('candidate_id')
        if not isinstance(key,str) or not key:
            raise ValueError('Stable candidate_id required for photo union')
        if key in unique:
            old=unique[key]
            if (any(old.get(k)!=box.get(k) for k in ('x0','y0','grid_shape','source_key')) or
                    any(not np.array_equal(old[k],box[k]) for k in ('data','weight','validity'))):
                raise ValueError('Conflicting duplicate photographic candidate')
            continue
        unique[key]=box
    ordered=[unique[k] for k in sorted(unique) if unique[k]['data'].size]
    if not ordered:
        return []
    first=ordered[0]
    if any(b.get('source_key')!=first.get('source_key') or b.get('grid_shape')!=first.get('grid_shape') for b in ordered):
        raise ValueError('Photo union requires same source, normalization and grid')
    groups=[]
    for box in ordered:
        x,y=box['x0'],box['y0'];h,w=box['data'].shape[1:]
        group={'bounds':[x,y,x+w,y+h],'boxes':[box]}
        while True:
            for old in groups:
                a,b=group['bounds'],old['bounds']
                if a[0]<b[2] and b[0]<a[2] and a[1]<b[3] and b[1]<a[3]:
                    group['bounds']=[min(a[0],b[0]),min(a[1],b[1]),max(a[2],b[2]),max(a[3],b[3])]
                    group['boxes']+=old['boxes'];groups.remove(old);break
            else:
                break
        groups.append(group)
    result=[]
    for group in groups:
        x0,y0,x1,y1=group['bounds'];shape=(y1-y0,x1-x0)
        maximum=np.zeros(shape,np.float64);den=np.zeros(shape,np.float64)
        numerator=np.zeros((3,*shape),np.float64);validity=np.ones(shape,bool)
        members=sorted(group['boxes'],key=lambda b:b['candidate_id'])
        for box in members:
            h,w=box['weight'].shape;y,x=box['y0']-y0,box['x0']-x0
            weight=box['weight'];valid=box['validity'];data=box['data']
            if (data.shape!=(3,h,w) or data.dtype!=np.float32 or valid.dtype!=bool or valid.shape!=(h,w)
                    or not np.isfinite(data).all() or not np.isfinite(weight).all() or np.any(weight<0) or np.any(weight>1)):
                raise ValueError('Invalid photographic box')
            sl=np.s_[y:y+h,x:x+w]
            np.maximum(maximum[sl],weight,out=maximum[sl])
            den[sl]+=weight
            numerator[:,y:y+h,x:x+w]+=data.astype(np.float64)
            validity[sl]&=(weight<=0)|valid
        multiplier=np.divide(maximum,den,out=np.zeros_like(den),where=den>0)
        combined=numerator*multiplier[None]
        if not np.isfinite(combined).all() or np.any(np.abs(combined)>np.finfo(np.float32).max):
            raise ValueError('Photographic union exceeds finite float32')
        data=combined.astype(np.float32)
        data[:,~validity]=0
        footprint=maximum>0
        result.append({'x0':x0,'y0':y0,'data':data,'weight':maximum,'validity':validity,
            'grid_shape':first.get('grid_shape'),'source_key':first.get('source_key'),
            'footprint_pixels':int(footprint.sum()),
            'source_support':{'basis':'photographic_same_exposure_weighted_estimate',
                'footprint_pixels':int(footprint.sum()),'supported_pixels':int((footprint&validity).sum()),
                'unsupported_pixels':int((footprint&~validity).sum())},
            'photographic_render':{'version':VERSION,'union':'signed weighted estimate times max footprint',
                'candidate_ids':[b['candidate_id'] for b in members],'complete_flux_recovery':False}})
    return sorted(result,key=lambda b:(b['y0'],b['x0']))
