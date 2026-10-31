"""Small display-only foreground adjustment; no recovered-detail claim.

The caller owns `output`, which may already contain other finishing. We multiply
its existing RGB by one source-guided scalar only in conservative pure ground.
Inputs use TOP-DOWN display-sRGB uint16; masks use the same native grid.
The caller must lock uncertain sky holes, branches and structures in protect.
sky==0 is a compositing weight, not proof of physical opaque ground: this is
not an automatic ground segmentation algorithm or permission to change all
pixels supplied by a temporal foreground layer.
"""
from pathlib import Path
import math
import os
import time
import numpy as np

VERSION='ground-photographic-v2'
GUARD=8
TAPER_RADIUS=16

def _backing_file(array):
    seen=set()
    while array is not None and id(array) not in seen:
        seen.add(id(array))
        filename=getattr(array,'filename',None)
        if filename is not None:return Path(filename).resolve()
        array=getattr(array,'base',None)
    return None

def _aliases(a,b):
    if np.shares_memory(a,b):return True
    x,y=_backing_file(a),_backing_file(b)
    if x is None or y is None:return False
    if x==y:return True
    try:return os.path.samefile(x,y)
    except OSError:return False

def _nonoverlapping_output(array):
    """Conservative sufficient layout check, including reversed/gapped views.

    Each axis must move by at least the entire byte span of all faster axes.
    Singleton axes do not create aliases. This accepts C/F-contiguous arrays,
    their transpose/reversal, and ordinary positive or negative step slices;
    exotic interleaved layouts not proved by this rule are rejected.
    """
    extent=array.itemsize
    for stride,size in sorted((abs(s),n) for s,n in zip(array.strides,array.shape) if n>1):
        if stride<extent:return False
        extent+=(size-1)*stride
    return True

def _mask_sum(sky,protect,start,stop,radius):
    """Exact counts in a square window; outside the image is ineligible."""
    h,w=sky.shape;lo=max(0,start-radius);hi=min(h,stop+radius)
    good=(sky[lo:hi]==0)&~protect[lo:hi]
    pad=np.pad(good,((max(0,radius-start),max(0,stop+radius-h)),(radius,radius)),constant_values=False)
    integral=np.pad(pad.astype(np.int32),((1,0),(1,0))).cumsum(axis=0,dtype=np.int32).cumsum(axis=1,dtype=np.int32)
    size=2*radius+1
    return integral[size:,size:]-integral[:-size,size:]-integral[size:,:-size]+integral[:-size,:-size]

def _luminance(a):
    f=np.asarray(a,np.float32)/np.float32(65535.)
    return f[...,0]*np.float32(.2126)+f[...,1]*np.float32(.7152)+f[...,2]*np.float32(.0722)

def finish_ground(image,sky,protect,output,*,strength=.6,rows=64):
    """Modify independent `output` in place and return an audit record.

    The image is immutable source guidance, not a replacement for output.
    output must already be initialized. Strength zero preserves it exactly.
    Read-only inputs and provably nonoverlapping output views are accepted;
    aliasing/backing-file
    overlap with output is rejected even for strength zero. Rows is 1..512.
    protect must also encode the complement of the caller's photographic
    allowed region; all uncertain semantic regions stay protected.
    """
    t=time.perf_counter()
    if not isinstance(image,np.ndarray) or image.dtype!=np.uint16 or image.ndim!=3 or image.shape[2]!=3 or min(image.shape[:2])<1:
        raise ValueError('Image must be nonempty HWC uint16 display-sRGB')
    if not isinstance(output,np.ndarray) or output.dtype!=np.uint16 or output.shape!=image.shape or not output.flags.writeable:
        raise ValueError('Output must be independent writable matching uint16 RGB')
    if not _nonoverlapping_output(output):
        raise ValueError('Output layout must prove nonoverlapping pixel/channel storage')
    if not isinstance(sky,np.ndarray) or sky.dtype!=np.float32 or sky.shape!=image.shape[:2]:
        raise ValueError('Sky must be matching float32 weights')
    if not isinstance(protect,np.ndarray) or protect.dtype!=bool or protect.shape!=sky.shape:
        raise ValueError('Protect must be a matching boolean mask')
    if type(rows) is not int or not 1<=rows<=512:
        raise ValueError('Rows must be a builtin integer in 1..512')
    if isinstance(strength,(bool,np.bool_)) or not isinstance(strength,(int,float,np.integer,np.floating)):
        raise ValueError('Strength must be a finite number in 0..1')
    try:s=float(strength)
    except (OverflowError,TypeError,ValueError):raise ValueError('Strength must be a finite number in 0..1') from None
    if not math.isfinite(s) or not 0<=s<=1:raise ValueError('Strength must be a finite number in 0..1')
    if any(_aliases(a,output) for a in (image,sky,protect)):
        raise ValueError('Output must not alias any input or its backing file')
    h,w=sky.shape
    # Validate the complete mask before the first output mutation.
    for y in range(0,h,128):
        a=sky[y:y+128]
        if not np.isfinite(a).all() or np.any(a<0) or np.any(a>1):raise ValueError('Sky weights must be finite in 0..1')
    report={'version':VERSION,'domain':'uint16 encoded display-sRGB','roworder':'TOP-DOWN',
      'strength':s,'rows':rows,'identity':s==0,'guard_pixels':GUARD,'taper_radius_pixels':TAPER_RADIUS,
      'allowed':'sky==0, unprotected, complete 17x17 ground neighbourhood',
      'method':'source-guided small luminance residual shrinkage and shadow lift; one scalar on existing output RGB',
      'maximum_gain':1.,'minimum_gain':1.,'eligible_pixels':0,'changed_pixels':0,
      'new_white_channels':0,'new_black_channels':0,'complete_detail_recovery':False}
    if s==0:
        report['seconds']=time.perf_counter()-t;return report
    for y in range(0,h,rows):
        end=min(h,y+rows)
        allowed=_mask_sum(sky,protect,y,end,GUARD)==(2*GUARD+1)**2
        if not np.any(allowed):continue
        density=_mask_sum(sky,protect,y,end,TAPER_RADIUS).astype(np.float32)/np.float32((2*TAPER_RADIUS+1)**2)
        taper=np.clip((density-np.float32(25/33))/np.float32(8/33),0,1)
        taper=taper*taper*(3-2*taper)
        allowed&=taper>0
        if not np.any(allowed):continue
        lo=max(0,y-1);hi=min(h,end+1)
        luma=_luminance(image[lo:hi])
        luma=np.pad(luma,((int(y==0),int(end==h)),(1,1)),mode='edge')
        centre=luma[1:-1,1:-1]
        blur=(luma[:-2,:-2]+2*luma[:-2,1:-1]+luma[:-2,2:]
             +2*luma[1:-1,:-2]+4*centre+2*luma[1:-1,2:]
             +luma[2:,:-2]+2*luma[2:,1:-1]+luma[2:,2:])*np.float32(1/16)
        detail=centre-blur
        threshold=np.float32(.003)+np.float32(.02)*centre
        gate=1/(1+(detail/threshold)**2)**2
        smooth=centre-np.float32(.20*s)*gate*detail
        shadow=np.clip((np.float32(.4)-centre)/np.float32(.35),0,1)**2
        lift=1+np.float32(.18*s)*shadow*centre/(centre+np.float32(.025))
        ratio=np.divide(smooth*lift,centre,out=np.ones_like(centre),where=centre>0)
        ratio=np.clip(ratio,.94,1.18)
        gain=1+(ratio-1)*taper
        current=output[y:end]
        maximum=current.max(axis=2).astype(np.float64)
        # No newly clipped whites; already-white observations remain permitted.
        ceiling=np.divide(65534.,maximum,out=np.full(maximum.shape,1.18),where=maximum>0)
        ceiling[maximum==65535]=1.
        gain=np.minimum(gain.astype(np.float64),ceiling)
        candidate=np.rint(current.astype(np.float64)*gain[...,None]).astype(np.uint16)
        report['eligible_pixels']+=int(allowed.sum())
        report['changed_pixels']+=int(np.count_nonzero(np.any(candidate!=current,axis=2)&allowed))
        report['new_white_channels']+=int(np.count_nonzero((candidate==65535)&(current!=65535)&allowed[...,None]))
        report['new_black_channels']+=int(np.count_nonzero((candidate==0)&(current!=0)&allowed[...,None]))
        report['minimum_gain']=min(report['minimum_gain'],float(gain[allowed].min()))
        report['maximum_gain']=max(report['maximum_gain'],float(gain[allowed].max()))
        current[allowed]=candidate[allowed]
    report['seconds']=time.perf_counter()-t
    return report
