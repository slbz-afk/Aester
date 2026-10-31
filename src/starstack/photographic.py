"""Isolated display-sRGB photographic finishing, never scene-linear processing.

NumPy/Pillow only. Inputs and masks are TOP-DOWN, zero-based pixel centres.
The caller retains input ownership. Source arrays are never mutated. All RGB
channels receive the same scalar gain before rounding. No per-channel WB.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import time
import hashlib
import json
import numbers
from pathlib import Path
import numpy as np
from PIL import Image
from . import hashing

VERSION = "display-photographic-prototype-4"


def real_parameter(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError(name + ' must be a real scalar (not bool)')
    try:
        finite = np.isfinite(float(value))
    except (ValueError, TypeError, OverflowError):
        finite = False
    if not finite or not low <= value <= high:
        raise ValueError(name + ' outside finite permitted range')


def integer_parameter(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(name + ' must be an exact builtin int in range')


def array_identity(array):
    """Hash the full logical array, in bounded row blocks, including its layout.

    SHA-256 of ``str((shape, dtype.str))`` in ASCII, then the C-ordered bytes in
    blocks of 128 rows (``hashing.array_digest``, the same bytes and errors).
    """
    return hashing.array_digest(array, source='photographic.array_identity')


def array_identities(*named):
    """``{name: array_identity(array)}`` for ``(name, array)`` pairs, in order.

    The arrays are hashed in parallel; the digests are the same and the error
    raised is the one of the first failing array, as in the serial loop.
    """
    digests = hashing.arrays_digest_many([array for _, array in named], source='photographic.array_identity')
    return dict(zip((name for name, _ in named), digests))


def prepared_identity(prepared):
    semantic = {k: prepared[k] for k in ('version', 'shape', 'domain', 'recipe', 'source_identity', 'star_threshold')}
    # Recipe accepts numpy real scalars; normalize only for serializing identity.
    h = hashlib.sha256(json.dumps(semantic, sort_keys=True, default=lambda v: v.item(), allow_nan=False).encode())
    for key in ('gain', 'taper'):
        h.update(array_identity(prepared[key]).encode())
    return h.hexdigest()


@dataclass(frozen=True)
class Recipe:
    mode: str = "photographic"
    detail_strength: float = 1.0
    star_strength: float = 1.0
    medium_gain: float = 1.0
    broad_gain: float = 1.25
    work_long_edge: int = 1280
    rows: int = 128

    def validate(self):
        if type(self.mode) is not str or self.mode not in ("natural", "photographic"):
            raise ValueError("mode must be natural or photographic")
        for name in ('detail_strength', 'star_strength'):
            real_parameter(getattr(self, name), name, 0, 2)
        for name in ('medium_gain', 'broad_gain'):
            real_parameter(getattr(self, name), name, 0, 3)
        integer_parameter(self.work_long_edge, 'work_long_edge', 64, 2560)
        integer_parameter(self.rows, 'rows', 8, 512)


def validate_inputs(rgb, sky_weight, protect, *, domain):
    if type(domain) is not str or domain != "display-sRGB":
        raise ValueError("explicit display-sRGB domain required; not scene-linear")
    if not isinstance(rgb, np.ndarray) or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("RGB must have H,W,3 shape")
    if min(rgb.shape[:2]) < 8 or rgb.dtype not in (np.dtype('uint16'), np.dtype('float32')):
        raise ValueError("need uint16 or float32 RGB, at least 8x8")
    if not isinstance(sky_weight, np.ndarray) or sky_weight.shape != rgb.shape[:2] or sky_weight.dtype != np.float32:
        raise ValueError("same-size float32 sky_weight required")
    if not isinstance(protect, np.ndarray) or protect.shape != rgb.shape[:2] or protect.dtype != np.bool_:
        raise ValueError("same-size bool protect mask required")
    for y in range(0, rgb.shape[0], 128):
        a, w = rgb[y:y+128], sky_weight[y:y+128]
        if not np.isfinite(a).all() or (rgb.dtype == np.float32 and (a.min() < 0 or a.max() > 1)):
            raise ValueError("RGB must be finite display 0..1 (uint16 code range)")
        if not np.isfinite(w).all() or w.min() < 0 or w.max() > 1:
            raise ValueError("sky_weight must be finite 0..1")


def box_mean(a, radius):
    """Separable clamped box, float64 prefix sums only on work-map/tile arrays."""
    if radius <= 0:
        return np.asarray(a, np.float32).copy()
    out = np.asarray(a, np.float32)
    k = 2 * radius + 1
    for axis in (0, 1):
        pads = [(0, 0), (0, 0)]
        pads[axis] = (radius, radius)
        p = np.pad(out, pads, mode='edge')
        cs = np.cumsum(p, axis=axis, dtype=np.float64)
        pads[axis] = (1, 0)
        cs = np.pad(cs, pads, mode='constant')
        hi, lo = [slice(None)]*2, [slice(None)]*2
        hi[axis], lo[axis] = slice(k, None), slice(None, -k)
        out = ((cs[tuple(hi)] - cs[tuple(lo)]) / k).astype(np.float32)
    return out


def gaussian_approx(a, sigma):
    # Three equal boxes have summed variance r*(r+1), near sigma**2.
    r = max(1, int(round((np.sqrt(1+4*sigma*sigma)-1)/2)))
    out = a
    for _ in range(3):
        out = box_mean(out, r)
    return out


def resize_plane(a, size):
    return np.asarray(Image.fromarray(a).resize(size, Image.Resampling.BILINEAR), np.float32)


def area_plane(a, size):
    return np.array(Image.fromarray(np.asarray(a, np.float32)).resize(size, Image.Resampling.BOX), dtype=np.float32, copy=True)


def luminance(a):
    # Encoded-domain brightness proxy, explicitly not radiometric luminance.
    return a[..., 0]*np.float32(.2126) + a[..., 1]*np.float32(.7152) + a[..., 2]*np.float32(.0722)


def prepare(rgb, sky_weight, protect, recipe, *, domain):
    if type(recipe) is not Recipe:
        raise ValueError('recipe must be a Recipe')
    recipe.validate()
    validate_inputs(rgb, sky_weight, protect, domain=domain)
    t = time.perf_counter()
    h, w = rgb.shape[:2]
    scale = min(1., recipe.work_long_edge/max(h, w))
    size = (max(8, round(w*scale)), max(8, round(h*scale)))
    thumb = np.stack([resize_plane(rgb[..., c], size) for c in range(3)], axis=-1)
    if rgb.dtype == np.uint16:
        thumb *= np.float32(1/65535)
    # Exclude mixed/ground samples BEFORE image reduction: their encoded RGB
    # already contains foreground, so masking a resized RGB is insufficient.
    pure = sky_weight == np.float32(1)
    weight = area_plane(pure, size)
    small_protect = area_plane(protect, size) > 0
    valid_native = pure & ~protect
    valid = area_plane(valid_native, size)
    plane = np.empty((h,w), np.float32)
    bright_native = np.empty((h,w), bool)
    for y in range(0,h,128):
        a = rgb[y:y+128].astype(np.float32)
        if rgb.dtype == np.uint16: a *= np.float32(1/65535)
        plane[y:y+128] = luminance(a) * valid_native[y:y+128]
        bright_native[y:y+128] = a.max(axis=-1) > .82
    weighted_lum = area_plane(plane, size)
    bright_seed = area_plane(bright_native, size) > 0
    # Exclude a conservatively expanded native PSF neighbourhood before band
    # estimation. The extra work pixel covers BOX footprints and rounding.
    bright_radius = int(np.ceil(32*max(size[0]/w,size[1]/h)))+1
    bright_guard = neighbour_extreme(bright_seed,bright_radius,maximum=True)
    valid[bright_guard] = 0
    weighted_lum[bright_guard] = 0
    small_protect |= bright_guard
    del plane, pure, valid_native, bright_native
    lum = np.divide(weighted_lum, valid, out=np.zeros_like(valid), where=valid>0)
    # Remove the relative compact-positive (white top-hat) response from the
    # STRUCTURE ESTIMATOR only. Grayscale opening has no brightness threshold:
    # an isolated unsaturated point source cannot suddenly enter a band-pass
    # when crossing .82. It can also remove compact real emission knots, hence
    # this is photographic background structure estimation, not photometry.
    point_radius = max(1,int(np.ceil(32*max(size[0]/w,size[1]/h))))
    eroded = neighbour_extreme(np.where(valid>0,lum,np.inf),point_radius)
    eroded[(valid<=0)|~np.isfinite(eroded)] = -np.inf
    opened = neighbour_extreme(eroded,point_radius,maximum=True)
    opened[~np.isfinite(opened)] = 0
    opened = np.minimum(np.maximum(opened,0),lum)
    point_response = lum-opened
    weighted_lum = opened*valid
    sigmas = [s * max(size)/2560. for s in (16, 48, 144)]
    smooth = []
    for s in sigmas:
        den = gaussian_approx(valid, s)
        num = gaussian_approx(weighted_lum, s)
        smooth.append(np.divide(num, den, out=np.zeros_like(lum), where=den > .02))
    bands = (smooth[0]-smooth[1], smooth[1]-smooth[2])
    delta = recipe.medium_gain*bands[0] + recipe.broad_gain*bands[1]
    delta[np.abs(delta) <= np.finfo(np.float32).eps*8*np.maximum(lum,.06)] = 0
    # Brighten/darken broad structure, without sharpening the finest residual.
    gain = np.clip(delta / np.maximum(lum, .06), -.32, .38).astype(np.float32)
    gain[valid==0] = 0
    # Retreat smoothly from ground and all protected corridors. Exact hard
    # protection is also applied on the original grid during rendering.
    sky_interior = np.clip((gaussian_approx(weight, max(2., sigmas[0]))-.88)/.12, 0, 1)
    protected_soft = np.clip(gaussian_approx(small_protect.astype(np.float32), max(2., sigmas[0]))*5, 0, 1)
    taper = (sky_interior * (1-protected_soft)).astype(np.float32)
    # Fixed spatial sample, independent of render strip boundaries.
    residual=[]
    xx=np.arange(2,w-2,8)
    for yy in np.linspace(2,h-3,min(64,h-4)).astype(int):
        c=luminance(rgb[yy,xx].astype(np.float32))
        b=(luminance(rgb[yy-1,xx].astype(np.float32))+luminance(rgb[yy+1,xx].astype(np.float32))+
           luminance(rgb[yy,xx-1].astype(np.float32))+luminance(rgb[yy,xx+1].astype(np.float32)))*.25
        ok=np.ones(len(xx),bool)
        for dy,dx in ((0,0),(-1,0),(1,0),(0,-1),(0,1)):
            ok &= (sky_weight[yy+dy,xx+dx]==1)&~protect[yy+dy,xx+dx]
        residual.append((c-b)[ok])
    residual=np.concatenate(residual)
    if rgb.dtype==np.uint16: residual=residual/65535
    noise=float(np.median(np.abs(residual-np.median(residual))))*1.4826 if residual.size else 0.
    result = {"gain": gain, "taper": taper, "shape": (h,w), "recipe": asdict(recipe),
            "version":VERSION, "domain":domain,
            "source_identity": array_identities(('rgb',rgb),('sky_weight',sky_weight),('protect',protect)),
            "prepare_seconds": time.perf_counter()-t, "work_shape": lum.shape,
            "sigma_work_px": sigmas, "thumbnail": thumb,"star_threshold":max(.012,noise*4),
            "noise_proxy":noise,"noise_sample_count":int(residual.size),
            "small_protected_pixels":int(small_protect.sum()),
            "bright_guard_work_pixels":int(bright_guard.sum()),
            "bright_guard_minimum_native_px":32,
            "point_response_method":"masked grayscale opening / white top-hat",
            "point_response_radius_work_px":point_radius,
            "point_response_minimum_native_px":32,
            "point_response_max":float(point_response.max())}
    result['seal'] = prepared_identity(result)
    result['prepare_seconds'] = time.perf_counter()-t
    return result


def map_rows(small, shape, start, stop):
    """Bilinear centre-aligned sampling; independent of rendering strip size."""
    h, w = shape
    sh, sw = small.shape
    xx = np.clip((np.arange(w, dtype=np.float64)+.5)*sw/w-.5, 0, sw-1)
    yy = np.clip((np.arange(start, stop, dtype=np.float64)+.5)*sh/h-.5, 0, sh-1)
    x0, y0 = xx.astype(np.int32), yy.astype(np.int32)
    fx, fy = (xx-x0).astype(np.float32), (yy-y0).astype(np.float32)
    x1, y1 = np.minimum(x0+1,sw-1), np.minimum(y0+1,sh-1)
    a = small[y0[:,None], x0] * (1-fx) + small[y0[:,None], x1] * fx
    b = small[y1[:,None], x0] * (1-fx) + small[y1[:,None], x1] * fx
    return a*(1-fy[:,None]) + b*fy[:,None]


def neighbour_extreme(a, radius, maximum=False):
    out = a.copy()
    for axis in (0,1):
        p = np.pad(out, [(radius,radius) if j==axis else (0,0) for j in (0,1)], mode='edge')
        result = out.copy()
        op = np.maximum if maximum else np.minimum
        for j in range(2*radius+1):
            sl = [slice(None), slice(None)]
            sl[axis] = slice(j,j+a.shape[axis])
            op(result, p[tuple(sl)], out=result)
        out = result
    return out


def validate_prepared(prepared, rgb, sky_weight, protect):
    if type(prepared) is not dict:
        raise ValueError('prepared must be a prepare result')
    try:
        recipe = Recipe(**prepared['recipe'])
        recipe.validate()
        if prepared['version'] != VERSION or prepared['domain'] != 'display-sRGB' or tuple(prepared['shape']) != rgb.shape[:2]:
            raise ValueError('prepared version/domain/shape mismatch')
        expected_scale = min(1., recipe.work_long_edge/max(rgb.shape[:2]))
        expected_shape = tuple(max(8,round(d*expected_scale)) for d in rgb.shape[:2])
        for name,low,high in (('gain',-.320001,.380001),('taper',0,1)):
            a=prepared[name]
            if not isinstance(a,np.ndarray) or a.dtype!=np.float32 or a.shape!=expected_shape or not np.isfinite(a).all() or a.min()<low or a.max()>high:
                raise ValueError('invalid prepared '+name)
        real_parameter(prepared['star_threshold'], 'star_threshold', .012, 4.)
        if type(prepared['seal']) is not str or prepared_identity(prepared) != prepared['seal']:
            raise ValueError('prepared content changed')
        current=array_identities(('rgb',rgb),('sky_weight',sky_weight),('protect',protect))
        if current != prepared['source_identity']:
            raise ValueError('prepared source/mask/protect identity mismatch')
    except (KeyError, TypeError, OverflowError, AttributeError) as exc:
        raise ValueError('malformed prepared data') from exc
    return recipe


def backing_file(array):
    while isinstance(array, np.ndarray):
        if isinstance(array, np.memmap) and array.filename is not None:
            return Path(array.filename).resolve()
        array = array.base
    return None


def render(rgb, sky_weight, protect, prepared, output, *, domain, detail_strength=None, star_strength=None, row_size=None):
    t=time.perf_counter()
    validate_inputs(rgb,sky_weight,protect,domain=domain)
    if row_size is not None: integer_parameter(row_size,'row_size',8,512)
    if not isinstance(output,np.ndarray) or output.shape != rgb.shape or output.dtype != rgb.dtype or not output.flags.writeable or not output.flags.c_contiguous:
        raise ValueError('output must be writable C-contiguous same-shape same-dtype storage')
    for source in (rgb,sky_weight,protect):
        if np.shares_memory(source,output) or (backing_file(source) is not None and backing_file(source)==backing_file(output)):
            raise ValueError('output must not alias any input storage')
    recipe = validate_prepared(prepared,rgb,sky_weight,protect)
    if detail_strength is not None or star_strength is not None:
        values=asdict(recipe)
        if detail_strength is not None: values['detail_strength']=detail_strength
        if star_strength is not None: values['star_strength']=star_strength
        recipe=Recipe(**values)
    recipe.validate()
    if recipe.detail_strength == 0 and recipe.star_strength == 0:
        output[:]=rgb
        return {'render_seconds':time.perf_counter()-t,'identity':True}
    factor=.42 if recipe.mode=='natural' else 1.
    rows = recipe.rows if row_size is None else row_size
    h,w=rgb.shape[:2]
    stats={'changed_pixels':0,'new_endpoint_channels':0,'protected_changed_channels':0,'ground_changed_channels':0,'mixed_changed_channels':0,'star_eligible_pixels':0,'gain_min':1.,'gain_max':1.}
    for y in range(0,h,rows):
        end=min(h,y+rows); lo=max(0,y-8); hi=min(h,end+8)
        a=rgb[lo:hi].astype(np.float32)
        if rgb.dtype==np.uint16: a*=np.float32(1/65535)
        lum=luminance(a)
        local=box_mean(lum,2)
        hp=lum-local
        # Robust threshold estimated from the unenhanced tile; no sharpening.
        threshold=prepared['star_threshold']
        stars=np.clip((hp-threshold)/(threshold*2),0,1)
        maxrgb=a.max(axis=-1)
        bright=neighbour_extreme(maxrgb,3,maximum=True)>.82
        # A candidate core needs itself and at least two connected neighbours
        # above half of the strongest local prominence. This holds even weak
        # unresolved 1/2-pixel cores, rather than only large single-pixel spikes.
        local_peak=neighbour_extreme(hp,1,maximum=True)
        half=np.maximum(local_peak*.5,threshold)
        support=np.zeros(hp.shape,np.uint8)
        padded=np.pad(hp,1,mode='edge')
        for dy in range(3):
            for dx in range(3):
                support += padded[dy:dy+hp.shape[0],dx:dx+hp.shape[1]] >= half
        unresolved=hp > np.maximum(.10, local*.8)
        stars[bright | unresolved | (support<3) | (hp<half)]=0
        eroded=neighbour_extreme(lum,1)
        star_delta=np.clip((eroded-lum)/np.maximum(lum,.03),-.3,0)*stars
        sl=slice(y-lo,end-lo)
        base=a[sl]; current=rgb[y:end]
        tap=map_rows(prepared['taper'],(h,w),y,end)
        gain=1 + factor*recipe.detail_strength*map_rows(prepared['gain'],(h,w),y,end)*tap
        gain += factor*recipe.star_strength*np.float32(.35)*star_delta[sl]*tap
        # All controls act on the same RGB multiplier. Smooth weight, then
        # exact protected/ground bypass. Bright star neighbourhood is protected.
        gain=1 + (gain-1)*sky_weight[y:end]
        locked=protect[y:end] | (sky_weight[y:end]!=1) | bright[sl]
        gain[locked]=1
        maximum=base.max(axis=-1)
        cap=np.float32(65534/65535) if rgb.dtype==np.uint16 else np.nextafter(np.float32(1),np.float32(0))
        # Hold old saturated pixels. Previously unsaturated channels cannot
        # become saturated through this operation, including uint16 rounding.
        gain[maximum>=cap]=1
        gain=np.minimum(gain,np.divide(cap,maximum,out=np.ones_like(maximum),where=maximum>0))
        if rgb.dtype==np.uint16:
            minimum=np.where(current>0,current,np.uint16(65535)).min(axis=-1).astype(np.float32)
            gain=np.maximum(gain,np.float32(.5001)/minimum)
        else:
            subnormal=np.any((base>0)&(base<np.finfo(np.float32).tiny*4),axis=-1)
            gain[subnormal]=1
        gain[locked | (maximum>=cap) | (maximum==0)]=1
        new=base*gain[...,None]
        if rgb.dtype==np.uint16: new=np.rint(new*65535).astype(np.uint16)
        else: new=new.astype(np.float32)
        # Bit-exact preservation for the bypass, including 16-bit round trips.
        new[gain==1]=current[gain==1]
        if not np.isfinite(new).all():
            raise ValueError('nonfinite rendered data')
        output[y:end]=new
        top=65535 if rgb.dtype==np.uint16 else 1.
        stats['changed_pixels']+=int(np.count_nonzero(np.any(new!=current,axis=-1)))
        stats['new_endpoint_channels']+=int(np.count_nonzero(((new==top)&(current<top))|((new==0)&(current>0))))
        stats['protected_changed_channels']+=int(np.count_nonzero(new[protect[y:end]]!=current[protect[y:end]]))
        stats['ground_changed_channels']+=int(np.count_nonzero(new[sky_weight[y:end]==0]!=current[sky_weight[y:end]==0]))
        mixed=(sky_weight[y:end]>0)&(sky_weight[y:end]<1)
        stats['mixed_changed_channels']+=int(np.count_nonzero(new[mixed]!=current[mixed]))
        stats['star_eligible_pixels']+=int(np.count_nonzero((stars[sl]>0)&~locked))
        stats['gain_min']=min(stats['gain_min'],float(gain.min()));stats['gain_max']=max(stats['gain_max'],float(gain.max()))
    stats.update(render_seconds=time.perf_counter()-t,identity=False,mode=recipe.mode)
    return stats
