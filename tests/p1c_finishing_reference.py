"""The kept serial references of the P1c finishing pool sites (docs/EFFICIENCY_P1C_PREREG_20261005.md §3.5, §6.2 A5
(a)): the loops of the P1c base (aab6f21) verbatim, each lifted into a function with the inputs its pooled form takes.
The site tests compare every pooled form with these, so the references must never be "fixed" along with the product.

_<site>_reference naming follows §6.2 (a): milkyway_render, star_reduction, trail_pair, star_layers, starless_scope,
advice_residuals, jpeg_rows, inward_weight; and milkyway._inward_feather at aab6f21 (_inward_feather_reference), which
inward_weight's tiles run and which P1c split into single steps with abort probes between them.
"""
from __future__ import annotations

import time

import numpy as np

from starstack import milkyway as mw
from starstack import photographic as p
from starstack import photographic_job as job
from starstack.signed_layer_codec import compose_layer_codes, split_signed_codes
from starstack.star_reduction import bright_control
from starstack.star_reduction_curve import reduction_delta


def _milkyway_render_reference(rgb, sky, protect, mask, prepared, output, *, domain, row_size=None):
    """milkyway.render at aab6f21 (its render_seconds is a timing: tests mask it on both sides)."""
    started = time.perf_counter()
    Recipe, VERSION, validate_mask, _seal, _identities = mw.Recipe, mw.VERSION, mw.validate_mask, mw._seal, mw._identities
    p.validate_inputs(rgb, sky, protect, domain=domain)
    validate_mask(mask, rgb.shape[:2])
    if row_size is not None:
        p.integer_parameter(row_size, 'row_size', 8, 512)
    if not isinstance(output, np.ndarray) or output.shape != rgb.shape or output.dtype != rgb.dtype or not output.flags.writeable or not output.flags.c_contiguous:
        raise ValueError('output must be writable same-shape same-dtype contiguous storage')
    for source in (rgb, sky, protect, mask):
        if np.shares_memory(source, output) or (p.backing_file(source) is not None and p.backing_file(source) == p.backing_file(output)):
            raise ValueError('output must not alias an input')
    try:
        recipe = Recipe(**prepared['recipe']); recipe.validate()
        scale = min(1., recipe.work_long_edge/max(rgb.shape[:2]))
        shape = tuple(max(8,round(d*scale)) for d in rgb.shape[:2])
        if prepared['version'] != VERSION or prepared['domain'] != domain or tuple(prepared['shape']) != rgb.shape[:2]:
            raise ValueError('prepared version/domain/shape mismatch')
        for name, expected, low, high in [('safe', shape, 0, 1), ('delta_rgb', shape+(3,), -.075001, .085001)]:
            a = prepared[name]
            if not isinstance(a, np.ndarray) or a.dtype != np.float32 or a.shape != expected or not np.isfinite(a).all() or a.min()<low or a.max()>high:
                raise ValueError('invalid prepared map')
        if _seal(prepared) != prepared['seal'] or _identities(rgb,sky,protect,mask) != prepared['identities']:
            raise ValueError('source, masks, or prepared content changed')
    except (KeyError, TypeError, AttributeError, OverflowError) as exc:
        raise ValueError('malformed prepared data') from exc
    stats = dict(changed_pixels=0, outside_mask_changed_channels=0,
                 protected_changed_channels=0, ground_or_mixed_changed_channels=0,
                 new_endpoint_channels=0, max_abs_change=0.)
    h,w = rgb.shape[:2]
    rows = recipe.rows if row_size is None else row_size
    for y in range(0,h,rows):
        end = min(h,y+rows)
        current = rgb[y:end]
        base = current.astype(np.float32)
        if rgb.dtype == np.uint16: base *= np.float32(1/65535)
        weight = mask[y:end] * p.map_rows(prepared['safe'],(h,w),y,end) * np.float32(recipe.strength)
        locked = (mask[y:end]==0) | (sky[y:end]!=1) | protect[y:end] | (base.min(-1)<=0) | (base.max(-1)>=.98)
        if rgb.dtype == np.float32:
            locked |= np.any((base>0)&(base<np.finfo(np.float32).tiny*4),axis=-1)
        weight[locked] = 0
        delta = np.stack([p.map_rows(prepared['delta_rgb'][...,c],(h,w),y,end) for c in range(3)], -1) * weight[...,None]
        lower = np.float32(1/65535) if rgb.dtype == np.uint16 else np.nextafter(np.float32(0),np.float32(1))
        upper = np.float32(65534/65535) if rgb.dtype == np.uint16 else np.nextafter(np.float32(1),np.float32(0))
        head = np.divide(upper-base, delta, out=np.full_like(delta,np.inf), where=delta>0)
        foot = np.divide(base-lower, -delta, out=np.full_like(delta,np.inf), where=delta<0)
        alpha = np.clip(np.minimum(head,foot).min(-1),0,1)
        if rgb.dtype == np.float32:
            alpha = np.where(alpha<1,alpha*np.float32(1-8*np.finfo(np.float32).eps),alpha)
        new = base + delta * alpha[...,None]
        if rgb.dtype == np.uint16: new = np.rint(new*65535).astype(np.uint16)
        else: new = new.astype(np.float32)
        bypass = (weight==0) | np.all(delta==0,axis=-1)
        new[bypass] = current[bypass]
        if not np.isfinite(new).all(): raise ValueError('nonfinite result')
        output[y:end] = new
        diff = new != current
        stats['changed_pixels'] += int(np.any(diff,-1).sum())
        stats['outside_mask_changed_channels'] += int(diff[mask[y:end]==0].sum())
        stats['protected_changed_channels'] += int(diff[protect[y:end]].sum())
        stats['ground_or_mixed_changed_channels'] += int(diff[sky[y:end]!=1].sum())
        top = 65535 if rgb.dtype == np.uint16 else 1.
        stats['new_endpoint_channels'] += int(np.count_nonzero(((new==0)&(current>0))|((new==top)&(current<top))))
        stats['max_abs_change'] = max(stats['max_abs_change'],float(np.max(np.abs(new.astype(np.float32)-current.astype(np.float32)))))
    stats.update(render_seconds=time.perf_counter()-started, identity=stats['changed_pixels']==0)
    return stats


def _star_reduction_reference(s, b, d, r, t, out, bout, rout, weight, local_strength, *, shape, radius, extended,
                              strength, low, high, paired):
    """The 16-row loop of star_reduction.reduce_stars at aab6f21; returns its ten statistics in order."""
    changed=0; removed_sum=0; positive_sum=0; negative_count=0
    full_requested=0; full_limited=0; full_remaining=0; new_zero=0; increased=0; restored_sum=0
    for y in range(0,shape[0],16):
        end=min(shape[0],y+16); lo=max(0,y-2*radius); hi=min(shape[0],end+2*radius)
        peak=np.maximum(b[lo:hi].astype(np.int32)-d[lo:hi].astype(np.int32),0).max(axis=2).astype(np.uint16)
        control=bright_control(peak,radius)[y-lo:end-lo]
        for x in range(0,shape[1],4096):
            sl=(slice(y,end),slice(x,x+4096)); B=b[sl]; S=s[sl]; D=d[sl]; R=r[sl]
            if not np.array_equal(R,B.astype(np.int32)-D.astype(np.int32)):
                raise ValueError('star residual differs from base minus diffuse')
            if paired and not np.array_equal(t[sl],S.astype(np.int32)-B.astype(np.int32)):
                raise ValueError('transient residual differs from source minus base')
            if extended:
                from starstack.star_controls_curve import full_range_delta
                delta=full_range_delta(B,S,D,control[:,x:x+4096],weight[sl],local_strength[sl],bright_start=low,bright_end=high)
                endpoint=(local_strength[sl]==1)&(weight[sl]==1)
                remaining=R-delta
                limited=endpoint & np.any(remaining!=0,axis=2)
                full_requested+=int(endpoint.sum()); full_limited+=int(limited.sum())
                if np.any(limited): full_remaining=max(full_remaining,int(np.abs(remaining[limited]).max()))
            else:
                delta=reduction_delta(B,S,D,control[:,x:x+4096],weight[sl],strength=strength,bright_start=low,bright_end=high)
            new_zero+=int(((S.astype(np.int32)-delta==0)&(S>0)).sum())
            increased+=int((delta<0).sum())
            out[sl]=S.astype(np.int32)-delta
            if bout is not None: bout[sl]=B.astype(np.int32)-delta
            rout[sl]=R-delta
            changed+=int(np.any(delta!=0,axis=2).sum()); removed_sum+=int(np.maximum(delta,0).sum(dtype=np.int64))
            restored_sum+=int(np.maximum(-delta,0).sum(dtype=np.int64))
            positive_sum+=int(np.maximum(R,0).sum(dtype=np.int64)); negative_count+=int((R<0).sum())
    return (changed, removed_sum, positive_sum, negative_count, full_requested, full_limited, full_remaining,
            new_zero, increased, restored_sum)


def _trail_pair_reference(source, background, finished, output, paired_background, *, rows=128):
    """workflow_edit.add_signed_trails at aab6f21."""
    arrays = (source, background, finished, output, paired_background)
    if any(a.dtype != np.uint16 or a.shape != source.shape or a.ndim != 3 or a.shape[-1] != 3 for a in arrays):
        raise ValueError("亮迹配对需要同尺寸 uint16 RGB")
    if any(not a.flags.writeable for a in (output, paired_background)):
        raise ValueError("配对输出不可写")
    if any(np.shares_memory(a,b) for i,a in enumerate(arrays) for b in arrays[i+1:]):
        raise ValueError("配对输入与输出不可共享内存")
    stats = {"residual_pixels": 0, "headroom_limited_pixels": 0, "residual_mismatched_channels": 0,
             "outside_residual_changed_channels": 0, "physical_flux_preserved": False}
    for y in range(0, source.shape[0], rows):
        sl = slice(y,y+rows)
        s, b, proposed = (a[sl].astype(np.int32) for a in (source, background, finished))
        t = s-b
        active = np.any(t != 0, axis=-1)
        change = proposed-b
        head = np.divide(65534-np.maximum(s,b), change,
                         out=np.full(s.shape,np.inf), where=change>0)
        foot = np.divide(np.minimum(s,b)-1, -change,
                         out=np.full(s.shape,np.inf), where=change<0)
        alpha = np.clip(np.minimum(head,foot).min(-1),0,1)
        alpha[np.any((s==0)|(s==65535)|(b==0)|(b==65535),axis=-1)] = 0
        q = b+np.trunc(change*alpha[...,None]).astype(np.int32)
        q[~active] = proposed[~active]
        f = q+t
        if f.min()<0 or f.max()>65535 or q.min()<0 or q.max()>65535:
            raise ValueError("配对输出越界")
        if np.any((f-q)!=t):
            raise ValueError("亮迹有符号残差改变")
        output[sl], paired_background[sl] = f.astype(np.uint16), q.astype(np.uint16)
        stats["residual_pixels"] += int(active.sum())
        stats["headroom_limited_pixels"] += int(np.count_nonzero(active & np.any(q!=proposed,axis=-1)))
    return stats


def _star_layers_reference(s, b, d, r, t, sky, protect, paired, writers, shape):
    """The 16 x 4096 tile loop of star_layer_export.export_star_layers at aab6f21; returns the two counts."""
    negative_stars=negative_trails=0
    for y in range(0,shape[0],16):
        for x in range(0,shape[1],4096):
            sl=(slice(y,y+16),slice(x,x+4096));D=d[sl];B=b[sl].astype(np.int32);S=s[sl].astype(np.int32)
            R=r[sl];T=t[sl] if t is not None else np.zeros(R.shape,np.int32)
            if not np.array_equal(R,B-D.astype(np.int32)) or not np.array_equal(T,S-B):
                raise ValueError('residual decomposition is inconsistent')
            forbidden=(sky[sl]!=1)
            if not paired:forbidden|=protect[sl]
            if np.any((D.astype(np.int32)-B)[forbidden]):
                raise ValueError('diffuse layer changed outside its permitted scope')
            P,N=split_signed_codes(R);TP,TN=split_signed_codes(T)
            reconstructed=compose_layer_codes(D,P,N,TP,TN)
            if not np.array_equal(reconstructed,S):raise ValueError('editable layer roundtrip failed')
            for out,data in zip(writers,(D,N,P,TN,TP)):out[sl]=data
            negative_stars+=int((R<0).sum());negative_trails+=int((T<0).sum())
    return negative_stars,negative_trails


def _starless_scope_reference(processing_base, starless, sky, protect, *, paired_corridors=False):
    """photographic_job._starless_scope at aab6f21."""
    base=diffuse=None
    stats={'residual_pixels':0,'residual_min_code':0,'residual_max_code':0,
           'residual_outside_allowed_pixels':0}
    try:
        base=job._read_display(processing_base);diffuse=job._read_display(starless)
        if base.shape!=diffuse.shape or base.shape[:2]!=sky.shape or sky.shape!=protect.shape:
            raise ValueError('去星背景与基底、天空和保护蒙版尺寸不一致')
        for y in range(0,base.shape[0],128):
            residual=base[y:y+128].astype(np.int32)-diffuse[y:y+128].astype(np.int32)
            present=np.any(residual!=0,axis=-1)
            forbidden=(sky[y:y+128]!=1)
            if not paired_corridors:forbidden|=protect[y:y+128]
            if np.any(present&forbidden):
                raise ValueError('去星背景超出已验证的纯天空分层范围')
            stats['residual_pixels']+=int(present.sum())
            stats['residual_min_code']=min(stats['residual_min_code'],int(residual.min()))
            stats['residual_max_code']=max(stats['residual_max_code'],int(residual.max()))
        return stats
    finally:
        job._close_map(diffuse);job._close_map(base)


def _advice_residuals_reference(r, b, d, s, t, shape, paired):
    """The 64-row signed-layer check of star_reduction.suggest_star_reduction at aab6f21."""
    for y in range(0,shape[0],64):
        sl=slice(y,y+64)
        if not np.array_equal(r[sl],b[sl].astype(np.int32)-d[sl].astype(np.int32)):
            raise ValueError('star residual differs from base minus diffuse')
        if paired and not np.array_equal(t[sl],s[sl].astype(np.int32)-b[sl].astype(np.int32)):
            raise ValueError('transient residual differs from source minus base')


def _jpeg_rows_reference_reduction(picture):
    """star_reduction.reduce_stars' whole-image JPEG codes at aab6f21."""
    return np.rint(picture.astype(np.float32)/np.float32(257)).astype(np.uint8)


def _jpeg_rows_reference_trail_pair(final):
    """workflow_edit._write_pair's whole-image JPEG codes at aab6f21."""
    return np.rint(final.astype(np.float32)/257).astype(np.uint8)


def _inward_feather_reference(unsafe, width):
    """milkyway._inward_feather at aab6f21."""
    p.integer_parameter(width, 'feather width', 1, 4096)
    if not isinstance(unsafe, np.ndarray) or unsafe.ndim != 2 or unsafe.dtype != np.bool_ or min(unsafe.shape) < 1:
        raise ValueError('unsafe must be a nonempty 2D bool array')
    d = np.where(unsafe, 0, width+1).astype(np.float32)
    for axis in (0, 1):
        shape = [1, 1]; shape[axis] = unsafe.shape[axis]
        index = np.arange(unsafe.shape[axis], dtype=np.float32).reshape(shape)
        d = np.minimum.accumulate(d-index, axis=axis)+index
        reverse = np.flip(d+index, axis=axis)
        d = np.flip(np.minimum.accumulate(reverse, axis=axis), axis=axis)-index
    t = np.clip((d-1)/np.float32(width), 0, 1)
    return (t*t*(3-2*t)).astype(np.float32)


def _inward_weight_reference(weight_top_down, *, width_px=None, check=None):
    """region_blend.inward_weight at aab6f21 (with the kept feather, _inward_feather_reference)."""
    _inward_feather = _inward_feather_reference
    RECIPE = 'canvas-inward-feather-1'
    if (not isinstance(weight_top_down, np.ndarray) or weight_top_down.ndim != 2
            or min(weight_top_down.shape) < 1 or weight_top_down.dtype != np.float32):
        raise ValueError('composition weight must be a nonempty float32 2D array')
    if width_px is None:
        width_px = max(2, min(128, round(min(weight_top_down.shape) * .012)))
    if isinstance(width_px, bool) or not isinstance(width_px, int) or not 1 <= width_px <= 512:
        raise ValueError('composition feather width must be an integer from 1 to 512')
    height = weight_top_down.shape[0]
    output = np.empty_like(weight_top_down)
    changed = mixed = 0
    halo = width_px + 1
    for y in range(0, height, 256):
        if check:
            check()
        end = min(height, y + 256)
        y0, y1 = max(0, y - halo), min(height, end + halo)
        source = weight_top_down[y0:y1]
        if not np.isfinite(source).all() or np.any(source < 0) or np.any(source > 1):
            raise ValueError('composition weight must contain finite values in 0..1')
        safe = _inward_feather(source <= 0, width_px)[y-y0:end-y0]
        np.minimum(weight_top_down[y:end], safe, out=output[y:end])
        changed += int(np.count_nonzero(output[y:end] != weight_top_down[y:end]))
        mixed += int(np.count_nonzero((output[y:end] > 0) & (output[y:end] < 1)))
    receipt = {'recipe': RECIPE, 'width_px': width_px, 'changed_pixels': changed,
               'mixed_pixels': mixed, 'metric': 'capped-Manhattan-distance',
               'first_supported_pixel_weight': 0,
               'scope': 'only decreases already supported sky weight; excluded ground and no-coverage remain zero'}
    return output, receipt
