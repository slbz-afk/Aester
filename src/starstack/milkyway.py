"""Explicit, local Milky Way presentation in encoded sRGB.

The supplied mask locates the subject; this module never guesses its identity.
Smooth background changes are ADDED to the original, retaining the fine
residual rather than multiplying stars/noise. Not deconvolution or photometry.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import time
import numpy as np
from . import frame_pool, performance, photographic as p

VERSION = 'masked-milkyway-4'
# P1c T-5a (docs/EFFICIENCY_P1C_PREREG_20261005.md §3.5): render row blocks on the frame pool, at most 8 workers; one
# block's memory measured at 128 rows x 8280 columns (uint16 RGB), scaled by rows x W (render_unit_bytes).
RENDER_WORKER_CAP = 8
RENDER_UNIT_BYTES = 130_000_000          # measured 110.1 MB on the F48 fixture (serial unit peak)
RENDER_UNIT_ROWS, RENDER_UNIT_WIDTH = 128, 8280


@dataclass(frozen=True)
class Recipe:
    brightness: float = .020
    contrast: float = .60
    structure: float = .90
    color: float = .35
    strength: float = .60
    work_long_edge: int = 1536
    rows: int = 128

    def validate(self):
        for name, high in [('brightness', .06), ('contrast', 1.5),
                           ('structure', 2.), ('color', 1.), ('strength', 1.)]:
            p.real_parameter(getattr(self, name), name, 0., high)
        p.integer_parameter(self.work_long_edge, 'work_long_edge', 256, 2560)
        p.integer_parameter(self.rows, 'rows', 8, 512)


def validate_mask(mask, shape):
    if not isinstance(mask, np.ndarray) or mask.dtype != np.float32 or mask.shape != shape:
        raise ValueError('explicit same-size float32 Milky Way mask required')
    for y in range(0, shape[0], 128):
        a = mask[y:y+128]
        if not np.isfinite(a).all() or a.min() < 0 or a.max() > 1:
            raise ValueError('Milky Way mask must be finite 0..1')


def _identities(rgb, sky, protect, mask):
    return p.array_identities(('rgb', rgb), ('sky', sky), ('protect', protect), ('mask', mask))


def _seal(data):
    keys = ['version', 'domain', 'shape', 'recipe', 'identities', 'pivot']
    h = hashlib.sha256(json.dumps({k: data[k] for k in keys}, sort_keys=True,
                     allow_nan=False, default=lambda v: v.item()).encode())
    for key in ['delta_rgb', 'safe']:
        h.update(p.array_identity(data[key]).encode())
    return h.hexdigest()


def _inward_feather(unsafe, width, probe=None):
    """Smooth only INSIDE supported sky; no new dependency or image blur.

    A capped Manhattan distance transform uses separable prefix minima. The
    width is measured in working pixels, not native pixels. At the first safe
    pixel the weight is zero; a whole transition of ``width`` pixels follows.
    Image edges are not invented exclusions. This also fades around small
    holes, where a wide box-average alone would still jump at the hard lock.
    ``probe`` (a frame-pool worker's abort probe, P1c R18; None everywhere
    else) is called between the whole-tile steps (the same operations in the
    same order, one statement each, so that 16 workers' 1282-row tiles stay
    within R18's 1 s); it never changes the result.
    """
    p.integer_parameter(width, 'feather width', 1, 4096)
    if not isinstance(unsafe, np.ndarray) or unsafe.ndim != 2 or unsafe.dtype != np.bool_ or min(unsafe.shape) < 1:
        raise ValueError('unsafe must be a nonempty 2D bool array')
    # float32 directly: the same values as np.where(unsafe, 0, width+1).astype(np.float32) (width + 1 <= 4097 is
    # exact in float32), without the int64 intermediate (twice the bytes, the slowest step of a tile on 16 workers)
    d = np.where(unsafe, np.float32(0), np.float32(width+1))
    for axis in (0, 1):
        if probe: probe()
        shape = [1, 1]; shape[axis] = unsafe.shape[axis]
        index = np.arange(unsafe.shape[axis], dtype=np.float32).reshape(shape)
        d = d-index
        if probe: probe()
        d = np.minimum.accumulate(d, axis=axis)
        if probe: probe()
        d = d+index
        if probe: probe()
        reverse = np.flip(d+index, axis=axis)
        d = None
        if probe: probe()
        reverse = np.minimum.accumulate(reverse, axis=axis)
        if probe: probe()
        d = np.flip(reverse, axis=axis)-index
        reverse = None
    if probe: probe()
    t = (d-1)/np.float32(width)
    d = None
    if probe: probe()
    t = np.clip(t, 0, 1)
    if probe: probe()
    square = t*t
    if probe: probe()
    ramp = 3-2*t
    t = None
    if probe: probe()
    ramp = square*ramp
    square = None
    if probe: probe()
    return ramp.astype(np.float32)


def prepare(rgb, sky, protect, mask, recipe, *, domain):
    if type(recipe) is not Recipe:
        raise ValueError('recipe must be a Milky Way Recipe')
    recipe.validate()
    p.validate_inputs(rgb, sky, protect, domain=domain)
    validate_mask(mask, rgb.shape[:2])
    started = time.perf_counter()
    h, w = rgb.shape[:2]
    scale = min(1., recipe.work_long_edge / max(h, w))
    size = (max(8, round(w*scale)), max(8, round(h*scale)))
    thumb = np.stack([p.area_plane(rgb[..., c], size) for c in range(3)], -1)
    if rgb.dtype == np.uint16:
        thumb *= np.float32(1/65535)
    # No support-normalized convolution: its truncation near a mask changes
    # smooth slopes into spurious bands. Estimate over an unbroken rectangle,
    # then reject every output whose filter footprint touches excluded samples.
    excluded = p.area_plane(sky != 1, size) > 0
    point_radius = max(1, round(max(size)*.0013))
    sigma_small = max(1., max(size)*.0025)
    sigma_large = max(2., max(size)*.010)
    radius_large = max(1, int(round((np.sqrt(1+4*sigma_large**2)-1)/2)))
    support = 2*point_radius + 3*radius_large + 2
    unsafe = p.neighbour_extreme(excluded, support, maximum=True)
    # Keep the v3 measurement footprint/pivot. Changing the OUTPUT taper must
    # not retune contrast across the entire galaxy or make statistical holes.
    sky_safe = np.clip((p.box_mean((~unsafe).astype(np.float32), max(2, round(max(size)*.004)))-.5)*2,0,1)
    sky_safe = sky_safe*sky_safe*(3-2*sky_safe)
    sky_safe[unsafe] = 0
    # The old six-work-pixel half-ramp could cut through a still-positive MW
    # gradient. Fade over 8% of the working long edge, fully inside support.
    # Never enlarge the previous edit permission, including thin islands.
    feather_width = max(2, round(max(size)*.08))
    safe = np.minimum(sky_safe, _inward_feather(unsafe, feather_width))
    # Thin protected trails do not punch statistical holes into the estimator.
    # Opening suppresses compact positive features; rendering additionally
    # feathers a small neighbourhood and exactly bypasses the native mask.
    small_protect = p.area_plane(protect, size) > 0
    guard = p.neighbour_extreme(small_protect, 2, maximum=True)
    safe *= 1-p.box_mean(guard.astype(np.float32), max(2, round(max(size)*.002)))
    safe[small_protect] = 0
    opened = np.stack([p.neighbour_extreme(p.neighbour_extreme(thumb[..., c], point_radius),
                                         point_radius, maximum=True) for c in range(3)], -1)
    small = np.stack([p.gaussian_approx(opened[..., c], sigma_small) for c in range(3)], -1)
    broad_rgb = np.stack([p.gaussian_approx(opened[..., c], sigma_large) for c in range(3)], -1)
    broad = p.luminance(broad_rgb)
    fine = p.luminance(small)
    weights = p.area_plane(mask, size) * sky_safe
    eligible = weights > 0
    if eligible.any():
        values = broad[eligible]
        order = np.argsort(values, kind='stable')
        cumulative = np.cumsum(weights[eligible][order], dtype=np.float64)
        middle = min(len(order)-1, np.searchsorted(cumulative, cumulative[-1]*.5))
        pivot = float(values[order[middle]])
    else:
        pivot = 0.  # No supported editing pixels: render is an exact bypass.
    # Four separate controls: uniform lift, whole-subject contrast about a
    # fixed pivot, medium-scale structures, and existing smooth chroma.
    structure = fine - broad
    structure[np.abs(structure) < 2e-7] = 0
    light = recipe.brightness + recipe.contrast*(broad-pivot) + recipe.structure*structure
    light = np.clip(light, -.055, .075)
    # Only temper NEW darkening. A smooth budget relative to the original
    # broad background avoids a fixed black floor and does not lift natural
    # dust lanes wholesale. Positive changes and existing chroma are unchanged.
    # The rational shoulder joins the positive branch with slope one at zero.
    dark = np.maximum(-light, 0)
    budget = np.float32(.35) * np.maximum(broad, 0)
    softened = np.divide(dark * budget, dark + budget,
                         out=np.zeros_like(dark), where=(dark + budget) > 0)
    light = np.where(light < 0, -softened, light)
    # Enhance differences of existing colours across cloud/dust structures,
    # not the shared low-frequency airglow or a new per-channel white balance.
    chroma = (small-fine[...,None]) - (broad_rgb-broad[...,None])
    delta = light[..., None] + recipe.color * chroma
    # Do not infer stronger chroma from nearly black/white or very saturated
    # samples. Bound the photographic change independently of image content.
    delta = np.clip(delta, -.075, .085).astype(np.float32)
    result = dict(version=VERSION, domain=domain, shape=(h,w), recipe=asdict(recipe),
                  identities=_identities(rgb, sky, protect, mask), pivot=pivot,
                  delta_rgb=delta, safe=safe, sigma_work=[sigma_small,sigma_large],
                  support_work_pixels=support, work_shape=small.shape[:2],
                  output_feather_work_pixels=feather_width,
                  output_feather_metric='capped_Manhattan_inside_supported_sky',
                  prepare_seconds=time.perf_counter()-started)
    result['seal'] = _seal(result)
    return result


def render_unit_bytes(rows, width):
    """unit_bytes of one render row block (P1c R10/R17): the measured 128-row block at W=8280 scaled by rows x W, never
    below the measured value (payload and result are a row index and six numbers)."""
    scaled = -(-RENDER_UNIT_BYTES * int(rows) * int(width) // (RENDER_UNIT_ROWS * RENDER_UNIT_WIDTH))
    return max(RENDER_UNIT_BYTES, scaled)


def render_block(rgb, sky, protect, mask, prepared, output, recipe, y, end, probe=None):
    """Render rows [y, end) into ``output``; returns that block's statistics as (changed pixels, outside-mask,
    protected, ground-or-mixed and new-endpoint changed channels, the block's max abs change).

    A plain function on purpose (P1c T-5a, MW-D-WHEN-WHO): the Milky Way D scheme replaces this body. It writes only
    ``output[y:end]`` and reads the inputs and ``prepared`` (R1), runs on a frame-pool worker or in the caller, and must
    stay free of global or context side effects (R6). ``probe`` (a pool worker's abort probe; None on the serial path,
    whose rhythm of checks stays one per block) is called between its steps, so a stopping pool returns soon (R18).
    """
    h, w = rgb.shape[:2]
    current = rgb[y:end]
    base = current.astype(np.float32)
    if rgb.dtype == np.uint16: base *= np.float32(1/65535)
    if probe: probe()
    weight = mask[y:end] * p.map_rows(prepared['safe'],(h,w),y,end) * np.float32(recipe.strength)
    if probe: probe()
    locked = (mask[y:end]==0) | (sky[y:end]!=1) | protect[y:end] | (base.min(-1)<=0) | (base.max(-1)>=.98)
    if rgb.dtype == np.float32:
        locked |= np.any((base>0)&(base<np.finfo(np.float32).tiny*4),axis=-1)
    weight[locked] = 0
    if probe: probe()
    channels = []
    for c in range(3):
        channels.append(p.map_rows(prepared['delta_rgb'][...,c],(h,w),y,end))
        if probe: probe()
    delta = np.stack(channels, -1) * weight[...,None]
    channels = None
    if probe: probe()
    # Fade all channels together to avoid clipping (and consequent hue
    # changes). No new zero/white values, including uint16 rounding.
    lower = np.float32(1/65535) if rgb.dtype == np.uint16 else np.nextafter(np.float32(0),np.float32(1))
    upper = np.float32(65534/65535) if rgb.dtype == np.uint16 else np.nextafter(np.float32(1),np.float32(0))
    head = np.divide(upper-base, delta, out=np.full_like(delta,np.inf), where=delta>0)
    if probe: probe()
    foot = np.divide(base-lower, -delta, out=np.full_like(delta,np.inf), where=delta<0)
    if probe: probe()
    alpha = np.clip(np.minimum(head,foot).min(-1),0,1)
    if probe: probe()
    if rgb.dtype == np.float32:
        # Leave an inward numerical margin when gamut limiting. The
        # smallest subnormal lower bound vanishes in base-lower, and an
        # exact boundary ratio can cancel to zero in float32 arithmetic.
        alpha = np.where(alpha<1,alpha*np.float32(1-8*np.finfo(np.float32).eps),alpha)
    new = base + delta * alpha[...,None]
    if rgb.dtype == np.uint16: new = np.rint(new*65535).astype(np.uint16)
    else: new = new.astype(np.float32)
    if probe: probe()
    bypass = (weight==0) | np.all(delta==0,axis=-1)
    new[bypass] = current[bypass]
    if not np.isfinite(new).all(): raise ValueError('nonfinite result')
    output[y:end] = new
    if probe: probe()
    diff = new != current
    if probe: probe()
    changed, outside = int(np.any(diff,-1).sum()), int(diff[mask[y:end]==0].sum())
    protected, ground = int(diff[protect[y:end]].sum()), int(diff[sky[y:end]!=1].sum())
    if probe: probe()
    top = 65535 if rgb.dtype == np.uint16 else 1.
    endpoints = int(np.count_nonzero(((new==0)&(current>0))|((new==top)&(current<top))))
    if probe: probe()
    return (changed, outside, protected, ground, endpoints,
            float(np.max(np.abs(new.astype(np.float32)-current.astype(np.float32)))))


def render(rgb, sky, protect, mask, prepared, output, *, domain, row_size=None, check=None):
    """Render ``prepared`` into ``output`` in row blocks on the frame pool (P1c T-5a); ``check`` (the workflow's cancel
    check, None elsewhere) is called before each block and polled while blocks run. Output and statistics are
    identical for any worker count."""
    started = time.perf_counter()
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
    starts = range(0,h,rows)

    def block(y, unit):
        return render_block(rgb, sky, protect, mask, prepared, output, recipe, y, min(h,y+rows),
                            probe=unit.check if frame_pool.in_pool() else None)

    def fold(i, value):          # R2: the serial loop's accumulation, in block order, on the coordinator
        changed, outside, protected, ground, endpoints, largest = value
        stats['changed_pixels'] += changed
        stats['outside_mask_changed_channels'] += outside
        stats['protected_changed_channels'] += protected
        stats['ground_or_mixed_changed_channels'] += ground
        stats['new_endpoint_channels'] += endpoints
        stats['max_abs_change'] = max(stats['max_abs_change'],largest)

    # P1c T-5a: row blocks on the frame pool; the decision and the pool's figures go to the timings notes only (R19)
    unit_bytes = render_unit_bytes(rows, w)
    workers = performance.frame_workers(len(starts), unit_bytes, RENDER_WORKER_CAP, blas=False)
    pool_stats = {}
    with performance.operation('milkyway_render'):
        frame_pool.run_ordered(block, starts, workers=workers, name='milkyway_render', check=check, on_result=fold,
                               unit_bytes=unit_bytes, blas=False, stats=pool_stats)
    performance.note_pool('milkyway_render', pool_stats)
    from . import resources
    resources.shadow_pool('py.milkyway_render', actual=pool_stats.get('workers'), items=len(starts),
                          width=-(-w*rows//RENDER_UNIT_ROWS))
    stats.update(render_seconds=time.perf_counter()-started, identity=stats['changed_pixels']==0)
    return stats


def preserve_transient_residual(composite, background, finished_background, output):
    """Apply the same integer background change beneath and beside each trail.

    S=composite, B=background, Q=finished_background. Output F satisfies
    F-Q == S-B exactly in signed display-code arithmetic. Q is writable and is
    adjusted in place only where shared headroom limits its proposed change.
    This preserves the encoded additive layer, not scene-linear photometry.
    """
    arrays=(composite,background,finished_background,output)
    if any(not isinstance(a,np.ndarray) or a.dtype!=np.uint16 or a.ndim!=3 or a.shape[-1]!=3 for a in arrays):
        raise ValueError('paired layer requires uint16 H,W,3 arrays')
    if min(composite.shape[:2])<8 or any(a.shape!=composite.shape for a in arrays):
        raise ValueError('paired layer shapes must match')
    for a in (finished_background,output):
        if not a.flags.writeable or not a.flags.c_contiguous:
            raise ValueError('paired outputs must be writable contiguous arrays')
    for i,a in enumerate(arrays):
        for b in arrays[i+1:]:
            if np.shares_memory(a,b) or (p.backing_file(a) is not None and p.backing_file(a)==p.backing_file(b)):
                raise ValueError('paired arrays must not alias')
    stats=dict(changed_pixels=0, residual_mismatched_channels=0, gamut_limited_pixels=0,
               new_endpoint_channels=0, max_abs_change=0., protection_mode='exact_signed_display_residual')
    for y in range(0,composite.shape[0],128):
        sl=slice(y,min(y+128,composite.shape[0]))
        s=composite[sl].astype(np.int32)
        b=background[sl].astype(np.int32)
        proposed=finished_background[sl].astype(np.int32)-b
        change=proposed.copy()
        # Whole-pixel endpoint protection avoids altering hue at existing
        # saturated or exact-zero samples. Ordinary corridor background is
        # free to follow the surrounding Milky Way adjustment.
        hold=np.any((s==0)|(s==65535)|(b==0)|(b==65535),axis=-1)
        head=np.divide(65534-np.maximum(s,b),change,
                       out=np.full(s.shape,np.inf),where=change>0)
        foot=np.divide(np.minimum(s,b)-1,-change,
                       out=np.full(s.shape,np.inf),where=change<0)
        alpha=np.clip(np.minimum(head,foot).min(-1),0,1)
        alpha[hold]=0
        change=np.trunc(change*alpha[...,None]).astype(np.int32)
        new=s+change; q=b+change
        if new.min()<0 or new.max()>65535 or q.min()<0 or q.max()>65535:
            raise ValueError('paired gamut bound violated')
        mismatch=(new-q)!=(s-b)
        if mismatch.any(): raise ValueError('transient residual changed')
        endpoints=(((new==0)&(s>0))|((new==65535)&(s<65535))|
                   ((q==0)&(b>0))|((q==65535)&(b<65535)))
        if endpoints.any(): raise ValueError('paired new endpoint introduced')
        output[sl]=new.astype(np.uint16)
        finished_background[sl]=q.astype(np.uint16)
        stats['changed_pixels']+=int(np.any(change!=0,axis=-1).sum())
        stats['gamut_limited_pixels']+=int(np.any(change!=proposed,axis=-1).sum())
        stats['max_abs_change']=max(stats['max_abs_change'],float(np.abs(change).max()))
    stats['identity']=stats['changed_pixels']==0
    return stats
