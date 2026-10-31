"""Bounded display-domain star controls on verified separated layers.

This reduces a positive star residual with one RGB factor; it does not measure
stellar magnitudes, deconvolve stars, or improve the star-removal model. Diffuse
signal stays unchanged; the explicit full-range mode also restores negative
residuals as strength approaches one. Paired transients retain
their exact signed code difference, including under constrained headroom.
"""
from __future__ import annotations
from pathlib import Path
from contextlib import ExitStack
import hashlib, json, math, time, uuid
import numpy as np
import tifffile
from PIL import Image
from . import core, frame_pool, hashing, performance, photographic as photo, photographic_job as job
from .milkyway_gate import assess_evidence
from .milkyway import _inward_feather
from .star_layer_export import _bounded
from .star_reduction_curve import reduction_delta

VERSION = 'star-reduction-job-1'
# P1c (docs/EFFICIENCY_P1C_PREREG_20261005.md §3.5): the reduction's 16-row blocks (T-5b) and the advice residual
# check's 64-row blocks (T-5x (b)) on the frame pool, at most 8 workers; one unit's peak measured at W=8280 on the
# F48 fixture (the reduction at r=8: 48 rows read), scaled by W (and the halo, reduction_unit_bytes).
REDUCTION_WORKER_CAP = 8
REDUCTION_UNIT_BYTES = 45_000_000      # measured 28.6 MB (r=8) and 63.4 MB (r=32: 144 rows read)
REDUCTION_UNIT_WIDTH = 8280
ADVICE_UNIT_BYTES = 24_000_000         # measured 19.2 MB (serial unit peak)


def _number(value, name, low=0., high=1.):
    if type(value) not in (int, float) or not low <= value <= high:
        raise ValueError(f'{name} must be a builtin finite number in [{low}, {high}]')
    value = float(value)
    if not math.isfinite(value): raise ValueError(f'{name} must be finite')
    return value


def _snapshot(path, root):
    path = _bounded(path, root)
    data = path.read_bytes()
    return path, json.loads(data), hashlib.sha256(data).hexdigest()


def _first_changed(expected):
    """Index of the first ``{path: sha256}`` entry whose file no longer has that digest (None if all match).

    Same order and failure semantics as ``any(sha(Path(p)) != h for p, h in expected.items())``, hashed in parallel.
    """
    return hashing.first_mismatch([Path(p) for p in expected], list(expected.values()), source='star_reduction.inputs')


def bright_control(peak, radius):
    """Feather a bright neighbourhood, retaining every pixel's own peak.

    The caller supplies a 2*radius halo before cropping a streaming tile.
    A raw maximum filter alone would leave a square protection boundary.
    """
    if radius == 0: return peak.astype(np.uint16)
    expanded = photo.neighbour_extreme(peak, radius, maximum=True)
    feathered = photo.box_mean(expanded.astype(np.float64), radius)
    return np.rint(np.maximum(peak, feathered)).astype(np.uint16)


def _prepare_inputs(separation_result, allowed_root, milkyway_mask, milkyway_evidence, pool_stats=None):
    if milkyway_evidence is not None and milkyway_mask is None:
        raise ValueError("Milky Way evidence requires its bound mask")
    rp, prior, digest = _snapshot(separation_result, allowed_root)
    if prior.get('state') != 'succeeded' or prior.get('version') != 'star-separation-experiment-3':
        raise ValueError('a succeeded current separation is required')
    expected = {str(rp): digest}
    for path, value in prior['source_hashes'].items():
        expected[str(_bounded(path, allowed_root))] = value
    for item in prior['artifacts'].values():
        expected[str(_bounded(item['path'], allowed_root))] = item['sha256']
    paths = {key: _bounded(prior[key], allowed_root) for key in ('source','processing_base')}
    paired = bool(prior.get('transient_background'))
    for key, name in [('diffuse','diffuse_srgb16.tif'),('stars','stars_signed_codes.tif')]:
        paths[key] = _bounded(prior['artifacts'][name]['path'], allowed_root)
    if paired: paths['trails'] = _bounded(prior['artifacts']['trails_signed_codes.tif']['path'], allowed_root)
    elif paths['source'] != paths['processing_base']: raise ValueError('unpaired base differs from source')
    sky_path = _bounded(prior['masks']['sky'], allowed_root)
    protect_path = _bounded(prior['masks']['protected'], allowed_root)
    if any(str(p) not in expected for p in [*paths.values(),sky_path,protect_path]):
        raise ValueError('unbound separation input')
    if _first_changed(expected) is not None:
        raise ValueError('separation source or artifact changed')
    shape, icc, _ = job._display_source(paths['source'])
    bshape, _, bmeta = job._display_source(paths['processing_base'])
    dshape, _, dmeta = job._display_source(paths['diffuse'])
    if bshape != shape or dshape != shape: raise ValueError('layer geometry differs')
    if (dmeta.get('star_separation_for_sha256') != expected[str(paths['processing_base'])]
            or dmeta.get('sky_mask_sha256') != expected[str(sky_path)]
            or dmeta.get('protected_mask_sha256') != expected[str(protect_path)]):
        raise ValueError('diffuse layer identity differs')
    sky = job._mask(sky_path, shape[:2]); protect = job._mask(protect_path, shape[:2], binary=True)
    if paired:
        if (bmeta.get('transient_background_for_sha256') != expected[str(paths['source'])]
                or dmeta.get('separation_scope') != 'pure-sky-on-bound-transient-background'
                or dmeta.get('transient_source_sha256') != expected[str(paths['source'])]):
            raise ValueError('unbound paired transient background')
        job._pair_scope(paths['source'], paths['processing_base'], protect)
    job._starless_scope(paths['processing_base'],paths['diffuse'],sky,protect,paired_corridors=paired,
                        pool_stats=pool_stats)
    gate = None; mw = None
    if milkyway_mask is not None:
        mp = _bounded(milkyway_mask, allowed_root); expected[str(mp)] = hashing.sha256_file(mp, source='star_reduction.inputs')
        mw = job._mask(mp, shape[:2])
        evidence = None
        if milkyway_evidence is not None:
            ep, evidence, eh = _snapshot(milkyway_evidence, allowed_root); expected[str(ep)] = eh
        gate = assess_evidence(evidence, source_sha256=expected[str(paths['source'])],mask_sha256=expected[str(mp)])
    return rp, prior, digest, expected, paths, paired, shape, icc, sky, protect, gate, mw


def reduction_unit_bytes(width, radius):
    """unit_bytes of one 16-row reduction block read with its 2 r halo rows (P1c T-5b, R17): measured at W=8280 and
    r=8 (48 rows read), scaled by W x (16 + 4 r) / 48, never below the measured value."""
    scaled = -(-REDUCTION_UNIT_BYTES * int(width) * (16 + 4 * int(radius)) // (REDUCTION_UNIT_WIDTH * 48))
    return max(REDUCTION_UNIT_BYTES, scaled)


def _reduce_rows(s, b, d, r, t, out, bout, rout, weight, local_strength, *, shape, radius, extended, strength, low,
                 high, paired, check=None):
    """The reduction's 16-row blocks, each read with its 2 r halo, on the frame pool (P1c T-5b).

    A block writes only its own rows of out / bout / rout (R1); its integer
    statistics are folded in block order on the coordinator (R2), so results
    and statistics are identical for any worker count. ``check`` is called
    before each block. Returns, in the serial loop's order: changed,
    removed_sum, positive_sum, negative_count, full_requested, full_limited,
    full_remaining, new_zero, increased, restored_sum.
    """
    if extended:
        from .star_controls_curve import full_range_delta   # imported once here, never on a worker
    totals = {'changed':0,'removed_sum':0,'positive_sum':0,'negative_count':0,'full_requested':0,'full_limited':0,
              'full_remaining':0,'new_zero':0,'increased':0,'restored_sum':0}

    def block(y, unit):
        changed=0; removed_sum=0; positive_sum=0; negative_count=0
        full_requested=0; full_limited=0; full_remaining=0; new_zero=0; increased=0; restored_sum=0
        end=min(shape[0],y+16); lo=max(0,y-2*radius); hi=min(shape[0],end+2*radius)
        peak=np.maximum(b[lo:hi].astype(np.int32)-d[lo:hi].astype(np.int32),0).max(axis=2).astype(np.uint16)
        control=bright_control(peak,radius)[y-lo:end-lo]
        probe = unit.check if frame_pool.in_pool() else None   # R18: parallel path only, per tile
        for x in range(0,shape[1],4096):
            if probe: probe()
            sl=(slice(y,end),slice(x,x+4096)); B=b[sl]; S=s[sl]; D=d[sl]; R=r[sl]
            if not np.array_equal(R,B.astype(np.int32)-D.astype(np.int32)):
                raise ValueError('star residual differs from base minus diffuse')
            if paired and not np.array_equal(t[sl],S.astype(np.int32)-B.astype(np.int32)):
                raise ValueError('transient residual differs from source minus base')
            if extended:
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

    def fold(i, value):         # Python integers: exact in any order; folded in block order all the same (R2)
        for key, number in zip(totals, value):
            totals[key] = max(totals[key], number) if key == 'full_remaining' else totals[key] + number

    starts = range(0, shape[0], 16)
    unit_bytes = reduction_unit_bytes(shape[1], radius)
    workers = performance.frame_workers(len(starts), unit_bytes, REDUCTION_WORKER_CAP, blas=False)
    pool_stats = {}
    with performance.operation('star_reduction'):
        frame_pool.run_ordered(block, starts, workers=workers, name='star_reduction', check=check, on_result=fold,
                               unit_bytes=unit_bytes, blas=False, stats=pool_stats)
    performance.note_pool('star_reduction', pool_stats)
    from . import resources
    resources.shadow_pool('py.star_reduction', actual=pool_stats.get('workers'), items=len(starts),
                          width=-(-shape[1] * (16 + 4 * radius) // 48))
    return tuple(totals.values())


def _advice_residuals(r, b, d, s, t, shape, paired):
    """suggest_star_reduction's signed-layer check in 64-row blocks on the frame pool (P1c T-5x (b)); raises the first
    failing block's error, as the serial loop did."""
    def block(y, unit):
        sl=slice(y,y+64)
        if not np.array_equal(r[sl],b[sl].astype(np.int32)-d[sl].astype(np.int32)):
            raise ValueError('star residual differs from base minus diffuse')
        if paired and not np.array_equal(t[sl],s[sl].astype(np.int32)-b[sl].astype(np.int32)):
            raise ValueError('transient residual differs from source minus base')

    starts = range(0, shape[0], 64)
    unit_bytes = max(ADVICE_UNIT_BYTES, -(-ADVICE_UNIT_BYTES * int(shape[1]) // REDUCTION_UNIT_WIDTH))
    workers = performance.frame_workers(len(starts), unit_bytes, REDUCTION_WORKER_CAP, blas=False)
    pool_stats = {}
    with performance.operation('advice_residuals'):
        frame_pool.run_ordered(block, starts, workers=workers, name='advice_residuals', unit_bytes=unit_bytes,
                               blas=False, stats=pool_stats)
    performance.note_pool('starless_scope.advice_residuals', pool_stats)
    from . import resources
    resources.shadow_pool('py.starless_scope', actual=pool_stats.get('workers'), items=len(starts),
                          width=-(-shape[1] * 64 // 128))


def reduce_stars(separation_result, output_root, *, strength=.55,
                 bright_start=.10, bright_end=.45, milkyway_mask=None,
                 milkyway_evidence=None, milkyway_relief=.5, bright_radius=None,
                 allowed_root=None, background_strength=None, milkyway_offset=-.2, check=None, jpeg_quality=None):
    # ``check``: the workflow worker's cancel check (P1c T-5b), None from the CLI, MCP and GUI. A cancel removes
    # the staging folder once every pool has ended and every map is closed.
    # ``jpeg_quality``: star_reduced.jpg at that quality with 4:4:4 chroma (the GUI's setting); None keeps the
    # historical 95 with Pillow's default 4:2:0 (the workflow stage passes none: its delivery re-encodes).
    if jpeg_quality is not None:
        from .jpeg_estimate import validate_quality
        jpeg_quality = validate_quality(jpeg_quality)
    frame_pool.validate_environment()   # P1c T-5b: a malformed AESTER_FRAME_WORKERS / AESTER_OVERLAP, before any work
    strength = _number(strength, 'strength')
    extended = background_strength is not None
    background = _number(background_strength, 'background_strength') if extended else None
    offset = _number(milkyway_offset, 'milkyway_offset', -1., 1.)
    low = _number(bright_start, 'bright_start')
    high = _number(bright_end, 'bright_end')
    relief = _number(milkyway_relief, 'milkyway_relief')
    if low >= high: raise ValueError('bright_start must be below bright_end')
    if bright_radius is not None and (type(bright_radius) is not int or not 0 <= bright_radius <= 32):
        raise ValueError('bright_radius must be an integer in [0,32] or None')
    if milkyway_evidence is not None and milkyway_mask is None:
        raise ValueError('Milky Way evidence requires its bound mask')
    pool_stats={}
    rp, prior, digest, expected, paths, paired, shape, icc, sky, protect, gate, mw = _prepare_inputs(
        separation_result, allowed_root, milkyway_mask, milkyway_evidence, pool_stats=pool_stats)
    performance.note_pool('starless_scope.reduction',pool_stats)   # P1c T-5x (b): one note key per call site
    from . import resources
    resources.shadow_pool('py.starless_scope',actual=pool_stats.get('workers'),items=-(-shape[0]//128),
                          width=shape[1])
    radius = min(12,max(2,round(max(shape[:2])/1000))) if bright_radius is None else bright_radius
    unsafe = sky != 1
    if not paired: unsafe |= protect
    feather = max(2,round(max(shape[:2])*.004))
    weight = _inward_feather(unsafe,feather)
    local_strength = None
    if extended:
        local_strength = np.full(shape[:2], background, np.float64)
        if gate and gate['enabled']:
            local_strength = np.clip(local_strength + offset * mw.astype(np.float64), 0., 1.)
    elif gate and gate['enabled']: weight *= (1-relief*mw)
    weight[unsafe] = 0
    recipe = {'strength':strength,'bright_start':low,'bright_end':high,'milkyway_relief':relief,
              'bright_radius':radius,'sky_inward_feather_pixels':feather,
              'curve':'positive-residual-common-RGB-smoothstep-bright-protection',
              'brightness_domain':'encoded-sRGB residual max channel / 65535, not stellar magnitude',
              'negative_residual':'unchanged','diffuse':'unchanged','transients':'same integer removal from source and paired base'}
    if extended:
        recipe.update(background_strength=background, milkyway_offset=offset,
            milkyway_strength=float(np.clip(background+offset,0,1)) if gate and gate['enabled'] else background,
            curve='signed-full-range-v1-bright-release-0.75-to-1',
            negative_residual='restored toward diffuse above local strength 0.75',
            complete_removal='current separated stars only; safety boundary and paired headroom may limit')
    root = Path(output_root).resolve()
    if allowed_root is not None and not root.is_relative_to(Path(allowed_root).resolve(strict=True)):
        raise ValueError('output root is outside the allowed root')
    root.mkdir(parents=True, exist_ok=True)
    job._sweep_discarded(root)
    name = 'star-reduction-'+uuid.uuid4().hex[:12]; stage=root/('.partial-'+name); target=root/name; stage.mkdir()
    record = {'version':('star-reduction-job-2' if extended else VERSION),'state':'running','processing_directory':str(stage),
              'source':str(paths['source']),'separation_result':str(rp),'input_hashes':expected,
              'recipe':recipe,'milkyway_gate':gate,'milkyway_relief_applied':bool(gate and gate['enabled'] and ((extended and offset != 0 and np.any(mw > 0)) or (not extended and relief and strength))),
              'paired_trails':paired,'backend':prior['backend'],'domain':'encoded-sRGB display code editing',
              'physical_recovery':False,'quality_status':'needs_visual_review',
              'unchanged_diffuse':str(paths['diffuse']),'unchanged_trails':str(paths.get('trails',''))}
    if jpeg_quality is not None:
        record['jpeg_quality'] = jpeg_quality      # absent: the historical 95 / 4:2:0 star_reduced.jpg
    core.atomic_json(stage/'result.json', record)
    readers=[]; writers=[]; start=time.perf_counter()
    guard=ExitStack()
    cancelled=failed=False
    try:
        s=job._read_display(paths['source']); readers.append(s)
        b=job._read_display(paths['processing_base']); readers.append(b)
        d=job._read_display(paths['diffuse']); readers.append(d)
        r=job._read_display(paths['stars']); readers.append(r)
        t=job._read_display(paths['trails']) if paired else None
        if t is not None: readers.append(t)
        if r.shape!=shape or r.dtype!=np.int32 or (t is not None and (t.shape!=shape or t.dtype!=np.int32)):
            raise ValueError('residuals must be same-shape int32')
        meta={'scene_linear':False,'display_encoding_applied':True,'roworder':'TOP-DOWN','color_space':'encoded_sRGB',
              'source_sha256':expected[str(paths['source'])],'recipe':recipe,'experimental':True}
        sp=stage/'star_reduced_srgb16.tif'
        out=job._new_map(guard,sp,shape=shape,dtype=np.uint16,photometric='rgb',metadata=meta,
                           extratags=[(34675,'B',len(icc),icc,False)]); writers.append(out)
        bp=stage/'background_srgb16.tif'
        bout=job._new_map(guard,bp,shape=shape,dtype=np.uint16,photometric='rgb',metadata=meta,
                           extratags=[(34675,'B',len(icc),icc,False)]) if paired else None
        if bout is not None: writers.append(bout)
        rout=job._new_map(guard,stage/'stars_reduced_signed_codes.tif',shape=shape,dtype=np.int32,photometric='rgb',
                 metadata={'roworder':'TOP-DOWN','domain':'signed-display-code-difference','viewable_RGB':False,
                           'original_separation_sha256':digest,'recipe':recipe}); writers.append(rout)
        (changed, removed_sum, positive_sum, negative_count, full_requested, full_limited, full_remaining,
         new_zero, increased, restored_sum) = _reduce_rows(
            s, b, d, r, t, out, bout, rout, weight, local_strength, shape=shape, radius=radius, extended=extended,
            strength=strength, low=low, high=high, paired=paired, check=check)
        for writer in writers: writer.flush(); job._close_map(writer)
        writers=[]
        guard.close()
        if paired:
            # Bind the finished paired background to the actual new composite.
            # Only this new TIFF description is updated, never an input file.
            comment=json.dumps({**meta,'transient_background_for_sha256':hashing.sha256_file(sp,source='star_reduction.artifacts')})
            with hashing.writing(bp):   # rewrites the description of this new TIFF in place
                tifffile.tiffcomment(bp,comment)
        picture=job._read_display(sp)
        try:
            # P1c T-5x (c): the same expression per 128-row block into one uint8 array (no whole-image floats)
            pool_stats={}
            preview=job._jpeg_rows(
                picture,lambda rows:np.rint(rows.astype(np.float32)/np.float32(257)).astype(np.uint8),
                pool_stats=pool_stats)
            performance.note_pool('jpeg_rows.reduction',pool_stats)
            from . import resources
            resources.shadow_pool('py.jpeg_rows',actual=pool_stats.get('workers'),items=-(-shape[0]//128),
                                  width=shape[1])
            from .master_export import save_image_with_digest
            encoding=({'quality':95} if jpeg_quality is None else {'quality':jpeg_quality,'subsampling':0})
            save_image_with_digest(Image.fromarray(preview),stage/'star_reduced.jpg',source='star_reduction.artifacts',
                                   icc_profile=icc,**encoding)
        finally: job._close_map(picture)
        record.update(state='succeeded',processing_directory=str(target),seconds=time.perf_counter()-start,
                      changed_pixels=changed,removed_positive_code_sum=removed_sum,positive_code_sum=positive_sum,
                      negative_residual_channels_preserved=(negative_count if not extended else None),
                      input_negative_residual_channels=negative_count,
                      exact_transient_residual=True,diffuse_unchanged=True,
                      new_zero_channels=new_zero,new_above_range_channels=0,
                      increased_channels=increased,
                      restored_negative_code_sum=restored_sum,
                      controls_mode='background-offset' if extended else 'legacy-positive',
                      full_removal_requested_interior_pixels=full_requested,
                      full_removal_limited_pixels=full_limited,
                      full_removal_max_remaining_code=full_remaining,
                      full_removal_achieved_on_requested_interior=bool(full_requested and full_limited==0))
        files=[p for p in stage.iterdir() if p.is_file() and p.name!='result.json']
        digests=hashing.sha256_many(files,source='star_reduction.artifacts',fresh=True)
        record['artifacts']={p.name:{'path':str(target/p.name),'sha256':digest,'bytes':p.stat().st_size,
                                  'kind':'jpeg' if p.suffix=='.jpg' else 'tiff'}
                             for p,digest in zip(files,digests)}
        core.atomic_json(stage/'result.json',record)
        if _first_changed(expected) is not None:
            raise ValueError('input changed before publication')
        hashing.rename(stage,target)   # carries the ledger entries of the artifacts (P1b-5a)
        return record
    except Exception as exc:
        failed=True   # the staged pixels go below, once the maps are closed; result.json stays
        record.update(state='failed',error=str(exc),artifacts={},seconds=time.perf_counter()-start)
        core.atomic_json(stage/'result.json',record)
        raise
    except KeyboardInterrupt:
        cancelled=True   # WorkflowCancelled (P1c T-5b): no failed record; the staging folder goes below
        raise
    finally:
        try:
            try:
                for array in writers+readers: job._close_map(array)
            finally:
                guard.close()
        finally:
            if cancelled:   # every pool has ended and every map is closed: no staging folder is left
                job._discard_partial(stage)
            elif failed:    # the same failed folder at 1 and N workers (P1c §3.5)
                job._discard_failed(stage)


def suggest_star_reduction(separation_result, output_root, *, milkyway_mask=None,
                           milkyway_evidence=None, allowed_root=None):
    """Measure display-scale stellar crowding and publish advice, never edit.

    Uses the same authenticated inputs as reduction. It checks signed layer
    arithmetic, then samples positive residuals at an area-averaged 2048-long
    display scale. All exclusions are conservative and no source is changed.
    """
    from .star_density import analyse_density, recommend_controls
    pool_stats={}
    rp, prior, digest, expected, paths, paired, shape, icc, sky, protect, gate, mw = _prepare_inputs(
        separation_result, allowed_root, milkyway_mask, milkyway_evidence, pool_stats=pool_stats)
    performance.note_pool('starless_scope.advice',pool_stats)   # P1c T-5x (b): one note key per call site
    from . import resources
    resources.shadow_pool('py.starless_scope',actual=pool_stats.get('workers'),items=-(-shape[0]//128),
                          width=shape[1])
    root = Path(output_root).resolve()
    if allowed_root is not None and not root.is_relative_to(Path(allowed_root).resolve(strict=True)):
        raise ValueError('output root is outside the allowed root')
    root.mkdir(parents=True, exist_ok=True)
    name = 'star-advice-'+uuid.uuid4().hex[:12]
    stage=root/('.partial-'+name); target=root/name; stage.mkdir()
    record={'version':'star-advice-job-1','state':'running','processing_directory':str(stage),
            'separation_result':str(rp),'input_hashes':expected,'milkyway_gate':gate,
            'advice_only':True,'backend':prior['backend']}
    core.atomic_json(stage/'result.json',record)
    readers=[]; start=time.perf_counter()
    try:
        b=job._read_display(paths['processing_base']); readers.append(b)
        d=job._read_display(paths['diffuse']); readers.append(d)
        r=job._read_display(paths['stars']); readers.append(r)
        s=job._read_display(paths['source']); readers.append(s)
        t=job._read_display(paths['trails']) if paired else None
        if t is not None: readers.append(t)
        if r.shape!=shape or r.dtype!=np.int32 or (t is not None and (t.shape!=shape or t.dtype!=np.int32)):
            raise ValueError('residuals must be same-shape int32')
        _advice_residuals(r,b,d,s,t,shape,paired)   # P1c T-5x (b): 64-row blocks on the frame pool
        scale=min(1.,2048/max(shape[:2])); size=(max(8,round(shape[1]*scale)),max(8,round(shape[0]*scale)))
        positive=np.empty((size[1],size[0],3),np.float32)
        # One full scalar plane at a time, not a full float64 RGB copy.
        for c in range(3):
            plane=np.maximum(r[...,c],0).astype(np.float32)/np.float32(65535)
            positive[...,c]=photo.area_plane(plane,size)
            del plane
        # Exclude even paired meteor corridors from the density estimator,
        # although their stellar background remains editable in the renderer.
        valid=photo.area_plane(((sky==1)&~protect).astype(np.float32),size)==1
        galaxy=photo.area_plane(mw,size) if gate and gate['enabled'] else None
        metrics=analyse_density(positive,valid,galaxy)
        recommendation=recommend_controls(metrics,galaxy_enabled=bool(gate and gate['enabled']))
        record.update(state='succeeded',processing_directory=str(target),seconds=time.perf_counter()-start,
            metrics=metrics,recommendation=recommendation,
            note='视觉密度启发式建议，不是星表或物理星等；未自动应用。')
        core.atomic_json(stage/'result.json',record)
        if _first_changed(expected) is not None:
            raise ValueError('input changed before publication')
        stage.rename(target)
        return record
    except Exception as exc:
        record.update(state='failed',error=str(exc),seconds=time.perf_counter()-start)
        core.atomic_json(stage/'result.json',record)
        raise
    finally:
        for reader in readers: job._close_map(reader)
