"""Experimental display-domain star/background decomposition.

The compact estimator is a limited morphological baseline, not complete star
removal. A provider can supply a different same-size uint16 estimate. Both
routes retain the exact SIGNED residual, including negative estimates, and
bind source/masks/backend identity. Roundtrip is not a quality certification.
"""
from __future__ import annotations
from pathlib import Path
from contextlib import ExitStack
import json, time, uuid, os
import numpy as np
import tifffile
from . import core, hashing, photographic as p
from . import photographic_job as job

VERSION = 'star-separation-experiment-3'


def compact_estimate(rgb, output, *, radius=None, rows=128):
    """Suppress compact positive structures; preserve unselected samples.

    Three-channel opening supplies the local floor; a smooth brightness
    response selects where to use it. This can also suppress compact real
    nebular knots. Large halos and crowded star fields remain limitations.
    Row overlap covers BOTH stages of opening and the one-pixel noise filter.
    """
    if not isinstance(rgb,np.ndarray) or rgb.dtype!=np.uint16 or rgb.ndim!=3 or rgb.shape[-1]!=3 or min(rgb.shape[:2])<8:
        raise ValueError('uint16 H,W,3 at least 8x8 required')
    if not isinstance(output,np.ndarray) or output.shape!=rgb.shape or output.dtype!=np.uint16 or not output.flags.c_contiguous or not output.flags.writeable:
        raise ValueError('writable contiguous uint16 output required')
    if np.shares_memory(rgb,output):
        raise ValueError('output aliases source')
    source_file=p.backing_file(rgb); output_file=p.backing_file(output)
    if source_file is not None and output_file is not None:
        try:
            aliases=os.path.samefile(source_file,output_file)
        except OSError as exc:
            raise ValueError('cannot verify source/output backing-file identity') from exc
        if aliases:
            raise ValueError('output aliases source')
    p.integer_parameter(rows,'rows',1,512)
    if radius is None: radius=max(2,round(max(rgb.shape[:2])*.0008))
    p.integer_parameter(radius,'radius',1,32)
    h,w=rgb.shape[:2]
    for y in range(0,h,rows):
        end=min(h,y+rows); lo=max(0,y-2*radius-2); hi=min(h,end+2*radius+2)
        tile=rgb[lo:hi].astype(np.float32)/np.float32(65535)
        opened=np.stack([p.neighbour_extreme(p.neighbour_extreme(tile[...,c],radius),radius,maximum=True) for c in range(3)],-1)
        lum=p.luminance(tile); floor=p.luminance(opened)
        # A fixed code floor plus local noise estimate avoids selecting every
        # tiny quantization/noise fluctuation in a dark smooth background.
        local_noise=p.box_mean(np.abs(lum-p.box_mean(lum,1)),radius)
        threshold=np.maximum(np.float32(4/65535),np.float32(1.5)*local_noise)
        response=np.maximum(lum-floor,0)
        t=np.clip((response-threshold)/(threshold*3+np.float32(.008)),0,1)
        weight=t*t*(3-2*t)
        estimate=tile+(opened-tile)*weight[...,None]
        converted=np.rint(np.clip(estimate,0,1)*65535).astype(np.uint16)
        converted[weight==0]=rgb[lo:hi][weight==0]
        output[y:end]=converted[y-lo:end-lo]
    return {'backend':'compact-opening-v1','radius_native':radius,'complete_star_removal':False}


def separate_stars(source,output_root,sky_mask,protected_mask,*,provider=None,provider_identity=None,
                   transient_background=None):
    """Write a bound diffuse estimate + int32 code residual to a NEW job.

    Optional provider(source_array, writable_estimate, stage_directory) returns
    a JSON-safe receipt. The caller supplies a nonempty identity including the
    backend/version/model hashes; no arbitrary backend runs automatically.
    Providers are an internal extension point, not a serialized executable.
    With a source-bound transient background, the provider sees that background
    and may remove stars throughout pure sky, including beneath trail corridors.
    Trails are saved separately; a corridor is not a star-removal exclusion.
    """
    if provider is not None and (not callable(provider) or not isinstance(provider_identity,dict) or not provider_identity):
        raise ValueError('provider requires a callable and explicit identity')
    identity={'backend':'compact-opening-v1','model':None,'uses_learned_model':False} if provider is None else provider_identity
    identity=json.loads(json.dumps(identity,allow_nan=False))
    learned=identity.get('uses_learned_model')
    if learned is not None and type(learned) is not bool:
        raise ValueError('uses_learned_model must be a boolean or unknown')
    paths=[Path(x).resolve(strict=True) for x in (source,sky_mask,protected_mask)]
    source,sky_mask,protected_mask=paths
    paired=transient_background is not None
    if paired:
        transient_background=Path(transient_background).resolve(strict=True)
        paths.append(transient_background)
    processing_base=transient_background if paired else source
    hashes=dict(zip(map(str,paths),hashing.sha256_many(paths,source='star_separation.inputs')))
    shape,icc,_=job._display_source(source)
    sky=job._mask(sky_mask,shape[:2]); protect=job._mask(protected_mask,shape[:2],binary=True)
    pair_validation=None
    if paired:
        bg_shape,_,bg_meta=job._display_source(processing_base)
        if bg_shape!=shape or bg_meta.get('transient_background_for_sha256')!=hashes[str(source)]:
            raise ValueError('transient background is not bound to this same-size source')
        pair_validation=job._pair_scope(source,processing_base,protect)
    scope='pure-sky-on-bound-transient-background' if paired else 'pure-sky-outside-protection'
    root=Path(output_root).resolve(); root.mkdir(parents=True,exist_ok=True)
    name='separation-'+uuid.uuid4().hex[:12]; stage=root/('.partial-'+name); stage.mkdir(); target=root/name
    record={'state':'running','version':VERSION,'source_hashes':hashes,'source':str(source),
            'backend':identity,'shape':list(shape),'domain':'display-sRGB','experimental':True,
            'physical_recovery':False,'quality_status':'not_accepted','roundtrip_is_quality_test':False,
            'learned_star_separation':learned,'processing_base':str(processing_base),
            'separation_scope':scope,'pair_validation':pair_validation,
            'masks':{'sky':str(sky_mask),'protected':str(protected_mask)}}
    if paired:record['transient_background']={'path':str(processing_base),'sha256':hashes[str(processing_base)]}
    core.atomic_json(stage/'result.json',record)
    image=diffuse=residual=composite=trails=None; start=time.perf_counter()
    guard=ExitStack()
    try:
        image=job._read_display(processing_base)
        composite=job._read_display(source) if paired else image
        metadata={'roworder':'TOP-DOWN','scene_linear':False,'display_encoding_applied':True,
                  'color_space':'encoded_sRGB','star_separation_for_sha256':hashes[str(processing_base)],
                  'sky_mask_sha256':hashes[str(sky_mask)],'protected_mask_sha256':hashes[str(protected_mask)],
                  'backend':identity,'experimental':True,'physical_recovery':False,
                  'learned_star_separation':learned,'separation_scope':scope}
        if paired:metadata['transient_source_sha256']=hashes[str(source)]
        diffuse=job._new_map(guard,stage/'diffuse_srgb16.tif',shape=shape,dtype=np.uint16,photometric='rgb',
                              metadata=metadata,extratags=[(34675,'B',len(icc),icc,False)])
        if provider is None:
            provider_receipt=compact_estimate(image,diffuse)
        else:
            # imread fallback (compressed TIFF) is writable, unlike a read-only
            # memmap. Never expose it as writable to an external estimator.
            source_view=image.view(); source_view.flags.writeable=False
            source_identity=p.array_identity(image)
            provider_receipt=provider(source_view,diffuse,stage)
            if p.array_identity(image)!=source_identity:
                raise ValueError('provider changed source samples in memory')
        # Freeze the receipt before attaching it to the record. A bad provider
        # receipt must not also poison the failed-state record in the handler.
        record['provider_receipt']=json.loads(json.dumps(provider_receipt,allow_nan=False))
        residual=job._new_map(guard,stage/'stars_signed_codes.tif',shape=shape,dtype=np.int32,photometric='rgb',
                    metadata={'roworder':'TOP-DOWN','domain':'signed-display-code-difference','viewable_RGB':False,
                              'source_sha256':hashes[str(processing_base)],'invariant':'processing_base = diffuse + signed stars'})
        if paired:
            trails=job._new_map(guard,stage/'trails_signed_codes.tif',shape=shape,dtype=np.int32,photometric='rgb',
                metadata={'roworder':'TOP-DOWN','domain':'signed-display-code-difference','viewable_RGB':False,
                          'source_sha256':hashes[str(source)],'background_sha256':hashes[str(processing_base)],
                          'invariant':'source = diffuse + signed stars + signed trails'})
        count=negative=0
        for y in range(0,shape[0],128):
            s=image[y:y+128]; d=diffuse[y:y+128]
            # A verified trail-free base needs no trail exclusion. Without
            # one, preserve the conservative whole-pixel protection policy.
            locked=(sky[y:y+128]!=1)
            if not paired:locked|=protect[y:y+128]
            d[locked]=s[locked]
            r=s.astype(np.int32)-d.astype(np.int32);residual[y:y+128]=r
            count+=int(np.any(r!=0,-1).sum());negative+=int((r<0).sum())
            assert np.array_equal(d.astype(np.int32)+r,s.astype(np.int32))
            if paired:
                t=composite[y:y+128].astype(np.int32)-s.astype(np.int32)
                trails[y:y+128]=t
                if not np.array_equal(d.astype(np.int32)+r+t,composite[y:y+128]):
                    raise ValueError('three-layer roundtrip mismatch')
        diffuse.flush();residual.flush()
        job._close_map(diffuse);diffuse=None;job._close_map(residual);residual=None
        if trails is not None:trails.flush();job._close_map(trails);trails=None
        guard.close()
        record.update(state='succeeded',seconds=time.perf_counter()-start,roundtrip_exact=True,
                      residual_pixels=count,negative_residual_channels=negative)
        files=list(stage.glob('*.tif'))
        digests=hashing.sha256_many(files,source='star_separation.artifacts',fresh=True)
        record['artifacts']={f.name:{'path':str(target/f.name),'sha256':digest,'bytes':f.stat().st_size}
                             for f,digest in zip(files,digests)}
        core.atomic_json(stage/'result.json',record)
        # The success record write may take time; bind inputs after that write,
        # immediately before publication, just as the finishing entry does.
        if hashing.first_mismatch(paths,[hashes[str(x)] for x in paths],source='star_separation.inputs') is not None:
            raise ValueError('input changed during separation')
        hashing.rename(stage,target)   # carries the ledger entries of the artifacts (P1b-5a)
        return record
    except KeyboardInterrupt as exc:
        record.update(state='cancelled',error=str(exc) or 'KeyboardInterrupt',
                      cancel_reason='keyboard_interrupt',seconds=time.perf_counter()-start,artifacts={})
        record.pop('roundtrip_exact',None)
        core.atomic_json(stage/'result.json',record)
        raise
    except Exception as exc:
        record.update(state='failed',error=str(exc),seconds=time.perf_counter()-start,artifacts={})
        core.atomic_json(stage/'result.json',record)
        raise
    finally:
        try:
            job._close_map(trails)
            if paired:job._close_map(composite)
            job._close_map(residual);job._close_map(diffuse);job._close_map(image)
        finally:
            guard.close()
