"""Editable RGB16 TIFF layer bundle; signed originals remain authoritative.

This is encoded-sRGB additive editing, not scene-linear photometry. Separate
positive/negative layers avoid silently throwing negative residuals away.
No claim is made about an untested editor's 16-bit blend rounding or clipping.
"""
from __future__ import annotations
from pathlib import Path
from contextlib import ExitStack
import hashlib,json,shutil,time,uuid
import numpy as np
import tifffile
from . import core,frame_pool,hashing,performance,photographic_job as job
from .signed_layer_codec import split_signed_codes,compose_layer_codes

VERSION='editable-star-layers-1'
LAYER_NAMES=('00_background.tif','01_stars_subtract.tif','02_stars_add.tif',
             '03_trails_subtract.tif','04_trails_add.tif')
# P1c T-5x (a) (docs/EFFICIENCY_P1C_PREREG_20261005.md §3.5): the 16-row layer blocks on the frame pool, at most 8
# workers; one block's memory measured at W=8280, scaled by W (layers_unit_bytes).
LAYERS_WORKER_CAP=8
LAYERS_UNIT_BYTES=16_000_000        # measured 8.9 MB on the F48 fixture (serial unit peak)
LAYERS_UNIT_WIDTH=8280


def _bounded(path,root):
    value=Path(path).resolve(strict=True)
    if root is not None and not value.is_relative_to(Path(root).resolve(strict=True)):
        raise ValueError('layer source is outside the allowed output root')
    return value


def _first_changed(expected):
    """``any(sha(Path(p))!=h for p,h in expected.items())`` as an index (None: all match), hashed in parallel."""
    return hashing.first_mismatch([Path(p) for p in expected],list(expected.values()),source='star_layer_export.inputs')


def layers_unit_bytes(width):
    """unit_bytes of one 16-row layer block (P1c T-5x (a), R17): measured at W=8280, scaled by W, never below the
    measured value."""
    return max(LAYERS_UNIT_BYTES,-(-LAYERS_UNIT_BYTES*int(width)//LAYERS_UNIT_WIDTH))


def _export_rows(s,b,d,r,t,sky,protect,paired,writers,shape):
    """Verify and write the five layers in 16 x 4096 tiles; 16-row blocks on the frame pool (P1c T-5x (a)).

    A block writes only its own rows of the five layer maps (R1) and checks its
    tiles in the serial order, so the first failing block raises the serial
    loop's error; the two negative-channel counts are folded in block order.
    Returns (negative star channels, negative trail channels).
    """
    counts=[0,0]

    def block(y,unit):
        negative_stars=negative_trails=0
        probe=unit.check if frame_pool.in_pool() else None   # R18: parallel path only, per tile
        # <=65536 pixels per codec call, including very wide images.
        for x in range(0,shape[1],4096):
            if probe:probe()
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

    def fold(i,value):
        counts[0]+=value[0];counts[1]+=value[1]

    starts=range(0,shape[0],16)
    unit_bytes=layers_unit_bytes(shape[1])
    workers=performance.frame_workers(len(starts),unit_bytes,LAYERS_WORKER_CAP,blas=False)
    pool_stats={}
    with performance.operation('star_layers'):
        frame_pool.run_ordered(block,starts,workers=workers,name='star_layers',on_result=fold,unit_bytes=unit_bytes,
                               blas=False,stats=pool_stats)
    performance.note_pool('star_layers',pool_stats)
    from . import resources
    resources.shadow_pool('py.star_layers',actual=pool_stats.get('workers'),items=len(starts),width=shape[1])
    return counts[0],counts[1]


def export_star_layers(separation_result,output_root,*,allowed_root=None):
    """Export a completed version-3 separation without inference or enhancement.

    Keep background/positive/negative stars/positive/negative trails as normal
    ICC-tagged RGB16 TIFFs. Each negative layer precedes its positive partner
    in the mathematical stack. Verify every arithmetic step before publishing.
    """
    result_path=_bounded(separation_result,allowed_root)
    encoded=result_path.read_bytes();result_hash=hashlib.sha256(encoded).hexdigest()
    source_record=json.loads(encoded)
    if (source_record.get('state')!='succeeded' or
            source_record.get('version')!='star-separation-experiment-3'):
        raise ValueError('a succeeded current separation result is required')
    # All referenced source and artifact files are checked; record hashes are
    # integrity evidence, not a cryptographic signature or backend certification.
    expected={str(result_path):result_hash}
    for path,digest in source_record['source_hashes'].items():
        expected[str(_bounded(path,allowed_root))]=digest
    for item in source_record['artifacts'].values():
        expected[str(_bounded(item['path'],allowed_root))]=item['sha256']
    if _first_changed(expected) is not None:
        raise ValueError('separation source or artifact changed')
    source=_bounded(source_record['source'],allowed_root)
    base=_bounded(source_record['processing_base'],allowed_root)
    artifacts=source_record['artifacts'];paired=bool(source_record.get('transient_background'))
    diffuse=_bounded(artifacts['diffuse_srgb16.tif']['path'],allowed_root)
    stars=_bounded(artifacts['stars_signed_codes.tif']['path'],allowed_root)
    trails=_bounded(artifacts['trails_signed_codes.tif']['path'],allowed_root) if paired else None
    for p in (source,base):
        if str(p) not in expected:raise ValueError('unbound source reference')
    shape,icc,_=job._display_source(source)
    if job._display_source(base)[0]!=shape or job._display_source(diffuse)[0]!=shape:
        raise ValueError('layer geometry differs')
    sky_path=_bounded(source_record['masks']['sky'],allowed_root)
    protect_path=_bounded(source_record['masks']['protected'],allowed_root)
    if any(str(p) not in expected for p in (sky_path,protect_path)):
        raise ValueError('unbound masks')
    sky=job._mask(sky_path,shape[:2]);protect=job._mask(protect_path,shape[:2],binary=True)
    if paired:
        if job._display_source(base)[2].get('transient_background_for_sha256')!=expected[str(source)]:
            raise ValueError('unbound transient base')
        job._pair_scope(source,base,protect)
    elif base!=source:raise ValueError('unpaired processing base differs from source')
    root=Path(output_root).resolve();root.mkdir(parents=True,exist_ok=True)
    name='star-layers-'+uuid.uuid4().hex[:12];target=root/name;stage=root/('.partial-'+name);stage.mkdir()
    record={'version':VERSION,'state':'running','processing_directory':str(stage),
            'source':str(source),'separation_result':str(result_path),'input_hashes':expected,
            'domain':'encoded-sRGB code arithmetic','scene_linear':False,
            'physical_recovery':False,'separation_quality':'experimental_not_certified',
            'editor_roundtrip_tested':False,'paired_trails':paired,
            'backend':source_record['backend'],
            'unresolved_protected_sky_pixels':0 if paired else int(((sky==1)&protect).sum())}
    core.atomic_json(stage/'result.json',record)
    readers=[];writers=[];started=time.perf_counter()
    guard=ExitStack()
    failed=False
    try:
        s=job._read_display(source);readers.append(s)
        b=job._read_display(base);readers.append(b)
        d=job._read_display(diffuse);readers.append(d)
        r=job._read_display(stars);readers.append(r)
        t=job._read_display(trails) if paired else None
        if t is not None:readers.append(t)
        if r.shape!=shape or r.dtype!=np.int32 or (t is not None and (t.shape!=shape or t.dtype!=np.int32)):
            raise ValueError('signed residuals must be same-shape int32')
        for index,filename in enumerate(LAYER_NAMES):
            metadata={'roworder':'TOP-DOWN','scene_linear':False,'display_encoding_applied':True,
                      'color_space':'encoded_sRGB','layer_index_bottom_to_top':index,
                      'blend_operation':('normal' if index==0 else 'subtract' if index in (1,3) else 'add'),
                      'source_sha256':expected[str(source)],'experimental':True}
            writers.append(job._new_map(guard,stage/filename,shape=shape,dtype=np.uint16,
                photometric='rgb',metadata=metadata,extratags=[(34675,'B',len(icc),icc,False)]))
        negative_stars,negative_trails=_export_rows(s,b,d,r,t,sky,protect,paired,writers,shape)
        for writer in writers:writer.flush();job._close_map(writer)
        writers=[]
        guard.close()
        shutil.copyfile(source,stage/'reference_srgb16.tif')
        shutil.copyfile(stars,stage/'stars_exact_signed_codes.tif')
        if paired:shutil.copyfile(trails,stage/'trails_exact_signed_codes.tif')
        instructions='''# 可编辑星空图层包

这是独立的16位RGB TIFF图层，不是单个PSD。背景保留地景；去星范围限已验证的纯天空。
恒星和亮迹分开，负残差没有裁掉。原片和原线性母版保持。

从底到顶：00背景（正常）、01恒星负层（减去/Subtract）、02恒星正层（线性减淡-添加/Add）、
03亮迹负层（减去）、04亮迹正层（添加）。每层100%不透明度。同尺寸对齐，使用相同sRGB配置。
不要用“滤色/Screen”代替相加。正负层必须成对保留；缩星应改恒星对，不要改独立亮迹对。
每组负层在正层下面；未编辑时按此顺序的整数重合成已验证。

本包是编码sRGB显示域，不是线性光子/物理通量图层。int32 exact文件供程序精确重放，
不是普通照片编辑器的可见RGB层。16位编辑器的混合精度、Gamma和中间裁切可能不同；
尚未在Photoshop中导入读回认证，应先与reference_srgb16.tif比较。
大幅调整后的重合成不保证无裁切。当前模型仍可能留小暗/彩环，图层可编辑不等于无星质量认证。
'''
        (stage/'READ_ME.md').write_text(instructions,encoding='utf8')
        record.update(state='succeeded',processing_directory=str(target),seconds=time.perf_counter()-started,
            exact_integer_roundtrip=True,intermediate_clipping=False,
            negative_star_channels=negative_stars,negative_trail_channels=negative_trails,
            layers=[{'file':n,'blend':('normal' if i==0 else 'subtract' if i in (1,3) else 'add')}
                    for i,n in enumerate(LAYER_NAMES)])
        files=[p for p in stage.iterdir() if p.is_file() and p.name!='result.json']
        digests=hashing.sha256_many(files,source='star_layer_export.artifacts',fresh=True)
        record['artifacts']={p.name:{'path':str(target/p.name),'sha256':digest,'bytes':p.stat().st_size}
                            for p,digest in zip(files,digests)}
        core.atomic_json(stage/'result.json',record)
        if _first_changed(expected) is not None:
            raise ValueError('layer source changed before publication')
        hashing.rename(stage,target)   # carries the ledger entries of the artifacts (P1b-5a)
        return record
    except Exception as exc:
        failed=True   # the staged pixels go below, once the maps are closed; result.json stays
        record.update(state='failed',error=str(exc),artifacts={},seconds=time.perf_counter()-started)
        core.atomic_json(stage/'result.json',record)
        raise
    finally:
        try:
            try:
                for array in writers+readers:job._close_map(array)
            finally:
                guard.close()
        finally:
            if failed:   # P1c §3.5: a pool stopped by a block's error may have written rows past it
                job._discard_failed(stage)
