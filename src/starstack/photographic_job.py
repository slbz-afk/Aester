"""Explicit display-sRGB finishing with immutable source and atomic publication."""
from __future__ import annotations
from pathlib import Path
from contextlib import ExitStack
from dataclasses import asdict,replace
import io,json,shutil,time,uuid,struct,hashlib
import numpy as np
from PIL import Image,ImageCms
import tifffile
from . import core,frame_pool,hashing,performance

VERSION='photographic-job-1'
MILKYWAY_VERSION='milkyway-job-3'
# P1c T-5x (b) (docs/EFFICIENCY_P1C_PREREG_20261005.md §3.5): _starless_scope's 128-row blocks on the frame pool, at
# most 8 workers; one block's memory measured at W=8280 (uint16 RGB), scaled by W (scope_unit_bytes).
SCOPE_WORKER_CAP=8
SCOPE_UNIT_BYTES=45_000_000         # measured 38.4 MB on the F48 fixture (serial unit peak)
SCOPE_UNIT_WIDTH=8280
# P1c T-5x (c): whole-image 16-to-8-bit JPEG conversions in 128-row blocks (star reduction, trail add-back) into one
# preallocated uint8 array, at most 8 workers; one block's memory measured at W=8280, scaled by W (jpeg_unit_bytes).
JPEG_WORKER_CAP=8
JPEG_UNIT_BYTES=32_000_000          # measured 25.6 MB on the F48 fixture (serial unit peak)
DISCARD_MARKER='cancelled.json'     # a cancelled step's staging folder that could not be removed at once
DISCARD_SECONDS=1.                  # its bounded removal retries: well inside the workflow's 3 s cancel budget (§6.4)
FINISH_JPEG_QUALITY=97              # the finished JPEG when the caller sets no jpeg_quality (4:4:4)


def _icc_tags(data):
    """Bounded read of the colorimetric tags; descriptions are not color proof."""
    if len(data)<132 or data[36:40]!=b'acsp' or data[16:20]!=b'RGB ' or data[20:24]!=b'XYZ ':
        raise ValueError('需要RGB至XYZ的本产品sRGB配置')
    size=struct.unpack_from('>I',data)[0]
    count=struct.unpack_from('>I',data,128)[0]
    if size>len(data) or size<132 or count>256 or 132+12*count>size:
        raise ValueError('无效ICC长度或标签表')
    tags={}
    for i in range(count):
        key,offset,length=struct.unpack_from('>4sII',data,132+12*i)
        if key in tags or offset<132+12*count or length<8 or offset+length>size:
            raise ValueError('无效ICC标签')
        tags[key]=data[offset:offset+length]
    return tags


def _validate_srgb(icc):
    profile=ImageCms.ImageCmsProfile(io.BytesIO(icc))
    if profile.profile.xcolor_space.strip()!='RGB':
        raise ValueError('需要RGB sRGB配置')
    actual=_icc_tags(icc)
    reference=_icc_tags(ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes())
    for key in (b'rXYZ',b'gXYZ',b'bXYZ',b'wtpt',b'chad',b'rTRC',b'gTRC',b'bTRC'):
        if key not in actual or actual[key]!=reference[key]:
            raise ValueError('当前仅接受本产品生成的标准sRGB色度和传递曲线')


def _mask(path,shape,*,binary=False):
    path=Path(path).resolve(strict=True)
    if path.suffix.lower() in ('.tif','.tiff'):
        with tifffile.TiffFile(path) as tf:
            if len(tf.pages)!=1:
                raise ValueError('蒙版需为单页')
            p=tf.pages[0]
            if int(p.tags['Orientation'].value if 'Orientation' in p.tags else 1)!=1:
                raise ValueError('蒙版朝向必须为TOP-DOWN')
            try:md=json.loads(p.description)
            except (ValueError,TypeError):md={}
            if isinstance(md,dict) and any(md.get(k,'TOP-DOWN')!='TOP-DOWN' for k in ('roworder','row_order')):
                raise ValueError('蒙版行序与照片不一致')
            a=p.asarray()
    else:
        with Image.open(path) as im:
            if im.getexif().get(274,1)!=1:
                raise ValueError('蒙版朝向必须为TOP-DOWN')
            a=np.array(im)
    if a.shape!=shape or a.ndim!=2:
        raise ValueError('蒙版必须与照片同尺寸、单通道、朝向一致')
    if a.dtype not in (np.dtype('uint8'),np.dtype('uint16'),np.dtype('float32'),np.dtype('float64'),np.dtype(bool)):
        raise ValueError('不支持的蒙版数值类型')
    if np.issubdtype(a.dtype,np.floating) and (not np.isfinite(a).all() or a.min()<0 or a.max()>1):
        raise ValueError('蒙版需为有限0至1权重')
    if binary:
        return a>0  # Preserve positive sub-float32 values as protected.
    if a.dtype==np.uint8:
        a=a.astype(np.float32)/255
    elif a.dtype==np.uint16:
        a=a.astype(np.float32)/65535
    elif a.dtype in (np.dtype('float32'),np.dtype('float64'),np.dtype(bool)):
        mixed=(a>0)&(a<1)
        a=a.astype(np.float32)
        a[mixed]=np.clip(a[mixed],np.nextafter(np.float32(0),np.float32(1)),
                        np.nextafter(np.float32(1),np.float32(0)))
    else:
        raise ValueError('不支持的蒙版数值类型')
    if not np.isfinite(a).all() or a.min()<0 or a.max()>1:
        raise ValueError('蒙版需为有限0至1权重')
    return a


def finish_photo(source,output_root,sky_mask,protected_mask,*,mode='photographic',
                 detail_strength=1.,star_strength=1.,ground_strength=0.,jpeg_quality=None):
    """Finish an explicitly encoded, 16-bit sRGB TIFF into a new directory.

    Masks are mandatory: no inferred empty protection for photos with meteors.
    Finishing never rewrites a RAW/master and never invokes a generative model.
    ``jpeg_quality`` (60..100) encodes the JPEG at that quality; None keeps 97 (both 4:4:4).
    """
    from . import photographic as photo
    recipe=photo.Recipe(mode=mode,detail_strength=detail_strength,star_strength=star_strength,
                        medium_gain=1.4,broad_gain=1.8)
    recipe.validate()
    if isinstance(ground_strength,(bool,np.bool_)) or not isinstance(ground_strength,(int,float)) or not np.isfinite(ground_strength) or not 0<=ground_strength<=1:
        raise ValueError('地景强度需为0至1')
    return _finish_display(source,output_root,sky_mask,protected_mask,processor=photo,
                           recipe=recipe,ground_strength=ground_strength,jpeg_quality=jpeg_quality)


class MilkywayConfirmationChanged(ValueError):
    """The source image or Milky Way mask changed after a person confirmed them."""


def milkyway_confirmation(source,mask):
    """What a person confirmed: the resolved source and mask paths and their SHA-256 (through ``core._sha``)."""
    return {"source_path":str(Path(source).resolve()),"mask_path":str(Path(mask).resolve()),
            "source_sha256":core._sha(Path(source)),"milkyway_mask_sha256":core._sha(Path(mask))}


def write_milkyway_review(confirmation,source,mask,directory,*,reviewer,rationale):
    """Write the human Milky Way evidence record for an earlier ``milkyway_confirmation``; return its path.

    Refuses with ``MilkywayConfirmationChanged`` when either file moved or changed since the confirmation.
    The record is the ``milkyway-subject-evidence`` format ``finish_milkyway`` accepts as evidence.
    """
    from .milkyway_gate import VERSION
    if (confirmation.get("source_path")!=str(Path(source).resolve())
            or confirmation.get("mask_path")!=str(Path(mask).resolve())
            or confirmation.get("source_sha256")!=core._sha(Path(source))
            or confirmation.get("milkyway_mask_sha256")!=core._sha(Path(mask))):
        raise MilkywayConfirmationChanged('源图或银河蒙版在确认后已改变，请重新查看并确认。')
    record={"version":VERSION,"status":"present","method":"human-review",
            "reviewer":reviewer,"rationale":rationale,
            "source_sha256":confirmation["source_sha256"],
            "milkyway_mask_sha256":confirmation["milkyway_mask_sha256"]}
    directory=Path(directory)
    directory.mkdir(parents=True)
    evidence=directory/"evidence.json"
    evidence.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding="utf8")
    return str(evidence)


def finish_milkyway(source,output_root,sky_mask,protected_mask,milkyway_mask,*,
                    brightness=.020,contrast=.60,structure=.90,color=.35,strength=.60,
                    transient_background=None,starless_background=None,milkyway_evidence=None,check=None,
                    jpeg_quality=None):
    """Make a local presentation derivative through three explicit input masks.

    The caller supplies the Milky Way location; there is no automatic fallback.
    A bound transient_background preserves the signed display residual while
    allowing the background beneath trails to follow the surrounding edit.
    Without it, the full protected pixels are held, which can leave dark bands.
    An optional bound starless_background edits an experimental diffuse layer
    and preserves the signed display star residual. Separation is not truth.
    This is display-domain editing, not physical signal reconstruction.
    A mask locates an edit; it does not prove the Milky Way is visible. Without
    source-and-mask-bound human evidence, this entry returns an exact no-op.
    ``check`` (the workflow worker's cancel check; None from the CLI, MCP and
    GUI) goes to the Milky Way renderer only (P1c T-5a); a cancel leaves no
    staging folder behind. ``jpeg_quality`` (60..100, the GUI's setting)
    encodes milkyway_srgb.jpg at that quality; None keeps 97 (both 4:4:4).
    """
    from . import milkyway
    # P1c §3.5: the render runs on the frame pool, so a malformed AESTER_FRAME_WORKERS / AESTER_OVERLAP is refused
    # here, before any input is read or any staging folder exists
    frame_pool.validate_environment()
    recipe=milkyway.Recipe(brightness=brightness,contrast=contrast,structure=structure,
                           color=color,strength=strength)
    recipe.validate()
    if any(value is None for value in (sky_mask,protected_mask,milkyway_mask)):
        raise ValueError('需显式提供天空、保护与同尺寸银河蒙版')
    return _finish_display(source,output_root,sky_mask,protected_mask,processor=milkyway,
                           recipe=recipe,milkyway_mask=milkyway_mask,
                           transient_background=transient_background,
                           starless_background=starless_background,
                           milkyway_evidence=milkyway_evidence,check=check,jpeg_quality=jpeg_quality)


def _display_source(source):
    """Validate the same display color and storage contract for S and B."""
    with tifffile.TiffFile(source) as tf:
        if len(tf.pages)!=1:
            raise ValueError('需要单页16位sRGB TIFF')
        p=tf.pages[0]
        if p.dtype!=np.uint16 or len(p.shape)!=3 or p.shape[2]!=3 or not 64<=p.shape[0]*p.shape[1]<=120_000_000:
            raise ValueError('需要本软件输出的16位RGB TIFF；此入口不接受RAW或线性母版')
        if int(p.photometric)!=2 or int(p.planarconfig)!=1:
            raise ValueError('需要交织的RGB TIFF')
        if int(p.tags['Orientation'].value if 'Orientation' in p.tags else 1)!=1:
            raise ValueError('请先统一照片与蒙版朝向')
        icc=p.tags[34675].value if 34675 in p.tags else None
        if not isinstance(icc,bytes):
            raise ValueError('输入缺少明确sRGB配置，不猜测颜色空间')
        _validate_srgb(icc)
        try: metadata=json.loads(p.description)
        except (TypeError,ValueError): metadata={}
        if not isinstance(metadata,dict):
            raise ValueError('TIFF描述必须为颜色元数据对象')
        if (metadata.get('scene_linear') is not False or metadata.get('display_encoding_applied') is not True
                or metadata.get('roworder')!='TOP-DOWN' or metadata.get('row_order','TOP-DOWN')!='TOP-DOWN'):
            raise ValueError('需要明确TOP-DOWN、非线性且已显示编码的本产品成片')
        if metadata.get('color_space') not in ('encoded_sRGB','display-sRGB','sRGB','nonlinear_sRGB','nonlinear sRGB'):
            raise ValueError('需要明确的显示sRGB颜色空间')
        return p.shape,icc,metadata


def _read_display(path):
    try:return tifffile.memmap(path,mode='r')
    except ValueError:return tifffile.imread(path)


def _close_map(array):
    if isinstance(array,np.memmap):array._mmap.close()


def _new_map(guard,path,**kwargs):
    """tifffile.memmap creating ``path``, held inside ``hashing.writing(path)`` until ``guard`` closes.

    Writes through a memory map keep the file's modification time, so no digest
    of the file may be used or remembered while it is writable; close the map
    before closing ``guard``.
    """
    guard.enter_context(hashing.writing(path))
    return tifffile.memmap(path,**kwargs)


def jpeg_unit_bytes(width):
    """unit_bytes of one 128-row JPEG conversion block (P1c T-5x (c), R17): measured at W=8280, scaled by W, never
    below the measured value."""
    return max(JPEG_UNIT_BYTES,-(-JPEG_UNIT_BYTES*int(width)//SCOPE_UNIT_WIDTH))


def _jpeg_rows(array,convert,*,pool_stats=None):
    """``convert(array)`` as one new uint8 array, built from 128-row blocks on the frame pool (P1c T-5x (c)).

    ``convert`` is the call site's own element-wise expression (16-bit codes
    to 8-bit), so every byte equals the whole-image conversion while the
    whole-image float32 temporaries (1.1-1.6 GB at 45.7 MP) are never made.
    ``pool_stats`` receives the pool's diagnostics for the caller's note.
    """
    codes=np.empty(array.shape,np.uint8)
    starts=range(0,array.shape[0],128)

    def block(y,unit):
        codes[y:y+128]=convert(array[y:y+128])

    unit_bytes=jpeg_unit_bytes(array.shape[1])
    workers=performance.frame_workers(len(starts),unit_bytes,JPEG_WORKER_CAP,blas=False)
    with performance.operation('jpeg_rows'):
        frame_pool.run_ordered(block,starts,workers=workers,name='jpeg_rows',unit_bytes=unit_bytes,blas=False,
                               stats=pool_stats)
    return codes


def _discard_partial(stage,*,attempts=8,seconds=DISCARD_SECONDS):
    """Remove a cancelled step's staging folder; True when it is gone (P1c §3.5).

    Called after every pool has ended and every memory map is closed, so that
    a cancelled and resumed run leaves the same tree as an uninterrupted one.
    Windows may hold a handle a little longer (an antivirus scan, a view still
    being unmapped): at most ``attempts`` tries within ``seconds`` (no wait
    after the last one, so the cancel is re-raised within the workflow's
    cancel budget), then the folder is marked (DISCARD_MARKER) and the next
    run of the same step removes it (_sweep_discarded). Never raises: the
    caller re-raises the cancel.
    """
    stage=Path(stage)
    deadline=time.perf_counter()+seconds
    delay=.02
    for attempt in range(attempts):
        if not stage.exists():
            return True
        try:
            shutil.rmtree(stage)
            return True
        except OSError:
            left=deadline-time.perf_counter()
            if attempt==attempts-1 or left<=0:
                break
            time.sleep(min(delay,left))
            delay=min(.25,delay*2)
    try:
        if stage.is_dir():
            (stage/DISCARD_MARKER).write_text(json.dumps({'state':'cancelled','remove':True}),encoding='utf-8')
    except OSError:
        pass
    return not stage.exists()


def _sweep_discarded(root,pattern='.partial-*'):
    """Remove the staging folders under ``root`` that a cancelled step marked (_discard_partial); never raises."""
    try:
        marked=sorted(p for p in Path(root).glob(pattern) if (p/DISCARD_MARKER).is_file())
    except OSError:
        return
    for path in marked:
        shutil.rmtree(path,ignore_errors=True)


def _discard_failed(stage):
    """Drop a failed attempt's staged pixel files and keep its JSON records (P1c §3.5, §5 item 2).

    Called once every pool has ended and every memory map is closed. A pool
    stopped by a block's error may already have written rows past that block
    into its unpublished maps, so the staged pixels would differ between 1
    and N workers; without them a failed attempt leaves the same folder at any
    worker count: its failure record (and the other records), for diagnosis.
    Nothing reads a failed attempt's pixels (its record lists no artifacts).
    Best effort, never raises: the caller returns or re-raises the failure.
    """
    try:
        paths=sorted(Path(stage).rglob('*'),reverse=True)
    except OSError:
        return
    for path in paths:
        try:
            if path.is_file() and path.suffix.lower()!='.json':
                path.unlink()
        except OSError:
            pass


def _pair_scope(source,background,protect):
    """Require all signed residual samples to lie inside the declared layer."""
    image=base=None
    stats={'residual_pixels':0,'residual_outside_protected_pixels':0,
           'residual_min_code':0,'residual_max_code':0}
    try:
        image=_read_display(source);base=_read_display(background)
        if image.shape!=base.shape or image.shape[:2]!=protect.shape:
            raise ValueError('亮迹背景与源照片、保护蒙版尺寸不一致')
        for y in range(0,image.shape[0],128):
            difference=image[y:y+128].astype(np.int32)-base[y:y+128].astype(np.int32)
            present=np.any(difference!=0,axis=-1)
            if np.any(present&~protect[y:y+128]):
                raise ValueError('亮迹背景配对失败：保护蒙版外存在非零残差')
            stats['residual_pixels']+=int(present.sum())
            stats['residual_min_code']=min(stats['residual_min_code'],int(difference.min()))
            stats['residual_max_code']=max(stats['residual_max_code'],int(difference.max()))
        return stats
    finally:
        _close_map(base);_close_map(image)


def scope_unit_bytes(width):
    """unit_bytes of one 128-row _starless_scope block (P1c T-5x (b), R17): measured at W=8280, scaled by W, never
    below the measured value."""
    return max(SCOPE_UNIT_BYTES,-(-SCOPE_UNIT_BYTES*int(width)//SCOPE_UNIT_WIDTH))


def _starless_scope(processing_base,starless,sky,protect,*,paired_corridors=False,pool_stats=None):
    """Only a verified trail-free paired base permits removing corridor stars.

    P1c T-5x (b): the 128-row blocks run on the frame pool; the first failing
    block's error is raised, the statistics are folded in block order (R2).
    ``pool_stats`` receives the pool's diagnostics for the caller's own
    timings note (one key per call site, R19).
    """
    base=diffuse=None
    stats={'residual_pixels':0,'residual_min_code':0,'residual_max_code':0,
           'residual_outside_allowed_pixels':0}
    try:
        base=_read_display(processing_base);diffuse=_read_display(starless)
        if base.shape!=diffuse.shape or base.shape[:2]!=sky.shape or sky.shape!=protect.shape:
            raise ValueError('去星背景与基底、天空和保护蒙版尺寸不一致')

        def block(y,unit):
            residual=base[y:y+128].astype(np.int32)-diffuse[y:y+128].astype(np.int32)
            present=np.any(residual!=0,axis=-1)
            forbidden=(sky[y:y+128]!=1)
            if not paired_corridors:forbidden|=protect[y:y+128]
            if np.any(present&forbidden):
                raise ValueError('去星背景超出已验证的纯天空分层范围')
            return int(present.sum()),int(residual.min()),int(residual.max())

        def fold(i,value):
            stats['residual_pixels']+=value[0]
            stats['residual_min_code']=min(stats['residual_min_code'],value[1])
            stats['residual_max_code']=max(stats['residual_max_code'],value[2])

        starts=range(0,base.shape[0],128)
        unit_bytes=scope_unit_bytes(base.shape[1])
        workers=performance.frame_workers(len(starts),unit_bytes,SCOPE_WORKER_CAP,blas=False)
        with performance.operation('starless_scope'):
            frame_pool.run_ordered(block,starts,workers=workers,name='starless_scope',on_result=fold,
                                   unit_bytes=unit_bytes,blas=False,stats=pool_stats)
        return stats
    finally:
        _close_map(diffuse);_close_map(base)


def _finalize_diffuse(source,starless,final,diffuse_output):
    """Apply the final common edit in signed arithmetic; never clip a layer."""
    for y in range(0,source.shape[0],128):
        original=starless[y:y+128].astype(np.int32)
        adjusted=original+final[y:y+128].astype(np.int32)-source[y:y+128].astype(np.int32)
        if np.any((adjusted<0)|(adjusted>65535)):
            raise ValueError('分层共同余量不足，弥散层超出16位范围')
        if np.any(((adjusted==0)&(original>0))|((adjusted==65535)&(original<65535))):
            raise ValueError('分层共同余量引入新端点')
        diffuse_output[y:y+128]=adjusted.astype(np.uint16)


def _finish_display(source,output_root,sky_mask,protected_mask,*,processor,recipe,
                    ground_strength=0.,milkyway_mask=None,transient_background=None,
                    starless_background=None,milkyway_evidence=None,check=None,jpeg_quality=None):
    """Shared validated display TIFF boundary and isolated publication.

    ``check`` reaches the Milky Way renderer only. A cancel (a
    KeyboardInterrupt, e.g. the workflow's WorkflowCancelled) bypasses the
    failed-record path; once every pool has ended and every map is closed the
    staging folder is removed and the cancel re-raised (P1c §3.5).
    """
    if jpeg_quality is not None:
        from .jpeg_estimate import validate_quality
        validate_quality(jpeg_quality)
    local_milkyway=milkyway_mask is not None
    paired=transient_background is not None
    separated=starless_background is not None
    if (paired or separated) and not local_milkyway:
        raise ValueError('配对亮迹或去星背景仅适用于显式银河蒙版入口')
    if milkyway_evidence is not None and not local_milkyway:
        raise ValueError('银河存在依据只适用于显式银河入口')
    source=Path(source).resolve(strict=True)
    sky_mask=Path(sky_mask).resolve(strict=True)
    protected_mask=Path(protected_mask).resolve(strict=True)
    paths=[source,sky_mask,protected_mask]
    if local_milkyway:
        milkyway_mask=Path(milkyway_mask).resolve(strict=True)
        paths.append(milkyway_mask)
    if paired:
        transient_background=Path(transient_background).resolve(strict=True)
        paths.append(transient_background)
    if separated:
        starless_background=Path(starless_background).resolve(strict=True)
        paths.append(starless_background)
    if milkyway_evidence is not None:
        milkyway_evidence=Path(milkyway_evidence).resolve(strict=True)
        paths.append(milkyway_evidence)
    processing_base=transient_background if paired else source
    source_hashes=dict(zip(map(str,paths),hashing.sha256_many(paths,source='photographic_job.inputs')))
    subject_gate=None
    requested_recipe=asdict(recipe)
    if local_milkyway:
        from .milkyway_gate import assess_evidence
        evidence=None
        if milkyway_evidence is not None:
            raw_evidence=milkyway_evidence.read_bytes()
            if hashlib.sha256(raw_evidence).hexdigest()!=source_hashes[str(milkyway_evidence)]:
                raise ValueError('银河存在依据在读取期间改变')
            evidence=json.loads(raw_evidence)
        subject_gate=assess_evidence(evidence,source_sha256=source_hashes[str(source)],
                                    mask_sha256=source_hashes[str(milkyway_mask)])
        if not subject_gate['enabled']:recipe=replace(recipe,strength=0.)
    shape,icc,_=_display_source(source)
    if paired:
        background_shape,_,background_metadata=_display_source(transient_background)
        if background_shape!=shape:
            raise ValueError('亮迹背景必须与源照片同尺寸')
        if background_metadata.get('transient_background_for_sha256')!=source_hashes[str(source)]:
            raise ValueError('亮迹背景未绑定当前源照片SHA256')
    paired_corridors=False
    if separated:
        diffuse_shape,_,diffuse_metadata=_display_source(starless_background)
        if diffuse_shape!=shape:
            raise ValueError('去星背景必须与源照片同尺寸')
        bindings={'star_separation_for_sha256':processing_base,'sky_mask_sha256':sky_mask,
                  'protected_mask_sha256':protected_mask}
        if any(diffuse_metadata.get(key)!=source_hashes[str(path)] for key,path in bindings.items()):
            raise ValueError('去星背景未绑定当前处理基底、天空及保护蒙版SHA256')
        if diffuse_metadata.get('separation_scope')=='pure-sky-on-bound-transient-background':
            if not paired or diffuse_metadata.get('transient_source_sha256')!=source_hashes[str(source)]:
                raise ValueError('去星走廊开放必须绑定当前源照片及配对亮迹背景')
            paired_corridors=True
    sky=_mask(sky_mask,shape[:2]);protect=_mask(protected_mask,shape[:2],binary=True)
    galaxy=_mask(milkyway_mask,shape[:2]) if local_milkyway else None
    pair_validation=_pair_scope(source,transient_background,protect) if paired else None
    separation_validation=None
    if separated:
        pool_stats={}
        separation_validation=_starless_scope(processing_base,starless_background,sky,protect,
                                             paired_corridors=paired_corridors,pool_stats=pool_stats)
        performance.note_pool('starless_scope.milkyway',pool_stats)
        from . import resources
        resources.shadow_pool('py.starless_scope',actual=pool_stats.get('workers'),items=-(-shape[0]//128),
                              width=shape[1])
    if (paired or separated) and hashing.first_mismatch(paths,[source_hashes[str(p)] for p in paths],
                                                        source='photographic_job.inputs') is not None:
        raise ValueError('输入在配对验证期间改变，未开始处理')
    root=Path(output_root).resolve()
    root.mkdir(parents=True,exist_ok=True)
    _sweep_discarded(root)
    prefix='milkyway' if local_milkyway else 'photographic'
    name=('milkyway-' if local_milkyway else 'photo-')+uuid.uuid4().hex[:12]
    stage=root/('.partial-'+name);stage.mkdir()
    target=root/name
    record={'state':'running','version':MILKYWAY_VERSION if local_milkyway else VERSION,
        'source':str(source),'source_hashes':source_hashes,
        'source_domain':'display-sRGB uint16 TOP-DOWN','scene_linear':False,'generative_model_used':False,
        'recipe':asdict(recipe),'ground_strength':float(ground_strength),'source_unchanged':False,
        'recipe_version':processor.VERSION,'row_order':'TOP-DOWN','shape':list(shape),'processing_directory':str(stage),
        'operation':'milkyway_local' if local_milkyway else 'experimental_all_sky',
        'presentation_only':True,'physical_recovery':False,
        'quality_status':'aesthetics_not_accepted',
        'message':('显式银河蒙版内的局部展示调整；不是物理信号恢复，摄影审美尚待验收。' if local_milkyway
                   else '实验全天空／星点／地景调整，摄影审美未通过；银河局部处理请使用显式银河蒙版入口。'),
        'masks':{name:{'path':str(path),'sha256':source_hashes[str(path)]}
                 for name,path in [('sky',sky_mask),('protected',protected_mask)]}}
    if local_milkyway:
        record.pop('ground_strength')
        record['subject_gate']=subject_gate
        record['requested_recipe']=requested_recipe
        record['operation_status']=subject_gate['decision']
        if not subject_gate['enabled']:
            record['message']='未确认可见银河，保持原图像素；蒙版不能替代对象存在依据。'
            record['quality_status']='not_applied'
        record['masks']['milkyway']={'path':str(milkyway_mask),'sha256':source_hashes[str(milkyway_mask)]}
        record['mask_mode']='explicit-mask'
        record['protection_mode']='exact_signed_display_residual' if paired else 'whole_pixel_hold'
        record['protected_object']='encoded_signed_residual' if paired else 'whole_protected_pixels'
        record['physical_flux_preserved']=False
        if paired:
            record['transient_background']={'path':str(transient_background),
                'sha256':source_hashes[str(transient_background)]}
            record['pair_validation']=pair_validation
            record['message']+='先增强同源背景，再保留编码显示域的有符号亮迹残差；不是物理通量保护。'
        else:
            record['message']+='当前冻结保护区域的全部像素，周围背景增强后可能形成相对暗带；审美未通过。'
        if separated:
            record.update(version='milkyway-starless-job-3',experimental_separation=True,
                          star_layer_physical_truth=False,separation_validation=separation_validation)
            record['upstream_star_separation']={
                'backend':diffuse_metadata.get('backend'),
                'learned_star_separation':diffuse_metadata.get('learned_star_separation')}
            record['generative_model_scope']='this finishing operation; upstream separation is recorded separately'
            record['starless_background']={'path':str(starless_background),
                'sha256':source_hashes[str(starless_background)],
                'processing_base_sha256':source_hashes[str(processing_base)]}
            record['message']+='实验分层：增强弥散层并保留有符号恒星残差；已有星层不是物理真值，分离质量尚待验收。'
    if jpeg_quality is not None:
        record['jpeg_quality']=jpeg_quality     # absent: the historical 97
    core.atomic_json(stage/'result.json',record)
    start=time.perf_counter()
    image=output=background=finished_background=starless=finished_diffuse=None
    render_options={'check':check} if local_milkyway else {}   # the photographic renderer takes no check
    cancelled=failed=False
    guard=ExitStack()
    try:
        image=_read_display(source)
        if paired:
            background=_read_display(transient_background)
            inputs=(background,sky,np.zeros(shape[:2],dtype=bool),galaxy)
        else:
            inputs=(image,sky,protect,galaxy) if local_milkyway else (image,sky,protect)
        if separated:
            starless=_read_display(starless_background)
            inputs=(starless,sky,inputs[2],galaxy)
        prepared=processor.prepare(*inputs,recipe,domain='display-sRGB')
        path=stage/(prefix+'_srgb16.tif')
        metadata={'roworder':'TOP-DOWN','color_space':'encoded_sRGB','scene_linear':False,
                  'display_encoding_applied':True,'photographic_derivative':True,
                  'source_sha256':source_hashes[str(source)],'recipe_version':processor.VERSION,
                  'operation':record['operation'],'presentation_only':True,'physical_recovery':False}
        if local_milkyway:
            metadata.update(mask_mode='explicit-mask',milkyway_mask_sha256=source_hashes[str(milkyway_mask)],
                            protection_mode=record['protection_mode'],protected_object=record['protected_object'],
                            physical_flux_preserved=False)
        if paired:
            metadata.update(transient_background_sha256=source_hashes[str(transient_background)],
                            paired_background_artifact='milkyway_background_srgb16.tif')
        if separated:
            metadata.update(experimental_separation=True,star_layer_physical_truth=False,
                            starless_background_sha256=source_hashes[str(starless_background)],
                            star_processing_base_sha256=source_hashes[str(processing_base)],
                            diffuse_artifact='milkyway_diffuse_srgb16.tif')
        output=_new_map(guard,path,shape=shape,dtype=np.uint16,photometric='rgb',
            metadata=metadata,
            extratags=[(34675,'B',len(icc),icc,False)])
        if paired:
            background_metadata={**metadata,'artifact_role':'finished_transient_background',
                'paired_composite_artifact':path.name,'headroom_adjusted':True}
            background_metadata.pop('paired_background_artifact')
            finished_background=_new_map(guard,stage/'milkyway_background_srgb16.tif',shape=shape,
                dtype=np.uint16,photometric='rgb',metadata=background_metadata,
                extratags=[(34675,'B',len(icc),icc,False)])
        if separated:
            finished_diffuse=_new_map(guard,stage/'milkyway_diffuse_srgb16.tif',shape=shape,
                dtype=np.uint16,photometric='rgb',
                metadata={**metadata,'artifact_role':'finished_diffuse_background','headroom_adjusted':True},
                extratags=[(34675,'B',len(icc),icc,False)])
            record['diffuse_statistics']=processor.render(*inputs,prepared,finished_diffuse,domain='display-sRGB',
                                                          **render_options)
            record['diffuse_statistics_scope']='D to proposed Q before shared star and transient headroom adjustment'
            stats=processor.preserve_transient_residual(background if paired else image,starless,
                finished_diffuse,finished_background if paired else output)
            record['star_statistics']=stats
            record['star_statistics_scope']='processing_base to first star-constrained enhancement; paired transient adjustment may further reduce it'
            if paired:
                stats=processor.preserve_transient_residual(image,background,finished_background,output)
                record['paired_statistics']=stats
                record['paired_artifacts']={'composite':path.name,'background':'milkyway_background_srgb16.tif',
                    'invariant':'F - Q == S - B exactly in signed display-code arithmetic'}
            _finalize_diffuse(image,starless,output,finished_diffuse)
            record['separated_artifacts']={'composite':path.name,'diffuse':'milkyway_diffuse_srgb16.tif',
                'star_residual':'processing_base - input_starless_background',
                'transient_residual':'source - processing_base',
                'invariant':'D_final + (processing_base - D) + (source - processing_base) == final in signed display-code arithmetic'}
            finished_diffuse.flush();_close_map(finished_diffuse);finished_diffuse=None
            if paired:
                finished_background.flush();_close_map(finished_background);finished_background=None
        elif paired:
            record['background_statistics']=processor.render(*inputs,prepared,finished_background,
                                                             domain='display-sRGB',**render_options)
            record['background_statistics_scope']='B to proposed Q before shared headroom adjustment'
            stats=processor.preserve_transient_residual(image,background,finished_background,output)
            record['paired_statistics']=stats
            record['paired_artifacts']={'composite':path.name,'background':'milkyway_background_srgb16.tif',
                'invariant':'F - Q == S - B exactly in signed display-code arithmetic'}
            finished_background.flush()
            _close_map(finished_background);finished_background=None
        else:
            stats=processor.render(*inputs,prepared,output,domain='display-sRGB',**render_options)
        if ground_strength and not local_milkyway:
            from .ground_finish import finish_ground
            record['ground']=finish_ground(image,sky,protect,output,strength=float(ground_strength))
        output.flush()
        byte8=np.empty(shape,np.uint8)
        for y in range(0,shape[0],128):
            byte8[y:y+128]=np.rint(output[y:y+128].astype(np.float32)/257).astype(np.uint8)
        # JPEGs are encoded in memory and written with their digests (same bytes as Image.save to the path).
        from .master_export import save_image_with_digest
        save_image_with_digest(Image.fromarray(byte8),stage/(prefix+'_srgb.jpg'),source='photographic_job.artifacts',
                               quality=FINISH_JPEG_QUALITY if jpeg_quality is None else jpeg_quality,
                               subsampling=0,icc_profile=icc)
        preview=Image.fromarray(byte8);preview.thumbnail((1920,1920),Image.Resampling.LANCZOS)
        save_image_with_digest(preview,stage/'preview.jpg',source='photographic_job.artifacts',quality=95,icc_profile=icc)
        del byte8
        _close_map(output)
        output=None
        guard.close()
        files=[p for p in stage.iterdir() if p.is_file() and p.name!='result.json']
        digests=hashing.sha256_many(files,source='photographic_job.artifacts',fresh=True)
        record['artifacts']={p.name:{'path':str(target/p.name),'sha256':digest,'bytes':p.stat().st_size}
            for p,digest in zip(files,digests)}
        record.update(state='succeeded',source_unchanged=True,statistics=stats,
                      prepare_seconds=prepared['prepare_seconds'],seconds=time.perf_counter()-start)
        record['processing_directory']=str(target)
        core.atomic_json(stage/'result.json',record)
        # Verify after all encoding and staged metadata writes, just before
        # publication. In-memory preparation seals are not file seals.
        if hashing.first_mismatch(paths,[source_hashes[str(p)] for p in paths],
                                  source='photographic_job.inputs') is not None:
            record['source_unchanged']=False
            raise ValueError('输入在处理期间改变，结果未发布')
        hashing.rename(stage,target)   # carries the ledger entries of the artifacts (P1b-5a)
        return record
    except Exception as exc:
        failed=True               # the staged pixels go below, once the maps are closed; result.json stays
        record.update(state='failed',error=str(exc),seconds=time.perf_counter()-start,
                      processing_directory=str(stage),artifacts={},source_unchanged=False)
        core.atomic_json(stage/'result.json',record)
        return record
    except KeyboardInterrupt:
        cancelled=True            # WorkflowCancelled: no failed record; the staging folder goes below
        raise
    finally:
        try:
            try:
                _close_map(finished_diffuse);_close_map(starless)
                _close_map(finished_background);_close_map(output)
                _close_map(background);_close_map(image)
            finally:
                guard.close()
        finally:
            if cancelled:         # every pool has ended (run_ordered) and every map is closed
                _discard_partial(stage)
            elif failed:          # the same failed folder at 1 and N workers (P1c §3.5)
                _discard_failed(stage)
