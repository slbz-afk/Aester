"""First-exposure preview and full-resolution manual mask template.

New explicit selections use the product's camera colour/display pipeline and
a compact verified linear cache. Legacy directory calls preserve their old
uncalibrated full-frame preview. The white template is never a segmentation.
"""

from importlib import import_module
from pathlib import Path
import json
import math
import shutil
import time
import traceback
import uuid

import numpy as np
from astropy.io import fits
from PIL import Image, ImageCms, PngImagePlugin

from . import core, siril
from .imaging import read_rgb, display_base


REFERENCE_CONVERSION_TIMEOUT_SECONDS = 300
PREVIEW_MAX_SIDE = 1024
INTERACTIVE_MAX_SIDE = 640


def _optional_raw_metadata(path: Path, kind: str) -> dict:
    if kind != "raw":
        return {"available": False, "status": "unknown", "reason": "input_is_fits"}
    try:
        reader = import_module(".raw_metadata", package=__package__).read_raw_metadata
        result = reader(path)
        if not isinstance(result, dict):
            raise TypeError("RAW metadata reader did not return a dictionary")
        json.dumps(result, allow_nan=False)
        if not result.get("available"):
            result = {**result, "reader_status": result.get("status"), "status": "unknown"}
        return result
    except Exception as exc:
        # Optional LibRaw metadata must not prevent preparing a reference.
        return {"available": False, "status": "unknown", "reason": str(exc)}


def _fits_metadata(path: Path) -> dict:
    header = fits.getheader(path)
    keys = ("INSTRUME", "TELESCOP", "EXPTIME", "EXPOSURE", "ISOSPEED", "ISO",
            "FOCALLEN", "APERTURE", "DATE-OBS", "DATE", "XPIXSZ", "YPIXSZ",
            "BITPIX", "BZERO", "BSCALE", "BUNIT", "ROWORDER", "NAXIS1", "NAXIS2")
    values = {}
    for key in keys:
        if key not in header:
            continue
        value = header[key]
        if isinstance(value, float) and not math.isfinite(value):
            value = None
        elif not isinstance(value, (str, int, float, bool, type(None))):
            value = str(value)
        values[key] = value
    return values


def prepare_reference(input_dir: str | None, output_root: str, siril_path: str | None = None, *,
                      input_files=None, color_adjustments=None, image_rotation=None) -> dict:
    """Decode first selected exposure (one allowed); old directory calls keep
    their minimum of three. Conversion matches the job. New RAW previews share
    camera colour and display intent, at up to 1024px; the mask and decoded FITS
    keep full geometry. Existing jobs and originals remain untouched.
    """
    modern = input_files is not None or color_adjustments is not None
    from . import orientation
    if image_rotation is not None:
        image_rotation = orientation.normalize_rotation(image_rotation)
    from . import color_adjustments as ca
    adjustments = ca.normalize(color_adjustments)
    selected = core.inspect_sources(input_dir, input_files=input_files, min_count=1) if modern else None
    source_directory = Path(selected['input_dir'] if selected else input_dir).expanduser().resolve()
    root = Path(output_root).expanduser().resolve()
    if root == source_directory:
        raise ValueError("参考图结果文件夹与素材文件夹必须分开")
    if selected and any(root.is_relative_to(Path(p).parent) for p in selected['files']):
        raise ValueError("参考图结果必须位于原片文件夹之外")
    root.mkdir(parents=True, exist_ok=True)
    directory = root / f"reference-{uuid.uuid4().hex}"
    directory.mkdir()
    manifest_path = directory / "reference.json"
    log_path = directory / "reference.log"
    record = {
        "schema_version": 1, "reference_dir": str(directory), "state": "preparing",
        "stage": "inspect", "created_at": core.now(), "updated_at": core.now(),
        "input_dir": str(source_directory), "metadata": {}, "error": None,
        "log_path": str(log_path),
    }

    def update(stage: str, **values) -> None:
        record.update(stage=stage, updated_at=core.now(), **values)
        core.atomic_json(manifest_path, record)

    try:
        update("inspect")
        log_path.write_text("Preparing a full-frame reference; source files are read-only.\n", encoding="utf-8")
        sources = selected or core.inspect_sources(str(source_directory))
        # inspect_sources already sorts its direct child files by filename.
        source = Path(sources["files"][0])
        update("engine", source_path=str(source), input_kind=sources["kind"],
               input_count=sources["count"], reference_index=1,
               selection="first_selected" if input_files is not None else "filename_first")
        engine = core.doctor(siril_path)
        if not engine.get("available"):
            raise ValueError(engine.get("message", "未找到符合版本要求的 Siril"))
        fits_contract = sources.get('fits_input')
        cfa = bool(fits_contract and fits_contract['layout'] == 'cfa')
        if cfa:
            from . import fits_input as fi
            fi.require_engine(engine)
            fits_contract = {**fits_contract, 'frames': fits_contract['frames'][:1]}
        update("copy", engine=engine)
        stage, work = directory / "stage", directory / "work"
        stage.mkdir()
        work.mkdir()
        source_hash = core._sha(source)
        staged = stage / f"frame00001{source.suffix.lower()}"
        update("copy", source_sha256=source_hash, staged_path=str(staged))
        shutil.copyfile(source, staged)
        staged_hash = core._sha(staged)
        if staged_hash != source_hash or core._sha(source) != source_hash:
            raise ValueError("首帧在复制时发生变化，不能生成可靠参考图；请检查素材后重试")
        capture = orientation.source_orientation(staged)
        rotation = (capture['rotation'] if modern else 0) if image_rotation is None else image_rotation
        update('copy', image_rotation=rotation, capture_rotation=capture['rotation'], capture_orientation=capture)
        # DNG colour comes from its tags below. The optional exhaustive LibRaw
        # metadata reader also unpacks CFA pixels; do not decode them twice just
        # to prepare a small preview. Legacy inspection keeps its old contract.
        metadata = {"raw": ({"available": False, "status": "not_requested",
                    "reason": "modern_dng_preview_uses_colour_tags_without_second_pixel_decode"}
                    if modern and staged.suffix.lower() == '.dng' else
                    _optional_raw_metadata(staged, sources["kind"]))}
        update("convert", staged_sha256=staged_hash, metadata=metadata,
               conversion_timeout_seconds=REFERENCE_CONVERSION_TIMEOUT_SECONDS)
        conversion = ("convertraw ref -debayer -out=../work" if sources["kind"] == "raw"
                      else "convert ref -out=../work")
        deadline = time.monotonic() + REFERENCE_CONVERSION_TIMEOUT_SECONDS

        def check_conversion_deadline() -> None:
            if time.monotonic() >= deadline:
                # run_script's finally block terminates its owned process;
                # this callback never searches for or kills unrelated Siril.
                raise TimeoutError("Siril 单帧参考转换超过 5 分钟，已请求停止；请查看转换日志")

        conversion_log = directory / "01_convert.log"
        log_start = conversion_log.stat().st_size if conversion_log.exists() else 0
        try:
            if cfa:
                audit = fi.convert_cfa(engine, directory, [staged], work, 'ref', fits_contract,
                                       check_conversion_deadline, siril.run_script)
                metadata['fits_input'] = fits_contract
                metadata['cfa_conversion_audit'] = str(audit)
            else:
                siril.run_script(engine["executable"], directory, "01_convert",
                                 ["setext fit", "set32bits", "setcompress 0", "cd stage", conversion],
                                 check_conversion_deadline)
        except siril.Cancelled:
            raise
        except RuntimeError as exc:
            if cfa:
                raise
            raise RuntimeError(siril.conversion_failure_message(
                conversion_log, str(exc), start_offset=log_start)) from exc
        converted = sorted(p for p in work.glob("ref_*.fit") if p.is_file())
        if len(converted) != 1:
            raise ValueError(siril.conversion_failure_message(
                conversion_log, f"Siril 必须生成唯一首帧 FITS，实际找到 {len(converted)} 个",
                start_offset=log_start))
        linear_path = converted[0]
        update("validate")
        if rotation:
            orientation.rotate_fits(linear_path, linear_path, rotation)
        shape = core._validate_frame(linear_path)
        data = read_rgb(linear_path)
        height, width = shape[1:]
        if data.shape != (height, width, 3) or height <= 0 or width <= 0:
            raise ValueError("首帧转换后的图像尺寸无效")
        metadata.update({
            "fits": _fits_metadata(linear_path),
            "geometry": {"width": width, "height": height,
                         "linear_row_order": "BOTTOM-UP", "preview_row_order": "TOP-DOWN",
                         "mask_row_order": "TOP-DOWN", "no_crop": True, "no_scale": True,
                         "basis": "full_siril_decoded_frame"},
            "rendering": {"method": "imaging.display_base", "display_space": "sRGB",
                          "color_calibrated": False, "additional_camera_white_balance_applied": False,
                          "additional_camera_matrix_applied": False,
                          "siril_decode_settings": "same_as_current_pipeline_defaults"},
            "mask": {"kind": "all_white_editing_template", "is_segmentation": False,
                     "white_value": 255, "message": "全白编辑模板，非天空分割结果；有地景时须人工绘制"},
        })
        update("render", metadata=metadata)
        # read_rgb returns bottom-up working pixels; display_base performs the
        # single flip to ordinary image coordinates. Do not flip a second time.
        cache = None
        if modern and sources['kind'] == 'raw':
            if core._sha(staged) != source_hash:
                raise ValueError('首帧缓存已改变')
            rendered, cache = _prepare_color_preview(data, staged, directory, adjustments)
            if core._sha(staged) != source_hash or core._sha(source) != source_hash:
                raise ValueError('首帧在准备颜色预览时发生变化')
            metadata['rendering'] = {
                'method': 'shared photo-default-1 + color-adjustments-1', 'display_space': 'sRGB',
                'color_calibrated': True, 'additional_camera_white_balance_applied': True,
                'additional_camera_matrix_applied': True, 'color_reference': str(source),
                'color_adjustments': adjustments, 'preview_kind': 'calibrated_single_exposure',
                'note': '同成片颜色逻辑；单帧噪声、堆栈锚点及缩小采样仍有区别'}
        else:
            if any(adjustments.values()):
                raise ValueError('此 FITS 参考尚无已校准的相机颜色底子，暂不支持白平衡／曝光滑块')
            rendered = np.round(display_base(data) * 255).astype(np.uint8)
        preview_path = directory / "reference_preview.png"
        mask_path = directory / "sky_mask_template.png"
        icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
        preview_info = PngImagePlugin.PngInfo()
        preview_info.add_text("Purpose", "Calibrated single-exposure preview" if cache else
                              "Uncalibrated sRGB visualization; full decoded frame, no crop or resize.")
        Image.fromarray(rendered).save(preview_path, icc_profile=icc, pnginfo=preview_info,
                                      compress_level=1 if cache else 6)
        mask_info = PngImagePlugin.PngInfo()
        mask_info.add_text("Purpose", "All-white editing template; not a sky segmentation result.")
        Image.new("L", (width, height), 255).save(mask_path, pnginfo=mask_info)
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(f"Artifacts prepared: {width} x {height}; full decoded frame; white mask is a template only.\n")
        # Publish success last, after both images, their hashes and the log exist.
        update("complete", state="succeeded", preview_path=str(preview_path),
               mask_template_path=str(mask_path), linear_path=str(linear_path),
               width=width, height=height, no_crop=True, no_scale=rendered.shape[:2] == (height, width),
               template_no_scale=True, preview_scale=rendered.shape[0] / height,
               linear_row_order="BOTTOM-UP", preview_row_order="TOP-DOWN", mask_row_order="TOP-DOWN",
               linear_sha256=core._sha(linear_path), preview_sha256=core._sha(preview_path),
               mask_template_sha256=core._sha(mask_path), color_cache=cache,
               color_adjustments=adjustments, preview_width=rendered.shape[1], preview_height=rendered.shape[0])
        return record
    except Exception as exc:
        failed_stage = record["stage"]
        record.update(state="failed", error=str(exc), updated_at=core.now())
        diagnostic_errors = []
        try:
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(traceback.format_exc())
        except Exception as write_error:
            diagnostic_errors.append(f"日志保存失败：{write_error}")
        try:
            core.atomic_json(manifest_path, record)
        except Exception as write_error:
            diagnostic_errors.append(f"错误记录保存失败：{write_error}")
        raise RuntimeError(f"参考图准备失败（{failed_stage}）：{exc}。诊断目录：{directory}"
                           + ("；" + "；".join(diagnostic_errors) if diagnostic_errors else "")) from exc


def _colour_proxy(camera, basis, max_side):
    """Reduce signed camera samples before the affine colour map.

    The all-zero nodata exception and sensor saturation are reduced separately,
    so a small saturated star and a missing-data border keep their meanings.
    Only this display proxy changes resolution; the decoded frame stays intact.
    Float32 averaging and the affine transform commute to rounding precision,
    not bit-for-bit (the full-size product still uses the original order).
    """
    from . import master_export as me
    from .whole_color import resize_signed
    small_camera = resize_signed(camera, max_side)
    small, _, _ = me.convert_camera_master(small_camera, basis, 'rec2020')
    h, w = camera.shape[:2]
    saturation = np.empty((h, w), dtype=bool)
    nodata = np.empty((h, w), dtype=bool)
    for start in range(0, h, 128):
        block = np.asarray(camera[start:start+128], dtype=np.float64)
        saturation[start:start+128] = (block * 65535.0 >= me.SATURATION_FRACTION * float(basis['white_DN'])).any(axis=2)
        nodata[start:start+128] = (block == 0).all(axis=2)
    size = small.shape[1], small.shape[0]
    # F-mode BOX preserves fractional coverage; do not round a faint hit to an
    # 8-bit zero before deciding whether the preview pixel contains saturation.
    sat_small = np.asarray(Image.fromarray(saturation.astype(np.float32)).resize(size, Image.Resampling.BOX)) > 0
    missing = np.asarray(Image.fromarray(nodata.astype(np.float32)).resize(size, Image.Resampling.BOX))
    affine_zero = me._apply_3x3(-np.asarray(basis['black_DN'], dtype=np.float64) / float(basis['scale_DN']),
                              me.master_matrix(basis, 'rec2020'))
    # convert_camera_master sets averaged all-zero samples to zero. Undo that
    # exception first, then apply exactly the area fraction of real nodata.
    zeros = (small_camera == 0).all(axis=2)
    small[zeros] = affine_zero.astype(np.float32)
    small = (small.astype(np.float64) - missing[..., None] * affine_zero).astype(np.float32)
    small[missing == 1] = 0.0
    return small, sat_small


def _prepare_color_preview(bottom_up, source, directory, adjustments):
    from dataclasses import replace
    from . import color_export as ce, master_export as me, photo_render as pr, display_transform as dt
    from . import color_adjustments as ca
    from .whole_color import resize_signed
    basis = ce.job_colour_basis(source)
    camera = np.flipud(bottom_up)
    meta = {'gamut': 'rec2020', 'colour_basis': {
        **{key: basis[key] for key in ('gains_rgb','wb_mode','as_shot_gains_rgb', 'daylight_gains_rgb',
                                     'camera_to_linear_srgb')},
        'master_matrix': me.master_matrix(basis, 'rec2020').tolist(),
        # the body, so the preview defaults to the same registered look as the job's D (camera_looks)
        'reference': {'camera': (basis.get('source') or {}).get('camera')}}}
    if max(camera.shape[:2]) <= PREVIEW_MAX_SIDE:
        small, sat_small, _ = me.convert_camera_master(camera, basis, 'rec2020')
        params = pr.default_params(meta, small)
    else:
        small, sat_small = _colour_proxy(camera, basis, PREVIEW_MAX_SIDE)
        # Preserve the old full-frame anchor and native-pixel noise samples.
        # Reducing before measuring noise would hide it and change the look.
        sampled, _, _ = me.convert_camera_master(camera[::8], basis, 'rec2020')
        _, matrix, _ = pr.wb_choice(meta, pr.DEFAULT_WB)
        anchor = dt.measure_anchor(sampled[:, ::8], 'rec2020', stride=1, wb_matrix=matrix)
        sigma = dt.measure_noise_sigma(sampled, 'rec2020', stride=1, wb_matrix=matrix)
        params = pr.default_params(meta, sampled, shadow_smoothing=False,
            anchor=anchor, noise_sigma=sigma,
            shadow_smooth_start=pr.SHADOW_SMOOTH_START_SIGMAS*sigma,
            shadow_smooth_end=pr.SHADOW_SMOOTH_END_SIGMAS*sigma)
        del sampled
    ratio = small.shape[0] / camera.shape[0]
    if params.shadow_smoothing:
        params = replace(params, shadow_smooth_radius=max(1, round(params.shadow_smooth_radius * ratio)))
    data_path, sat_path = directory/'preview_linear.npy', directory/'preview_saturated.npy'
    np.save(data_path, small, allow_pickle=False)
    np.save(sat_path, sat_small, allow_pickle=False)
    quick = resize_signed(small, INTERACTIVE_MAX_SIDE) if max(small.shape[:2]) > INTERACTIVE_MAX_SIDE else small
    quick_sat = np.asarray(Image.fromarray(sat_small.astype(np.float32)).resize(
        (quick.shape[1], quick.shape[0]), Image.Resampling.BOX)) > 0
    quick_path, quick_sat_path = directory/'interactive_linear.npy', directory/'interactive_saturated.npy'
    np.save(quick_path, quick, allow_pickle=False)
    np.save(quick_sat_path, quick_sat, allow_pickle=False)
    cache = {'version': 2, 'linear': str(data_path), 'saturated': str(sat_path),
             'linear_sha256': core._sha(data_path), 'saturated_sha256': core._sha(sat_path),
             'master_metadata': meta, 'base_params': params.to_json(),
             'basis': ce.basis_summary(basis), 'preview_scale': ratio,
             'full_geometry': {'width': camera.shape[1], 'height': camera.shape[0]},
             'proxy_geometry': {'width': small.shape[1], 'height': small.shape[0]},
             'proxy_method': 'signed_camera_box_then_affine_with_nodata_coverage',
             'statistics': 'original_pixel_scale_rows_stride8_anchor_columns_stride8',
             'interactive': {'linear': str(quick_path), 'saturated': str(quick_sat_path),
                 'linear_sha256': core._sha(quick_path), 'saturated_sha256': core._sha(quick_sat_path),
                 'width': quick.shape[1], 'height': quick.shape[0]}}
    # Metadata belongs to the cache identity too, not just the two arrays.
    import hashlib
    cache['signature'] = hashlib.sha256(json.dumps(cache, sort_keys=True, allow_nan=False).encode()).hexdigest()
    shown = ca.apply(params, meta, adjustments)
    rgb16, _ = dt.render(small, shown, camera_saturated=sat_small,
                         workers=pr.display_workers(shown, small.shape[0], small.shape[1], preview=True))
    return dt.to_8bit(rgb16), cache


def rotate_reference(reference_dir, *, image_rotation, _record=None):
    """Reorient an existing reference without decoding RAW again.

    Publish a new immutable reference with matching full-size FITS, proxies,
    preview and white mask. Old references/regions remain valid in their old
    direction; the GUI explicitly discards its old marks when choosing this.
    """
    import copy
    import hashlib
    from io import BytesIO
    from . import orientation
    angle = orientation.normalize_rotation(image_rotation)
    root = Path(reference_dir).expanduser().resolve(strict=True)
    record = copy.deepcopy(_record) if _record is not None else json.loads((root/'reference.json').read_text(encoding='utf-8'))
    if record.get('state') != 'succeeded' or Path(record['reference_dir']).resolve() != root:
        raise ValueError('不是已完成的原位预览')
    old_angle = orientation.normalize_rotation(record.get('image_rotation', 0))

    def verified(key, container=record):
        path = Path(container[key]).resolve(strict=True)
        # Decoded linear FITS is in work/ for the initial reference; proxies
        # and all rotated copies are direct children of their reference root.
        if not path.is_relative_to(root) or core._sha(path) != container[key.replace('_path', '')+'_sha256']:
            raise ValueError('方向参考缓存已改变或超出此参考目录')
        return path

    linear = verified('linear_path')
    preview = verified('preview_path')
    verified('mask_template_path')
    if angle == old_angle:
        return record
    delta = (angle - old_angle) % 360
    new_root = root.parent / ('reference-' + uuid.uuid4().hex)
    new_root.mkdir()
    out = copy.deepcopy(record)
    out.update(reference_dir=str(new_root), image_rotation=angle,
               created_at=core.now(), updated_at=core.now(), state='preparing', stage='rotate',
               rotated_from={'reference_dir': str(root), 'image_rotation': old_angle})
    out.pop('staged_path', None)
    out.pop('staged_sha256', None)
    if delta in (90, 270):
        out['width'], out['height'] = record['height'], record['width']
    out['linear_path'] = str(new_root/'reference.fit')
    orientation.rotate_fits(linear, out['linear_path'], delta)
    out['linear_sha256'] = core._sha(Path(out['linear_path']))
    cache = out.get('color_cache')
    if cache:
        signature = cache.pop('signature', None)
        if hashlib.sha256(json.dumps(cache, sort_keys=True, allow_nan=False).encode()).hexdigest() != signature:
            raise ValueError('预览颜色配方已改变')
        for name, arrays in [('standard', cache), ('interactive', cache.get('interactive'))]:
            if not arrays:
                continue
            for key in ('linear', 'saturated'):
                path = Path(arrays[key]).resolve(strict=True)
                if path.parent != root:
                    raise ValueError('预览缓存超出此参考目录')
                payload = path.read_bytes()
                if hashlib.sha256(payload).hexdigest() != arrays[key+'_sha256']:
                    raise ValueError('预览缓存已改变')
                data = np.load(BytesIO(payload), allow_pickle=False)
                data = orientation.rotate_array(data, delta)
                target = new_root / (name + '_' + key + '.npy')
                np.save(target, data, allow_pickle=False)
                arrays[key], arrays[key+'_sha256'] = str(target), core._sha(target)
            if name == 'interactive':
                arrays['width'], arrays['height'] = data.shape[1], data.shape[0]
            else:
                cache['proxy_geometry'] = {'width': data.shape[1], 'height': data.shape[0]}
                cache['preview_scale'] = data.shape[0] / out['height']
        cache['full_geometry'] = {'width': out['width'], 'height': out['height']}
        cache['signature'] = hashlib.sha256(json.dumps(cache, sort_keys=True, allow_nan=False).encode()).hexdigest()
    with Image.open(preview) as image:
        pixels = orientation.rotate_array(np.asarray(image), delta)
        icc = image.info.get('icc_profile')
        target = new_root / 'reference_preview.png'
        Image.fromarray(pixels).save(target, icc_profile=icc, compress_level=1)
    out.update(preview_path=str(target), preview_sha256=core._sha(target),
               preview_width=pixels.shape[1], preview_height=pixels.shape[0],
               preview_scale=pixels.shape[0]/out['height'],
               no_scale=pixels.shape[:2] == (out['height'], out['width']))
    target = new_root/'sky_mask_template.png'
    Image.new('L', (out['width'], out['height']), 255).save(target)
    out.update(mask_template_path=str(target), mask_template_sha256=core._sha(target))
    geometry = out.get('metadata', {}).get('geometry')
    if geometry:
        geometry.update(width=out['width'], height=out['height'])
    if 'fits' in out.get('metadata', {}):
        out['metadata']['fits'] = _fits_metadata(Path(out['linear_path']))
    log = new_root/'reference.log'
    log.write_text(f'RGB clockwise rotation {delta} degrees from {root}; no RAW re-decode.\n', encoding='utf-8')
    out.update(log_path=str(log), state='succeeded', stage='complete')
    core.atomic_json(new_root/'reference.json', out)
    return out


def render_reference(reference_dir, *, color_adjustments=None, quality='standard'):
    """Re-render verified linear proxies; ``interactive`` is a 640px quick view.

    ``width``/``height`` always remain original-frame geometry for the canvas.
    Every result is a separate file, so an obsolete worker cannot overwrite the
    preview another request is displaying. Original RAW and FITS are not read.
    """
    import hashlib
    from io import BytesIO
    from . import color_adjustments as ca, display_transform as dt
    if quality not in ('standard', 'interactive'):
        raise ValueError('预览质量须为 standard 或 interactive')
    root = Path(reference_dir).expanduser().resolve(strict=True)
    record = json.loads((root/'reference.json').read_text(encoding='utf-8'))
    if record.get('state') != 'succeeded' or Path(record['reference_dir']).resolve() != root:
        raise ValueError('不是已完成的原位预览')
    values = ca.normalize(color_adjustments)
    cache = record.get('color_cache')
    if not cache:
        if any(values.values()):
            raise ValueError('此参考没有已校准颜色缓存，暂不支持颜色滑块')
        return record
    signed = {k:v for k,v in cache.items() if k != 'signature'}
    if hashlib.sha256(json.dumps(signed, sort_keys=True, allow_nan=False).encode()).hexdigest() != cache['signature']:
        raise ValueError('预览颜色配方已改变')
    arrays = {}
    selected = cache.get('interactive', cache) if quality == 'interactive' else cache
    for key in ('linear','saturated'):
        path = Path(selected[key]).resolve(strict=True)
        if path.parent != root:
            raise ValueError('预览缓存超出此参考目录')
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != selected[key+'_sha256']:
            raise ValueError('预览缓存已改变或超出此参考目录')
        arrays[key] = np.load(BytesIO(payload), allow_pickle=False)
    params = ca.apply(dt.Params.from_json(cache['base_params']), cache['master_metadata'], values)
    if quality == 'interactive' and max(arrays['linear'].shape[:2]) > INTERACTIVE_MAX_SIDE:
        # Older references remain usable and do not require another RAW decode.
        from .whole_color import resize_signed
        arrays['linear'] = resize_signed(arrays['linear'], INTERACTIVE_MAX_SIDE)
        arrays['saturated'] = np.asarray(Image.fromarray(arrays['saturated'].astype(np.float32)).resize(
            (arrays['linear'].shape[1], arrays['linear'].shape[0]), Image.Resampling.BOX)) > 0
    if quality == 'interactive' and params.shadow_smoothing:
        from dataclasses import replace
        standard_height = cache.get('proxy_geometry', {}).get('height')
        if standard_height is None:
            standard_height = max(arrays['linear'].shape[0], record.get('preview_height', arrays['linear'].shape[0]))
        params = replace(params, shadow_smooth_radius=max(1, round(params.shadow_smooth_radius * arrays['linear'].shape[0] / standard_height)))
    from . import photo_render as pr
    rgb16, _ = dt.render(arrays['linear'], params, camera_saturated=arrays['saturated'],
                         workers=pr.display_workers(params, arrays['linear'].shape[0], arrays['linear'].shape[1], preview=True))
    destination = root / ('adjusted-' + uuid.uuid4().hex + '.png')
    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
    Image.fromarray(dt.to_8bit(rgb16)).save(destination, icc_profile=icc, compress_level=1)
    out = {**record, 'preview_path':str(destination), 'preview_sha256':core._sha(destination),
           'color_adjustments':values, 'display_transform':params.to_json(), 'preview_quality':quality,
           'preview_width':rgb16.shape[1], 'preview_height':rgb16.shape[0]}
    if record.get('height'):
        out['preview_scale'] = rgb16.shape[0] / record['height']
        out['no_scale'] = rgb16.shape[:2] == (record['height'], record['width'])
    core.atomic_json(destination.with_suffix('.json'), out)
    return out
