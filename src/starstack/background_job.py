"""Immutable background/display derivatives of a completed stack job."""
from pathlib import Path
import hashlib
import io
import json
import time
import uuid

import numpy as np
from PIL import Image, ImageCms
import tifffile

from . import background as bg, core, hashing
from .compose_evidence import read_weight
from .imaging import read_rgb
from .preview import _real_path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _inside(value, directory):
    p = Path(value)
    if not p.is_absolute():
        p = directory/p
    if '..' in p.parts or not p.is_relative_to(directory):
        raise ValueError('Background source must be inside the completed job')
    p = _real_path(p)
    if not p.is_file() or not p.is_relative_to(directory):
        raise ValueError('Background source is not a regular job file')
    return p


def render_background(job_dir, mask_path=None, recipe_path=None, *, mode=None,
                      strength=None, red_gain=None, blue_gain=None):
    """Create a unique derivative; replay uses frozen model/display, never refits."""
    started = time.perf_counter()
    directory = _real_path(Path(job_dir).expanduser().absolute())
    job_path = _inside('job.json', directory)
    job_bytes = job_path.read_bytes()
    job_hash = hashing.sha256_bytes(job_bytes, source="background_job.job")
    job = json.loads(job_bytes)
    if job.get('state') != 'succeeded':
        raise ValueError('Background coordination needs a completed stack job')
    candidates = [a['path'] for a in job.get('artifacts', []) if a.get('kind') == 'fits' and Path(a['path']).name == 'master_linear.fit']
    if len(candidates) != 1:
        raise ValueError('Need one published base composite, not an external single master')
    source = _inside(candidates[0], directory)
    core._validate_master(source)
    source_hash = core._sha(source)
    data = np.flipud(read_rgb(source)).copy()  # this branch is exclusively TOP-DOWN
    h, w = data.shape[:2]
    descriptor = ((job.get('diagnostics') or {}).get('sky_validity') or {}).get('compose_weight')
    weight, weight_path, weight_hash = None, None, None
    if job.get('mask') or descriptor is not None:
        if not isinstance(descriptor, dict):
            raise ValueError('A ground composite needs its recorded actual sky weight')
        weight_path = _inside(descriptor['path'], directory)
        _inside(descriptor['composite'], directory)
        weight = read_weight(directory, descriptor, source)
        bg.weight_array(weight, (h, w))
        weight_hash = core._sha(weight_path)
        if weight_hash != descriptor['sha256']:
            raise ValueError('Sky weight changed during loading')
    identity = {'source_sha256': source_hash, 'job_sha256': job_hash,
                'shape_hw': [h, w], 'roworder': 'TOP-DOWN', 'weight_sha256': weight_hash}
    if recipe_path is not None:
        if any(v is not None for v in (mask_path, mode, strength, red_gain, blue_gain)):
            raise ValueError('Replay cannot combine a frozen recipe with new controls')
        previous = _inside(recipe_path, directory)
        previous_bytes = previous.read_bytes()
        previous_record = json.loads(previous_bytes)
        payload = previous_record['payload']
        if previous_record.get('state') != 'succeeded' or digest(payload) != previous_record.get('payload_sha256'):
            raise ValueError('Replay recipe is incomplete or has changed')
        if payload.get('version') != bg.VERSION or payload.get('identity') != identity:
            raise ValueError('Replay source, job, geometry, weight or algorithm differs')
        frozen_mask = _inside(previous.parent/'background_mask.png', directory)
        mask_bytes = frozen_mask.read_bytes()
        if hashing.sha256_bytes(mask_bytes, source="background_job.mask") != payload.get('mask_sha256'):
            raise ValueError('Replay sample mask differs')
        mask_original = frozen_mask
    else:
        if mask_path is None:
            raise ValueError('Choose explicit background samples; automatic sky/nebula selection is not yet validated')
        mode = mode or 'both'
        if mode not in ('gradient', 'neutralize', 'both'):
            raise ValueError('Unknown background mode')
        strength = bg.number(1. if strength is None else strength, 0, 1, 'strength')
        red_gain = bg.number(1. if red_gain is None else red_gain, .25, 4, 'red gain')
        blue_gain = bg.number(1. if blue_gain is None else blue_gain, .25, 4, 'blue gain')
        mask_original = _real_path(Path(mask_path).expanduser().absolute())
        mask_bytes = mask_original.read_bytes()
        payload = {'version': bg.VERSION, 'identity': identity, 'mode': mode,
                   'strength': strength, 'mask_sha256': hashing.sha256_bytes(mask_bytes, source="background_job.mask")}
    with Image.open(io.BytesIO(mask_bytes)) as image:
        if image.format != 'PNG' or image.mode not in ('L', '1') or image.size != (w, h):
            raise ValueError('Background mask must be a full-size grayscale PNG, TOP-DOWN')
        samples = np.asarray(image.convert('L'))
        if not np.isin(samples, [0, 255]).all():
            raise ValueError('Background sample mask must be binary, not feathered')
        samples = samples == 255
    variations = directory/'background-variations'
    if variations.exists():
        _real_path(variations)
    variations.mkdir(exist_ok=True)
    output = variations/uuid.uuid4().hex
    output.mkdir()
    record_path = output/'recipe.json'
    record = {'state': 'running', 'created_at': core.now(), 'directory': str(output),
              'recipe_path': str(record_path), 'job_dir': str(directory), 'source_path': str(source),
              'original_modified': False, 'artifacts': [], 'color_calibrated': False,
              'input_color_state': 'uncalibrated_camera_or_input_RGB',
              'scope': 'base composite only; protected bright-trail derivative is not included',
              'limitations': ['Explicit samples may contain real emission; flat/gray background is not truth.',
                              'No camera profile, physical white balance, narrowband extraction or SHO here.',
                              'No-ground jobs apply to the full image; geometric edge coverage is not certified.',
                              'Linear TIFF is a float derivative; JPG has display clipping and quantization.']}
    core.atomic_json(record_path, record)
    try:
        (output/'background_mask.png').write_bytes(mask_bytes)
        if recipe_path is None:
            payload['model'] = bg.fit_background(data, samples, weight)
        corrected = bg.apply_background(data, payload['model'], mode=payload['mode'], strength=payload['strength'], weight=weight)
        if recipe_path is None:
            payload['display'] = bg.display_parameters(corrected)
            payload['display']['gains_rgb'] = [red_gain, 1., blue_gain]
        # Fail display validation before committing a large TIFF.
        before = bg.render_display(data, payload['display'])
        after = bg.render_display(corrected, payload['display'])
        linear_path = output/'coordinated_linear_32bit.tif'
        tifffile.imwrite(linear_path, corrected, photometric='rgb', metadata={
            'roworder': 'TOP-DOWN', 'color_space': 'uncalibrated_camera_or_input_RGB',
            'linear': True, 'background_modified': True, 'source_sha256': source_hash})
        icc = ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
        artifacts = [{'path': str(linear_path), 'kind': 'tiff', 'label': '背景协调浮点派生（未校色）'}]
        for name, pixels, label in [('before', before, '协调前（共同显示）'), ('after', after, '协调后（共同显示）')]:
            path = output/f'{name}.jpg'
            Image.fromarray(pixels).save(path, quality=95, icc_profile=icc)
            artifacts.append({'path': str(path), 'kind': 'jpeg', 'label': label})
        if core._sha(source) != source_hash or core._sha(job_path) != job_hash:
            raise ValueError('Source or job changed while producing derivative; not published')
        if core._sha(mask_original) != payload['mask_sha256'] or (weight_path is not None and core._sha(weight_path) != weight_hash):
            raise ValueError('Background samples or weights changed while producing derivative')
        if recipe_path is not None and previous.read_bytes() != previous_bytes:
            raise ValueError('Replay recipe changed while producing derivative')
        for a in artifacts:
            a['sha256'] = core._sha(Path(a['path']))
        record.update(state='succeeded', payload=payload, payload_sha256=digest(payload), artifacts=artifacts,
                      completed_at=core.now(), elapsed_seconds=time.perf_counter()-started,
                      statistics={'input_range': [float(data.min()), float(data.max())],
                                  'output_range': [float(corrected.min()), float(corrected.max())],
                                  'negative_channels': int(np.count_nonzero(corrected < 0)),
                                  'above_one_channels': int(np.count_nonzero(corrected > 1)),
                                  'preview_size_wh': [after.shape[1], after.shape[0]],
                                  'ground_unchanged': (bool(np.array_equal(data[weight == 0].view(np.uint32), corrected[weight == 0].view(np.uint32))) if weight is not None else None)})
        core.atomic_json(record_path, record)
    except Exception as exc:
        record.update(state='failed', artifacts=[], error=str(exc), payload=payload)
        core.atomic_json(record_path, record)
        raise
    return record
