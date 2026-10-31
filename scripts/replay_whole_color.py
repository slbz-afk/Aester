"""Bounded development replay of audited stacks; not a general import path.

Creates a new directory, never rewrites jobs or masters. Camera appearance is
explicitly post-stack compatibility; the NGC branch uses an existing, frozen
background hypothesis. It does not implement automatic background selection.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import platform

import numpy as np
from PIL import Image, ImageCms
import tifffile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from starstack import whole_color as wc, raw_color
from starstack.imaging import read_rgb


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def read_json_snapshot(path):
    data = path.read_bytes()
    return json.loads(data), hashlib.sha256(data).hexdigest()


def record_digest(value):
    payload = {k:v for k,v in value.items() if k!='record_sha256'}
    return hashlib.sha256(json.dumps(payload,sort_keys=True,allow_nan=False).encode()).hexdigest()


def reference_recipe(case):
    fields = case['actual_reference_metadata_after_unpack']
    meta = {'available': True, 'white_balance_applied': False, 'color_correction_applied': False,
            'versions': {'rawpy': '0.27.1', 'libraw': '0.22.1'}, 'fields': {}}
    for k, v in [('num_colors', fields['num_colors']), ('color_desc', fields['color_desc']),
                 ('raw_pattern', fields['raw_pattern']), ('rgb_xyz_matrix', fields['cam_xyz']),
                 ('camera_whitebalance', fields['camera_wb'])]:
        meta['fields'][k] = {'value': v}
    if case['black_variants'] != [[1008, 1008, 1008, 1008]]:
        raise ValueError('This bounded experiment requires independently verified common black')
    ref = case['metadata'][case['reference_position']]
    if sha(ref['path']) != ref['sha256']:
        raise ValueError('Audited reference changed')
    return raw_color.build_recipe(meta, ref['sha256'], fields['white'], 1008,
                                  camera_identity={'name': 'Nikon Z 8', 'scope': 'two audited development sequences'})


def save_preview(path, pixels, icc):
    Image.fromarray(pixels).save(path.with_suffix('.png'), icc_profile=icc)
    Image.fromarray(pixels).save(path, quality=95, icc_profile=icc)


def run(out, replay=None):
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    audit_path = ROOT/'outputs/whole-color-20260920/input-audit/RAW_STATE_RESULT.json'
    audit, audit_sha = read_json_snapshot(audit_path)
    previous = None
    if replay is not None:
        previous, replay_sha = read_json_snapshot(replay)
        if previous.get('state') != 'succeeded' or previous.get('version') != wc.VERSION:
            raise ValueError('Only completed matching-version recipes can be replayed')
        if record_digest(previous) != previous.get('record_sha256'):
            raise ValueError('Replay record changed or is incomplete')
        if any(sha(p) != v for p,v in previous['sources'].items()):
            raise ValueError('Frozen replay source identity changed')
    b1path = ROOT/'outputs/background-color-20260920/real-v1/RESULTS.json'
    b1, b1_sha = read_json_snapshot(b1path)
    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
    record = {'state': 'running', 'scope': 'audited stack replay; experimental, no default change',
              'version': wc.VERSION, 'sources': {str(audit_path): audit_sha, str(b1path): b1_sha},
              'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                              'Pillow': Image.__version__, 'tifffile': tifffile.__version__, 'platform':platform.platform()},
              'cases': [], 'limitations': ['No new decoding, registration or stacking in this replay.',
              'Camera matrices applied after old per-channel normalization/rejection are approximate.',
              'Display clips/quantizes; float camera derivative and old master remain separate.',
              'Camera base composites exclude protected bright-trail layers.',
              'No DCP HueSatMap, denoising, SHO or artistic spectral separation implemented.']}
    for p in [Path(__file__), ROOT/'src/starstack/whole_color.py', ROOT/'src/starstack/linked_tone.py',
              ROOT/'src/starstack/raw_color.py', ROOT/'src/starstack/background.py', ROOT/'src/starstack/imaging.py']:
        record['sources'][str(p)] = sha(p)
    if replay is not None:
        if previous.get('environment') != record['environment']:
            raise ValueError('Exact development replay requires the recorded environment')
        record['sources'][str(replay)] = replay_sha
        record['replayed_from'] = str(replay)
    write_json(out/'RESULT.json', record)
    try:
        for case in audit['cases']:
            key = case['key']
            d = out/key
            d.mkdir()
            t0 = time.perf_counter()
            jobp = Path(case['job'])/'job.json'
            job, job_sha = read_json_snapshot(jobp)
            if job_sha != case['job_sha_before']:
                raise ValueError('Audited camera job changed')
            paths = [Path(a['path']) for a in job['artifacts'] if Path(a['path']).name == 'master_linear.fit']
            if job['state'] != 'succeeded' or len(paths) != 1:
                raise ValueError('Need published successful base stack')
            src = paths[0]
            record['sources'][str(src)] = sha(src)
            record['sources'][str(jobp)] = job_sha
            ref = case['metadata'][case['reference_position']]
            record['sources'][ref['path']] = ref['sha256']
            data = np.flipud(read_rgb(src)).copy()
            color = reference_recipe(case)
            if previous is not None:
                prior = next(c for c in previous['cases'] if c['key']==key)
                if prior['raw_color'] != color or prior['source'] != str(src):
                    raise ValueError('Camera reference/geometry evidence differs')
                color = prior['raw_color']
            linear = wc.camera_from_cache(data, color, black_dn=1008, input_state=wc.CACHE_STATE)
            tifffile.imwrite(d/'approximate_linear_srgb.tif', linear, photometric='rgb', metadata={
                'roworder': 'TOP-DOWN', 'color_space': 'approximate_linear_sRGB_D65',
                'input_state': wc.CACHE_STATE, 'source_sha256': record['sources'][str(src)],
                'gamma_applied': False, 'clipped': False, 'post_stack_approximation': True})
            small = wc.resize_signed(linear)
            sample = linear[::12, ::12]
            peak = sample.max(axis=2)
            # Dark whole-image target inspired by the user's work, not pixel fitting.
            white = float(np.quantile(peak, .9999))
            anchor = float(np.median(peak))
            display = wc.make_display_recipe(black=0., white=white, anchor=anchor, target_srgb=.22)
            if previous is not None:
                display = prior['display']
            preview = wc.render_display(small, display)
            save_preview(d/'candidate.jpg', preview, icc)
            look_gains = [.94, 1., 1.08]
            look_peak = (sample*np.array(look_gains)).max(axis=2)
            look = wc.make_display_recipe(black=0., white=float(np.quantile(look_peak,.9999)),
                        anchor=float(np.median(look_peak)),target_srgb=.19,gains_rgb=look_gains)
            if previous is not None:
                look = prior['reference_look']
            save_preview(d/'reference_look.jpg', wc.render_display(small,look),icc)
            old = [Path(a['path']) for a in job['artifacts'] if Path(a['path']).name == 'natural.jpg' and a.get('group') != 'transients'][0]
            record['sources'][str(old)] = sha(old)
            with Image.open(old) as im:
                im = im.convert('RGB'); im.thumbnail((1600,1600), Image.Resampling.LANCZOS)
                im.save(d/'old_default.jpg', quality=95, icc_profile=icc)
            item = {'key': key, 'source': str(src), 'raw_color': color, 'display': display, 'reference_look':look,
                    'reference_look_policy': 'explicit optional global look: RGB gains .94/1/1.08, peak median -> encoded .19; no pixel fitting to reference',
                    'display_parameter_policy': 'black=0; full image strided peak median -> encoded .22; white peak quantile .9999',
                    'spatial_color_masks': None, 'camera_calibration_state': 'audited_cache_approximation',
                    'statistics': {'linear_minmax': [float(linear.min()), float(linear.max())],
                        'linear_negative_channels': int((linear<0).sum()), 'linear_above_one_channels': int((linear>1).sum()),
                        'preview_zero_pixel_fraction': float((preview.max(axis=2)==0).mean()),
                        'preview_any_255_fraction': float((preview.max(axis=2)==255).mean())},
                    'elapsed_seconds': time.perf_counter()-t0}
            write_json(d/'recipe.json', item)
            record['cases'].append(item)
            del data, linear, small, sample
        # Deep-sky display uses the same B1 linear background correction. Only tone differs.
        ngc = next(x['first'] for x in b1 if x['key']=='ngc281')
        src = Path(ngc['directory'])/'coordinated_linear_32bit.tif'
        maskp = Path(ngc['directory'])/'background_mask.png'
        d = out/'ngc281'; d.mkdir()
        record['sources'][str(src)] = sha(src)
        record['sources'][str(maskp)] = sha(maskp)
        if sha(src) != next(a['sha256'] for a in ngc['artifacts'] if a['kind']=='tiff'):
            raise ValueError('Frozen NGC background derivative changed')
        linear = tifffile.imread(src)
        with Image.open(maskp) as im:
            mask = np.asarray(im)==255
        sm = wc.resize_signed(linear,1600)
        bm = np.array(Image.fromarray(mask).resize((sm.shape[1],sm.shape[0]),Image.Resampling.NEAREST))
        selected = sm[bm]
        med = np.median(selected,axis=0)
        sigma = 1.4826*np.median(np.abs(selected-med),axis=0)
        black = float(med.min()-2.8*sigma.max())
        white = float(np.quantile(linear[::8,::8].max(axis=2), .9999))
        anchor = float(np.median(selected.max(axis=1)))
        display = wc.make_display_recipe(black=black, white=white, anchor=anchor, target_srgb=.18)
        if previous is not None:
            prior = next(c for c in previous['cases'] if c['key']=='ngc281')
            if prior['source'] != str(src):
                raise ValueError('NGC source differs')
            display = prior['display']
        preview = wc.render_display(linear, display)
        save_preview(d/'candidate.jpg', preview, icc)
        oldp = Path(ngc['directory'])/'after.jpg'
        record['sources'][str(oldp)] = sha(oldp)
        with Image.open(oldp) as im:
            im.save(d/'old_background_display.jpg', quality=95, icc_profile=icc)
        # A common display on the original uncorrrected stack exposes the spatial/colour change.
        basep = Path(ngc['source_path'])
        record['sources'][str(basep)] = sha(basep)
        before = wc.render_display(np.flipud(read_rgb(basep)).copy(),display)
        save_preview(d/'before_common_tone.jpg',before,icc)
        pm = np.array(Image.fromarray(mask).resize((preview.shape[1],preview.shape[0]),Image.Resampling.NEAREST))
        pix = preview[pm]
        item = {'key':'ngc281', 'source':str(src), 'display':display,
                'background_recipe':ngc['recipe_path'], 'background_sample_mask':str(maskp),
                'display_parameter_policy':'display-size BOX signed RGB; mask channel medians minus 2.8*largest robust sigma for common black; background peak median -> encoded .18; white full-image strided peak quantile .9999',
                'statistics': {'sample_median_rgb':med.tolist(), 'sample_robust_sigma_rgb':sigma.tolist(),
                    'background_median_encoded_rgb':np.median(pix,axis=0).tolist(),
                    'background_all_zero_fraction':float((pix.max(axis=1)==0).mean()),
                    'background_any_zero_fraction':float((pix.min(axis=1)==0).mean()),
                    'preview_any_255_fraction':float((preview.max(axis=2)==255).mean())}}
        write_json(d/'recipe.json',item); record['cases'].append(item)
        if any(sha(p)!=digest for p,digest in record['sources'].items()):
            raise ValueError('A source changed during rendering')
        record['state']='succeeded'
        record['elapsed_seconds']=time.perf_counter()-started
        record['artifacts'] = {str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name!='RESULT.json'}
    except Exception as exc:
        record.update(state='failed', error=str(exc))
        raise
    finally:
        record['record_sha256'] = record_digest(record)
        write_json(out/'RESULT.json', record)
    print(json.dumps({'state':record['state'], 'directory':str(out), 'seconds':record['elapsed_seconds']}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--recipe', type=Path, help='Completed RESULT.json to replay with frozen display parameters')
    args=p.parse_args()
    run(args.out.resolve(), args.recipe.resolve() if args.recipe else None)
