"""Isolated registration experiments with global, preassigned star identities.

The trainer receives training observations only. It neither reads Siril caches
nor chooses a candidate from test scores. This is a diagnostic entry point,
not a replacement for production calibration or proof of real-star identity.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from . import registration_distortion as rd
from .registration_models import fit_homography, apply_matrix, region_of
from .holdout_catalog import validate_catalog, training_view, holdout_view, canonical_sha256
from .acceptance_status import summarize_checks

HOLDOUT_VERSION = '1.0.0'
FIT_RECIPE = {'direction': 'u_to_p', 'fit_centre': True, 'iterations': 40, 'min_pairs': 12}
REGIONS = ('overall', 'center', 'corner', 'other')


def _training_data(view):
    """Validate this separate schema without manufacturing holdout observations."""
    keys = {'version','width','height','reference','identity_basis','role','stars','frames'}
    if type(view) is not dict or set(view) != keys or view['version'] != '1.0.0' or view['role'] != 'train':
        raise ValueError('trainer accepts a training view only')
    canonical_sha256(view)  # Strict JSON types, cycles and finite numbers.
    w,h = view['width'],view['height']
    if any(type(v) is not int or v < 2 for v in (w,h)):
        raise ValueError('invalid training dimensions')
    if view['identity_basis'] not in ('synthetic_truth','external_verified'):
        raise ValueError('unknown identity basis')
    ids = view['stars']
    if type(ids) is not list or not ids or any(type(s) is not str or not s.strip() for s in ids) or len(set(ids)) != len(ids):
        raise ValueError('training identities must be unique nonempty strings')
    if type(view['frames']) is not list or len(view['frames']) < 2:
        raise ValueError('training requires reference and at least one frame')
    frames = {}
    for frame in view['frames']:
        if type(frame) is not dict or set(frame) != {'id','observations'}:
            raise ValueError('malformed training frame')
        fid = frame['id']
        if type(fid) is not str or not fid.strip() or fid in frames or type(frame['observations']) is not list:
            raise ValueError('invalid training frame identity or observations')
        observations = {}
        for p in frame['observations']:
            if type(p) is not dict or set(p) != {'star_id','x','y'}:
                raise ValueError('malformed training observation')
            sid = p['star_id']
            if type(sid) is not str or sid not in ids or sid in observations:
                raise ValueError('unknown or duplicate training identity')
            if any(type(p[c]) not in (int,float) for c in ('x','y')):
                raise ValueError('invalid training coordinates')
            try:
                x,y = float(p['x']),float(p['y'])
            except (ValueError,OverflowError) as exc:
                raise ValueError('invalid training coordinates') from exc
            if not np.isfinite([x,y]).all() or not (0 <= x <= w-1 and 0 <= y <= h-1):
                raise ValueError('training coordinates outside image')
            observations[sid] = (x,y)
        frames[fid] = observations
    reference = view['reference']
    if type(reference) is not str or reference not in frames or set(frames[reference]) != set(ids):
        raise ValueError('reference must observe all declared training stars')
    return frames


def fit_training(view: dict) -> dict:
    """Fit both predetermined models from train-only coordinates; no I/O or test access."""
    frames = _training_data(view)
    view = copy.deepcopy(view)
    reference = view['reference']
    ref = frames[reference]
    inputs, unavailable = {}, {}
    used = {}
    for index,fid in enumerate(sorted(frames)):
        if fid == reference:
            continue
        ids = sorted(frames[fid])
        used[fid] = ids
        if len(ids) < FIT_RECIPE['min_pairs']:
            unavailable[fid] = f"only {len(ids)} training pairs; require {FIT_RECIPE['min_pairs']}"
            continue
        inputs[fid] = {'span': index, 'src': np.array([frames[fid][sid] for sid in ids]),
                       'dst': np.array([ref[sid] for sid in ids])}
    candidate = {'version': HOLDOUT_VERSION, 'recipe': dict(FIT_RECIPE),
                 'reference': reference, 'width': view['width'], 'height': view['height'],
                 'identity_basis': view['identity_basis'], 'training_sha256': canonical_sha256(view),
                 'training_ids': sorted(view['stars']), 'frame_training_ids': used,
                 'excluded_training_frames': unavailable, 'models': {},
                 'selection': 'both candidates fixed before test evaluation; no winner selected'}
    if not inputs:
        return candidate
    hs = {fid: fit_homography(f['src'],f['dst']) for fid,f in inputs.items()}
    candidate['models']['homography'] = {'frames': {fid:h.tolist() for fid,h in hs.items()}}
    radial = rd.fit_joint_radial(inputs, view['width'], view['height'],
                                masks={fid:np.ones(len(f['src']),bool) for fid,f in inputs.items()},
                                direction=FIT_RECIPE['direction'], fit_centre=FIT_RECIPE['fit_centre'],
                                iterations=FIT_RECIPE['iterations'])
    candidate['models']['radial'] = {'parameters': radial['distortion'].as_dict(),
                                    'frames': {fid:h.tolist() for fid,h in radial['H'].items()}}
    canonical_sha256(candidate)  # A non-finite candidate is never published.
    return candidate


def freeze_candidate(path: Path, candidate: dict) -> str:
    """Write-once local evidence. Hashes bind bytes; they are not signatures."""
    canonical_sha256(candidate)
    data = json.dumps(candidate,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('utf-8')
    with Path(path).open('xb') as stream:
        stream.write(data)
        stream.flush()
        import os
        os.fsync(stream.fileno())
    import hashlib
    return hashlib.sha256(data).hexdigest()


def _stats(records, eligible):
    errors = np.array([r['error_px'] for r in records],dtype=np.float64)
    return {'eligible': eligible, 'measured': len(records), 'missing': eligible-len(records),
            'rms_px': float(np.sqrt(np.mean(errors**2))) if len(errors) else None,
            'p95_px': float(np.percentile(errors,95)) if len(errors) else None,
            'max_px': float(errors.max()) if len(errors) else None}


def evaluate_frozen(catalog: dict, candidate: dict, *, tolerance_px=None) -> dict:
    """Evaluate every known test identity without matching, rejection or refitting.

    A tolerance must be fixed before evaluation to claim numerical precision.
    No tolerance means measured-only/insufficient_data, never an invented pass.
    External physical identity verification remains outside this tool.
    """
    document = validate_catalog(catalog)
    train, test = training_view(document), holdout_view(document)
    if not isinstance(candidate,dict) or candidate.get('version') != HOLDOUT_VERSION:
        raise ValueError('unknown frozen candidate format')
    canonical_sha256(candidate)
    if candidate.get('training_sha256') != canonical_sha256(train):
        raise ValueError('frozen candidate training data mismatch')
    if any(candidate.get(k) != train[k] for k in ('width','height','reference','identity_basis')):
        raise ValueError('frozen candidate geometry or provenance mismatch')
    if candidate.get('training_ids') != sorted(train['stars']):
        raise ValueError('frozen candidate identity ledger mismatch')
    if candidate.get('recipe') != FIT_RECIPE:
        raise ValueError('unknown frozen fit recipe')
    if tolerance_px is not None:
        if type(tolerance_px) not in (int,float):
            raise ValueError('tolerance must be finite and nonnegative')
        try:
            tolerance_px = float(tolerance_px)
        except (ValueError,OverflowError) as exc:
            raise ValueError('tolerance must be finite and nonnegative') from exc
        if not np.isfinite(tolerance_px) or tolerance_px < 0:
            raise ValueError('tolerance must be finite and nonnegative')
    expected_used = {f['id']:sorted(p['star_id'] for p in f['observations']) for f in train['frames'] if f['id'] != train['reference']}
    if candidate.get('frame_training_ids') != expected_used:
        raise ValueError('frozen per-frame training ledger mismatch')
    test_frames = {f['id']:{p['star_id']:(p['x'],p['y']) for p in f['observations']} for f in test['frames']}
    ref = test_frames[test['reference']]
    regions = {sid:region_of(*xy,train['width'],train['height']) for sid,xy in ref.items()}
    checks, results = [], {}
    for name in ('homography','radial'):
        model = candidate.get('models',{}).get(name)
        model_frames = {} if model is None else model.get('frames',{})
        distortion = rd.RadialDistortion(**model['parameters']) if name == 'radial' and model is not None else None
        results[name] = {}
        for fid,observed in test_frames.items():
            if fid == test['reference']:
                continue
            matrix = model_frames.get(fid)
            records = []
            if matrix is not None:
                h = np.asarray(matrix,dtype=np.float64)
                if h.shape != (3,3) or not np.isfinite(h).all():
                    raise ValueError('invalid frozen matrix')
                ids = sorted(observed)
                points = np.array([observed[sid] for sid in ids],dtype=np.float64).reshape(-1,2)
                prediction = apply_matrix(h,points if distortion is None else distortion.apply(points))
                if distortion is not None:
                    prediction = distortion.inverse(prediction)
                if not np.isfinite(prediction).all():
                    raise ValueError('non-finite frozen-model prediction')
                for sid,pred in zip(ids,prediction):
                    delta = pred-np.asarray(ref[sid])
                    records.append({'star_id':sid,'region':regions[sid], 'residual_xy':delta.tolist(),
                                    'error_px':float(np.hypot(*delta))})
            measurements = {}
            for region in REGIONS:
                expected = [sid for sid in ref if region == 'overall' or regions[sid] == region]
                selected = [r for r in records if region == 'overall' or r['region'] == region]
                stat = _stats(selected,len(expected)); measurements[region] = stat
                if not expected:
                    status, reason = 'insufficient_data','requested spatial region has no eligible test stars'
                elif tolerance_px is not None and stat['max_px'] is not None and stat['max_px'] > tolerance_px:
                    status, reason = 'failed',f'maximum residual exceeds predeclared {tolerance_px:g} px'
                elif stat['missing']:
                    status, reason = 'insufficient_data','requested test stars or training model missing'
                elif tolerance_px is None:
                    status, reason = 'insufficient_data','measured; no predeclared numerical/photographic threshold'
                elif train['identity_basis'] != 'synthetic_truth':
                    status, reason = 'insufficient_data','external identity declaration not independently verified by this tool'
                else:
                    status, reason = 'passed','all declared synthetic test identities satisfy the fixed numerical tolerance'
                checks.append({'name':f'{name}/{fid}/{region}','status':status,'samples':len(selected),'reason':reason,'metrics':stat})
            corners = {}
            measured_ids = {r['star_id'] for r in records}
            for corner in ('top_left','top_right','bottom_left','bottom_right'):
                expected = [sid for sid,(x,y) in ref.items() if regions[sid] == 'corner'
                            and ('left' in corner) == (x < train['width']/2)
                            and ('top' in corner) == (y < train['height']/2)]
                measured = len(set(expected) & measured_ids)
                corners[corner] = {'eligible':len(expected),'measured':measured,'missing':len(expected)-measured,
                                   'status':'passed' if expected and measured == len(expected) else 'insufficient_data',
                                   'scope':'sampling coverage only, not positional accuracy'}
            results[name][fid] = {'regions':measurements,'residuals':records,'corner_coverage':corners,
                                  'missing_star_ids':sorted(set(ref)-{r['star_id'] for r in records})}
    return {'version':HOLDOUT_VERSION,'catalog_sha256':canonical_sha256(document),
            'candidate_sha256':canonical_sha256(candidate), 'training_sha256':candidate['training_sha256'],
            'tolerance_px':tolerance_px, 'identity_basis':train['identity_basis'],
            'scope':'known-identity diagnostic; no detector, matcher, image warp or photographic certification',
            'accuracy_regions':list(REGIONS), 'corner_coverage_in_accuracy_verdict':False,
            'models':results, 'judgement':summarize_checks(checks)}


def evaluate_file(catalog_path: Path, candidate_path: Path, expected_sha256: str, *, tolerance_px=None):
    """Verify and decode the same frozen bytes before reading the evaluation catalog."""
    import hashlib
    data = Path(candidate_path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError('frozen candidate changed before evaluation')
    candidate = json.loads(data)
    catalog = json.loads(Path(catalog_path).read_bytes())
    return evaluate_frozen(catalog,candidate,tolerance_px=tolerance_px)
