"""Reproducible train/freeze/test experiment; never relabel old Siril fits."""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from starstack.holdout_catalog import training_view, canonical_sha256
from starstack.registration_holdout import fit_training, freeze_candidate, evaluate_file, FIT_RECIPE
from starstack.records import atomic_json

NUMERICAL_TOLERANCE_PX = 0.001  # Noiseless analytic fixture only, declared before its fit.


def synthetic_catalog():
    """Analytic native coordinates and true physical IDs, independent of fitted helpers."""
    w,h = 960,720
    center = np.array([w/2-0.5+14,h/2-0.5-10])
    scale = np.hypot(w,h)/2
    k1,k2 = -.08,.015
    stars, undistorted = [], []
    for row,y in enumerate(np.linspace(65,655,10)):
        for col,x in enumerate(np.linspace(75,885,14)):
            stars.append({'id':f'star-{row:02}-{col:02}',
                          'role':'holdout' if (col*7+row*3)%4 == 0 else 'train'})
            undistorted.append([x,y])
    u = np.array(undistorted)
    frames,truth_h = [],{}
    # Polynomial u->p is evaluated explicitly; no RadialDistortion generator.
    def native(points):
        q = (points-center)/scale
        r2 = (q*q).sum(axis=1)
        return center+(points-center)*(1+k1*r2+k2*r2*r2)[:,None]
    for index,t in enumerate((-2,-1,0,1,2)):
        angle = np.deg2rad(t*2.5)
        ca,sa = np.cos(angle),np.sin(angle)
        translate_from = np.array([[1.,0.,-center[0]],[0.,1.,-center[1]],[0.,0.,1.]])
        rotate = np.array([[ca,-sa,12*t],[sa,ca,7*t],[3e-5*t,-2e-5*t,1.]])
        translate_back = np.array([[1.,0.,center[0]],[0.,1.,center[1]],[0.,0.,1.]])
        matrix = translate_back@rotate@translate_from
        inv = np.linalg.inv(matrix)
        homogeneous = np.column_stack([u,np.ones(len(u))])@inv.T
        src = native(homogeneous[:,:2]/homogeneous[:,2,None])
        fid = f'frame-{index}'
        observations = [{'star_id':star['id'],'x':float(p[0]),'y':float(p[1])} for star,p in zip(stars,src)
                        if 0 <= p[0] <= w-1 and 0 <= p[1] <= h-1]
        frames.append({'id':fid,'observations':observations})
        truth_h[fid] = matrix.tolist()
    catalog = {'version':'1.0.0','width':w,'height':h,'reference':'frame-2',
               'identity_basis':'synthetic_truth','stars':stars,'frames':frames}
    truth = {'k1':k1,'k2':k2,'center':center.tolist(),'scale':float(scale),
             'source_to_reference_undistorted':truth_h,'noise_px':0.0,
             'scope':'coordinate-only polynomial fixture; not RAW detections or Siril matches'}
    return catalog,truth


def run_synthetic(output_root):
    run = Path(output_root).resolve()/('holdout-'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ-')+uuid.uuid4().hex[:6])
    run.mkdir(parents=True,exist_ok=False)
    plan = {'recipe':FIT_RECIPE,'tolerance_px':NUMERICAL_TOLERANCE_PX,
            'reference':'frame-2','roles':'physical star ID, global across all five frames',
            'selection':'both fixed candidates evaluated; no candidate chosen by test scores',
            'expected':'radial numerical recovery; homography comparator has unmodelled distortion',
            'regions':'center/corner/other by fixed 30%-70% native reference rule'}
    atomic_json(run/'plan.json',plan)
    catalog,truth = synthetic_catalog()
    atomic_json(run/'catalog.json',catalog)
    atomic_json(run/'truth.json',truth)
    train = training_view(catalog)
    atomic_json(run/'training.json',train)
    # The fitting entry receives this view only; no catalog path, test callbacks,
    # correspondence search, saved .seq matrix or prior fitted model.
    candidate = fit_training(train)
    frozen_sha = freeze_candidate(run/'candidate.json',candidate)
    atomic_json(run/'freeze.json',{'candidate_sha256':frozen_sha,
                                 'training_sha256':canonical_sha256(train),
                                 'plan_sha256':hashlib.sha256((run/'plan.json').read_bytes()).hexdigest()})
    report = evaluate_file(run/'catalog.json',run/'candidate.json',frozen_sha,tolerance_px=NUMERICAL_TOLERANCE_PX)
    atomic_json(run/'evaluation.json',report)

    # Change every test coordinate in every frame; no refit feedback or selection.
    changed = copy.deepcopy(catalog)
    held = {s['id'] for s in changed['stars'] if s['role'] == 'holdout'}
    for i,frame in enumerate(changed['frames']):
        for p in frame['observations']:
            if p['star_id'] in held:
                p['x'] += .31*(i+1)
                p['y'] -= .19*(i+1)
    contaminated_train = training_view(changed)
    changed_candidate = fit_training(contaminated_train)
    isolated = canonical_sha256(candidate) == canonical_sha256(changed_candidate)
    atomic_json(run/'altered_test_catalog.json',changed)
    changed_report = evaluate_file(run/'altered_test_catalog.json',run/'candidate.json',frozen_sha,tolerance_px=NUMERICAL_TOLERANCE_PX)
    atomic_json(run/'altered_test_evaluation.json',changed_report)
    numerical = [c for c in report['judgement']['checks'] if c['name'].startswith('radial/')]
    residuals_changed = report['models'] != changed_report['models']
    result = {'training_hash_unchanged':canonical_sha256(train)==canonical_sha256(contaminated_train),
              'frozen_model_unchanged_after_all_test_coordinates_changed':isolated,
              'evaluation_responds_to_changed_test_coordinates':residuals_changed,
              'radial_recovery_passed':bool(numerical) and all(c['status']=='passed' for c in numerical),
              'reference_self_fit_evaluated':False,'photographic_quality_claim':False,
              'candidate_sha256':frozen_sha,'plan':plan,'run':str(run)}
    result['expected_contract_passed'] = all(result[k] for k in ('training_hash_unchanged','frozen_model_unchanged_after_all_test_coordinates_changed','evaluation_responds_to_changed_test_coordinates','radial_recovery_passed'))
    atomic_json(run/'RESULT.json',result)
    lines = ['# A7 独立留出受控实验','',f"隔离及固定数值约定：{'通过' if result['expected_contract_passed'] else '未通过'}",'',
             '140 个已知物理身份，全序列角色固定；参考帧 frame-2。候选先冻结，留出评估不回流训练。',
             f'无噪声坐标真值容差固定 {NUMERICAL_TOLERANCE_PX} px，不作为实拍画质门槛。','',
             '| 模型 | 帧 | 整体 RMS (px) | 角区 RMS (px) | 应测 / 已测 |','|---|---|---|---|---|']
    for name,frames in report['models'].items():
        for fid,f in frames.items():
            a,c = f['regions']['overall'],f['regions']['corner']
            lines.append(f"| {name} | {fid} | {a['rms_px']} | {c['rms_px']} | {a['eligible']} / {a['measured']} |")
    lines += ['',f'所有留出坐标改变后训练哈希不变：{result["training_hash_unchanged"]}；冻结模型不变：{isolated}；测试结果随之变化：{residuals_changed}。',
              '', '单应比较项的精度失败如实保留；不按测试结果选模型。该实验不含检测／自动匹配、像素重采样或实拍身份核实，默认生产标定仍是条件残差。']
    (run/'HOLDOUT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))
    return 0 if result['expected_contract_passed'] else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    p = sub.add_parser('synthetic')
    p.add_argument('--output-root',default=str(ROOT/'outputs'/'a7-support-20260919'))
    p = sub.add_parser('fit')
    p.add_argument('--training',required=True)
    p.add_argument('--candidate',required=True)
    p = sub.add_parser('evaluate')
    p.add_argument('--catalog',required=True); p.add_argument('--candidate',required=True)
    p.add_argument('--sha256',required=True); p.add_argument('--output',required=True)
    p.add_argument('--tolerance',type=float,default=None)
    args = parser.parse_args(argv)
    if args.command == 'synthetic': return run_synthetic(args.output_root)
    if args.command == 'fit':
        view = json.loads(Path(args.training).read_bytes())
        print(freeze_candidate(Path(args.candidate),fit_training(view)))
        return 0
    result = evaluate_file(Path(args.catalog),Path(args.candidate),args.sha256,tolerance_px=args.tolerance)
    # Write-once output: repeated evaluation must name another report.
    with Path(args.output).open('x',encoding='utf-8') as stream:
        json.dump(result,stream,ensure_ascii=False,allow_nan=False,indent=2)
    return {'passed':0,'failed':2,'insufficient_data':3,'not_applicable':4}[result['judgement']['status']]


if __name__ == '__main__':
    raise SystemExit(main())
