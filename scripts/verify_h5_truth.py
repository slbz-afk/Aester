"""Bounded H5 acceptance examples with an independent scalar blend oracle."""
import argparse
import datetime as dt
import json
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
from starstack import horizon_acceptance as ha
from starstack.records import atomic_json


def scalar_truth(sky, ground, weight):
    out = np.empty(sky.shape, np.float32)
    for y in range(weight.shape[0]):
        for x in range(weight.shape[1]):
            for c in range(3):
                alpha = float(weight[y, x])
                out[y, x, c] = alpha*float(sky[y, x, c])+(1-alpha)*float(ground[y, x, c])
    return out


def cases():
    h, w = 72, 96
    sky = np.full((h,w,3), [.4,.3,.2], np.float32)
    ground = np.full_like(sky, [.01,.005,.02])
    mask = np.ones((h,w), np.float32)
    mask[48:54] = np.linspace(.9,.1,6)[:,None]; mask[54:] = 0
    def item(name, expected, *, a=sky, b=ground, m=mask, weight=None, mutate=None, sample_step=1, missing=False):
        weight = m.copy() if weight is None else weight.copy()
        truth = scalar_truth(a,b,weight)
        observed = truth.copy()
        if mutate: mutate(observed)
        return dict(name=name, expected=expected, composite=observed, sky=a, ground=b, mask=m,
                    effective_weight=None if missing else weight, sample_step=sample_step)
    yield item('natural_dark_ground', 'passed')
    no_sky = mask.copy(); no_sky[42:54] = 0
    yield item('unobserved_sky_returns_ground', 'passed', weight=no_sky)
    hard = (mask >= 1).astype(np.float32)
    yield item('hard_edge_feather_not_applicable', 'passed', m=hard)
    thin = hard.copy(); thin[47] = .5
    yield item('one_pixel_feather', 'passed', m=thin)
    hdr = sky.copy(); hdr[...,0] = 8.; hdr[...,1] = -.02
    yield item('signed_residual_and_hdr', 'passed', a=hdr)
    yield item('both_sources_legitimately_zero', 'passed', a=np.zeros_like(sky), b=np.zeros_like(ground))
    yield item('injected_black_seam', 'failed', mutate=lambda a: a.__setitem__((slice(50,52),slice(None),slice(None)),0))
    yield item('single_pixel_between_old_sample_rows', 'failed', m=thin,
               mutate=lambda a: a.__setitem__((47,3,1),.15))
    yield item('coarse_sampling_is_incomplete', 'insufficient_data', m=thin, sample_step=4)
    yield item('failure_wins_even_with_coarse_sampling', 'failed', sample_step=4,
               mutate=lambda a: a.__setitem__((48,4,1),0))
    yield item('missing_recorded_weight', 'insufficient_data', missing=True)
    yield item('wrong_weight_application', 'failed', mutate=lambda a: a.__setitem__((slice(48,54),slice(None),slice(None)),sky[48:54]))
    yield item('nonfinite_input', 'failed', mutate=lambda a: a.__setitem__((47,3,1),np.nan))
    empty = np.empty((0,0,3),np.float32)
    yield item('empty_image', 'insufficient_data', a=empty,b=empty,m=np.empty((0,0),np.float32))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', default=str(ROOT/'outputs'/'h5-validation-20260919'))
    args = parser.parse_args(argv)
    out = Path(args.output_root)/('truth-'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ-')+uuid.uuid4().hex[:6])
    out.mkdir(parents=True,exist_ok=False)
    records = []
    for case in cases():
        name, expected = case.pop('name'), case.pop('expected')
        try:
            metrics = ha.seam(**case)
            judgement = ha.judge_seam(metrics,ha.CRITERIA['H5'])
            actual = judgement['status']
            record = dict(name=name,expected=expected,actual=actual,metrics=metrics,judgement=judgement)
        except ValueError as exc:
            actual = 'failed'
            record = dict(name=name,expected=expected,actual=actual,input_rejection=str(exc))
        record['matches_expectation'] = expected == actual
        records.append(record)
    report = {'horizon_acceptance_version': ha.HORIZON_ACCEPTANCE_VERSION,
              'compose_tolerance': ha.CRITERIA['H5']['compose_tolerance'],
              'oracle': 'scalar Python float blend, then one float32 storage rounding; no production blend helper',
              'scope': 'bounded synthetic numerical validation, no photograph or interpolation-support claim',
              'cases': records, 'passed': all(r['matches_expectation'] for r in records)}
    atomic_json(out/'truth.json',report)
    lines = ['# H5 已知真值案例','',f"版本 {ha.HORIZON_ACCEPTANCE_VERSION}，固定容差 1e-5。",
             '', '检验是否符合预期：'+('全部符合' if report['passed'] else '存在不符'),'',
             '| 案例 | 应有状态 | 实际状态 | 符合预期 |','|---|---|---|---|']
    for r in records:
        lines.append(f"| {r['name']} | {r['expected']} | {r['actual']} | {r['matches_expectation']} |")
    lines += ['', '错误案例的 failed 是预期检出，不能当作验证套件失败。合成正确不证明权重本身正确；未验证真实摄影接边或插值支持。']
    (out/'TRUTH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'passed':report['passed'],'cases':len(records),'report':str(out.resolve()/'TRUTH.md')},ensure_ascii=False))
    return 0 if report['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
