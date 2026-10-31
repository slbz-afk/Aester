"""Source-fixed starless guards and bound injection truth; see the metrics-3 protocol.

NumPy only. No model execution or file I/O. Older metric reports are not rerun.
Memory is bounded by a small number of float32 planes, row histograms, batched
star windows, and local saturation components (no full-image RGB float64).
"""
from __future__ import annotations

import numpy as np
from . import photographic as p
from . import starless_metrics as m1
from . import starless_metrics2 as m2

VERSION = "starless-metrics-3.1"
TRUTH_VERSION = "starless-injection-3"
E1_VERSION = "starless-e1-guards-3.1"
MAX_COMPONENTS = 256
MAX_COMPONENT_SIDE = 1024
_HIST_EDGES = np.r_[0., np.arange(1, 4097) / 64., np.geomspace(64., 131072., 129)[1:]]


def summary(values):
    values = np.asarray(values, np.float64)
    values = values[np.isfinite(values)]
    if not values.size:
        return dict(count=0, median=None, p10=None, p90=None, p95=None, max=None)
    return dict(count=int(values.size), median=float(np.median(values)),
                p10=float(np.percentile(values, 10)), p90=float(np.percentile(values, 90)),
                p95=float(np.percentile(values, 95)), max=float(values.max()))


def _validate(source, estimate, region):
    m1._check_rgb(source, "source")
    m1._check_rgb(estimate, "estimate")
    if source.shape != estimate.shape:
        raise ValueError("source/estimate shapes differ")
    if not isinstance(region, np.ndarray) or region.dtype != np.bool_ or region.shape != source.shape[:2]:
        raise ValueError("region must be same-size bool")


def _bundle(stats, keep, ys, xs):
    vals = {k: np.asarray(v)[keep] for k, v in stats.items()}
    rr = vals['residual_signed']
    ll = vals['luminance_signed']
    coords = np.column_stack((ys[keep], xs[keep])).astype(np.int64)
    out = {k: summary(v) for k, v in vals.items() if k != 'snr'}
    out.update(residual_absolute=summary(abs(rr)), residual_negative=summary(np.maximum(-rr, 0)),
               residual_positive=summary(np.maximum(rr, 0)),
               luminance_absolute=summary(abs(ll)), luminance_negative=summary(np.maximum(-ll, 0)),
               luminance_positive=summary(np.maximum(ll, 0)),
               negative_fraction=float((rr < -.05).mean()) if rr.size else None,
               candidates=int(len(ys)), count=int(keep.sum()), sample_sha256=p.array_identity(coords))
    return out


def _stars(ys, xs, LS, LD, sigma, estimate, hpD, *, dense=False, bgS=None, bgD=None):
    """Statistics in batches; selection and scaling use source sigma only."""
    radius, ap_r, inner_r = (8, 2, 1) if dense else (12, 5, 3)
    _, _, dist = m1._offsets(radius)
    ap, inner = dist <= ap_r, dist <= inner_r
    ring = ((dist >= 6) & (dist <= 8)) if dense else ((dist >= 8) & (dist <= 12))
    chunks = {k: [] for k in ('snr', 'residual_signed', 'luminance_signed', 'colour_shift', 'fill_noise_ratio')}
    for start in range(0, len(ys), 256):
        y, x = ys[start:start + 256], xs[start:start + 256]
        ls, ld = m1._gather(LS, y, x, radius), m1._gather(LD, y, x, radius)
        noise = sigma[y, x].astype(np.float64)
        if dense:
            bs, bd = m1._gather(bgS, y, x, radius), m1._gather(bgD, y, x, radius)
            fs = (ls[:, ap] - bs[:, ap]).sum(1, dtype=np.float64)
            fd = (ld[:, ap] - bd[:, ap]).sum(1, dtype=np.float64)
            offset = (ld[:, inner] - bd[:, inner]).mean(1) / noise
        else:
            bs, bd = np.median(ls[:, ring], 1), np.median(ld[:, ring], 1)
            fs = (ls[:, ap] - bs[:, None]).sum(1, dtype=np.float64)
            fd = (ld[:, ap] - bd[:, None]).sum(1, dtype=np.float64)
            offset = (ld[:, inner].mean(1) - bd) / noise
        shift = np.zeros(len(y))
        for c in range(3):
            colour = m1._gather(estimate[..., c], y, x, radius).astype(np.float32) / 65535 - ld
            shift = np.maximum(shift, abs(colour[:, inner].mean(1) - colour[:, ring].mean(1)) / noise)
        hp = m1._gather(hpD, y, x, radius)
        rs = hp[:, ring].std(1)
        with np.errstate(divide='ignore', invalid='ignore'):
            chunks['residual_signed'].append(np.where(fs > 0, fd / fs, np.nan))
            chunks['fill_noise_ratio'].append(np.where(rs > 0, hp[:, inner].std(1) / rs, np.nan))
        chunks['snr'].append(fs / (noise * np.sqrt(ap.sum())))
        chunks['luminance_signed'].append(offset)
        chunks['colour_shift'].append(shift)
    stats = {k: np.concatenate(v) if v else np.zeros(0) for k, v in chunks.items()}
    keep = stats['snr'] >= (10 if dense else 20)
    return _bundle(stats, keep, ys, xs)


def _density(exclusion, region):
    """Same source density thresholds as m2 without its full int64 pixel-id map."""
    h, w = region.shape
    labels = np.full((-( -h // 64), -( -w // 64)), -1, np.int8)
    totals = {name: dict(pixels=0, blocks=0) for name in m2.STRATA}
    for by in range(labels.shape[0]):
        for bx in range(labels.shape[1]):
            sl = np.s_[by * 64:(by + 1) * 64, bx * 64:(bx + 1) * 64]
            r = region[sl]
            n = int(r.sum())
            if n < 1024:
                continue
            density = int((exclusion[sl] & r).sum()) * 1000 / n
            label = 0 if density < 3 else 1 if density < 10 else 2
            labels[by, bx] = label
            totals[m2.STRATA[label]]['pixels'] += n
            totals[m2.STRATA[label]]['blocks'] += 1
    return labels, totals


class _Histogram:
    def __init__(self):
        self.counts = np.zeros(len(_HIST_EDGES), np.int64)
        self.maximum = 0.
        self.total = 0

    def add(self, a):
        if not a.size:
            return
        a = np.asarray(a)
        self.maximum = max(self.maximum, float(a.max()))
        idx = np.searchsorted(_HIST_EDGES, a, side='left')
        idx = np.minimum(idx, len(_HIST_EDGES) - 1)
        self.counts += np.bincount(idx.ravel(), minlength=len(self.counts))
        self.total += a.size

    def report(self):
        out = dict(count=int(self.total), max=self.maximum if self.total else None,
                   quantiles='upper histogram bin: 1/64 sigma to 64, logarithmic above')
        cumulative = np.cumsum(self.counts)
        for name, q in (('median', .5), ('p10', .1), ('p90', .9), ('p95', .95)):
            out[name] = (min(self.maximum, float(_HIST_EDGES[np.searchsorted(cumulative, max(1, int(np.ceil(q * self.total))))]))
                         if self.total else None)
        return out


def _changes(source, estimate, LS, LD, sigma, mask):
    hist = {k: _Histogram() for k in ('luminance', 'rgb', 'chroma')}
    for y in range(0, source.shape[0], 128):
        sl = np.s_[y:y + 128]
        keep = mask[sl]
        if not keep.any():
            continue
        delta = (estimate[sl].astype(np.float32) - source[sl].astype(np.float32)) / 65535
        dl = LD[sl] - LS[sl]
        scale = sigma[sl][keep]
        hist['luminance'].add(abs(dl[keep]) / scale)
        hist['rgb'].add(np.max(abs(delta), axis=-1)[keep] / scale)
        hist['chroma'].add(np.max(abs(delta - dl[..., None]), axis=-1)[keep] / scale)
    return dict(pixels=int(mask.sum()), **{k: v.report() for k, v in hist.items()})


def _components(mask):
    """8-connected saturated cores as row runs; avoid a full integer label plane."""
    parents, runs = [], []

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    prev = []
    for y in range(mask.shape[0]):
        edges = np.flatnonzero(np.diff(np.r_[False, mask[y], False]))
        current = []
        cursor = 0
        for x0, x1 in edges.reshape(-1, 2):
            rid = len(parents)
            parents.append(rid)
            runs.append((y, int(x0), int(x1)))
            while cursor < len(prev) and prev[cursor][1] < x0:
                cursor += 1
            j = cursor
            while j < len(prev) and prev[j][0] <= x1:
                parents[root(rid)] = root(prev[j][2])
                j += 1
            current.append((int(x0), int(x1), rid))
        prev = current
    groups = {}
    for rid, run in enumerate(runs):
        groups.setdefault(root(rid), []).append(run)
    items = []
    for rs in groups.values():
        x0, x1 = min(r[1] for r in rs), max(r[2] for r in rs)
        y0, y1 = rs[0][0], rs[-1][0] + 1
        items.append(dict(runs=rs, core_pixels=sum(r[2] - r[1] for r in rs), bbox=[x0, y0, x1, y1]))
    return sorted(items, key=lambda a: (-a['core_pixels'], a['bbox'][1], a['bbox'][0]))


def _saturated(sat, region, LD, sigma, estimate, hpD):
    components = _components(sat & region)
    excluded_ring = m1.dilate(sat, 6)
    records, skipped = [], []
    h, w = region.shape
    for index, item in enumerate(components):
        x0, y0, x1, y1 = item['bbox']
        record = {k: item[k] for k in ('bbox', 'core_pixels')}
        if index >= MAX_COMPONENTS or max(x1 - x0, y1 - y0) + 24 > MAX_COMPONENT_SIDE:
            record['reason'] = 'component_limit' if index >= MAX_COMPONENTS else 'extent_limit'
            skipped.append(record)
            continue
        left, top, right, bottom = max(0, x0 - 12), max(0, y0 - 12), min(w, x1 + 12), min(h, y1 + 12)
        sl = np.s_[top:bottom, left:right]
        core = np.zeros((bottom - top, right - left), bool)
        for y, a, b in item['runs']:
            core[y - top, a - left:b - left] = True
        footprint = m1.dilate(core, 4)
        inner = m1.erode(footprint, 2) & region[sl]
        ring = m1.dilate(core, 12) & ~excluded_ring[sl] & region[sl]
        if inner.sum() < 8 or ring.sum() < 8:
            record['reason'] = 'insufficient_support'
            skipped.append(record)
            continue
        s = float(np.median(sigma[sl][ring]))
        ld = LD[sl]
        offset = float((ld[inner].mean() - np.median(ld[ring])) / s)
        colour = 0.
        for c in range(3):
            cc = estimate[sl][..., c].astype(np.float32) / 65535 - ld
            colour = max(colour, float(abs(cc[inner].mean() - cc[ring].mean()) / s))
        record.update(inner_pixels=int(inner.sum()), ring_pixels=int(ring.sum()),
                      luminance_signed=offset, luminance_absolute=abs(offset),
                      luminance_positive=max(offset, 0), luminance_negative=max(-offset, 0), colour_shift=colour)
        records.append(record)
    all_core = sum(item['core_pixels'] for item in components)
    covered = sum(item['core_pixels'] for item in records)
    keys = ('luminance_signed', 'luminance_absolute', 'luminance_positive', 'luminance_negative', 'colour_shift')
    identity = np.array([r['bbox'] + [r['core_pixels']] for r in records], np.int64).reshape(-1, 5)
    return dict(components=len(components), measured=len(records), core_pixels=all_core,
                covered_core_pixels=covered, core_coverage=covered / all_core if all_core else None,
                objects=records, skipped=skipped, sample_sha256=p.array_identity(identity),
                **{key: summary([r[key] for r in records]) for key in keys})


def evaluate3(source, estimate, region, *, truth=None, base=None, base_estimate=None, pairing=None):
    _validate(source, estimate, region)
    if truth is not None:
        validate_truth(source, base, region, truth)
        if truth['base_kind'] == 'paired-response':
            validate_pairing(source, estimate, base, base_estimate, region, pairing)
    det = m1.detect_stars(source, region)
    LS, sigma, bgS = det['L'], det['sigma'], det['background']
    LD = m1.luminance16(estimate)
    hpD = m1.highpass3(LD)
    ys, xs = det['tables']['measure']
    isolated = _stars(ys, xs, LS, LD, sigma, estimate, hpD)
    labels, totals = _density(det['exclusion'], region)
    ys, xs, _ = m2.dense_candidates(LS, sigma, bgS, region, source)
    del det['hp']
    bgD = m1.block_median(LD, 64)
    star_labels = labels[ys // 64, xs // 64]
    strata = {}
    for i, name in enumerate(m2.STRATA):
        select = np.flatnonzero(star_labels == i)
        if len(select) > 20000:
            select = select[::int(np.ceil(len(select) / 20000))]
        strata[name] = dict(**totals[name], stars=_stars(ys[select], xs[select], LS, LD, sigma,
                                                      estimate, hpD, dense=True, bgS=bgS, bgD=bgD))
    del bgD, bgS
    sat = _saturated(det['saturated_pixels'], region, LD, sigma, estimate, hpD)
    del hpD
    footprints = m1.dilate(det['exclusion'], 8) | m1.dilate(det['saturated_pixels'], 16)
    nonstar = _changes(source, estimate, LS, LD, sigma, region & ~footprints)
    out = dict(version=VERSION, shape=list(source.shape), region_pixels=int(region.sum()),
               source_array_sha256=p.array_identity(source), region_array_sha256=p.array_identity(region),
               selection='source-only detection, noise, density, SNR and sampling',
               basis=dict(isolated=m1.VERSION, density=m2.VERSION, legacy_reports_recomputed=False),
               guards=dict(isolated=isolated, dense_strata=strata, nonstar=nonstar, saturated_regions=sat))
    if truth is not None:
        out['injected'] = injection_metrics(source, estimate, region, base, truth, base_estimate=base_estimate, pairing=pairing)
    return out


def inject_stars3(rgb, rng, *, count, region=None, base_kind, saturated=0,
                  sigma_range=(.7, 1.4), amplitude_range=(.02, .6), min_separation=30, colour_jitter=.1):
    m1._check_rgb(rgb, 'base')
    if base_kind not in ('known-background', 'paired-response'):
        raise ValueError('base_kind must explicitly be known-background or paired-response')
    if region is None:
        region = np.ones(rgb.shape[:2], bool)
    _validate(rgb, rgb, region)
    p.integer_parameter(count, 'count', 0, 100000)
    p.integer_parameter(saturated, 'saturated', 0, 100000)
    if not (0 < sigma_range[0] <= sigma_range[1] <= 4 and 0 < amplitude_range[0] <= amplitude_range[1]):
        raise ValueError('invalid sigma/amplitude range')
    if not np.isfinite([*sigma_range, *amplitude_range, min_separation, colour_jitter]).all() or not 0 <= colour_jitter < 1:
        raise ValueError('injection parameters must be finite with 0 <= colour_jitter < 1')
    max_radius = int(np.ceil(4 * sigma_range[1])) + 3
    if min_separation < 2 * np.sqrt(2) * max_radius + 1:
        raise ValueError('metrics-3 requires non-overlapping supports; increase min_separation')
    h, w = region.shape
    margin = max(17, max_radius + 1)
    allowed = m1.erode(region, margin)
    allowed[:margin] = allowed[-margin:] = False
    allowed[:, :margin] = allowed[:, -margin:] = False
    if count + saturated and not allowed.any():
        raise ValueError('no valid injection support in region')
    image = rgb.astype(np.float32) / 65535
    positions, stars = [], []
    attempts = 0
    while len(positions) < count + saturated and attempts < max(1, (count + saturated) * 2000):
        attempts += 1
        y, x = rng.uniform(margin, h - margin), rng.uniform(margin, w - margin)
        iy, ix = int(round(y)), int(round(x))
        if not allowed[iy, ix] or any((y - yy) ** 2 + (x - xx) ** 2 < min_separation ** 2 for yy, xx in positions):
            continue
        positions.append((y, x))
    if len(positions) != count + saturated:
        raise ValueError('cannot place requested count in valid region')
    for index, (y, x) in enumerate(positions):
        requested_sat = index >= count
        s = float(rng.uniform(*sigma_range))
        amp = 3. if requested_sat else float(np.exp(rng.uniform(*np.log(amplitude_range))))
        gains = 1 + rng.uniform(-colour_jitter, colour_jitter, 3)
        gains /= float(gains @ m1._LUM)
        r = int(np.ceil(4 * s)) + 3
        iy, ix = int(round(y)), int(round(x))
        yy, xx = np.mgrid[iy - r:iy + r + 1, ix - r:ix + r + 1]
        sl = np.s_[iy - r:iy + r + 1, ix - r:ix + r + 1]
        added = amp * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * s * s))
        image[sl] += added[..., None].astype(np.float32) * gains.astype(np.float32)
        clipped = int((image[sl] > 1).any(-1).sum())
        stars.append(dict(y=y, x=x, sigma=s, amplitude=amp, gains=gains.tolist(),
                          support=[ix - r, iy - r, ix + r + 1, iy + r + 1],
                          requested_saturated=requested_sat, saturated=bool(clipped), clipped_pixels=clipped,
                          intrinsic_flux_luminance=float(2 * np.pi * s * s * amp)))
    output = np.rint(np.clip(image, 0, 1) * 65535).astype(np.uint16)
    for star in stars:
        x0, y0, x1, y1 = star['support']
        delta = (output[y0:y1, x0:x1].astype(np.float64) - rgb[y0:y1, x0:x1].astype(np.float64)) / 65535
        star['actual_flux_luminance'] = float((delta @ m1._LUM.astype(np.float64)).sum())
        star['saturated'] = bool((output[y0:y1, x0:x1] == 65535).any())
    truth = dict(version=TRUTH_VERSION, base_kind=base_kind, shape=list(rgb.shape), stars=stars,
                 base_array_sha256=p.array_identity(rgb), injected_array_sha256=p.array_identity(output),
                 region_array_sha256=p.array_identity(region), count=len(stars),
                 saturated=sum(s['saturated'] for s in stars), image_model='display-domain sampled Gaussian, quantized and clipped')
    return output, truth


def validate_truth(source, base, region, truth):
    if truth.get('version') != TRUTH_VERSION or base is None:
        raise ValueError('metrics-3 needs v3 truth and explicit base')
    _validate(source, base, region)
    expected = (('base_array_sha256', base), ('injected_array_sha256', source), ('region_array_sha256', region))
    if list(source.shape) != truth.get('shape'):
        raise ValueError('truth shape mismatch (cropped or different source)')
    for key, array in expected:
        if truth.get(key) != p.array_identity(array):
            raise ValueError(f'truth identity mismatch: {key}')
    if truth.get('base_kind') not in ('known-background', 'paired-response'):
        raise ValueError('truth has no valid base_kind')


def validate_pairing(source, estimate, base, base_estimate, region, pairing):
    """Require explicit inference provenance; shapes alone never certify a pair.

    CLI builds this only after checking two successful separation receipts and
    their input/output file hashes and stable backend/mask identity. In-memory
    callers must provide equivalent provenance from their controlled runner.
    """
    if base_estimate is None:
        raise ValueError('paired-response requires base_estimate')
    _validate(source, base_estimate, region)
    if not isinstance(pairing, dict) or pairing.get('version') != 'starless-pairing-3' or not pairing.get('backend'):
        raise ValueError('paired-response requires verified pairing provenance')
    checks = {'source_array_sha256': source, 'estimate_array_sha256': estimate,
              'base_array_sha256': base, 'base_estimate_array_sha256': base_estimate,
              'region_array_sha256': region}
    for key, array in checks.items():
        if pairing.get(key) != p.array_identity(array):
            raise ValueError('pairing identity mismatch: ' + key)


def injection_metrics(source, estimate, region, base, truth, *, base_estimate=None, pairing=None):
    validate_truth(source, base, region, truth)
    if truth['base_kind'] == 'paired-response':
        validate_pairing(source, estimate, base, base_estimate, region, pairing)
        target = base_estimate
    else:
        if base_estimate is not None:
            raise ValueError('base_estimate is only for paired-response')
        target = base
    groups = {'unsaturated': [], 'saturated': []}
    for star in truth['stars']:
        x0, y0, x1, y1 = star['support']
        if not (0 <= x0 < x1 <= source.shape[1] and 0 <= y0 < y1 <= source.shape[0]):
            raise ValueError('truth support out of bounds')
        sl = np.s_[y0:y1, x0:x1]
        if not region[sl].all():
            raise ValueError('truth support outside bound region')
        delta = (source[sl].astype(np.float64) - base[sl].astype(np.float64)) / 65535
        if bool(star['saturated']) != bool((source[sl] == 65535).any()):
            raise ValueError('truth saturation label disagrees with observed endpoint')
        denom = float((delta @ m1._LUM.astype(np.float64)).sum())
        if not np.isclose(denom, star['actual_flux_luminance'], rtol=1e-10, atol=1e-10):
            raise ValueError('truth actual increment mismatch')
        error = (estimate[sl].astype(np.float64) - target[sl].astype(np.float64)) / 65535
        residual = float((error @ m1._LUM.astype(np.float64)).sum())
        groups['saturated' if star['saturated'] else 'unsaturated'].append(dict(
            signed=residual / denom if denom > 0 else np.nan, rmse=float(np.sqrt(np.mean(error * error)))))
    # Exact target-error aggregates over every region pixel, not just star supports.
    ss, absolute, n, maximum = 0., 0., 0, 0.
    for y in range(0, source.shape[0], 128):
        keep = region[y:y + 128]
        err = (estimate[y:y + 128].astype(np.float64) - target[y:y + 128].astype(np.float64))[keep] / 65535
        if err.size:
            ss += float(np.sum(err * err)); absolute += float(np.sum(abs(err))); n += err.size
            maximum = max(maximum, float(np.max(abs(err))))
    out = dict(kind=truth['base_kind'], target_array_sha256=p.array_identity(target),
               region_pixels=int(region.sum()), target_error=dict(rmse=np.sqrt(ss / n) if n else None,
               mae=absolute / n if n else None, max=maximum if n else None))
    if pairing is not None:
        out['pairing_sha256'] = m1.report_digest(pairing)
    for name, items in groups.items():
        signed = np.array([r['signed'] for r in items])
        out[name] = dict(count=len(items), residual_signed=summary(signed), residual_absolute=summary(abs(signed)),
                         residual_negative=summary(np.maximum(-signed, 0)), support_rgb_rmse=summary([r['rmse'] for r in items]))
    return out


def e1_guard(reference, candidate):
    """Per-case deterministic screen; final two-track/visual decision is external."""
    for key in ('version', 'source_array_sha256', 'region_array_sha256'):
        if reference.get(key) != candidate.get(key) or (key == 'version' and reference.get(key) != VERSION):
            raise ValueError('E1 reports must share v3, source and region')
    failures, missing, worst, legacy_signed = [], [], {}, []

    def check(name, b, c, metric, tolerance, field='p90'):
        bv, cv = b.get(metric, {}).get(field), c.get(metric, {}).get(field)
        if bv is None or cv is None:
            missing.append(name + '.' + metric)
        elif cv > bv + tolerance:
            failures.append(dict(group=name, metric=metric, reference=bv, candidate=cv, tolerance=tolerance))

    bg, cg = reference['guards'], candidate['guards']
    groups = [('isolated', bg['isolated'], cg['isolated'])]
    groups += [(name, bg['dense_strata'][name]['stars'], cg['dense_strata'][name]['stars']) for name in m2.STRATA]
    for name, b, c in groups:
        if b['sample_sha256'] != c['sample_sha256'] or b['count'] != c['count']:
            raise ValueError('E1 sample selection changed: ' + name)
        if b['count'] < 20:
            if name == 'isolated' or b['candidates']:
                missing.append(name + '.count<20')
            if not b['count']:
                continue
        old, new = b['residual_signed']['p90'], c['residual_signed']['p90']
        legacy_signed.append(dict(group=name, reference=old, candidate=new, tolerance=.05,
                                  passes=old is not None and new is not None and new <= old + .05))
        for metric, tolerance in (('residual_absolute', .02), ('residual_negative', .02), ('residual_positive', .05),
                                  ('luminance_absolute', .25), ('luminance_negative', .25), ('luminance_positive', .25), ('colour_shift', .10)):
            check(name, b, c, metric, tolerance)
        if c['negative_fraction'] > b['negative_fraction'] + .01:
            failures.append(dict(group=name, metric='negative_fraction', reference=b['negative_fraction'], candidate=c['negative_fraction'], tolerance=.01))
    for metric in ('luminance', 'rgb', 'chroma'):
        check('nonstar', bg['nonstar'], cg['nonstar'], metric, .25, 'p95')
    if bg['nonstar']['pixels'] != cg['nonstar']['pixels']:
        raise ValueError('E1 nonstar support changed')
    b, c = bg['saturated_regions'], cg['saturated_regions']
    if b['sample_sha256'] != c['sample_sha256'] or b['skipped'] != c['skipped']:
        raise ValueError('E1 saturation support changed')
    if b['components']:
        if b['core_coverage'] < .8:
            missing.append('saturation.core_coverage<0.8')
        for metric in ('colour_shift', 'luminance_absolute', 'luminance_negative', 'luminance_positive'):
            check('saturation', b, c, metric, .25)
            differences = [new[metric] - old[metric] for old, new in zip(b['objects'], c['objects'])]
            if differences:
                idx = int(np.argmax(differences))
                worst[metric] = dict(delta=float(differences[idx]), bbox=b['objects'][idx]['bbox'], tolerance=.50)
                if differences[idx] > .50:
                    failures.append(dict(group='saturation_worst', metric=metric, reference=b['objects'][idx][metric],
                                         candidate=c['objects'][idx][metric], tolerance=.50, bbox=b['objects'][idx]['bbox']))
    b, c = bg['isolated']['colour_shift']['p90'], cg['isolated']['colour_shift']['p90']
    effect = b is not None and c is not None and b - c >= .10 and c <= .8 * b
    return dict(version=E1_VERSION, status='reject' if failures else 'indeterminate' if missing else 'candidate' if effect else 'no_practical_effect',
                practical_colour_effect=bool(effect), failures=failures, missing=missing,
                saturation_worst=worst, legacy_signed_p90_gate=legacy_signed,
                status_meaning='conservative screen; rejection is not proof of inferior true background',
                automatic_default_switch=False)


report_digest = m1.report_digest
