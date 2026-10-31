"""Display-domain starless evaluation metrics (starless package step 1).

NumPy only (the Windows lock has no scipy). Inputs are uint16 H×W×3 display-sRGB
arrays — the source S and a starless estimate D — plus a bool evaluation region.
The numbers are the numeric shadows of the six defect classes listed in
docs/STARLESS_PACKAGE_20260923.md §三; they are frozen as baselines, never used as
quality certification on their own, and never replace looking at the picture.
"""
from __future__ import annotations
import hashlib
import json
import numpy as np
from . import photographic as p

VERSION = "starless-metrics-1"
DEFECTS = ("残核", "暗坑", "彩环", "纹理编造", "噪声断层", "星云被吃")
SAT_CODE = 64224  # 0.98 × 65535: any channel at or above this is treated as saturated
PARAMS = {
    "block": 64,            # background / noise block size
    "detect_k": 6.0,        # measurement candidates: hp ≥ detect_k × σ
    "exclude_k": 4.0,       # exclusion / isolation candidates: hp ≥ exclude_k × σ
    "isolation": 12,        # no other exclusion candidate within this radius
    "moment_min": 0.4, "moment_max": 2.5,   # second moment of hp in r ≤ 3 (pixels)
    "aperture": 5, "annulus": (8, 12), "inner": 3,
    "sat_aperture": 8, "sat_annulus": (12, 16), "sat_inner": 6,
    "margin": 17,           # distance from region / image border
    "max_stars": 5000, "snr_min": 20.0,
    "star_exclude": 8, "sat_exclude": 16,       # dilations for the non-star region
    "sat_footprint": 4, "sat_ring": (6, 12),    # saturation footprint statistics
    "bands": ((0, 2), (2, 8), (8, 32)),         # (r1, r2): box(r1) − box(r2)
}
BAND_NAMES = ("fine", "medium", "broad")
_LUM = np.array([.2126, .7152, .0722], np.float32)
_MAD_TO_SIGMA = 1.4826 / 0.943   # MAD → σ, then undo the 3×3 high-pass variance factor 8/9


def _check_rgb(a, name):
    if not isinstance(a, np.ndarray) or a.dtype != np.uint16 or a.ndim != 3 or a.shape[-1] != 3 or min(a.shape[:2]) < 64:
        raise ValueError(name + ' must be uint16 H,W,3 at least 64x64')


def luminance16(rgb):
    """Encoded-domain brightness proxy in [0, 1], computed in row blocks."""
    out = np.empty(rgb.shape[:2], np.float32)
    for y in range(0, rgb.shape[0], 256):
        out[y:y + 256] = p.luminance(rgb[y:y + 256].astype(np.float32) / np.float32(65535))
    return out


def _upsample(small, block, shape):
    """Bilinear upsampling of a block grid (PIL, deterministic), cropped to the image."""
    ny, nx = small.shape
    full = p.resize_plane(np.ascontiguousarray(small, np.float32), (nx * block, ny * block))
    return np.ascontiguousarray(full[:shape[0], :shape[1]])


def block_median(a, block):
    """Median of each block×block tile (ragged edges are their own tiles), bilinearly upsampled."""
    h, w = a.shape
    ny, nx = -(-h // block), -(-w // block)
    small = np.empty((ny, nx), np.float32)
    for by in range(ny):
        for bx in range(nx):
            small[by, bx] = np.median(a[by * block:(by + 1) * block, bx * block:(bx + 1) * block])
    return _upsample(small, block, (h, w))


def highpass3(L):
    return L - p.box_mean(L, 1)


def noise_sigma(L, block):
    """Local robust σ from the 3×3 high-pass residual (MAD per block, nearest-upsampled)."""
    hp = highpass3(L)
    h, w = L.shape
    ny, nx = -(-h // block), -(-w // block)
    small = np.empty((ny, nx), np.float32)
    for by in range(ny):
        for bx in range(nx):
            tile = hp[by * block:(by + 1) * block, bx * block:(bx + 1) * block]
            small[by, bx] = np.median(np.abs(tile - np.median(tile)))
    small = np.maximum(small * np.float32(_MAD_TO_SIGMA), np.float32(1 / 65535))
    return np.maximum(_upsample(small, block, (h, w)), np.float32(1 / 65535))


def dilate(mask, radius):
    if radius <= 0:
        return mask.copy()
    return p.neighbour_extreme(mask.astype(np.uint8), radius, maximum=True).astype(bool)


def erode(mask, radius):
    if radius <= 0:
        return mask.copy()
    return p.neighbour_extreme(mask.astype(np.uint8), radius, maximum=False).astype(bool)


def local_maxima(hp):
    """3×3 local maxima; plateaus are broken strictly towards the right / bottom neighbours."""
    a = np.pad(hp, 1, mode='constant', constant_values=-np.inf)
    c = a[1:-1, 1:-1]
    strict = (c > a[1:-1, 2:]) & (c > a[2:, 1:-1]) & (c > a[2:, :-2]) & (c > a[2:, 2:])
    loose = (c >= a[1:-1, :-2]) & (c >= a[:-2, 1:-1]) & (c >= a[:-2, :-2]) & (c >= a[:-2, 2:])
    return strict & loose


def _offsets(radius):
    d = np.arange(-radius, radius + 1)
    dy, dx = np.meshgrid(d, d, indexing='ij')
    return dy, dx, np.sqrt(dy * dy + dx * dx, dtype=np.float32)


def _gather(plane, ys, xs, radius):
    dy, dx, _ = _offsets(radius)
    return plane[ys[:, None, None] + dy[None], xs[:, None, None] + dx[None]]


def detect_stars(source, region, *, params=PARAMS):
    """Isolated star candidates on the source only (both images share them).

    Returns a dict with the luminance, σ map, exclusion masks and candidate
    tables. Detection is a fixed recipe, not a catalogue match; a candidate is a
    compact local maximum, not a certified star.
    """
    _check_rgb(source, 'source')
    h, w = source.shape[:2]
    if not isinstance(region, np.ndarray) or region.shape != (h, w) or region.dtype != np.bool_:
        raise ValueError('region must be a same-size bool array')
    L = luminance16(source)
    sigma = noise_sigma(L, params['block'])
    background = block_median(L, params['block'])
    hp = L - background
    peaks = local_maxima(hp)
    exclusion = peaks & (hp >= params['exclude_k'] * sigma)
    margin = params['margin']
    inside = erode(region, margin)
    inside[:margin] = inside[-margin:] = False
    inside[:, :margin] = inside[:, -margin:] = False
    candidates = exclusion & (hp >= params['detect_k'] * sigma) & inside
    iso = params['isolation']
    neighbours = np.rint(p.box_mean(exclusion.astype(np.float32), iso) * (2 * iso + 1) ** 2)
    isolated = candidates & (neighbours <= 1)
    ys, xs = np.nonzero(isolated)
    maxchannel = source.max(-1)
    satmax = p.neighbour_extreme(maxchannel, 1, maximum=True)
    saturated_flag = satmax[ys, xs] >= SAT_CODE
    dy, dx, _ = _offsets(3)
    patch = np.clip(_gather(hp, ys, xs, 3), 0, None)
    total = patch.sum((1, 2)) + np.float32(1e-12)
    mx = (patch * dx[None]).sum((1, 2)) / total
    my = (patch * dy[None]).sum((1, 2)) / total
    vx = (patch * (dx[None] - mx[:, None, None]) ** 2).sum((1, 2)) / total
    vy = (patch * (dy[None] - my[:, None, None]) ** 2).sum((1, 2)) / total
    moment = np.sqrt((vx + vy) / 2)
    compact = (moment >= params['moment_min']) & (moment <= params['moment_max'])
    saturated_pixels = maxchannel >= SAT_CODE
    tables = {'measure': _subsample(ys, xs, (~saturated_flag) & compact, params['max_stars']),
              'saturated': _subsample(ys, xs, saturated_flag, params['max_stars'])}
    return {'L': L, 'sigma': sigma, 'hp': hp, 'background': background, 'exclusion': exclusion, 'saturated_pixels': saturated_pixels,
            'candidates': {'peaks': int(peaks.sum()), 'exclusion': int(exclusion.sum()), 'inside': int(candidates.sum()),
                           'isolated': int(len(ys)), 'measure': int(tables['measure'][0].size),
                           'saturated': int(tables['saturated'][0].size), 'compact_rejected': int((~compact & ~saturated_flag).sum())},
            'tables': tables, 'moment': moment}


def _subsample(ys, xs, keep, limit):
    ys, xs = ys[keep], xs[keep]
    if ys.size > limit:
        step = -(-ys.size // limit)
        ys, xs = ys[::step], xs[::step]
    return ys, xs


def _summary(values):
    values = np.asarray(values, np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {'count': 0, 'median': None, 'p10': None, 'p90': None}
    return {'count': int(values.size), 'median': float(np.median(values)),
            'p10': float(np.percentile(values, 10)), 'p90': float(np.percentile(values, 90))}


def star_statistics(ys, xs, planes, sigma, *, aperture, annulus, inner):
    """Per-star aperture photometry and fill statistics on gathered windows."""
    radius = annulus[1]
    _, _, dist = _offsets(radius)
    ap = dist <= aperture
    ann = (dist >= annulus[0]) & (dist <= annulus[1])
    inn = dist <= inner
    n = ys.size
    out = {}
    if n == 0:
        return {'residual_ratio': np.zeros(0), 'snr': np.zeros(0), 'fill_noise_ratio': np.zeros(0),
                'luminance_offset': np.zeros(0), 'colour_shift': np.zeros(0), 'flux_source': np.zeros(0), 'flux_estimate': np.zeros(0)}
    LS = _gather(planes['L_S'], ys, xs, radius)
    LD = _gather(planes['L_D'], ys, xs, radius)
    HD = _gather(planes['hp3_D'], ys, xs, radius)
    s = sigma[ys, xs].astype(np.float64)
    bgS = np.median(LS[:, ann], 1)
    bgD = np.median(LD[:, ann], 1)
    FS = (LS[:, ap] - bgS[:, None]).sum(1).astype(np.float64)
    FD = (LD[:, ap] - bgD[:, None]).sum(1).astype(np.float64)
    out['flux_source'], out['flux_estimate'] = FS, FD
    out['snr'] = FS / (s * np.sqrt(ap.sum()))
    with np.errstate(divide='ignore', invalid='ignore'):
        out['residual_ratio'] = np.where(FS > 0, FD / FS, np.nan)
        ring_std = HD[:, ann].std(1).astype(np.float64)
        out['fill_noise_ratio'] = np.where(ring_std > 0, HD[:, inn].std(1) / ring_std, np.nan)
    out['luminance_offset'] = (LD[:, inn].mean(1) - bgD) / s
    shift = np.zeros(n)
    for c in range(3):
        CD = _gather(planes['C_D'][c], ys, xs, radius)
        ci = (CD[:, inn] - LD[:, inn]).mean(1)
        cr = (CD[:, ann] - LD[:, ann]).mean(1)
        shift = np.maximum(shift, np.abs(ci - cr) / s)
    out['colour_shift'] = shift
    return out


def _channel_planes(rgb):
    return [rgb[..., c].astype(np.float32) / np.float32(65535) for c in range(3)]


def evaluate(source, estimate, region, *, truth=None, params=PARAMS):
    """All metrics of docs/STARLESS_PACKAGE_20260923.md §五 for one (S, D, region)."""
    _check_rgb(source, 'source')
    _check_rgb(estimate, 'estimate')
    if estimate.shape != source.shape:
        raise ValueError('estimate must match the source shape')
    det = detect_stars(source, region, params=params)
    L_S, sigma = det['L'], det['sigma']
    L_D = luminance16(estimate)
    planes = {'L_S': L_S, 'L_D': L_D, 'hp3_D': highpass3(L_D), 'C_D': _channel_planes(estimate)}
    report = {'version': VERSION, 'params': _json_params(params), 'shape': list(source.shape),
              'region_pixels': int(region.sum()), 'candidates': det['candidates']}
    ys, xs = det['tables']['measure']
    stars = star_statistics(ys, xs, planes, sigma, aperture=params['aperture'], annulus=params['annulus'], inner=params['inner'])
    keep = stars['snr'] >= params['snr_min']
    report['stars'] = {'measured': int(ys.size), 'snr_kept': int(keep.sum()),
                       'residual_ratio': _summary(stars['residual_ratio'][keep]),
                       'fill_noise_ratio': _summary(stars['fill_noise_ratio'][keep]),
                       'luminance_offset': _summary(stars['luminance_offset'][keep]),
                       'colour_shift': _summary(stars['colour_shift'][keep])}
    sy, sx = det['tables']['saturated']
    sat = star_statistics(sy, sx, planes, sigma, aperture=params['sat_aperture'], annulus=params['sat_annulus'], inner=params['sat_inner'])
    report['saturated_stars'] = {'measured': int(sy.size), 'residual_ratio': _summary(sat['residual_ratio']),
                                 'fill_noise_ratio': _summary(sat['fill_noise_ratio']),
                                 'luminance_offset': _summary(sat['luminance_offset']),
                                 'colour_shift': _summary(sat['colour_shift'])}
    footprints = dilate(det['exclusion'], params['star_exclude']) | dilate(det['saturated_pixels'], params['sat_exclude'])
    nonstar = region & ~footprints
    report['nonstar'] = _perturbation(L_S, L_D, sigma, nonstar)
    report['bands'] = _band_ratios(L_S, L_D, footprints, (det['background'], block_median(L_D, params['block'])), nonstar, params['bands'])
    del footprints
    report['saturation_footprint'] = _footprint(det['saturated_pixels'], region, L_S, L_D, planes, sigma, params)
    if truth is not None:
        report['injected'] = _truth_metrics(truth, planes, sigma, params)
    report['visual'] = {name: None for name in DEFECTS}
    return report


def _json_params(params):
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in params.items()}


def _perturbation(L_S, L_D, sigma, mask):
    n = int(mask.sum())
    if n == 0:
        return {'pixels': 0, 'p50': None, 'p95': None, 'p99': None, 'fraction_over_3sigma': None}
    d = np.abs(L_D[mask] - L_S[mask]) / sigma[mask]
    return {'pixels': n, 'p50': float(np.percentile(d, 50)), 'p95': float(np.percentile(d, 95)),
            'p99': float(np.percentile(d, 99)), 'fraction_over_3sigma': float((d > 3).mean())}


def _band_ratios(L_S, L_D, footprints, backgrounds, mask, bands):
    """Band energies outside star footprints. Each image's footprints are filled with its
    own smooth (block-median) background first, so removed stars do not leak into the
    ratio through the box filters and a missing nebula does not create false edges."""
    out = {}
    if not mask.any():
        return {name: None for name in BAND_NAMES}
    filled = []
    for L, background in zip((L_S, L_D), backgrounds):
        f = L.copy()
        f[footprints] = background[footprints]
        filled.append(f)
    for name, (r1, r2) in zip(BAND_NAMES, bands):
        energies = []
        for L in filled:
            band = (L if r1 == 0 else p.box_mean(L, r1)) - p.box_mean(L, r2)
            energies.append(float(np.mean(band[mask].astype(np.float64) ** 2)))
        out[name] = {'source': energies[0], 'estimate': energies[1],
                     'ratio': (energies[1] / energies[0]) if energies[0] > 0 else None}
    return out


def _footprint(sat, region, L_S, L_D, planes, sigma, params):
    if not sat.any():
        return {'pixels': 0}
    foot = dilate(sat, params['sat_footprint']) & region
    inner = erode(foot, 2) & region
    r0, r1 = params['sat_ring']
    ring = dilate(sat, r1) & ~dilate(sat, r0) & region
    out = {'pixels': int(foot.sum()), 'inner_pixels': int(inner.sum()), 'ring_pixels': int(ring.sum())}
    out['perturbation'] = _perturbation(L_S, L_D, sigma, foot)
    if inner.sum() < 8 or ring.sum() < 8:
        return out
    s = float(np.median(sigma[ring]))
    hp = planes['hp3_D']
    ring_std = float(hp[ring].std())
    out['fill_noise_ratio'] = (float(hp[inner].std()) / ring_std) if ring_std > 0 else None
    out['luminance_offset'] = (float(L_D[inner].mean()) - float(np.median(L_D[ring]))) / s
    shift = 0.
    for C in planes['C_D']:
        ci = float((C[inner] - L_D[inner]).mean())
        cr = float((C[ring] - L_D[ring]).mean())
        shift = max(shift, abs(ci - cr) / s)
    out['colour_shift'] = shift
    return out


def _truth_metrics(truth, planes, sigma, params):
    stars = [t for t in truth['stars'] if not t['saturated']]
    h, w = planes['L_S'].shape
    m = params['margin']
    stars = [t for t in stars if m <= t['y'] < h - m and m <= t['x'] < w - m]
    ys = np.array([int(round(t['y'])) for t in stars], np.int64)
    xs = np.array([int(round(t['x'])) for t in stars], np.int64)
    st = star_statistics(ys, xs, planes, sigma, aperture=params['aperture'], annulus=params['annulus'], inner=params['inner'])
    analytic = np.array([t['flux_luminance'] for t in stars], np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio_analytic = np.where(analytic > 0, st['flux_estimate'] / analytic, np.nan)
    return {'stars': int(ys.size), 'residual_ratio_measured': _summary(st['residual_ratio']),
            'residual_ratio_analytic': _summary(ratio_analytic), 'fill_noise_ratio': _summary(st['fill_noise_ratio']),
            'luminance_offset': _summary(st['luminance_offset']), 'colour_shift': _summary(st['colour_shift'])}


def inject_stars(rgb, rng, *, count, sigma_range=(0.7, 1.4), amplitude_range=(0.02, 0.6), saturated=0,
                 min_separation=30, margin=20, colour_jitter=0.1):
    """Add Gaussian stars of known flux to a display-domain image; returns (image, truth).

    Amplitudes are luminance peaks in [0, 1] units (log-uniform); saturated stars
    use a peak of 3.0 and clip. Positions have sub-pixel offsets and keep a
    minimum separation from each other (not from the image's own stars).
    """
    _check_rgb(rgb, 'rgb')
    h, w = rgb.shape[:2]
    p.integer_parameter(count, 'count', 0, 100000)
    p.integer_parameter(saturated, 'saturated', 0, 100000)
    image = rgb.astype(np.float32) / np.float32(65535)
    total = count + saturated
    py = np.empty(total)
    px = np.empty(total)
    placed = 0
    attempts = 0
    while placed < total and attempts < total * 200:
        attempts += 1
        y = rng.uniform(margin, h - margin)
        x = rng.uniform(margin, w - margin)
        if placed and np.min((py[:placed] - y) ** 2 + (px[:placed] - x) ** 2) < min_separation ** 2:
            continue
        py[placed], px[placed] = y, x
        placed += 1
    if placed < total:
        raise ValueError('cannot place the requested stars with this separation')
    placed = list(zip(py.tolist(), px.tolist()))
    truth = []
    for index, (y, x) in enumerate(placed):
        sat = index >= count
        s = float(rng.uniform(*sigma_range))
        amp = 3.0 if sat else float(np.exp(rng.uniform(np.log(amplitude_range[0]), np.log(amplitude_range[1]))))
        gains = 1 + rng.uniform(-colour_jitter, colour_jitter, 3)
        gains = gains / float(gains @ _LUM)          # luminance-neutral colour jitter
        r = int(np.ceil(4 * s)) + 3
        y0, x0 = int(round(y)), int(round(x))
        ys = np.arange(max(0, y0 - r), min(h, y0 + r + 1))
        xs = np.arange(max(0, x0 - r), min(w, x0 + r + 1))
        g = np.exp(-(((ys[:, None] - y) ** 2 + (xs[None, :] - x) ** 2) / (2 * s * s))).astype(np.float32)
        image[ys[0]:ys[-1] + 1, xs[0]:xs[-1] + 1] += amp * g[..., None] * gains.astype(np.float32)[None, None]
        truth.append({'y': y, 'x': x, 'sigma': s, 'amplitude': amp, 'gains': [float(v) for v in gains],
                      'flux_luminance': float(2 * np.pi * s * s * amp), 'saturated': bool(sat)})
    out = np.rint(np.clip(image, 0, 1) * 65535).astype(np.uint16)
    return out, {'version': VERSION, 'count': count, 'saturated': saturated, 'stars': truth}


def array_sha256(a):
    return p.array_identity(a)


def report_digest(report):
    """SHA-256 of the report without volatile fields (timestamps, paths)."""
    frozen = {k: v for k, v in report.items() if k not in ('timestamp', 'seconds', 'paths')}
    return hashlib.sha256(json.dumps(frozen, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()
