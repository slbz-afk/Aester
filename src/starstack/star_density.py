"""Conservative display-scale compact-star density, not a stellar catalogue.

Inputs are area-averaged positive RGB residuals at a common display scale.
The heuristic recommends controls; it neither edits an image nor certifies
that a star-removal model assigned every object correctly.
"""
from __future__ import annotations
import math
import numpy as np
from . import photographic as photo

VERSION = 'compact-star-density-1'


def analyse_density(positive_rgb, valid, milkyway=None):
    if (type(positive_rgb) is not np.ndarray or positive_rgb.dtype != np.float32
            or positive_rgb.ndim != 3 or positive_rgb.shape[2] != 3
            or min(positive_rgb.shape[:2]) < 8 or max(positive_rgb.shape[:2]) > 2048
            or not np.isfinite(positive_rgb).all() or np.any(positive_rgb < 0)
            or np.any(positive_rgb > 1)):
        raise ValueError('positive RGB must be finite float32 display 0..1, 8..2048 pixels per side')
    shape = positive_rgb.shape[:2]
    if type(valid) is not np.ndarray or valid.dtype != np.bool_ or valid.shape != shape:
        raise ValueError('valid must be a same-shape bool array')
    if milkyway is not None and (type(milkyway) is not np.ndarray or milkyway.shape != shape
            or milkyway.dtype not in (np.float32, np.float64)
            or not np.isfinite(milkyway).all() or np.any(milkyway < 0) or np.any(milkyway > 1)):
        raise ValueError('Milky Way must be finite same-shape float weight 0..1')
    # Stay away from excluded cells. A missing/zero layer isn't evidence of sky.
    safe = photo.neighbour_extreme(valid.astype(np.float32), 4) == 1
    safe[:4] = False; safe[-4:] = False; safe[:, :4] = False; safe[:, -4:] = False
    peak = positive_rgb.max(axis=2)
    hp = peak - photo.box_mean(peak, 4)
    threshold = np.empty(shape, np.float32)
    # Robust local threshold adapts to the actual residual noise floor.
    for y in range(0, shape[0], 128):
        for x in range(0, shape[1], 128):
            sl = (slice(y,y+128), slice(x,x+128))
            a = hp[sl][safe[sl]]
            med = float(np.median(a)) if a.size else 0.
            mad = float(np.median(np.abs(a-med))) if a.size else 0.
            threshold[sl] = max(64/65535, med + 5.5 * 1.4826 * mad)
    maximum = photo.neighbour_extreme(hp, 2, maximum=True)
    candidate = safe & (hp == maximum) & (hp > threshold)
    # Two-channel response rejects one-channel hotspots. At least 3 nearby
    # signal pixels reject isolated impulses; long straight ridges lack a
    # unique peak across a 5x5 neighbourhood and are conservatively excluded.
    second = np.partition(positive_rgb, 1, axis=2)[..., 1]
    candidate &= second >= .20 * peak
    support = photo.box_mean((hp > threshold*.35).astype(np.float32), 1)*9
    candidate &= support >= 2.9
    # Keep the first of equal compact peaks instead of rejecting a flat
    # 2x2 core (which is common after resampling or saturation).
    for dy, dx in [(0,1),(1,-1),(1,0),(1,1)]:
        other = np.roll(np.roll(hp, dy, axis=0), dx, axis=1)
        candidate &= hp > other
    # Reject elongated ridges by their local second moments, rather than
    # confusing every equal-brightness core with a trail.
    yy,xx=np.mgrid[-4:5,-4:5]
    for y,x in zip(*np.nonzero(candidate)):
        patch=np.maximum(hp[y-4:y+5,x-4:x+5],0).astype(np.float64)
        total=float(patch.sum())
        cx=float((patch*xx).sum()/total); cy=float((patch*yy).sum()/total)
        vx=float((patch*(xx-cx)**2).sum()/total)
        vy=float((patch*(yy-cy)**2).sum()/total)
        xy=float((patch*(xx-cx)*(yy-cy)).sum()/total)
        gap=math.sqrt((vx-vy)**2+4*xy*xy)
        minor=max(0.,(vx+vy-gap)/2); major=(vx+vy+gap)/2
        if minor < .10 or major > 5*max(minor,1e-12): candidate[y,x]=False
    background = safe if milkyway is None else safe & (milkyway <= .10)
    galaxy = np.zeros(shape, bool) if milkyway is None else safe & (milkyway >= .80)
    def summarize(region):
        area = int(region.sum()); count = int((candidate & region).sum())
        # Density uses a canonical long side of 2048 even for smaller inputs.
        reference_area=area*(2048/max(shape))**2
        # Numerous barely visible residual specks must not dominate advice.
        # A 0.10 display-code prominence contributes one visible-star unit;
        # lower prominences contribute quadratically. No physical flux claim.
        visible=float(np.minimum((hp[candidate & region]/.10)**2,1).sum(dtype=np.float64))
        return {'area_pixels':area, 'candidate_stars':count,
                'reference_area_pixels':reference_area,
                'density_per_megapixel':float(count * 1e6 / reference_area) if area else None,
                'visible_star_units':visible,
                'visible_density_per_megapixel':float(visible*1e6/reference_area) if area else None,
                'signal_coverage':float(np.mean((hp > threshold)[region])) if area else None,
                'median_threshold_code':float(np.median(threshold[region])*65535) if area else None}
    return {'version':VERSION, 'work_shape':list(shape), 'background':summarize(background),
            'milkyway':summarize(galaxy), 'all_sky':summarize(safe),
            'method':'2048-scale compact peaks; local MAD, colour, spatial and shape checks; squared display prominence weights',
            'physical_star_density':False}


def recommend_controls(metrics, *, galaxy_enabled=False):
    bg = metrics['background']; mw = metrics['milkyway']
    area = bg['area_pixels']; count = bg['candidate_stars']; density = bg['visible_density_per_megapixel']
    if area < 16384 or count < 8 or bg['visible_star_units'] < 8 or density is None:
        return {'available':False, 'background_strength':None, 'milkyway_offset':None,
                'milkyway_strength':None,
                'reason':'背景有效面积或可靠星点不足，暂不给自动力度；仍可手动调整。'}
    # Frozen aesthetic heuristic, deliberately capped below full removal.
    # Higher measured density monotonically increases the recommendation.
    strength = float(np.clip(.35 + .45*math.log1p(density/150)/math.log1p(5000/150), .35, .80))
    strength = round(strength, 2)
    offset = 0.
    detail = '未使用银河分区；相对偏移建议为0。'
    if galaxy_enabled:
        if mw['area_pixels'] >= 16384 and mw['candidate_stars'] >= 8 and mw['visible_star_units'] >= 8:
            # Retain up to 1.5x the background density to keep the Milky Way
            # populated. This is a first-order visual proxy, not an exact
            # inverse of a nonlinear curve nor a predicted star count.
            desired = 1 - 1.5*density*(1-strength)/mw['visible_density_per_megapixel']
            galaxy = float(np.clip(desired, max(0.,strength-.25), max(0.,strength-.05)))
            offset = round(galaxy-strength, 2)
            detail = '结合银河密度建议低5至25个百分点，以渐变选区过渡；不保证两区星数相等。'
        else:
            offset = -min(.15, strength)
            detail = '银河密度样本不足，保守建议比背景低15个百分点。'
    return {'available':True, 'background_strength':strength, 'milkyway_offset':offset,
            'milkyway_strength':round(float(np.clip(strength+offset,0,1)),2),
            'reason':f'背景检测到{count}个紧致星点候选，按显眼程度加权约{density:.0f}单位/百万参考像素；建议背景{strength:.0%}。'+detail,
            'automatic_range':[.35,.80], 'aesthetic_heuristic':True,
            'advice_only':True, 'predicted_physical_star_count':False}
