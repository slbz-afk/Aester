"""Per-exposure validity for additive trail layers; not raw saturation recovery.

The source validity is a conservative finite/nonzero RGB observation predicate.
Outside an intentional footprint the additive layer is known zero, not missing
sky. Inside it, a missing exposure must never borrow another frame's support.
"""
from __future__ import annotations

import numpy as np

from .resample import DEFAULT_KERNEL, _taps

VERSION = '1.0.0'


def attach_source_support(box, validity):
    data = np.asarray(box['data'])
    valid = np.asarray(validity)
    weight = np.asarray(box['weight'])
    if data.ndim != 3 or data.shape[0] != 3 or data.dtype.kind != 'f':
        raise ValueError('trail box requires floating RGB planes')
    if valid.dtype != np.dtype(bool) or valid.shape != data.shape[1:] or weight.shape != valid.shape:
        raise ValueError('trail validity and feather weights must match the RGB box')
    if not np.isfinite(weight).all() or np.any((weight < 0) | (weight > 1)):
        raise ValueError('invalid trail feather weights')
    footprint = weight > 0
    observed = valid & np.isfinite(data).all(axis=0)
    dependency = (~footprint) | observed
    result = dict(box)
    clean = np.array(data, copy=True)
    clean[:, ~dependency | ~footprint] = 0
    result['data'] = clean
    result['validity'] = dependency
    result['source_support'] = {
        'version': VERSION, 'basis': 'registered_rgb_finite_nonzero',
        'footprint_pixels': int(footprint.sum()),
        'supported_pixels': int(np.count_nonzero(footprint & observed)),
        'unsupported_pixels': int(np.count_nonzero(footprint & ~observed)),
        'completeness': 'conservative_registered_observations_not_raw_or_flux_completeness',
    }
    return result


def dependency_support(validity, points, *, origin, grid_shape, kernel=DEFAULT_KERNEL):
    """Match actual warp kernel taps, crop clamp, and reject real grid extension.

Local positions outside the box are not contributions. For interior positions,
clamped crop edges follow warp_image; global out-of-frame taps remain invalid.
Known zero feather margins are valid, so support is not confused with a mask
that selects the trail's shape. All actual nonzero (also negative) taps count.
"""
    valid = np.asarray(validity)
    pts = np.asarray(points, dtype=np.float64)
    if valid.dtype != np.dtype(bool) or valid.ndim != 2 or not valid.size:
        raise ValueError('nonempty boolean dependency map required')
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError('points must have shape (N,2)')
    if len(origin) != 2 or len(grid_shape) != 2:
        raise ValueError('origin and grid shape require two integers')
    if any(type(v) is not int for v in (*origin, *grid_shape)):
        raise ValueError('origin and grid dimensions must be integers')
    ox, oy = origin
    gh, gw = grid_shape
    h, w = valid.shape
    if min(ox, oy) < 0 or min(gh, gw) <= 0 or ox+w > gw or oy+h > gh:
        raise ValueError('trail box outside registered grid')
    radius, fn = _taps(kernel)
    good = np.isfinite(pts).all(axis=1)
    good &= ((pts[:, 0] >= -.5) & (pts[:, 0] < w-.5)
             & (pts[:, 1] >= -.5) & (pts[:, 1] < h-.5))
    out = np.zeros(len(pts), dtype=bool)
    take = np.flatnonzero(good)
    if not len(take):
        return out
    p = pts[take]
    base = np.floor(p).astype(np.int64)
    frac = p-base
    ok = np.ones(len(take), dtype=bool)
    offsets = np.arange(-radius+1, radius+1)
    wx = np.stack([fn(frac[:, 0]-k) for k in offsets], axis=1)
    wy = np.stack([fn(frac[:, 1]-k) for k in offsets], axis=1)
    for j, dy in enumerate(offsets):
        yy = base[:, 1]+dy
        for i, dx in enumerate(offsets):
            xx = base[:, 0]+dx
            used = wx[:, i]*wy[:, j] != 0
            within_grid = (xx+ox >= 0) & (xx+ox < gw) & (yy+oy >= 0) & (yy+oy < gh)
            observed = valid[np.clip(yy, 0, h-1), np.clip(xx, 0, w-1)]
            ok &= ~used | (within_grid & observed)
    out[take] = ok
    return out


def mask_warped_box(source_box, warped_box, model, grid_shape, *, kernel=DEFAULT_KERNEL, check=None):
    """Filter an already warped layer by its own exposure's complete support."""
    result = dict(warped_box)
    data = np.array(warped_box['data'], copy=True)
    result['data'] = data
    if 'validity' not in source_box:
        result['native_support'] = {'version': VERSION, 'status': 'unavailable',
                                    'reason': 'source_dependency_map_absent'}
        return result
    if data.ndim != 3 or data.shape[0] != 3:
        raise ValueError('warped layer must have three planes')
    h, w = data.shape[1:]
    supported = np.zeros((h, w), dtype=bool)
    x0, y0 = int(source_box['x0']), int(source_box['y0'])
    nx0, ny0 = int(result['x0']), int(result['y0'])
    before = np.any(data != 0, axis=0)
    for y in range(0, h, 64):
        if check is not None:
            check()
        for x in range(0, w, 256):
            yy, xx = np.mgrid[y:min(h,y+64), x:min(w,x+256)]
            native = np.column_stack([(xx+nx0).ravel(), (yy+ny0).ravel()]).astype(np.float64)
            global_grid = native if model is None else np.asarray(model.apply(native), dtype=np.float64)
            local = global_grid-np.array([x0, y0])
            if model is None:
                # Identity copy does not perform interpolation; no unnecessary
                # edge erosion from a hypothetical kernel in this branch.
                ix, iy = local[:, 0].astype(np.int64), local[:, 1].astype(np.int64)
                v = np.asarray(source_box['validity'])
                inside = (ix >= 0) & (iy >= 0) & (ix < v.shape[1]) & (iy < v.shape[0])
                ok = np.zeros(len(ix), bool)
                ok[inside] = v[iy[inside], ix[inside]]
            else:
                ok = dependency_support(source_box['validity'], local, origin=(x0,y0),
                                        grid_shape=grid_shape, kernel=kernel)
            supported[y:y+yy.shape[0], x:x+xx.shape[1]] = ok.reshape(yy.shape)
    supported &= np.isfinite(data).all(axis=0)
    data[:, ~supported] = 0
    result['validity'] = supported
    result['native_support'] = {
        'version': VERSION, 'status': 'applied',
        'basis': 'this_exposure_registered_rgb_then_actual_warp_dependencies',
        'kernel': kernel if model is not None else 'identity_copy',
        'nonzero_pixels_before': int(before.sum()),
        'nonzero_pixels_removed': int(np.count_nonzero(before & ~supported)),
        'nonzero_pixels_after': int(np.count_nonzero(np.any(data != 0, axis=0))),
        'does_not_certify': 'upstream_unrecorded_interpolation_or_raw_clipping_or_full_flux_recovery',
    }
    return result
