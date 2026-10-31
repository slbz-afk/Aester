"""Explicit background hypotheses and display-only coordination, TOP-DOWN.

This module does not characterize camera colour or distinguish light pollution
from real extended emission. Samples are a caller-selected background hypothesis.
"""
import math
import numpy as np
from PIL import Image

VERSION = 'background-plane-display-1'


def number(value, low, high, name):
    if type(value) not in (int, float) or not low <= value <= high or not math.isfinite(value):
        raise ValueError(f'{name} needs a finite number in [{low}, {high}]')
    return float(value)


def image_array(data):
    if (type(data) is not np.ndarray or data.dtype != np.float32 or data.ndim != 3
            or data.shape[2] != 3 or min(data.shape[:2]) < 2 or not np.isfinite(data).all()):
        raise ValueError('Expected finite TOP-DOWN float32 RGB, minimum 2x2')


def weight_array(weight, shape):
    if weight is not None and (not isinstance(weight, np.ndarray) or weight.shape != shape
            or weight.dtype.kind != 'f' or not np.isfinite(weight).all()
            or np.any(weight < 0) or np.any(weight > 1)):
        raise ValueError('Invalid TOP-DOWN sky weights')


def fit_background(data, mask, weight=None):
    """Fixed 16x16 cells; explicit pixels only, no adaptive colour rejection."""
    from .background_plane import fit_plane
    image_array(data)
    h, w = data.shape[:2]
    weight_array(weight, (h, w))
    if not isinstance(mask, np.ndarray) or mask.dtype != bool or mask.shape != (h, w):
        raise ValueError('Background mask must be a matching bool array')
    xy, rgb, cells = [], [], []
    ys, xs = np.linspace(0, h, 17, dtype=int), np.linspace(0, w, 17, dtype=int)
    for iy, (y0, y1) in enumerate(zip(ys[:-1], ys[1:])):
        for ix, (x0, x1) in enumerate(zip(xs[:-1], xs[1:])):
            # Bounded sampling cost; coordinates refer to actual selected pixels.
            dy, dx = max(1, math.ceil((y1-y0)/64)), max(1, math.ceil((x1-x0)/64))
            selected = mask[y0:y1:dy, x0:x1:dx].copy()
            if weight is not None:
                selected &= weight[y0:y1:dy, x0:x1:dx] == 1
            yy, xx = np.nonzero(selected)
            # Complete rectangular sampling windows are centrally symmetric:
            # median of a true plane is then evaluated at this coordinate.
            # Partial masks must not pair a median value with an unrelated centre.
            if len(xx) < 16 or not selected.all():
                continue
            xy.append([2*float(np.median(x0+xx*dx))/(w-1)-1,
                       2*float(np.median(y0+yy*dy))/(h-1)-1])
            rgb.append(np.median(data[y0:y1:dy, x0:x1:dx][selected], axis=0).tolist())
            cells.append({'grid_xy': [ix, iy], 'pixels_sampled': len(xx)})
    if len(xy) < 12:
        raise ValueError('Need at least 12 complete background cells, each with 16 fully-sky samples')
    points = np.asarray(xy, dtype=np.float64)
    if np.any(np.ptp(points, axis=0) < .5):
        raise ValueError('Background samples cover too little of the image')
    model = fit_plane(points, np.asarray(rgb, dtype=np.float64))
    return {'version': VERSION, 'shape_hw': [h, w], 'fit': model,
            'samples_xy': xy, 'samples_rgb': rgb, 'cells': cells,
            'sampling': '16x16 grid; complete rectangular windows only; up to 64x64 strided pixels per cell; channel medians',
            'meaning': 'explicit background hypothesis; not proof of unwanted light'}


def apply_background(data, model, *, mode='both', strength=1., weight=None):
    """C' = C - strength * actual_sky_weight * correction, no clipping."""
    image_array(data)
    h, w = data.shape[:2]
    weight_array(weight, (h, w))
    strength = number(strength, 0, 1, 'strength')
    if mode not in ('gradient', 'neutralize', 'both'):
        raise ValueError('Unknown background mode')
    if model.get('version') != VERSION or model.get('shape_hw') != [h, w]:
        raise ValueError('Background model shape/version differs')
    coefficients = np.asarray(model['fit']['coefficients'], dtype=np.float64)
    if coefficients.shape != (3, 3) or not np.isfinite(coefficients).all():
        raise ValueError('Invalid background coefficients')
    out = data.copy()
    if strength == 0:
        return out
    x = np.linspace(-1, 1, w, dtype=np.float64)
    y = np.linspace(-1, 1, h, dtype=np.float64)
    try:
        with np.errstate(over='raise', invalid='raise'):
            delta = coefficients[:, 0]-coefficients[0, 0]
            offsets = delta-delta.mean() if mode != 'gradient' else np.zeros(3)
            for y0 in range(0, h, 128):
                y1 = min(h, y0+128)
                change = np.broadcast_to(offsets, (y1-y0, w, 3)).copy()
                if mode != 'neutralize':
                    change += x[None, :, None]*coefficients[:, 1] + y[y0:y1, None, None]*coefficients[:, 2]
                change *= strength
                if weight is not None:
                    change *= weight[y0:y1, :, None]
                out[y0:y1] = (data[y0:y1].astype(np.float64)-change).astype(np.float32)
                unchanged = change == 0
                out[y0:y1][unchanged] = data[y0:y1][unchanged]
                if weight is not None:
                    # Also preserve signed zeros at untouched ground pixels.
                    ground = weight[y0:y1] == 0
                    out[y0:y1][ground] = data[y0:y1][ground]
    except FloatingPointError as exc:
        raise ValueError('Background derivative exceeds float32; original is unchanged') from exc
    return out


def display_parameters(data):
    image_array(data)
    sample = data[::max(1, math.ceil(data.shape[0]/512)), ::max(1, math.ceil(data.shape[1]/512))]
    black = float(np.quantile(sample, .003))
    white = float(np.quantile(sample, .998))
    return {'version': VERSION, 'black': black, 'scale': max(white-black, 1e-6),
            'gains_rgb': [1., 1., 1.], 'asinh_strength': 12.,
            'method': 'maxRGB_asinh_exponential_shoulder_then_sRGB_encoding',
            'color_calibrated': False}


def render_display(data, recipe, *, max_side=1600):
    """Average signed input before display clipping; preserve positive ratios."""
    image_array(data)
    if type(max_side) is not int or not 2 <= max_side <= 4096:
        raise ValueError('Invalid display dimensions')
    if recipe.get('version') != VERSION or recipe.get('method') != 'maxRGB_asinh_exponential_shoulder_then_sRGB_encoding' or recipe.get('color_calibrated') is not False:
        raise ValueError('Unknown display recipe')
    black = number(recipe['black'], -1e30, 1e30, 'black')
    scale = number(recipe['scale'], 1e-30, 1e30, 'scale')
    k = number(recipe['asinh_strength'], 1, 100, 'asinh_strength')
    gains = recipe['gains_rgb']
    if type(gains) is not list or len(gains) != 3:
        raise ValueError('Expected three display gains')
    gains = np.array([number(v, .25, 4, 'display gain') for v in gains])
    h, w = data.shape[:2]
    factor = min(1., max_side/max(h, w))
    size = (max(2, round(w*factor)), max(2, round(h*factor)))
    small = np.stack([np.asarray(Image.fromarray(data[..., c]).resize(size, Image.Resampling.BOX)) for c in range(3)], axis=2).astype(np.float64)
    x = np.maximum((small-black)/scale, 0)*gains
    peak = x.max(axis=2)
    value = -np.expm1(-np.arcsinh(k*peak)/np.arcsinh(k))
    ratio = np.divide(value, peak, out=np.zeros_like(peak), where=peak > 0)
    linear = x*ratio[..., None]
    encoded = np.where(linear <= .0031308, 12.92*linear, 1.055*linear**(1/2.4)-.055)
    return np.round(np.clip(encoded, 0, 1)*255).astype(np.uint8)
