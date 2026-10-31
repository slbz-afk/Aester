"""Independent streaming comparison metrics for uint16 display-sRGB outputs."""

import math

import numpy as np

_VERSION = 'display-comparison-1'
_ALLOWED_DOMAIN = 'display-sRGB'
_MAX_TILE_PIXELS = 1_000_000
_GROUP_NAMES = ('all', 'allowed', 'outside')


def _new_accumulator():
    return {
        'pixel_count': 0,
        'changed_pixels': 0,
        'changed_channels': 0,
        'decreased_channels': 0,
        'increased_channels': 0,
        'new_zero_channels': 0,
        'new_white_channels': 0,
        'min_delta_code': None,
        'max_delta_code': None,
        'sum_delta_code': 0,
        'sum_sq_delta_code': 0,
        'luma_decreased_pixels': 0,
        'luma_increased_pixels': 0,
        'sum_luma_numerator': 0,
    }


def _accumulate(acc, delta, luma, new_zero, new_white, mask):
    if mask is None:
        n_pix = delta.shape[0] * delta.shape[1]
        if n_pix == 0:
            return
        channels = delta.reshape(-1, 3)
        luma_values = luma.reshape(-1)
        zero_values = new_zero.reshape(-1, 3)
        white_values = new_white.reshape(-1, 3)
    else:
        n_pix = int(np.count_nonzero(mask))
        if n_pix == 0:
            return
        channels = delta[mask]
        luma_values = luma[mask]
        zero_values = new_zero[mask]
        white_values = new_white[mask]

    acc['pixel_count'] += n_pix

    changed_pixels = np.any(channels != 0, axis=1)
    acc['changed_pixels'] += int(np.count_nonzero(changed_pixels))
    acc['changed_channels'] += int(np.count_nonzero(channels))
    acc['decreased_channels'] += int(np.count_nonzero(channels < 0))
    acc['increased_channels'] += int(np.count_nonzero(channels > 0))
    acc['new_zero_channels'] += int(np.count_nonzero(zero_values))
    acc['new_white_channels'] += int(np.count_nonzero(white_values))

    low = int(channels.min())
    high = int(channels.max())
    if acc['min_delta_code'] is None or low < acc['min_delta_code']:
        acc['min_delta_code'] = low
    if acc['max_delta_code'] is None or high > acc['max_delta_code']:
        acc['max_delta_code'] = high

    acc['sum_delta_code'] += int(channels.sum(dtype=np.int64))
    acc['sum_sq_delta_code'] += int(np.sum(channels * channels, dtype=np.int64))
    acc['luma_decreased_pixels'] += int(np.count_nonzero(luma_values < 0))
    acc['luma_increased_pixels'] += int(np.count_nonzero(luma_values > 0))
    acc['sum_luma_numerator'] += int(luma_values.sum(dtype=np.int64))


def _finish(acc):
    n_pix = acc['pixel_count']
    channel_count = n_pix * 3
    return {
        'pixel_count': int(n_pix),
        'changed_pixels': int(acc['changed_pixels']),
        'changed_channels': int(acc['changed_channels']),
        'decreased_channels': int(acc['decreased_channels']),
        'increased_channels': int(acc['increased_channels']),
        'new_zero_channels': int(acc['new_zero_channels']),
        'new_white_channels': int(acc['new_white_channels']),
        'min_delta_code': acc['min_delta_code'],
        'max_delta_code': acc['max_delta_code'],
        'mean_delta_code': (acc['sum_delta_code'] / channel_count) if channel_count else None,
        'rms_delta_code': math.sqrt(acc['sum_sq_delta_code'] / channel_count) if channel_count else None,
        'luma_decreased_pixels': int(acc['luma_decreased_pixels']),
        'luma_increased_pixels': int(acc['luma_increased_pixels']),
        'mean_luma_delta_code': (acc['sum_luma_numerator'] / (10000 * n_pix)) if n_pix else None,
    }


def _tile_ranges(height, width, rows):
    """Yield row-major tiles bounded to at most _MAX_TILE_PIXELS pixels."""
    if width <= _MAX_TILE_PIXELS:
        tile_rows = max(1, min(rows, _MAX_TILE_PIXELS // width))
        for y0 in range(0, height, tile_rows):
            yield y0, min(height, y0 + tile_rows), 0, width
    else:
        for y0 in range(height):
            for x0 in range(0, width, _MAX_TILE_PIXELS):
                yield y0, y0 + 1, x0, min(width, x0 + _MAX_TILE_PIXELS)


def _validate(source, candidate, allowed, domain, rows):
    if (not isinstance(source, np.ndarray) or source.dtype != np.dtype(np.uint16)
            or source.ndim != 3 or source.shape[2] != 3):
        raise ValueError('source must be a H,W,3 uint16 ndarray')
    if (not isinstance(candidate, np.ndarray) or candidate.dtype != np.dtype(np.uint16)
            or candidate.ndim != 3 or candidate.shape[2] != 3):
        raise ValueError('candidate must be a H,W,3 uint16 ndarray')
    if source.shape != candidate.shape:
        raise ValueError('source and candidate must have the same shape')

    height, width = source.shape[:2]
    if height < 1 or width < 1:
        raise ValueError('source and candidate must have positive height and width')

    if (not isinstance(allowed, np.ndarray) or allowed.dtype != np.dtype(bool)
            or allowed.shape != (height, width)):
        raise ValueError('allowed must be a bool ndarray with shape H,W')

    if type(domain) is not str or domain != _ALLOWED_DOMAIN:
        raise ValueError("domain must be exactly 'display-sRGB'")
    if type(rows) is not int or rows < 1 or rows > 512:
        raise ValueError('rows must be a builtin int in 1..512')

    return height, width


def compare_uint16(source, candidate, allowed, *, domain, rows=128) -> dict:
    """Compare two frozen uint16 display-sRGB arrays over the declared mask."""
    height, width = _validate(source, candidate, allowed, domain, rows)

    accumulators = {name: _new_accumulator() for name in _GROUP_NAMES}

    for y0, y1, x0, x1 in _tile_ranges(height, width, rows):
        source_tile = source[y0:y1, x0:x1]
        candidate_tile = candidate[y0:y1, x0:x1]

        delta = np.subtract(candidate_tile, source_tile, dtype=np.int64)
        luma = 2126 * delta[..., 0] + 7152 * delta[..., 1] + 722 * delta[..., 2]
        new_zero = (candidate_tile == 0) & (source_tile > 0)
        new_white = (candidate_tile == 65535) & (source_tile < 65535)
        allowed_tile = allowed[y0:y1, x0:x1]

        _accumulate(accumulators['all'], delta, luma, new_zero, new_white, None)
        _accumulate(accumulators['allowed'], delta, luma, new_zero, new_white, allowed_tile)
        _accumulate(accumulators['outside'], delta, luma, new_zero, new_white, ~allowed_tile)

    groups = {name: _finish(accumulators[name]) for name in _GROUP_NAMES}

    return {
        'version': _VERSION,
        'domain': domain,
        'shape': [height, width, 3],
        'groups': groups,
        'all_changes_within_allowed': bool(groups['outside']['changed_channels'] == 0),
    }
