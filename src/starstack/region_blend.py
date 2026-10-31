"""Conservative inward blending for explicitly drawn camera/sky regions.

The annotation remains a binary exclusion when registering the sky. Only the
actual final composition weight is softened, inside its already observed sky.
This cannot recover obscured sky or remove moving-cloud ghosts.
"""
from __future__ import annotations

import math
import numpy as np

from . import frame_pool, performance

RECIPE = 'canvas-inward-feather-1'
# P1c T-5x (d) (docs/EFFICIENCY_P1C_PREREG_20261005.md §3.5): the 256-row tiles (each read with its halo) on the
# frame pool, at most 8 workers; one tile's memory measured at W=8280 and width_px=128 (514 rows read), scaled by
# W x (256 + 2 (width_px + 1)) / 514 (inward_unit_bytes).
INWARD_WORKER_CAP = 8
INWARD_UNIT_BYTES = 120_000_000         # measured 72.7 MB (width_px=128; 181.0 MB at 512) on F48 size
INWARD_UNIT_WIDTH, INWARD_UNIT_ROWS = 8280, 514


def inward_unit_bytes(width, width_px):
    """unit_bytes of one inward_weight tile (P1c R10/R17), never below the measured value."""
    rows = 256 + 2 * (int(width_px) + 1)
    return max(INWARD_UNIT_BYTES, -(-INWARD_UNIT_BYTES * int(width) * rows // (INWARD_UNIT_WIDTH * INWARD_UNIT_ROWS)))


def inward_weight(weight_top_down: np.ndarray, *, width_px: int | None = None, check=None):
    """Return (weight, receipt), preserving every excluded or unsupported pixel.

    Capped Manhattan distance from existing exact-zero weights gives a smooth
    transition. A one-pixel inset also suppresses the interpolation rim. Image
    borders are not exclusions. Full-resolution work uses 256-row tiles with
    sufficient halo; the result matches a whole-array distance calculation.
    P1c T-5x (d): the tiles run on the frame pool, each writing only its own
    rows; the counts are folded in tile order and ``check`` is still called
    before every tile, so the result is identical for any worker count.
    """
    from .milkyway import _inward_feather
    if (not isinstance(weight_top_down, np.ndarray) or weight_top_down.ndim != 2
            or min(weight_top_down.shape) < 1 or weight_top_down.dtype != np.float32):
        raise ValueError('composition weight must be a nonempty float32 2D array')
    if width_px is None:
        width_px = max(2, min(128, round(min(weight_top_down.shape) * .012)))
    if isinstance(width_px, bool) or not isinstance(width_px, int) or not 1 <= width_px <= 512:
        raise ValueError('composition feather width must be an integer from 1 to 512')
    height = weight_top_down.shape[0]
    output = np.empty_like(weight_top_down)
    counts = [0, 0]                    # changed, mixed
    halo = width_px + 1

    def tile(y, unit):
        end = min(height, y + 256)
        y0, y1 = max(0, y - halo), min(height, end + halo)
        source = weight_top_down[y0:y1]
        if not np.isfinite(source).all() or np.any(source < 0) or np.any(source > 1):
            raise ValueError('composition weight must contain finite values in 0..1')
        # R18: abort probes on the parallel path only, also between the feather's passes (the serial rhythm stays
        # one check() per tile)
        probe = unit.check if frame_pool.in_pool() else None
        if probe: probe()
        safe = _inward_feather(source <= 0, width_px, probe=probe)[y-y0:end-y0]
        if probe: probe()
        np.minimum(weight_top_down[y:end], safe, out=output[y:end])
        return (int(np.count_nonzero(output[y:end] != weight_top_down[y:end])),
                int(np.count_nonzero((output[y:end] > 0) & (output[y:end] < 1))))

    def fold(i, value):
        counts[0] += value[0]
        counts[1] += value[1]

    starts = range(0, height, 256)
    unit_bytes = inward_unit_bytes(weight_top_down.shape[1], width_px)
    workers = performance.frame_workers(len(starts), unit_bytes, INWARD_WORKER_CAP, blas=False)
    pool_stats = {}
    with performance.operation('inward_weight'):
        frame_pool.run_ordered(tile, starts, workers=workers, name='inward_weight', check=check, on_result=fold,
                               unit_bytes=unit_bytes, blas=False, stats=pool_stats)
    performance.note_pool('inward_weight', pool_stats)
    from . import resources
    resources.shadow_pool('py.inward_weight', actual=pool_stats.get('workers'), items=len(starts),
                          width=-(-weight_top_down.shape[1] * (256 + 2 * halo) // INWARD_UNIT_ROWS))
    changed, mixed = counts
    receipt = {'recipe': RECIPE, 'width_px': width_px, 'changed_pixels': changed,
               'mixed_pixels': mixed, 'metric': 'capped-Manhattan-distance',
               'first_supported_pixel_weight': 0,
               'scope': 'only decreases already supported sky weight; excluded ground and no-coverage remain zero'}
    return output, receipt
