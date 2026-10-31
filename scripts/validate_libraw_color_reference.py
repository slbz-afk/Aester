"""Compare real-image unsaturated interiors with an independent LibRaw render.

Black/white/WB are deliberately fixed for both sides to isolate color math.
This direct LSB comparison requires green to be the minimum WB coefficient:
our G1=1 convention then matches LibRaw HighlightMode.Clip's min-gain scale.
Canon uses decoder-reported levels for this comparison only; that does not
validate its file black model or enable the experimental product front end.
"""
from pathlib import Path
import json
import sys
import time

import numpy as np
import rawpy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'scripts'))
from starstack import core, raw_color
from starstack.raw_metadata import read_raw_metadata
from starstack.float_demosaic import demosaic_bilinear, rgb_pattern
from development_assets import dng_for_original


def matching_wb_scale(gains) -> dict:
    """Reject scale-mismatched references before the full reference render.

    LibRaw Clip divides all four pre_mul slots by their minimum. The product
    fixes G1 to one instead. Do not rescale results to hide this distinction:
    keep this comparator narrow and record the checked convention in evidence.
    """
    values = np.asarray(gains)
    if (values.shape != (3,) or values.dtype.kind not in 'iuf'
            or not np.isfinite(values).all() or np.any(values <= 0)):
        raise ValueError('LibRaw reference requires three finite positive RGB WB gains')
    red, green, blue = map(float, values)
    user_wb = [red, green, blue, green]
    minimum = min(user_wb)
    if green != 1.0 or minimum != green:
        raise ValueError(
            'LibRaw reference scale mismatch: product WB uses G1=1, while '
            'HighlightMode.Clip divides by the minimum of four WB gains. '
            f'This direct comparator requires min(user_wb) == G1 == 1; '
            f'got user_wb={user_wb}, minimum={minimum}. '
            'A common brightness-scale difference is not a color-matrix error.')
    return {
        'product_normalization': 'G1_equals_one',
        'libraw_highlight_mode': 'Clip',
        'libraw_normalization': 'minimum_of_four_pre_mul_slots',
        'user_wb': user_wb, 'green_coefficient': green,
        'minimum_coefficient': minimum,
        'precondition': 'min(user_wb) == G1 == 1',
        'precondition_satisfied': True,
        'libraw_to_product_common_scale': 1.0,
        'source': 'https://github.com/LibRaw/LibRaw/blob/0.22.1/src/postprocessing/postprocessing_utils_dcrdefs.cpp#L190-L203',
    }


def compare(source, name):
    metadata = read_raw_metadata(source)
    levels = metadata['fields']['black_level_per_channel']['value']
    assert len(set(levels)) == 1  # Narrow common-black reference exercise.
    black = levels[0]
    white = metadata['fields']['white_level']['value']
    recipe = raw_color.build_recipe(metadata, core._sha(source), white, black)
    gains = np.array(recipe['white_balance']['gains_rgb'])
    wb_scale = matching_wb_scale(gains)
    with rawpy.imread(str(source)) as raw:
        codes = raw.raw_image_visible.copy()
        pattern = np.roll(raw.raw_pattern, (-raw.sizes.top_margin % 2, -raw.sizes.left_margin % 2), axis=(0, 1))
        sizes = {'width': raw.sizes.width, 'height': raw.sizes.height,
                 'top_margin': raw.sizes.top_margin, 'left_margin': raw.sizes.left_margin}
        params = {'user_wb': wb_scale['user_wb'], 'use_camera_wb': False,
            'use_auto_wb': False, 'output_bps': 16, 'gamma': (1, 1), 'no_auto_bright': True,
            'no_auto_scale': False, 'adjust_maximum_thr': 0., 'bright': 1.,
            'user_flip': 0, 'median_filter_passes': 0, 'user_black': black,
            # LibRaw dcraw_process assigns user_sat AFTER subtracting black.
            # Supply the effective range, not the original file WhiteLevel.
            'user_cblack': [0, 0, 0, 0], 'user_sat': white-black}
        reference = raw.postprocess(demosaic_algorithm=rawpy.DemosaicAlgorithm.LINEAR,
            output_color=rawpy.ColorSpace.sRGB, highlight_mode=rawpy.HighlightMode.Clip,
            **params)
    signal = codes.astype(np.float32) - black
    camera = demosaic_bilinear(signal, pattern)
    ours = raw_color.apply_color(camera, recipe)
    assert ours.shape == reference.shape
    p = rgb_pattern(pattern)
    h, w = codes.shape
    # Every contributing CFA sample must remain safely inside LibRaw's integer
    # WB scaling range. Erode by the bilinear support and exclude image edges.
    good_input = np.ones(codes.shape, bool)
    for y in range(2):
        for x in range(2):
            scaled = signal[y::2, x::2] * (gains[p[y, x]] / (white-black))
            good_input[y::2, x::2] = (scaled > .002) & (scaled < .95)
    valid = np.zeros(codes.shape, bool)
    interior = np.ones((h-8, w-8), bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            interior &= good_input[4+dy:h-4+dy, 4+dx:w-4+dx]
    valid[4:-4, 4:-4] = interior
    valid &= np.all((ours > .002) & (ours < .95), axis=2)
    # Integer WB and conversion add rounding; retain error in 16-bit output LSB.
    differences = np.abs(ours[valid].astype(np.float64)*65535 - reference[valid])
    assert len(differences) > 10_000
    quantiles = np.quantile(differences, [.5, .9, .99, 1]).tolist()
    assert core._sha(source) == recipe['reference']['source_sha256']
    result = {'source': {'name': source.name, 'sha256': core._sha(source)}, 'frame': name,
        'scope': 'fixed_common_black_white_color_comparison_not_sensor_calibration',
        'black_DN': black, 'white_DN': white, 'sizes': sizes, 'versions': metadata['versions'],
        'reference_settings': {**params, 'demosaic': 'LINEAR', 'output_color': 'sRGB', 'highlight_mode': 'Clip'},
        'float_recipe': recipe, 'tested_pixels': int(np.count_nonzero(valid)),
        'channel_error_16bit_LSB': dict(zip(('median', 'p90', 'p99', 'maximum'), quantiles)),
        'threshold_16bit_LSB': 8., 'passed': bool(quantiles[-1] <= 8.),
        'scale_contract_source': 'https://github.com/LibRaw/LibRaw/blob/0.22.1/src/postprocessing/dcraw_process.cpp#L78-L88',
        'scale_contract': 'user_sat overrides maximum after black subtraction; set WhiteLevel minus common black',
        'wb_scale_contract': wb_scale,
        'limitations': ['Excluded black/bright-clipped samples and outer 4-pixel edge',
                        'Direct LSB comparison requires green to be the minimum WB coefficient',
                        'Same decoder source is not proof of true camera color',
                        'Canon declared file black still unsupported in product float entry']}
    return result


def main():
    began = time.monotonic()
    report = {'state': 'running', 'created_at': core.now(), 'comparisons': {}}
    target = ROOT/'docs/diagnostics/libraw-color-reference-20260925.json'
    for name, source in [('z8', dng_for_original('DSC_2206.NEF')),
                         ('canon', ROOT/'data/external/photographingspace/CSM30799.CR2')]:
        print('Comparing', name, flush=True)
        report['comparisons'][name] = compare(source, name)
        core.atomic_json(target, report)
        print(name, report['comparisons'][name]['channel_error_16bit_LSB'], flush=True)
    report.update(state='succeeded' if all(c['passed'] for c in report['comparisons'].values()) else 'failed',
                  elapsed_seconds=time.monotonic()-began, script_sha256=core._sha(Path(__file__)))
    core.atomic_json(target, report)
    print('Result', report['state'], target, flush=True)
    if report['state'] != 'succeeded':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
