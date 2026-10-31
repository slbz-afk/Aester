"""Independent bounded Z8 DCP checks, with scalar formulae and binary parsing.

The public profile is a local test asset, not fetched by this suite.  Tests which
need it skip explicitly when absent. Product functions never create expected
DCP pixels or HueSatMap coefficients.
"""
import colorsys
import copy
from fractions import Fraction
import hashlib
import math
from pathlib import Path
import struct

import numpy as np
import pytest

from starstack import experimental_dcp as dcp
from starstack.experimental_huesat import lookup_huesat
from starstack.raw_color import build_recipe

ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / 'outputs/whole-color-20260920/references/rawtherapee_NIKON_Z_8.dcp'
SHA = 'eef2ad2709fcaeb3830a25f63bd9079d68166d4bcac5419bcefe21753cc3b228'
CAM_XYZ = np.asarray([[1.1423, -.4564, -.1123], [-.4816, 1.2895, .2119],
                      [-.021, .1061, .7282], [0, 0, 0]], dtype=np.float32).tolist()
# Independent literal transcription of named working-space matrices from the
# Adobe SDK reference audit.  No product matrix or precomputed inverse is used.
W = [0.3457 / 0.3585, 1.0, (1 - 0.3457 - 0.3585) / 0.3585]
PP = [[.7977, .1352, .0313], [.2880, .7119, .0001], [0., 0., .8249]]
SR = [[.4361, .3851, .1431], [.2225, .7169, .0606], [.0139, .0971, .7141]]


def row_normalize(a):
    return [[x * W[i] / math.fsum(row) for x in row] for i, row in enumerate(a)]


def inverse3(a):
    """Independent small Gauss-Jordan inverse; no NumPy linear algebra."""
    m = [[float(x) for x in row] + [float(i == j) for j in range(3)]
         for i, row in enumerate(a)]
    for col in range(3):
        pivot = max(range(col, 3), key=lambda r: abs(m[r][col]))
        m[col], m[pivot] = m[pivot], m[col]
        factor = m[col][col]
        if factor == 0:
            raise ValueError('singular independent reference')
        m[col] = [v / factor for v in m[col]]
        for row in range(3):
            if row != col:
                factor = m[row][col]
                m[row] = [x - factor*y for x, y in zip(m[row], m[col])]
    return [row[3:] for row in m]


def mv(a, x):
    return [math.fsum(v*c for v, c in zip(row, x)) for row in a]


def decode_independent(path):
    """Read only the two numeric tags we need, independently of product parser."""
    b = Path(path).read_bytes()
    assert len(b) == 65350 and hashlib.sha256(b).hexdigest() == SHA
    byteorder = 'little' if b[0:2] == b'II' else 'big'
    prefix = '<' if byteorder == 'little' else '>'
    u = lambda start, n: int.from_bytes(b[start:start+n], byteorder)
    ifd = u(4, 4)
    offsets = {}
    for pos in range(ifd+2, ifd+2+12*u(ifd, 2), 12):
        tag = u(pos, 2)
        assert tag not in offsets
        offsets[tag] = (u(pos+2, 2), u(pos+4, 4), u(pos+8, 4))
    typ, count, start = offsets[50965]
    assert (typ, count) == (10, 9)
    vals = []
    for k in range(9):
        n, den = struct.unpack_from(prefix+'ii', b, start+8*k)
        vals.append(float(Fraction(n, den)))
    fm = [vals[i:i+3] for i in range(0, 9, 3)]
    typ, count, start = offsets[50939]
    assert (typ, count) == (11, 8100)
    raw = struct.unpack_from(prefix+'f'*count, b, start)
    table = [[list(raw[3*(h*30+s):3*(h*30+s)+3]) for s in range(30)] for h in range(90)]
    return row_normalize(fm), table


def table_scalar(table, hue, sat):
    """Interpolate S first then H; tuple indexing expresses S-fastest storage."""
    hdim, sdim = len(table), len(table[0])
    u = (float(hue) % 360.) * hdim / 360.
    low_h = int(math.floor(u)) % hdim
    frac_h = u - math.floor(u)
    v = float(sat) * (sdim-1)
    low_s = min(int(math.floor(v)), sdim-2)
    frac_s = v-low_s
    answer = []
    for c in range(3):
        a = table[low_h][low_s][c] + frac_s*(table[low_h][low_s+1][c]-table[low_h][low_s][c])
        b = table[(low_h+1) % hdim][low_s][c] + frac_s*(table[(low_h+1) % hdim][low_s+1][c]-table[(low_h+1) % hdim][low_s][c])
        answer.append(a + frac_h*(b-a))
    return answer


def scalar_render(pixel, recipe, fm, table, use_table):
    gains = recipe['white_balance']['gains_rgb']
    scale = recipe['scale']['common_denominator_DN']
    balanced = [float(pixel[i])*gains[i]/scale for i in range(3)]
    xyz = mv(fm, balanced)
    pro = mv(inverse3(row_normalize(PP)), xyz)
    if use_table and min(pro) >= 0:
        h, s, v = colorsys.rgb_to_hsv(*pro)
        dh, sm, vm = table_scalar(table, h*360, s)
        adjusted = colorsys.hsv_to_rgb(((h*360+dh) % 360)/360, s*sm, v*vm)
        xyz = mv(row_normalize(PP), adjusted)
    return mv(inverse3(row_normalize(SR)), xyz)


def make_recipe(*, scale=14884., cam_xyz=None):
    values = {'num_colors': 3, 'color_desc': 'RGBG', 'raw_pattern': [[0, 1], [3, 2]],
              'rgb_xyz_matrix': CAM_XYZ if cam_xyz is None else cam_xyz,
              'camera_whitebalance': [1.7285157442092896, 1., 1.5253901481628418, 0.]}
    meta = {'available': True, 'white_balance_applied': False, 'color_correction_applied': False,
            'versions': {'test': 'independent'}, 'fields': {k: {'value': v} for k, v in values.items()}}
    # This creates an input contract only, not expected DCP colour values.
    return build_recipe(meta, 'a'*64, common_scale=scale)


@pytest.fixture(scope='module')
def asset():
    if not ASSET.is_file():
        pytest.skip('Pinned public Z8 DCP local asset is unavailable; no network fetch')
    assert hashlib.sha256(ASSET.read_bytes()).hexdigest() == SHA
    return ASSET


@pytest.fixture(scope='module')
def decoded(asset):
    return decode_independent(asset)


def test_pinned_parser_independent(asset, decoded):
    fm, table = decoded
    p = dcp.load_z8_profile(asset)
    np.testing.assert_allclose(p['forward_matrix'], fm, rtol=2e-15, atol=1e-16)
    np.testing.assert_array_equal(p['huesat'], np.asarray(table))
    assert not p['forward_matrix'].flags.writeable and not p['huesat'].flags.writeable
    assert p['default_black_render'] == 'None (51110=1)'
    assert not p['huesat_encoding_tag_present'] and not p['look_table_present']


def test_working_matrices_common_pcs():
    np.testing.assert_allclose(dcp.PROPHOTO_TO_XYZ, row_normalize(PP), atol=2e-16, rtol=2e-15)
    np.testing.assert_allclose(dcp.XYZ_TO_PROPHOTO, inverse3(row_normalize(PP)), atol=2e-16, rtol=2e-15)
    np.testing.assert_allclose(dcp.XYZ_TO_SRGB, inverse3(row_normalize(SR)), atol=8e-16, rtol=2e-15)
    np.testing.assert_allclose(dcp.PROPHOTO_TO_XYZ @ np.ones(3), W, atol=2e-16)
    np.testing.assert_allclose(dcp.XYZ_TO_SRGB @ np.asarray(W), np.ones(3), atol=8e-16)


@pytest.mark.parametrize('use_table', [False, True])
def test_random_signed_pixels_scalar(asset, decoded, use_table):
    rng = np.random.default_rng(716091)
    pixels = rng.uniform(-6000, 48000, (17, 23, 3)).astype(np.float32)
    pixels[0, :4] = [[0, 0, 0], [-0., -0., -0.], [1008, 1008, 1008], [100, 200, 50]]
    recipe = make_recipe()
    before, before_recipe = pixels.copy(), copy.deepcopy(recipe)
    actual, record = dcp.render_camera(pixels, recipe, asset, apply_huesat=use_table)
    fm, table = decoded
    expected = np.asarray([scalar_render(p, recipe, fm, table, use_table) for p in pixels.reshape(-1, 3)],
                          dtype=np.float32).reshape(pixels.shape)
    np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=3e-7)
    np.testing.assert_array_equal(pixels, before)
    assert recipe == before_recipe
    assert actual.dtype == np.float32 and not np.shares_memory(actual, pixels)
    assert np.isfinite(actual).all() and (actual < 0).any() and (actual > 1).any()
    assert record['huesat_applied'] is use_table
    assert record['baseline_exposure_applied'] is False
    assert record['automatic_black_applied'] is False
    assert record['look_table_applied'] is False
    assert record['tone_curve_applied'] is False
    assert record['profile_sha256'] == SHA


@pytest.mark.parametrize('use_table', [False, True])
def test_gray_axis_and_linear_exposure(asset, use_table):
    recipe = make_recipe()
    g = np.array(recipe['white_balance']['gains_rgb'])
    levels = np.array([0, 1e-9, .01, .1, .5, 1, 2, 7])
    camera = (levels[:, None, None] * 14884. / g).astype(np.float32)
    out, _ = dcp.render_camera(camera, recipe, asset, apply_huesat=use_table)
    np.testing.assert_allclose(out, np.broadcast_to(levels[:, None, None], out.shape), rtol=3e-7, atol=2e-7)
    doubled, _ = dcp.render_camera(camera*2, recipe, asset, apply_huesat=use_table)
    np.testing.assert_allclose(doubled, out*2, rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize('use_table', [False, True])
def test_coloured_exposure_homogeneity_not_additivity(asset, use_table):
    camera = np.array([[[100, 200, 300], [14000, 5000, 16000], [-40, 100, 30]]], np.float32)
    a, _ = dcp.render_camera(camera, make_recipe(), asset, apply_huesat=use_table)
    b, _ = dcp.render_camera(camera*8, make_recipe(), asset, apply_huesat=use_table)
    np.testing.assert_array_equal(a*8, b)


def test_negative_prophoto_bypasses_exactly(asset, decoded):
    fm, _ = decoded
    recipe = make_recipe()
    pixels = np.asarray([[[-100, 0, 0], [0, -100, 0], [0, 0, -100],
                          [-1, -2, -3], [1000, 1, 1], [1, 1000, 1]]], np.float32)
    # Compute membership independently, rather than trusting the returned count.
    pp_inv = inverse3(row_normalize(PP))
    gains = recipe['white_balance']['gains_rgb']
    good = []
    for p in pixels.reshape(-1, 3):
        xyz = mv(fm, [float(p[i])*gains[i]/14884 for i in range(3)])
        good.append(min(mv(pp_inv, xyz)) >= 0)
    bad = ~np.array(good).reshape(pixels.shape[:2])
    assert bad.any()
    matrix, _ = dcp.render_camera(pixels, recipe, asset, apply_huesat=False)
    mapped, info = dcp.render_camera(pixels, recipe, asset, apply_huesat=True)
    np.testing.assert_array_equal(mapped[bad], matrix[bad])
    assert info['huesat_bypassed_negative_pixels'] == int(bad.sum())
    assert info['huesat_processed_pixels'] == int((~bad).sum())


def test_real_hsm_saturation_above_one_without_clipping(asset, decoded):
    fm, table = decoded
    recipe = make_recipe()
    # Construct camera values from a specified positive ProPhoto colour through
    # independent scalar inverse matrices. This avoids guessing branch membership.
    candidates = [(table_scalar(table, h, .95)[1], h) for h in range(0, 360, 4)]
    factor, hue = max(candidates)
    assert factor*.95 > 1
    pro = colorsys.hsv_to_rgb(hue/360, .95, 2.5)
    balanced = mv(inverse3(fm), mv(row_normalize(PP), pro))
    gains = recipe['white_balance']['gains_rgb']
    camera = np.array([[[balanced[i]*14884/gains[i] for i in range(3)]]], np.float32)
    actual, rec = dcp.render_camera(camera, recipe, asset, apply_huesat=True)
    expected = np.asarray(scalar_render(camera[0, 0], recipe, fm, table, True), np.float32)
    np.testing.assert_allclose(actual[0, 0], expected, rtol=2e-6, atol=5e-7)
    assert rec['huesat_saturation_above_one_pixels'] == 1
    assert rec['huesat_processed_pixels'] == 1
    assert actual.min() < 0 and actual.max() > 1


def test_rgb_hsv_helpers_match_independent_stdlib():
    colors = np.array([[0, 0, 0], [1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                       [1, 1, 0], [0, 1, 1], [1, 0, 1], [1e-200, 2e-200, 3e-200],
                       [3, 2, 1], [3, 3, 2]], np.float64)
    h, s, v = dcp.rgb_to_hsv(colors)
    expect = np.array([colorsys.rgb_to_hsv(*p) for p in colors])
    np.testing.assert_allclose(h/360, expect[:, 0], rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(s, expect[:, 1], rtol=1e-14, atol=1e-14)
    np.testing.assert_array_equal(v, expect[:, 2])
    h2 = np.array([-1., 0., 30., 60., 120., 180., 240., 300., 359.9, 360., 721.])
    s2 = np.linspace(0., 1.8, len(h2))
    v2 = np.linspace(.1, 4., len(h2))
    expected = [colorsys.hsv_to_rgb((h % 360)/360, s, v) for h, s, v in zip(h2, s2, v2)]
    np.testing.assert_allclose(dcp.hsv_to_rgb(h2, s2, v2), expected, rtol=1e-14, atol=1e-14)


def test_hsm_exposure_homogeneity_does_not_mean_additivity(asset):
    a = np.array([[[5000, 1000, 5000]]], np.float32)
    b = np.array([[[1000, 6000, 1000]]], np.float32)
    recipe = make_recipe()
    fa, _ = dcp.render_camera(a, recipe, asset, apply_huesat=True)
    fb, _ = dcp.render_camera(b, recipe, asset, apply_huesat=True)
    fab, _ = dcp.render_camera(a+b, recipe, asset, apply_huesat=True)
    assert np.max(np.abs(fab-fa-fb)) > .01


def test_changed_fixed_recipe_rejected(asset):
    for field in ('gain', 'state'):
        recipe = make_recipe()
        if field == 'gain':
            recipe['white_balance']['gains_rgb'][0] *= 1.01
        else:
            recipe['conditions']['input_WB_applied_required'] = True
        with pytest.raises(ValueError):
            dcp.render_camera(np.ones((2, 2, 3), np.float32), recipe, asset)


def test_zero_and_already_black_subtracted_cfa(asset, decoded):
    from starstack.raw_audit import subtract_declared_black
    from starstack.float_demosaic import demosaic_bilinear
    # Measured R/G/B are exactly 100/200/50 DN above 1008; same-colour constant
    # fields make the independent missing-colour truth exact, even at edges.
    cfa = np.tile(np.array([[1108, 1208], [1208, 1058]], np.uint16), (4, 5))
    # The model contract is deliberately explicit; no profile affects it.
    model = {'status': 'available', 'repeat_shape': [1, 1], 'black_values': [1008.]}
    signal = subtract_declared_black(cfa, model)
    camera = demosaic_bilinear(signal, [[0, 1], [3, 2]])
    np.testing.assert_array_equal(camera, np.broadcast_to(np.array([100., 200., 50.], np.float32), camera.shape))
    out, _ = dcp.render_camera(camera, make_recipe(), asset, apply_huesat=False)
    wanted = np.asarray(scalar_render([100, 200, 50], make_recipe(), *decoded, False), np.float32)
    np.testing.assert_allclose(out, np.broadcast_to(wanted, out.shape), rtol=2e-6, atol=2e-8)
    zero, _ = dcp.render_camera(np.zeros((2, 3, 3), np.float32), make_recipe(), asset, apply_huesat=True)
    np.testing.assert_array_equal(zero, 0)


def test_user_selected_z8_default_has_hsm_and_linear_opt_out(asset, decoded):
    """The photographic default follows user choice; a linear master is explicit."""
    camera = np.array([[[5000., 1000., 5000.], [1000., 6000., 1000.]]], np.float32)
    recipe = make_recipe()
    actual, record = dcp.render_camera(camera, recipe, asset)
    wanted = np.asarray([scalar_render(p, recipe, *decoded, True)
                         for p in camera.reshape(-1, 3)], np.float32).reshape(camera.shape)
    np.testing.assert_allclose(actual, wanted, rtol=2e-6, atol=2e-8)
    assert record['huesat_applied'] is True
    linear, info = dcp.render_camera(camera, recipe, asset, apply_huesat=False)
    expected_linear = np.asarray([scalar_render(p, recipe, *decoded, False)
                                  for p in camera.reshape(-1, 3)], np.float32).reshape(camera.shape)
    np.testing.assert_allclose(linear, expected_linear, rtol=2e-6, atol=2e-8)
    assert info['huesat_applied'] is False
    assert np.max(np.abs(actual-linear)) > .01


@pytest.mark.parametrize('shape', [(0, 2, 3), (2, 0, 3), (2, 3), (2, 2, 4), (2, 2, 3, 1)])
def test_wrong_shape(asset, shape):
    with pytest.raises(ValueError):
        dcp.render_camera(np.zeros(shape, np.float32), make_recipe(), asset)


@pytest.mark.parametrize('dtype', [np.float64, np.int16, np.uint16, bool, object])
def test_wrong_dtype(asset, dtype):
    with pytest.raises(ValueError):
        dcp.render_camera(np.ones((2, 2, 3), dtype), make_recipe(), asset)


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -float('inf')])
@pytest.mark.parametrize('use_table', [False, True])
def test_nonfinite_later_block_keeps_input(asset, bad, use_table):
    a = np.ones((129, 2, 3), np.float32)
    a[-1, -1, -1] = bad
    before = a.tobytes()
    with pytest.raises(ValueError):
        dcp.render_camera(a, make_recipe(), asset, apply_huesat=use_table)
    assert a.tobytes() == before


def test_wrong_state_flag_and_nonarray(asset):
    a = np.ones((2, 2, 3), np.float32)
    for kwargs in ({'input_state': 'already_WB'}, {'input_state': None}, {'apply_huesat': 1},
                   {'apply_huesat': np.bool_(True)}):
        with pytest.raises(ValueError):
            dcp.render_camera(a, make_recipe(), asset, **kwargs)
    with pytest.raises(ValueError):
        dcp.render_camera(a.tolist(), make_recipe(), asset)


@pytest.mark.parametrize('use_table', [False, True])
def test_readonly_strided_and_block_boundaries(asset, use_table):
    a = np.arange(130*4*3, dtype=np.float32).reshape(130, 4, 3)[::-1, ::-1]
    a.setflags(write=False)
    out, _ = dcp.render_camera(a, make_recipe(), asset, apply_huesat=use_table)
    pieces = [dcp.render_camera(a[i:i+1], make_recipe(), asset, apply_huesat=use_table)[0] for i in [0, 127, 128, 129]]
    for i, p in zip([0, 127, 128, 129], pieces):
        np.testing.assert_array_equal(out[i:i+1], p)


def test_non_z8_matrix_family_rejected(asset):
    other = copy.deepcopy(CAM_XYZ)
    other[0][0] += .02
    with pytest.raises(ValueError, match='Z8'):
        dcp.render_camera(np.ones((2, 2, 3), np.float32), make_recipe(cam_xyz=other), asset)


def test_profile_byte_change_rejected(asset, tmp_path):
    b = bytearray(asset.read_bytes())
    b[-1] ^= 1
    p = tmp_path/'changed.dcp'
    p.write_bytes(b)
    with pytest.raises(ValueError, match='pinned'):
        dcp.load_z8_profile(p)
    p.write_bytes(b[:20])
    with pytest.raises(ValueError, match='profile size|pinned'):
        dcp.load_z8_profile(p)


@pytest.mark.parametrize('use_table', [False, True])
def test_float32_overflow_is_valueerror(asset, use_table):
    a = np.full((2, 2, 3), np.finfo(np.float32).max, np.float32)
    before = a.copy()
    with pytest.raises(ValueError, match='overflow|Nonfinite'):
        dcp.render_camera(a, make_recipe(scale=1), asset, apply_huesat=use_table)
    np.testing.assert_array_equal(a, before)


def test_real_table_all_nodes_and_gray_rows(decoded):
    _, table = decoded
    a = np.asarray(table)
    hues, sats = np.meshgrid(np.arange(90, dtype=np.float64)*4,
                             np.arange(30, dtype=np.float64)/29, indexing='ij')
    np.testing.assert_allclose(lookup_huesat(a, hues, sats), a, rtol=3e-14, atol=3e-13)
    np.testing.assert_array_equal(a[:, 0, 0], 0)
    np.testing.assert_array_equal(a[:, 0, 2], 1)


def test_table_axes_wrap_and_scalar_reference(decoded):
    _, table = decoded
    hue = np.array([-721., -360., -.01, 0., 1., 2., 4., 123.7, 356., 358.,
                    np.nextafter(360., 0), 360., 721., 1e12])
    sat = np.array([0., .01, .5, 1., .999, .2, .25, .7, .3, .4, 1., 0., .2, .4])
    actual = lookup_huesat(np.asarray(table), hue, sat)
    expect = [table_scalar(table, h, s) for h, s in zip(hue, sat)]
    np.testing.assert_allclose(actual, expect, rtol=2e-13, atol=3e-13)


def test_distinct_h_s_axes_and_hue_shift_not_circular():
    table = np.empty((4, 3, 3), np.float64)
    for h in range(4):
        for s in range(3):
            table[h, s] = [100*h+10*s, 1+2*h+3*s, 2+5*h+7*s]
    hue = np.array([45., 315., 0., 90.])
    sat = np.array([.25, .75, 1., 0.])
    expect = np.array([table_scalar(table.tolist(), h, s) for h, s in zip(hue, sat)])
    np.testing.assert_allclose(lookup_huesat(table, hue, sat), expect, rtol=0, atol=1e-13)
    assert expect[0, 0] == 55. and expect[1, 0] == 165.


@pytest.mark.parametrize('shape', [(), (0,), (2, 0), (2, 3)])
def test_huesat_arbitrary_query_shape_no_mutation(shape):
    table = np.empty((2, 2, 3), np.float32)
    table[:] = [0, 1, 1]
    h, s = np.zeros(shape, np.float32), np.ones(shape, np.float32)
    before = (table.tobytes(), h.tobytes(), s.tobytes())
    actual = lookup_huesat(table, h, s)
    assert actual.shape == shape+(3,) and actual.dtype == np.float64
    assert before == (table.tobytes(), h.tobytes(), s.tobytes())
    np.testing.assert_array_equal(actual, np.broadcast_to([0., 1., 1.], actual.shape))


@pytest.mark.parametrize('kind', ['int', 'object', 'badshape', 'few_h', 'few_s', 'nan', 'negative_multiplier'])
def test_huesat_bad_table(kind):
    table = np.zeros((3, 4, 3), np.float64)
    table[..., 1:] = 1
    if kind == 'int': table = table.astype(np.int32)
    elif kind == 'object': table = table.astype(object)
    elif kind == 'badshape': table = table[..., :2]
    elif kind == 'few_h': table = table[:1]
    elif kind == 'few_s': table = table[:, :1]
    elif kind == 'nan': table[0, 0, 0] = np.nan
    elif kind == 'negative_multiplier': table[0, 0, 1] = -.1
    with pytest.raises(ValueError):
        lookup_huesat(table, np.array([0.]), np.array([0.]))


@pytest.mark.parametrize('h,s', [([0.], np.array([0.])), (np.array([0]), np.array([0.])),
                                (np.array([np.inf]), np.array([0.])),
                                (np.array([0.]), np.array([np.nan])),
                                (np.array([0.]), np.array([-.1])),
                                (np.array([0.]), np.array([1.01])),
                                (np.array([0., 1.]), np.array([.5]))])
def test_huesat_bad_query(h, s):
    table = np.ones((2, 2, 3), np.float64)
    with pytest.raises(ValueError):
        lookup_huesat(table, h, s)
