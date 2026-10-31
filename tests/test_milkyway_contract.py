"""Codex local fallback API tests; no DeepSeek delivery or image-quality claims.

The module imports the public package normally so it can later move to tests/.
All arrays are deterministic synthetic data and all disk IO uses pytest tmp_path.
"""
from copy import deepcopy

import numpy as np
import pytest

from starstack import milkyway as mw

DOMAIN = 'display-sRGB'
DTYPES = [np.float32, np.uint16]


def scene(dtype=np.float32, shape=(96, 128), exclusions=True):
    h, w = shape
    y, x = np.mgrid[:h, :w].astype(np.float32)
    x /= max(w - 1, 1)
    y /= max(h - 1, 1)
    base = .19 + .12*x + .035*y + .009*np.sin(x*19)
    rgb = np.stack((base*1.06, base, base*.92), axis=-1).astype(np.float32)
    if dtype == np.uint16:
        rgb = np.rint(rgb*65535).astype(np.uint16)
    sky = np.ones((h, w), np.float32)
    protect = np.zeros((h, w), bool)
    mask = np.ones((h, w), np.float32)
    if exclusions:
        sky[-12:] = 0
        sky[-14:-12] = .5
        mask[:, :9] = 0
        mask[:8] = 0
        mask[:, -7:] = 0
        protect[h//2, w//4:3*w//4] = True
    return rgb, sky, protect, mask


def prepare(inputs, recipe=None):
    return mw.prepare(*inputs, recipe or mw.Recipe(work_long_edge=256), domain=DOMAIN)


def render(inputs, prepared, **kwargs):
    output = np.empty_like(inputs[0], order='C')
    stats = mw.render(*inputs, prepared, output, domain=DOMAIN, **kwargs)
    return output, stats


def bytes_of(inputs):
    return tuple(a.tobytes() for a in inputs)


def reject_before_write(inputs, prepared, **kwargs):
    output = np.full_like(inputs[0], 71, order='C')
    before = output.tobytes()
    with pytest.raises(ValueError):
        mw.render(*inputs, prepared, output, domain=DOMAIN, **kwargs)
    assert output.tobytes() == before, 'rejection must precede output writes'


def test_real_recipe_parameters_reject_nonreal_nonfinite_and_out_of_range():
    for name, high in [('brightness', .06), ('contrast', 1.5),
                       ('structure', 2.), ('color', 1.), ('strength', 1.)]:
        for bad in [True, np.bool_(False), '0.1', None, complex(.1),
                    np.nan, np.inf, -np.inf, -.001, high+.001]:
            with pytest.raises(ValueError, match=name):
                mw.Recipe(**{name: bad}).validate()
        mw.Recipe(**{name: 0.}).validate()
        mw.Recipe(**{name: high}).validate()


def test_integer_recipe_parameters_require_exact_builtin_int_in_bounds():
    for name, low, high in [('work_long_edge', 256, 2560), ('rows', 8, 512)]:
        for bad in [True, np.int64(low), float(low), str(low), None, low-1, high+1]:
            with pytest.raises(ValueError, match=name):
                mw.Recipe(**{name: bad}).validate()
        mw.Recipe(**{name: low}).validate()
        mw.Recipe(**{name: high}).validate()
    inputs = scene()
    prepared = prepare(inputs)
    for value in [True, np.int64(8), 8., '8', 0, 7, 513]:
        reject_before_write(inputs, prepared, row_size=value)


def test_recipe_class_and_domain_are_explicit():
    inputs = scene()

    class Derived(mw.Recipe):
        pass

    for bad in [None, {}, Derived()]:
        with pytest.raises(ValueError, match='recipe'):
            mw.prepare(*inputs, bad, domain=DOMAIN)
    for bad in [None, 1, 'scene-linear', 'sRGB', np.str_(DOMAIN)]:
        with pytest.raises(ValueError, match='domain'):
            mw.prepare(*inputs, mw.Recipe(), domain=bad)


def test_rgb_rejects_invalid_dtype_shape_and_values():
    inputs = scene()
    rgb = inputs[0]
    bad_arrays = [rgb.tolist(), rgb[..., 0], rgb[..., :2], rgb[:7],
                  rgb.astype(np.float64), rgb.astype(np.uint8)]
    for bad_value in [np.nan, np.inf, -.01, 1.01]:
        a = rgb.copy()
        a[33, 44, 1] = bad_value
        bad_arrays.append(a)
    for bad in bad_arrays:
        with pytest.raises(ValueError):
            prepare((bad, *inputs[1:]))


def test_sky_and_protect_require_matching_shapes_types_and_valid_weights():
    inputs = scene()
    bad_sky = [inputs[1].tolist(), inputs[1][:-1], inputs[1].astype(np.float64)]
    for value in [np.nan, np.inf, -.01, 1.01]:
        a = inputs[1].copy()
        a[30, 40] = value
        bad_sky.append(a)
    for bad in bad_sky:
        with pytest.raises(ValueError):
            prepare((inputs[0], bad, inputs[2], inputs[3]))
    for bad in [inputs[2].tolist(), inputs[2][:-1], inputs[2].astype(np.uint8)]:
        with pytest.raises(ValueError):
            prepare((inputs[0], inputs[1], bad, inputs[3]))


def test_explicit_milkyway_mask_validation_applies_to_prepare_and_render():
    inputs = scene()
    prepared = prepare(inputs)
    bad_masks = [inputs[3].tolist(), inputs[3][:-1],
                 inputs[3].astype(np.float64), inputs[3][..., None]]
    for value in [np.nan, np.inf, -.001, 1.001]:
        a = inputs[3].copy()
        a[50, 60] = value
        bad_masks.append(a)
    for bad in bad_masks:
        altered = (*inputs[:3], bad)
        with pytest.raises(ValueError):
            prepare(altered)
        reject_before_write(altered, prepared)


@pytest.mark.parametrize('dtype', DTYPES)
def test_prepare_and_render_preserve_readonly_source_arrays(dtype):
    inputs = scene(dtype)
    for a in inputs:
        a.setflags(write=False)
    before = bytes_of(inputs)
    prepared = prepare(inputs)
    output, stats = render(inputs, prepared)
    assert bytes_of(inputs) == before
    assert all(not a.flags.writeable for a in inputs)
    assert stats['changed_pixels'] > 0
    assert not np.shares_memory(output, inputs[0])


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('disabled', ['strength', 'mask'])
def test_disabled_processing_preserves_full_bytes(dtype, disabled):
    inputs = scene(dtype)
    if dtype == np.float32:
        inputs[0][0, 0] = np.float32(-0.)
        inputs[0][0, 1] = np.float32(1.)
    recipe = mw.Recipe(work_long_edge=256, strength=0. if disabled == 'strength' else 1.)
    if disabled == 'mask':
        inputs[3][:] = 0
    before = bytes_of(inputs)
    output, stats = render(inputs, prepare(inputs, recipe))
    assert output.tobytes() == before[0]
    assert bytes_of(inputs) == before
    assert stats['identity'] and stats['changed_pixels'] == 0


@pytest.mark.parametrize('dtype', DTYPES)
def test_all_excluded_pixel_categories_are_exact_while_interior_changes(dtype):
    inputs = scene(dtype)
    rgb, sky, protect, mask = inputs
    before = bytes_of(inputs)
    output, stats = render(inputs, prepare(inputs))
    categories = {'outside': mask == 0, 'ground': sky == 0,
                  'mixed': (sky > 0) & (sky < 1), 'protect': protect}
    for name, selected in categories.items():
        assert selected.any(), name
        assert output[selected].tobytes() == rgb[selected].tobytes(), name
    allowed = (mask > 0) & (sky == 1) & ~protect
    assert np.any(output[allowed] != rgb[allowed])
    assert stats['changed_pixels'] == np.count_nonzero(np.any(output != rgb, axis=-1))
    assert stats['outside_mask_changed_channels'] == 0
    assert stats['protected_changed_channels'] == 0
    assert stats['ground_or_mixed_changed_channels'] == 0
    assert bytes_of(inputs) == before


def test_every_source_identity_is_rechecked_before_output_write():
    for index in range(4):
        inputs = scene()
        prepared = prepare(inputs)
        if index == 0:
            inputs[0][25, 30, 0] += np.float32(.001)
        elif index == 2:
            inputs[2][25, 30] = True
        else:
            inputs[index][25, 30] = np.float32(.4)
        reject_before_write(inputs, prepared)


def test_prepared_maps_reject_valid_range_tampering_and_malformed_arrays():
    inputs = scene()
    original = prepare(inputs)
    for name in ['delta_rgb', 'safe']:
        replacement = original[name].copy()
        replacement.flat[0] = .01 if name == 'delta_rgb' else .123
        cases = [replacement, original[name].astype(np.float64),
                 original[name][:-1], original[name].tolist()]
        for value in [np.nan, np.inf, -1., 2.]:
            a = original[name].copy()
            a.flat[0] = value
            cases.append(a)
        for bad in cases:
            prepared = dict(original)
            prepared[name] = bad
            reject_before_write(inputs, prepared)


def test_prepared_metadata_and_missing_fields_are_rejected():
    inputs = scene()
    original = prepare(inputs)
    cases = [('version', 'other'), ('domain', 'scene-linear'), ('shape', (8, 8)),
             ('pivot', original['pivot']+.0001), ('seal', 'altered'),
             ('identities', dict(original['identities'], rgb='altered')),
             ('recipe', dict(original['recipe'], brightness=.001))]
    for name, value in cases:
        prepared = deepcopy(original)
        prepared[name] = value
        reject_before_write(inputs, prepared)
    for name in ['version', 'recipe', 'identities', 'pivot', 'safe', 'delta_rgb', 'seal']:
        prepared = dict(original)
        del prepared[name]
        reject_before_write(inputs, prepared)
    for bad in [None, [], 1]:
        reject_before_write(inputs, bad)


def test_output_same_array_and_shared_view_aliases_are_rejected():
    inputs = scene()
    prepared = prepare(inputs)
    before = bytes_of(inputs)
    for output in [inputs[0], inputs[0].view()]:
        with pytest.raises(ValueError, match='alias'):
            mw.render(*inputs, prepared, output, domain=DOMAIN)
    assert bytes_of(inputs) == before
    # A mask can share an otherwise valid RGB output buffer too.
    for mask_index in [1, 3]:
        backing = np.zeros(inputs[0].shape, np.float32)
        shared = backing[..., 0]
        shared[:] = 1
        altered = list(scene())
        altered[mask_index] = shared
        prepared = prepare(altered)
        before = backing.tobytes()
        with pytest.raises(ValueError, match='alias'):
            mw.render(*altered, prepared, backing, domain=DOMAIN)
        assert backing.tobytes() == before


def test_distinct_memmaps_of_same_file_are_rejected(tmp_path):
    inputs = list(scene())
    path = tmp_path / 'shared-storage.bin'
    source = np.memmap(path, mode='w+', dtype=np.float32, shape=inputs[0].shape)
    source[:] = inputs[0]
    source.flush()
    output = np.memmap(path, mode='r+', dtype=np.float32, shape=inputs[0].shape)
    try:
        inputs[0] = source
        prepared = prepare(inputs)
        before = source.tobytes()
        assert not np.shares_memory(source, output)
        with pytest.raises(ValueError, match='alias'):
            mw.render(*inputs, prepared, output, domain=DOMAIN)
        assert source.tobytes() == before
    finally:
        output._mmap.close()
        source._mmap.close()


def test_output_rejects_incompatible_or_unwritable_buffers():
    inputs = scene()
    prepared = prepare(inputs)
    readonly = np.empty_like(inputs[0])
    readonly.setflags(write=False)
    h, w = inputs[0].shape[:2]
    noncontiguous = np.empty((h, w*2, 3), np.float32)[:, ::2]
    cases = [None, np.empty((h, w, 2), np.float32),
             np.empty_like(inputs[0], dtype=np.float64), readonly, noncontiguous]
    before = bytes_of(inputs)
    for output in cases:
        with pytest.raises(ValueError, match='output'):
            mw.render(*inputs, prepared, output, domain=DOMAIN)
    assert bytes_of(inputs) == before


def test_memory_only_memmap_subclass_is_valid_output(tmp_path):
    inputs=list(scene())
    source=np.memmap(tmp_path/'source.bin',mode='w+',dtype=np.float32,shape=inputs[0].shape)
    source[:]=inputs[0];inputs[0]=source
    try:
        output=np.empty_like(source)
        assert isinstance(output,np.memmap) and output.filename is None
        prepared=prepare(inputs)
        stats=mw.render(*inputs,prepared,output,domain=DOMAIN)
        assert stats['changed_pixels']>0
        assert np.isfinite(output).all()
    finally:
        source._mmap.close()


def test_contrast_only_preserves_constant_colour_across_mask_half_threshold():
    inputs = list(scene(exclusions=False))
    inputs[0][:] = (.17, .23, .31)
    recipe = mw.Recipe(brightness=0., contrast=1.5, structure=0., color=0.,
                       strength=1., work_long_edge=256)
    for weight in [.01, .49, .5, .51, 1.]:
        inputs[3][:] = np.float32(weight)
        before = bytes_of(inputs)
        output, _ = render(inputs, prepare(inputs, recipe))
        np.testing.assert_allclose(output, inputs[0], rtol=0., atol=1e-7,
                                   err_msg=f'constant-colour shift at mask={weight}')
        assert bytes_of(inputs) == before


@pytest.mark.parametrize('dtype', DTYPES)
def test_recipe_and_override_row_sizes_do_not_change_pixels(dtype):
    inputs = scene(dtype, shape=(269, 321))
    before = bytes_of(inputs)
    reference, stats = render(inputs, prepare(inputs, mw.Recipe(work_long_edge=256, rows=8)))
    assert stats['changed_pixels'] > 0
    for rows in [17, 64, 128, 512]:
        prepared = prepare(inputs, mw.Recipe(work_long_edge=256, rows=rows))
        output, _ = render(inputs, prepared)
        override, _ = render(inputs, prepared, row_size=31)
        assert output.tobytes() == reference.tobytes(), rows
        assert override.tobytes() == reference.tobytes(), rows
    assert bytes_of(inputs) == before


@pytest.mark.parametrize('dtype', DTYPES)
def test_strong_controls_do_not_create_zero_or_white_endpoints(dtype):
    inputs = list(scene(np.float32, exclusions=False))
    rgb = inputs[0]
    rgb[25:49, 25:49] = (.0004, .0006, .0008)
    rgb[25:49, 75:99] = (.965, .94, .90)
    rgb[60:75, 30:50] = (.85, .004, .02)
    rgb[1, 1] = 0
    rgb[1, 2] = 1
    rgb[1, 3] = np.nextafter(np.float32(0), np.float32(1))
    rgb[1, 4] = np.float32(np.finfo(np.float32).tiny*2)
    if dtype == np.uint16:
        inputs[0] = np.rint(rgb*65535).astype(np.uint16)
    else:
        rgb[1, 1, 0] = np.float32(-0.)
    before = bytes_of(inputs)
    for brightness in [0., .06]:
        recipe = mw.Recipe(brightness=brightness, contrast=1.5, structure=2.,
                           color=1., strength=1., work_long_edge=256)
        output, stats = render(inputs, prepare(inputs, recipe))
        source = inputs[0]
        top = 65535 if dtype == np.uint16 else 1.
        new_endpoints = ((output == 0) & (source > 0)) | ((output == top) & (source < top))
        assert np.isfinite(output).all()
        assert output.min() >= 0 and output.max() <= top
        assert not new_endpoints.any(), f'{np.count_nonzero(new_endpoints)} new endpoints'
        assert stats['new_endpoint_channels'] == 0
        assert stats['changed_pixels'] > 0
        assert output[1, 1:5].tobytes() == source[1, 1:5].tobytes()
    assert bytes_of(inputs) == before
