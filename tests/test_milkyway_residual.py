"""Independent integer-residual and storage contracts for paired finishing."""
import hashlib
from itertools import combinations

import numpy as np
import pytest

from starstack.milkyway import preserve_transient_residual


def signed(array):
    return array.astype(np.int32)


def test_signed_residual_is_exact_across_row_blocks():
    rng = np.random.default_rng(47021)
    shape = (257, 13, 3)  # Exercise both sides of the renderer's block boundary.
    background = rng.integers(1000, 63000, shape, dtype=np.uint16)
    source = (signed(background) + rng.integers(-700, 701, shape)).astype(np.uint16)
    proposal = np.clip(signed(background) + rng.integers(-1800, 1801, shape), 1, 65534).astype(np.uint16)
    before_source, before_background = source.copy(), background.copy()
    source.flags.writeable = background.flags.writeable = False
    output = np.empty_like(source)

    stats = preserve_transient_residual(source, background, proposal, output)

    expected_residual = signed(before_source) - signed(before_background)
    assert expected_residual.min() < 0 < expected_residual.max()
    np.testing.assert_array_equal(signed(output) - signed(proposal), expected_residual)
    np.testing.assert_array_equal(source, before_source)
    np.testing.assert_array_equal(background, before_background)
    assert np.any(output != source)
    assert stats['changed_pixels'] == np.any(output != source, axis=-1).sum()
    assert stats['residual_mismatched_channels'] == stats['new_endpoint_channels'] == 0
    assert stats['protection_mode'] == 'exact_signed_display_residual'


def test_flat_background_and_trail_corridor_receive_identical_change():
    background = np.full((24, 32, 3), 14000, np.uint16)
    source = background.copy()
    source[:, 14:17] += np.array([4500, 6000, 3500], np.uint16)
    proposal = background + np.array([1600, 1200, 800], np.uint16)
    output = np.empty_like(source)

    preserve_transient_residual(source, background, proposal, output)

    expected_change = np.broadcast_to(np.array([1600, 1200, 800]), source.shape)
    np.testing.assert_array_equal(signed(output) - signed(source), expected_change)
    np.testing.assert_array_equal(signed(proposal) - signed(background), expected_change)
    # The original bright corridor is retained; no dark band is made beside it.
    np.testing.assert_array_equal(signed(output) - signed(proposal), signed(source) - signed(background))
    np.testing.assert_array_equal(output[:, 13], output[:, 17])


@pytest.mark.parametrize('source_pixel,background_pixel,delta,expected', [
    ([65000, 40000, 1000], [64000, 39000, 2000], [1068, 600, -600], [534, 300, -300]),
    ([100, 30000, 40000], [1000, 31000, 39000], [-396, 800, 1200], [-99, 200, 300]),
])
def test_near_endpoint_uses_shared_headroom_for_all_channels(source_pixel, background_pixel, delta, expected):
    source = np.broadcast_to(np.array(source_pixel, np.uint16), (8, 8, 3)).copy()
    background = np.broadcast_to(np.array(background_pixel, np.uint16), source.shape).copy()
    proposal = (signed(background) + delta).astype(np.uint16)
    output = np.empty_like(source)

    stats = preserve_transient_residual(source, background, proposal, output)

    change = np.broadcast_to(np.array(expected), source.shape)
    np.testing.assert_array_equal(signed(output) - signed(source), change)
    np.testing.assert_array_equal(signed(proposal) - signed(background), change)
    assert output.min() > 0 and output.max() < 65535
    assert proposal.min() > 0 and proposal.max() < 65535
    assert stats['gamut_limited_pixels'] == 64


def test_existing_endpoint_in_either_input_holds_the_entire_pixel():
    source = np.full((8, 8, 3), 15000, np.uint16)
    background = np.full_like(source, 14000)
    source[0, 0, 0] = 0
    source[0, 1, 2] = 65535
    background[0, 2, 1] = 0
    background[0, 3, 0] = 65535
    proposal = np.full_like(source, 16000)
    output = np.empty_like(source)

    stats = preserve_transient_residual(source, background, proposal, output)

    np.testing.assert_array_equal(output[0, :4], source[0, :4])
    np.testing.assert_array_equal(proposal[0, :4], background[0, :4])
    np.testing.assert_array_equal(output[1:], np.full_like(output[1:], 17000))
    assert stats['gamut_limited_pixels'] == 4
    assert stats['new_endpoint_channels'] == 0


def test_zero_background_change_is_exact_identity_including_signed_endpoints():
    rng = np.random.default_rng(81)
    source = rng.integers(0, 65536, (8, 11, 3), dtype=np.uint16)
    background = rng.integers(0, 65536, source.shape, dtype=np.uint16)
    source[0, 0] = [0, 65535, 1]
    background[0, 0] = [65535, 0, 65534]
    proposal = background.copy()
    output = np.full_like(source, 123)

    stats = preserve_transient_residual(source, background, proposal, output)

    np.testing.assert_array_equal(output, source)
    np.testing.assert_array_equal(proposal, background)
    assert stats['identity'] and stats['changed_pixels'] == 0
    assert stats['max_abs_change'] == stats['gamut_limited_pixels'] == 0


def test_readonly_mapped_sources_remain_byte_identical(tmp_path):
    source_path, background_path = tmp_path/'source.npy', tmp_path/'background.npy'
    np.save(source_path, np.full((8, 9, 3), 16000, np.uint16))
    np.save(background_path, np.full((8, 9, 3), 14000, np.uint16))
    before = {path: hashlib.sha256(path.read_bytes()).digest() for path in (source_path, background_path)}
    source = np.load(source_path, mmap_mode='r')
    background = np.load(background_path, mmap_mode='r')
    try:
        assert not source.flags.writeable and not background.flags.writeable
        proposal = np.full(source.shape, 15000, np.uint16)
        output = np.empty_like(proposal)
        preserve_transient_residual(source, background, proposal, output)
        np.testing.assert_array_equal(output, np.full_like(output, 17000))
        np.testing.assert_array_equal(proposal, np.full_like(proposal, 15000))
    finally:
        source._mmap.close()
        background._mmap.close()
    assert before == {path: hashlib.sha256(path.read_bytes()).digest() for path in before}


@pytest.mark.parametrize('left,right', list(combinations(range(4), 2)))
def test_every_memory_alias_is_rejected_before_mutation(left, right):
    arrays = [np.full((8, 8, 3), 10000 + index*1000, np.uint16) for index in range(4)]
    arrays[right] = arrays[left].view()
    before = [array.copy() for array in arrays]

    with pytest.raises(ValueError, match='alias'):
        preserve_transient_residual(*arrays)

    for actual, expected in zip(arrays, before):
        np.testing.assert_array_equal(actual, expected)


def test_separate_mappings_of_one_backing_file_are_rejected(tmp_path):
    path = tmp_path/'same-output.npy'
    np.save(path, np.full((8, 8, 3), 15000, np.uint16))
    before = path.read_bytes()
    proposal = np.load(path, mmap_mode='r+')
    output = np.load(path, mmap_mode='r+')
    try:
        assert not np.shares_memory(proposal, output)
        with pytest.raises(ValueError, match='alias'):
            preserve_transient_residual(np.full(proposal.shape, 16000, np.uint16),
                                        np.full(proposal.shape, 14000, np.uint16), proposal, output)
    finally:
        proposal._mmap.close()
        output._mmap.close()
    assert path.read_bytes() == before


@pytest.mark.parametrize('kind', ['dtype', 'channels', 'shape', 'small', 'readonly_q', 'readonly_f', 'strided_q', 'strided_f'])
def test_invalid_array_boundary_is_rejected_before_mutation(kind):
    arrays = [np.full((8, 9, 3), 10000 + index*1000, np.uint16) for index in range(4)]
    if kind == 'dtype': arrays[1] = arrays[1].astype(np.float32)
    elif kind == 'channels': arrays[0] = arrays[0][..., :2].copy()
    elif kind == 'shape': arrays[1] = arrays[1][:, :8].copy()
    elif kind == 'small': arrays = [array[:7].copy() for array in arrays]
    elif kind.startswith('readonly'): arrays[2 if kind.endswith('q') else 3].flags.writeable = False
    elif kind.startswith('strided'):
        index = 2 if kind.endswith('q') else 3
        arrays[index] = arrays[index][:, ::-1]
    before = [array.copy() for array in arrays]

    with pytest.raises(ValueError):
        preserve_transient_residual(*arrays)

    for actual, expected in zip(arrays, before):
        np.testing.assert_array_equal(actual, expected)


def test_extreme_proposals_preserve_residual_and_only_reduce_requested_change():
    rng = np.random.default_rng(210921)
    values = np.array([0, 1, 2, 99, 32768, 65436, 65533, 65534, 65535], np.uint16)
    source = rng.choice(values, (27, 27, 3))
    background = rng.choice(values, source.shape)
    proposal = rng.integers(0, 65536, source.shape, dtype=np.uint16)
    requested = signed(proposal) - signed(background)
    output = np.empty_like(source)

    stats = preserve_transient_residual(source, background, proposal, output)

    change = signed(output) - signed(source)
    np.testing.assert_array_equal(signed(output) - signed(proposal), signed(source) - signed(background))
    np.testing.assert_array_equal(change, signed(proposal) - signed(background))
    assert np.all(np.abs(change) <= np.abs(requested))
    assert np.all((change == 0) | (np.sign(change) == np.sign(requested)))
    held = np.any((source == 0) | (source == 65535) | (background == 0) | (background == 65535), axis=-1)
    np.testing.assert_array_equal(change[held], np.zeros_like(change[held]))
    for old, new in ((source, output), (background, proposal)):
        assert not np.any(((new == 0) & (old > 0)) | ((new == 65535) & (old < 65535)))
    assert stats['gamut_limited_pixels'] == np.any(change != requested, axis=-1).sum()
    assert stats['changed_pixels'] == np.any(change != 0, axis=-1).sum()
