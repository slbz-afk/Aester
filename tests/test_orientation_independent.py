"""Independent pixel/metadata contracts for whole-selection orientation.

Expected permutations are written as source-to-destination pixel assignments,
not via the production helper or numpy's rot90 implementation.
"""
from pathlib import Path
import hashlib

import numpy as np
import pytest
import tifffile
from astropy.io import fits

from starstack import orientation as subject


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _expected(top, angle):
    height, width = top.shape[:2]
    shape = (width, height) if angle in (90, 270) else (height, width)
    out = np.empty((*shape, *top.shape[2:]), dtype=top.dtype)
    for y in range(height):
        for x in range(width):
            if angle == 90:
                target = (x, height - 1 - y)
            elif angle == 180:
                target = (height - 1 - y, width - 1 - x)
            elif angle == 270:
                target = (width - 1 - x, y)
            else:
                target = (y, x)
            out[target] = top[y, x]
    return out


@pytest.mark.parametrize('angle', [0, 90, 180, 270])
@pytest.mark.parametrize('bottom_up', [False, True])
@pytest.mark.parametrize('dtype', [np.uint16, np.int16, np.float32])
def test_non_square_rgb_permutation_and_channel_order(angle, bottom_up, dtype):
    top = np.arange(4 * 7 * 3).reshape(4, 7, 3).astype(dtype)
    stored = top[::-1] if bottom_up else top
    before = stored.copy()
    actual = subject.rotate_array(stored, angle, bottom_up=bottom_up)
    expected = _expected(top, angle)
    if bottom_up:
        expected = expected[::-1]
    assert actual.dtype == stored.dtype
    assert actual.shape == expected.shape
    assert np.array_equal(actual, expected)
    assert np.array_equal(stored, before)


@pytest.mark.parametrize('angle', [0, 90, 180, 270])
@pytest.mark.parametrize('roworder', ['TOP-DOWN', 'BOTTOM-UP', None])
def test_float_fits_preserves_bits_negative_zero_and_source(tmp_path, angle, roworder):
    # Includes negative zero, subnormal, finite max, +/- infinity and a NaN
    # payload. The geometry helper is not allowed to modify sample bits.
    pattern = np.array([0, 0x80000000, 0x3f800000, 0xbf800000, 1,
                        0x7f7fffff, 0x7f800000, 0xff800000, 0x7fc01234], np.uint32)
    top = np.resize(pattern, (3, 5, 3)).view(np.float32)
    stored = top if roworder == 'TOP-DOWN' else top[::-1]
    header = fits.Header({'EXPTIME': 12.5, 'INSTRUME': 'Independent fixture'})
    if roworder:
        header['ROWORDER'] = roworder
    source, target = tmp_path/'original.fit', tmp_path/'rotated.fit'
    fits.writeto(source, np.moveaxis(stored, -1, 0), header)
    before = _sha(source)
    subject.rotate_fits(source, target, angle)
    with fits.open(target, do_not_scale_image_data=True) as hdus:
        actual = np.moveaxis(hdus[0].data, 0, -1)
        if roworder != 'TOP-DOWN':
            actual = actual[::-1]
        expected = _expected(top, angle)
        assert np.array_equal(actual.view('>u4'), expected.astype('>f4').view('>u4'))
        assert hdus[0].header['EXPTIME'] == 12.5
        assert hdus[0].header['INSTRUME'] == 'Independent fixture'
        assert hdus[0].header.get('ROWORDER') == roworder
        assert hdus[0].header['NAXIS1'] == expected.shape[1]
        assert hdus[0].header['NAXIS2'] == expected.shape[0]
    assert _sha(source) == before
    assert not list(tmp_path.glob('.rotate-*'))


@pytest.mark.parametrize('angle', [90, 180, 270])
@pytest.mark.parametrize('roworder', ['TOP-DOWN', 'BOTTOM-UP'])
@pytest.mark.parametrize('bscale,bzero', [(1, 32768), (.125, -32.5)])
def test_integer_storage_and_physical_scaling_preserved(tmp_path, angle, roworder, bscale, bzero):
    top = (np.arange(5 * 8 * 3).reshape(5, 8, 3) * 537 - 32000).astype(np.int16)
    top[0, 0] = [-32768, 32767, 0]
    stored = top if roworder == 'TOP-DOWN' else top[::-1]
    header = fits.Header({'ROWORDER': roworder})
    hdu = fits.PrimaryHDU(np.moveaxis(stored, -1, 0), header=header)
    hdu.header['BSCALE'], hdu.header['BZERO'] = bscale, bzero
    source, target = tmp_path/'integer.fit', tmp_path/'rotated.fit'
    hdu.writeto(source)
    before = _sha(source)
    subject.rotate_fits(source, target, angle)
    expected = _expected(top, angle)
    with fits.open(target, do_not_scale_image_data=True) as hdus:
        actual = np.moveaxis(hdus[0].data, 0, -1)
        if roworder == 'BOTTOM-UP':
            actual = actual[::-1]
        assert actual.dtype == np.dtype('>i2')
        assert np.array_equal(actual, expected)
        assert hdus[0].header['BITPIX'] == 16
        assert hdus[0].header['BSCALE'] == bscale
        assert hdus[0].header['BZERO'] == bzero
    with fits.open(target, memmap=False) as hdus:
        physical = np.moveaxis(hdus[0].data, 0, -1)
        if roworder == 'BOTTOM-UP':
            physical = physical[::-1]
        assert np.array_equal(physical, expected.astype(np.float64) * bscale + bzero)
    assert _sha(source) == before


def test_wcs_and_checksums_invalidated_without_losing_capture_metadata(tmp_path):
    header = fits.Header({'ROWORDER': 'BOTTOM-UP', 'EXPTIME': 30., 'ISOSPEED': 1600,
                          'FILTER': 'L', 'INSTRUME': 'Camera', 'DATE-OBS': '2026-09-25T00:00:00'})
    geometry = {'CTYPE1': 'RA---TAN-SIP', 'CTYPE2': 'DEC--TAN-SIP', 'CUNIT1': 'deg',
                'CRVAL1': 12., 'CRPIX2': 5., 'CD1_2': .002, 'PC2_1': .01,
                'CDELT1': .001, 'CROTA2': 45., 'PV1_0': 1., 'PS2_0': 'x',
                'WCSAXES': 2, 'WCSNAME': 'test', 'LONPOLE': 180., 'LATPOLE': 90.,
                'A_ORDER': 2, 'A_2_0': .01, 'B_0_2': .02, 'AP_ORDER': 2, 'BP_1_1': .01,
                'CTYPE1A': 'RA---TAN', 'CRPIX1A': 2., 'CD1_1A': .1, 'WCSNAMEA': 'alternate'}
    header.update(geometry)
    source, target = tmp_path/'wcs.fit', tmp_path/'result.fit'
    fits.writeto(source, np.ones((3, 4, 7), np.float32), header, checksum=True)
    subject.rotate_fits(source, target, 90)
    result = fits.getheader(target)
    assert not any(key in result for key in geometry)
    assert 'CHECKSUM' not in result and 'DATASUM' not in result
    for key in ('EXPTIME', 'ISOSPEED', 'FILTER', 'INSTRUME', 'DATE-OBS', 'ROWORDER'):
        assert result[key] == header[key]
    assert 'clockwise rotation 90' in str(result['HISTORY'])


def test_four_inplace_quarter_turns_restore_storage_exactly(tmp_path):
    top = np.arange(3 * 4 * 7, dtype=np.float32).reshape(3, 4, 7)
    top[0, 0, 0] = -0.
    target = tmp_path/'owned.fit'
    fits.writeto(target, top, fits.Header({'ROWORDER': 'BOTTOM-UP'}))
    for _ in range(4):
        subject.rotate_fits(target, target, 90)
    actual = fits.getdata(target)
    assert np.array_equal(actual.view('>u4'), top.astype('>f4').view('>u4'))


@pytest.mark.parametrize('angle', [0, 90, 180, 270])
def test_non_square_pixel_and_binning_axes_follow_geometry(tmp_path, angle):
    source, target = tmp_path/'scale.fit', tmp_path/'rotated.fit'
    header = fits.Header({'XPIXSZ': 3.2, 'YPIXSZ': 4.8, 'XBINNING': 1, 'YBINNING': 2})
    fits.writeto(source, np.ones((3, 4, 7), np.float32), header)
    subject.rotate_fits(source, target, angle)
    actual = fits.getheader(target)
    expected = (4.8, 3.2, 2, 1) if angle in (90, 270) else (3.2, 4.8, 1, 2)
    assert tuple(actual[k] for k in ('XPIXSZ', 'YPIXSZ', 'XBINNING', 'YBINNING')) == expected


@pytest.mark.parametrize('key', ['XPIXSZ', 'YPIXSZ', 'XBINNING', 'YBINNING'])
def test_incomplete_axis_metadata_not_mislabelled_on_quarter_turn(tmp_path, key):
    source, target = tmp_path/'scale.fit', tmp_path/'rotated.fit'
    fits.writeto(source, np.ones((3, 4, 7), np.float32), fits.Header({key: 2}))
    subject.rotate_fits(source, target, 270)
    actual = fits.getheader(target)
    assert not any(k in actual for k in ('XPIXSZ', 'YPIXSZ', 'XBINNING', 'YBINNING'))


@pytest.mark.parametrize('value', [None, True, False, -90, 360, 45, 90., '90', np.int64(90)])
def test_rotation_admission_is_explicit(value):
    with pytest.raises(ValueError):
        subject.normalize_rotation(value)


@pytest.mark.parametrize('tag,expected,status', [(1, 0, 'known'), (3, 180, 'known'),
    (6, 90, 'known'), (8, 270, 'known'), (2, 0, 'manual_required'),
    (4, 0, 'manual_required'), (5, 0, 'manual_required'), (7, 0, 'manual_required')])
def test_dng_orientation_never_confuses_mirroring_with_rotation(tmp_path, tag, expected, status):
    p = tmp_path/'header.dng'
    tifffile.imwrite(p, np.zeros((32, 48), np.uint16),
                     extratags=[(274, 'H', 1, tag, False)])
    before = _sha(p)
    result = subject.source_orientation(p)
    assert result['rotation'] == expected and result['status'] == status
    assert result['source'] == 'dng_orientation'
    assert _sha(p) == before


@pytest.mark.parametrize('flip,expected,status', [(0, 0, 'known'), (3, 180, 'known'),
    (5, 270, 'known'), (6, 90, 'known'), (1, 0, 'manual_required'), (2, 0, 'manual_required')])
def test_native_raw_header_only_and_flip_mapping(monkeypatch, tmp_path, flip, expected, status):
    import rawpy
    opened = []
    class HeaderOnly:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def open_file(self, path):
            opened.append(path)
        @property
        def sizes(self):
            class Sizes:
                pass
            result = Sizes()
            result.flip = flip
            return result
        def unpack(self):
            pytest.fail('orientation must not decode pixels')
    monkeypatch.setattr(rawpy, 'RawPy', HeaderOnly)
    monkeypatch.setattr(rawpy, 'imread', lambda *a: pytest.fail('must use header-only open_file'))
    p = tmp_path/'native.nef'
    p.write_bytes(b'not read by the controlled header reader')
    before = _sha(p)
    result = subject.source_orientation(p)
    assert result['rotation'] == expected and result['status'] == status
    assert opened == [str(p)] and _sha(p) == before


def test_missing_raw_and_fits_do_not_guess_orientation(tmp_path):
    assert subject.source_orientation(tmp_path/'missing.nef')['status'] == 'unavailable'
    result = subject.source_orientation(tmp_path/'missing.fits')
    assert result['rotation'] == 0 and result['status'] == 'not_applicable'


@pytest.mark.parametrize('failure', ['before', 'after'])
def test_cancel_preserves_existing_destination_and_removes_temporary(tmp_path, failure):
    source, destination = tmp_path/'source.fit', tmp_path/'destination.fit'
    fits.writeto(source, np.arange(3 * 4 * 7, dtype=np.float32).reshape(3, 4, 7))
    destination.write_bytes(b'existing destination sentinel')
    source_hash, destination_hash = _sha(source), _sha(destination)
    calls = []
    def check():
        calls.append(True)
        if len(calls) == (1 if failure == 'before' else 2):
            raise InterruptedError('independent cancellation')
    with pytest.raises(InterruptedError):
        subject.rotate_fits(source, destination, 90, check=check)
    assert _sha(source) == source_hash and _sha(destination) == destination_hash
    assert not list(tmp_path.glob('.rotate-*'))


def test_unknown_roworder_rejected_without_destination(tmp_path):
    source = tmp_path/'source.fit'
    fits.writeto(source, np.ones((3, 4, 7), np.float32), fits.Header({'ROWORDER': 'SIDEWAYS'}))
    with pytest.raises(ValueError):
        subject.rotate_fits(source, tmp_path/'result.fit', 90)
    assert not (tmp_path/'result.fit').exists()
    assert not list(tmp_path.glob('.rotate-*'))
