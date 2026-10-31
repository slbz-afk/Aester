"""Lossless group orientation after demosaicing, before masks/registration.

Angles are clockwise in ordinary TOP-DOWN image coordinates. Siril 1.4.4
explicitly decodes RAW with user_flip=0, so capture orientation is applied once
here, to RGB pixels, never to the CFA mosaic or original files.
"""
from __future__ import annotations

import os
from pathlib import Path
import re
import uuid

import numpy as np

from . import hashing


def normalize_rotation(value):
    if type(value) is not int or value not in (0, 90, 180, 270):
        raise ValueError('图像方向须为顺时针 0、90、180 或 270 度')
    return value


def source_orientation(path):
    """Read capture metadata only; unavailable/mirrored metadata never guessed."""
    path = Path(path)
    if path.suffix.lower() in ('.fit', '.fits', '.fts'):
        return {'rotation': 0, 'status': 'not_applicable', 'source': 'fits_row_order_only'}
    try:
        if path.suffix.lower() == '.dng':
            import tifffile
            with tifffile.TiffFile(path) as image:
                tag = image.pages[0].tags.get('Orientation')
                if tag is not None:
                    value = int(tag.value)
                    mapping = {1: 0, 3: 180, 6: 90, 8: 270}
                    return {'rotation': mapping.get(value, 0),
                            'status': 'known' if value in mapping else 'manual_required',
                            'source': 'dng_orientation', 'value': value}
        import rawpy
        # Header access only: no unpack or postprocess is needed here.
        with rawpy.RawPy() as raw:
            raw.open_file(str(path))
            value = int(raw.sizes.flip)
        mapping = {0: 0, 3: 180, 5: 270, 6: 90}
        return {'rotation': mapping.get(value, 0),
                'status': 'known' if value in mapping else 'manual_required',
                'source': 'libraw_flip', 'value': value}
    except Exception as exc:
        return {'rotation': 0, 'status': 'unavailable', 'source': 'capture_metadata',
                'reason': str(exc)}


def source_rotation(path):
    return source_orientation(path)['rotation']


def rotate_array(array, image_rotation, *, bottom_up=False, axes=(0, 1)):
    """Permutation only: retain signed values, dtype and all channel samples."""
    angle = normalize_rotation(image_rotation)
    steps = angle // 90 if bottom_up else -angle // 90
    return np.rot90(array, steps, axes=axes) if angle else array


# Converted camera frames normally have no WCS. A previously plate-solved FITS
# may have one; invalidate its pixel-coordinate solution before registration.
_WCS_PIXEL = re.compile(r'^(?:CTYPE|CUNIT|CRPIX|CRVAL|CDELT|CROTA)[12][A-Z]?$|'
                        r'^(?:CD|PC)[12]_[12][A-Z]?$|^(?:PV|PS)[12]_\d+[A-Z]?$|'
                        r'^(?:A|B|AP|BP)_(?:ORDER|\d+_\d+)$|'
                        r'^(?:LONPOLE|LATPOLE|WCSAXES|WCSNAME)[A-Z]?$')


def rotate_fits(source, destination, image_rotation, *, check=None):
    """Write a rotated copy (or atomically replace an owned working FITS).

    No interpolation, no rescaling, and no FITS row-order guessing beyond the
    product's existing missing-ROWORDER = BOTTOM-UP convention.
    """
    from astropy.io import fits
    angle = normalize_rotation(image_rotation)
    source, destination = Path(source), Path(destination)
    temp = destination.with_name('.rotate-' + uuid.uuid4().hex + '.fit')
    if check:
        check()
    try:
        with fits.open(source, memmap=True, do_not_scale_image_data=True) as hdus:
            if len(hdus) != 1 or hdus[0].data is None:
                raise ValueError('方向调整需要单幅已解码 FITS')
            data, header = hdus[0].data, hdus[0].header.copy()
            if data.ndim not in (2, 3) or (data.ndim == 3 and data.shape[0] != 3):
                raise ValueError('方向调整需要已去马赛克 RGB 或灰度图像')
            order = header.get('ROWORDER', 'BOTTOM-UP')
            if order not in ('TOP-DOWN', 'BOTTOM-UP'):
                raise ValueError('方向调整遇到未知 FITS 行方向')
            pixels = rotate_array(data, angle, bottom_up=order == 'BOTTOM-UP', axes=(-2, -1))
            if angle:
                for key in list(header):
                    if _WCS_PIXEL.match(key) or key in ('CHECKSUM', 'DATASUM'):
                        del header[key]
                header.add_history(f'Aester RGB clockwise rotation {angle} deg; pixel permutation, no interpolation.')
                if angle in (90, 270):
                    for x_key, y_key in (('XPIXSZ', 'YPIXSZ'), ('XBINNING', 'YBINNING')):
                        if x_key in header and y_key in header:
                            header[x_key], header[y_key] = header[y_key], header[x_key]
                        else:
                            header.remove(x_key, ignore_missing=True)
                            header.remove(y_key, ignore_missing=True)
            hdu = fits.PrimaryHDU(data=np.ascontiguousarray(pixels), header=header)
            # Keep integer storage scaling verbatim, not a float round trip.
            for key in ('BSCALE', 'BZERO'):
                if key in header:
                    hdu.header[key] = header[key]
            hdu.writeto(temp, overwrite=False)
            del pixels, data, hdu
        if check:
            check()
        with hashing.writing(destination):   # may replace a file the run ledger knows (in place rotation)
            os.replace(temp, destination)
    finally:
        temp.unlink(missing_ok=True)
