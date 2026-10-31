"""Independent exposure-hole truth through FITS, layer construction and warp."""
import numpy as np

from starstack import transient_trails as tt
from starstack.synthetic_starfield import write_fits_rgb32


class FractionalTranslation:
    def apply(self, points):
        return np.asarray(points) + np.array([.25, .375])

    def inverse(self, points):
        return np.asarray(points) - np.array([.25, .375])


def test_real_fits_single_missing_channel_cannot_borrow_master_or_other_rgb(tmp_path):
    shape = (128, 192)
    master = np.ones((3, *shape), np.float32) * .25
    frame = master.copy()
    # Constant signed residual, including a negative channel. Green is absent
    # on a known vertical hole; the other channels and stack remain present.
    frame[:, 62:67, 35:155] += np.array([.125, .0625, -.0625], np.float32)[:, None, None]
    frame[1, :, 91:94] = 0
    fp, mp = tmp_path/'frame.fit', tmp_path/'master.fit'
    write_fits_rgb32(fp, frame)
    write_fits_rgb32(mp, master)
    candidate = {'measurement': {'x0':35., 'y0':64., 'x1':154., 'y1':64., 'half_width_px':2}}
    box = tt.layer_box(fp, mp, candidate, [1]*3, [.25]*3, [.25]*3)
    x, y = box['x0'], box['y0']
    assert box['source_support']['unsupported_pixels'] > 0
    assert np.all(box['data'][:, :, 91-x:94-x] == 0)
    assert box['data'][2, 64-y, 60-x] == -.0625
    assert box['grid_shape'] == shape
    native = tt.warp_box_to_native(box, FractionalTranslation(), shape)
    # Cubic dependency [floor(s)-1, floor(s)+2] touches missing x=91..93.
    # At native integer x=89..94 the interpolation must therefore be withheld.
    nx, ny = native['x0'], native['y0']
    assert np.all(native['data'][:, 62-ny:66-ny, 89-nx:95-nx] == 0)
    assert native['native_support']['nonzero_pixels_removed'] > 0
    assert native['data'][2, 64-ny, 60-nx] < 0
    # The known clean plateau is analytically unchanged by a normalized cubic.
    np.testing.assert_array_equal(native['data'][:, 64-ny, 60-nx], [.125, .0625, -.0625])


def test_native_frame_boundary_requires_real_observation_but_identity_does_not(tmp_path):
    shape = (40, 80)
    master = np.full((3, *shape), .25, np.float32)
    frame = master.copy()
    frame[:, 19:22, :] += .125
    fp, mp = tmp_path/'frame.fit', tmp_path/'master.fit'
    write_fits_rgb32(fp, frame)
    write_fits_rgb32(mp, master)
    c = {'measurement': {'x0':0., 'y0':20., 'x1':79., 'y1':20., 'half_width_px':1}}
    box = tt.layer_box(fp, mp, c, [1]*3, [.25]*3, [.25]*3)
    identity = tt.warp_box_to_native(box, None, shape)
    shifted = tt.warp_box_to_native(box, FractionalTranslation(), shape)
    assert identity['data'][0, 20-identity['y0'], 0] == .125
    assert shifted['data'][0, 20-shifted['y0'], 0] == 0
    assert shifted['native_support']['nonzero_pixels_removed'] > 0
