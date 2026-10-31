import numpy as np


def lookup_huesat(table, hue_degrees, saturation):
    if type(table) is not np.ndarray or table.dtype.type not in (np.float32, np.float64):
        raise ValueError("table must be an exact float32 or float64 numpy.ndarray")
    if table.ndim != 3 or table.shape[2] != 3:
        raise ValueError("table must have shape (H, S, 3)")
    H, S = int(table.shape[0]), int(table.shape[1])
    if H < 2 or S < 2:
        raise ValueError("table must have H >= 2 and S >= 2")
    if not np.isfinite(table).all():
        raise ValueError("table entries must be finite")
    if not np.all(table[..., 1:] >= 0):
        raise ValueError("table multipliers must be nonnegative")

    if type(hue_degrees) is not np.ndarray or type(saturation) is not np.ndarray:
        raise ValueError("hue_degrees and saturation must be exact numpy.ndarrays")
    if (hue_degrees.dtype.type not in (np.float32, np.float64) or
            saturation.dtype.type not in (np.float32, np.float64)):
        raise ValueError("hue_degrees and saturation must be float32 or float64")
    if hue_degrees.shape != saturation.shape:
        raise ValueError("hue_degrees and saturation must have identical shapes")
    if not np.isfinite(hue_degrees).all() or not np.isfinite(saturation).all():
        raise ValueError("hue_degrees and saturation must be finite")
    if np.any(saturation < 0) or np.any(saturation > 1):
        raise ValueError("saturation must be in [0, 1]")

    t = table.astype(np.float64, copy=False)
    hue = hue_degrees.astype(np.float64, copy=False)
    sat = saturation.astype(np.float64, copy=False)

    with np.errstate(over="ignore", invalid="ignore"):
        wrapped = np.remainder(hue, 360.0)
        h = (wrapped * H) / 360.0
        h = np.where(h >= H, 0.0, h)

        i0 = np.floor(h).astype(np.intp)
        i1 = (i0 + 1) % H
        fh = h - i0

        s = sat * (S - 1)
        j0 = np.minimum(np.floor(s).astype(np.intp), S - 2)
        j1 = j0 + 1
        fs = s - j0

        v00 = t[i0, j0]
        v01 = t[i0, j1]
        v10 = t[i1, j0]
        v11 = t[i1, j1]

        w00 = (1.0 - fh) * (1.0 - fs)
        w01 = (1.0 - fh) * fs
        w10 = fh * (1.0 - fs)
        w11 = fh * fs

        out = (v00 * w00[..., None] + v01 * w01[..., None] +
               v10 * w10[..., None] + v11 * w11[..., None])

    if not np.isfinite(out).all():
        raise ValueError("interpolation produced nonfinite output")
    return np.ascontiguousarray(out, dtype=np.float64)
