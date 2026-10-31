"""star_layer_audit.py -- independent signed star-layer contract audit.

audit_layers() recomputes, from raw arrays using bounded int64 tiles, whether
    source == diffuse + residual
holds exactly at code level and whether any diffuse change or any nonzero
residual falls outside the independently declared `allowed` edit scope.
The returned booleans are arithmetic facts only: they never imply physical
recovery, a valid star extraction, or an approved edit.  No thresholds,
percentiles, approvals, or project imports are used.
"""

import numpy as np

_VERSION = "signed-star-layer-audit-1"
_GROUPS = ("all", "allowed", "outside")
_TILE_PIXELS = 65536    # max spatial pixels per row x column block
_CODE_LIMIT = 65535     # uint16 codes span 0..65535 -> residuals in [-65535, 65535]

_KEYS = (
    "pixel_count", "nonzero_residual_pixels", "negative_residual_channels",
    "positive_residual_channels", "residual_min", "residual_max",
    "residual_mean_code", "roundtrip_mismatch_channels",
    "decomposition_mismatch_channels", "diffuse_changed_channels",
    "squared_residual_sum", "residual_sum",
)
_NONE_KEYS = frozenset(("residual_min", "residual_max", "residual_mean_code"))


def _checked(x, dtype, name):
    """Cheap validation; returns a base-class view (never a full-image copy)."""
    if not isinstance(x, np.ndarray):
        raise ValueError(f"{name} must be a numpy ndarray")
    want = np.dtype(dtype)
    if x.dtype.kind != want.kind or x.dtype.itemsize != want.itemsize:
        raise ValueError(f"{name} must have dtype {want.name}")
    return np.asarray(x)  # view only; readonly/memmap/strided inputs are untouched


def _empty_stats():
    return {k: (None if k in _NONE_KEYS else 0) for k in _KEYS}


def audit_layers(source, diffuse, residual, allowed, *, domain, rows=128) -> dict:
    """Audit a signed star-layer decomposition with bounded row/column blocks.

    `rows` is the row-block height; column blocks are sized so every tile is at
    most 65536 spatial pixels.  Invalid arguments raise ValueError.  Inputs are
    only read (readonly/memmap/strided arrays are fine) and never mutated.
    """
    if type(domain) is not str or domain != "display-sRGB":
        raise ValueError("domain must be the builtin str 'display-sRGB'")
    if type(rows) is not int or not 1 <= rows <= 512:
        raise ValueError("rows must be a builtin int in 1..512")

    src = _checked(source, np.uint16, "source")
    dif = _checked(diffuse, np.uint16, "diffuse")
    res = _checked(residual, np.int32, "residual")
    ins = _checked(allowed, np.bool_, "allowed")

    if src.ndim != 3 or src.shape[2] != 3 or src.shape[0] < 1 or src.shape[1] < 1:
        raise ValueError("source must be a nonempty H,W,3 uint16 array")
    if dif.shape != src.shape:
        raise ValueError("diffuse must match source shape")
    if res.shape != src.shape:
        raise ValueError("residual must match source shape")
    H, W = int(src.shape[0]), int(src.shape[1])
    if ins.shape != (H, W):
        raise ValueError("allowed must be bool with shape (H, W) matching source")

    stats = {g: _empty_stats() for g in _GROUPS}
    row_block = min(rows, H)
    col_block = min(W, _TILE_PIXELS // row_block)   # tile <= 65536 spatial pixels

    for r0 in range(0, H, row_block):
        r1 = min(r0 + row_block, H)
        for c0 in range(0, W, col_block):
            c1 = min(c0 + col_block, W)

            # Bounded int64 tile copies; the caller's arrays are only read.
            s64 = src[r0:r1, c0:c1, :].astype(np.int64)
            d64 = dif[r0:r1, c0:c1, :].astype(np.int64)
            raw = res[r0:r1, c0:c1, :]
            # Full-scan range check, including pixels outside `allowed`.
            if int(raw.min()) < -_CODE_LIMIT or int(raw.max()) > _CODE_LIMIT:
                raise ValueError("residual codes must lie in [-65535, 65535]")
            r64 = raw.astype(np.int64)

            inside = ins[r0:r1, c0:c1]      # bool view of the declared scope
            outside = ~inside               # bounded complement copy
            roundtrip = (d64 + r64) != s64
            decomposed = r64 != (s64 - d64)
            changed = d64 != s64
            nonzero = r64 != 0
            negative = r64 < 0
            positive = r64 > 0
            any_nonzero = nonzero.any(axis=2)   # per spatial pixel

            for name, pixel_mask in (("all", None), ("allowed", inside),
                                     ("outside", outside)):
                st = stats[name]
                if pixel_mask is None:
                    st["pixel_count"] += (r1 - r0) * (c1 - c0)
                    st["nonzero_residual_pixels"] += int(any_nonzero.sum())
                    st["negative_residual_channels"] += int(negative.sum())
                    st["positive_residual_channels"] += int(positive.sum())
                    st["roundtrip_mismatch_channels"] += int(roundtrip.sum())
                    st["decomposition_mismatch_channels"] += int(decomposed.sum())
                    st["diffuse_changed_channels"] += int(changed.sum())
                    vals = r64.reshape(-1)
                else:
                    count = int(pixel_mask.sum())
                    if count == 0:
                        continue                # empty group stays 0/None
                    m3 = pixel_mask[..., None]
                    st["pixel_count"] += count
                    st["nonzero_residual_pixels"] += int(
                        (any_nonzero & pixel_mask).sum())
                    st["negative_residual_channels"] += int((negative & m3).sum())
                    st["positive_residual_channels"] += int((positive & m3).sum())
                    st["roundtrip_mismatch_channels"] += int((roundtrip & m3).sum())
                    st["decomposition_mismatch_channels"] += int(
                        (decomposed & m3).sum())
                    st["diffuse_changed_channels"] += int((changed & m3).sum())
                    vals = r64[pixel_mask].reshape(-1)

                # A tile holds <= 196608 codes in [-65535, 65535], so per-tile
                # int64 sums cannot overflow; Python-int accumulation is exact.
                st["residual_sum"] += int(vals.sum(dtype=np.int64))
                st["squared_residual_sum"] += int(
                    np.multiply(vals, vals, dtype=np.int64).sum(dtype=np.int64))
                vmin, vmax = int(vals.min()), int(vals.max())
                if st["residual_min"] is None or vmin < st["residual_min"]:
                    st["residual_min"] = vmin   # zeros are ordinary codes
                if st["residual_max"] is None or vmax > st["residual_max"]:
                    st["residual_max"] = vmax

    for st in stats.values():
        if st["pixel_count"] > 0:
            # Mean over all residual channel codes (3 codes per spatial pixel),
            # computed once from exact totals so block sizes cannot shift it.
            st["residual_mean_code"] = st["residual_sum"] / (3 * st["pixel_count"])

    return {
        "version": _VERSION,
        "domain": domain,
        "shape": [H, W, 3],
        "groups": stats,
        "exact_roundtrip": bool(all(
            stats[g]["roundtrip_mismatch_channels"] == 0 for g in _GROUPS)),
        "changes_within_allowed": bool(
            stats["outside"]["diffuse_changed_channels"] == 0
            and stats["outside"]["nonzero_residual_pixels"] == 0),
    }
