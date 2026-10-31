"""Dense-field starless metrics (metrics-2), docs/STARLESS_METRICS2_20260923.md.

metrics-1 measures only isolated stars and star-free sky, which in a Milky Way core is
almost nothing. metrics-2 stratifies the evaluation region by local star density and
reports, per stratum, per-star statistics that need no isolation (small aperture, block
background, no annulus) and block-level high-pass energy ratios that need no detection
at all. NumPy only. A supplement to the frozen baseline, never a replacement.
"""
from __future__ import annotations
import numpy as np
from . import photographic as p
from . import starless_metrics as m1

VERSION = "starless-metrics-2"
STRATA = ("sparse", "medium", "dense")
PARAMS = {
    "block": 64,                     # density / background / energy block
    "density_thresholds": (3.0, 10.0),   # candidates per 1000 evaluation pixels: sparse < 3, medium 3–10, dense ≥ 10
    "min_block_fraction": 0.25,      # blocks with less evaluation-region coverage are not stratified
    "detect_k": 6.0, "exclude_k": 4.0,
    "moment_radius": 2, "moment_min": 0.3, "moment_max": 2.0,
    "aperture": 2, "inner": 1, "chroma_reference": 8,
    "snr_min": 10.0, "margin": 17, "max_stars": 20000,
    "sigma_mask_radius": 2, "lowpass": 16,
}


def block_index(h, w, block):
    """Per-pixel block id (row-major) for an h×w image."""
    by = np.arange(h) // block
    bx = np.arange(w) // block
    nx = -(-w // block)
    return by[:, None] * nx + bx[None, :], (-(-h // block), nx)


def block_sums(values, ids, nblocks, mask=None):
    """Sum of ``values`` per block id (float64), optionally restricted to ``mask``."""
    v = values.astype(np.float64, copy=False)
    if mask is not None:
        return np.bincount(ids[mask], weights=v[mask], minlength=nblocks)
    return np.bincount(ids.ravel(), weights=v.ravel(), minlength=nblocks)


def stratify(exclusion, region, params=PARAMS):
    """Density per block (candidates per 1000 evaluation pixels) and the stratum of each block / pixel."""
    h, w = region.shape
    ids, (ny, nx) = block_index(h, w, params["block"])
    nblocks = ny * nx
    region_px = np.bincount(ids[region], minlength=nblocks).astype(np.float64)
    counts = np.bincount(ids[exclusion & region], minlength=nblocks).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        density = np.where(region_px > 0, counts / region_px * 1000.0, np.nan)
    full = float(params["block"] ** 2)
    assigned = region_px >= params["min_block_fraction"] * full
    lo, hi = params["density_thresholds"]
    stratum = np.full(nblocks, -1, np.int8)       # -1: not stratified
    stratum[assigned & (density < lo)] = 0
    stratum[assigned & (density >= lo) & (density < hi)] = 1
    stratum[assigned & (density >= hi)] = 2
    pixel_stratum = stratum[ids]
    pixel_stratum[~region] = -1
    return {"ids": ids, "nblocks": nblocks, "shape": (ny, nx), "region_px": region_px, "density": density,
            "block_stratum": stratum, "pixel_stratum": pixel_stratum}


def noise_sigma_masked(L, block, mask, minimum=200):
    """Local robust σ like metrics-1, but the MAD ignores ``mask`` (star pixels) whenever a block
    keeps at least ``minimum`` unmasked pixels; dense fields otherwise inflate σ with star light."""
    hp = m1.highpass3(L)
    h, w = L.shape
    ny, nx = -(-h // block), -(-w // block)
    small = np.empty((ny, nx), np.float32)
    for by in range(ny):
        for bx in range(nx):
            tile = hp[by * block:(by + 1) * block, bx * block:(bx + 1) * block]
            keep = ~mask[by * block:(by + 1) * block, bx * block:(bx + 1) * block]
            values = tile[keep] if keep.sum() >= minimum else tile
            small[by, bx] = np.median(np.abs(values - np.median(values)))
    small = np.maximum(small * np.float32(m1._MAD_TO_SIGMA), np.float32(1 / 65535))
    return np.maximum(m1._upsample(small, block, (h, w)), np.float32(1 / 65535))


def dense_candidates(L, sigma, background, region, source, params=PARAMS):
    """All compact, non-saturated ≥ detect_k σ local maxima inside the margin — no isolation."""
    hp = L - background
    peaks = m1.local_maxima(hp)
    margin = params["margin"]
    inside = m1.erode(region, margin)
    inside[:margin] = inside[-margin:] = False
    inside[:, :margin] = inside[:, -margin:] = False
    candidates = peaks & (hp >= params["detect_k"] * sigma) & inside
    ys, xs = np.nonzero(candidates)
    if ys.size == 0:
        return ys, xs, np.zeros(0, bool)
    maxchannel = source.max(-1)
    satmax = p.neighbour_extreme(maxchannel, 1, maximum=True)
    saturated = satmax[ys, xs] >= m1.SAT_CODE
    r = params["moment_radius"]
    dy, dx, _ = m1._offsets(r)
    patch = np.clip(m1._gather(hp, ys, xs, r), 0, None)
    total = patch.sum((1, 2)) + np.float32(1e-12)
    mx = (patch * dx[None]).sum((1, 2)) / total
    my = (patch * dy[None]).sum((1, 2)) / total
    vx = (patch * (dx[None] - mx[:, None, None]) ** 2).sum((1, 2)) / total
    vy = (patch * (dy[None] - my[:, None, None]) ** 2).sum((1, 2)) / total
    moment = np.sqrt((vx + vy) / 2)
    compact = (moment >= params["moment_min"]) & (moment <= params["moment_max"])
    keep = compact & ~saturated
    return ys[keep], xs[keep], keep


def _star_statistics(ys, xs, planes, sigma, params=PARAMS):
    ap_r, in_r = params["aperture"], params["inner"]
    _, _, dist = m1._offsets(ap_r)
    ap = dist <= ap_r
    inn = dist <= in_r
    LS = m1._gather(planes["L_S"], ys, xs, ap_r)
    LD = m1._gather(planes["L_D"], ys, xs, ap_r)
    BS = m1._gather(planes["bg_S"], ys, xs, ap_r)
    BD = m1._gather(planes["bg_D"], ys, xs, ap_r)
    s = sigma[ys, xs].astype(np.float64)
    FS = (LS[:, ap] - BS[:, ap]).sum(1).astype(np.float64)
    FD = (LD[:, ap] - BD[:, ap]).sum(1).astype(np.float64)
    snr = FS / (s * np.sqrt(ap.sum()))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(FS > 0, FD / FS, np.nan)
    offset = (LD[:, inn].mean(1) - BD[:, inn].mean(1)) / s
    shift = np.zeros(ys.size)
    for c in range(3):
        CH = m1._gather(planes["chroma_D"][c], ys, xs, ap_r)
        ref = planes["chroma_ref_D"][c][ys, xs].astype(np.float64)
        shift = np.maximum(shift, np.abs(CH[:, inn].mean(1) - ref) / s)
    return {"snr": snr, "residual_ratio": ratio, "luminance_offset": offset, "colour_shift": shift}


def evaluate2(source, estimate, region, *, params=PARAMS):
    """metrics-2 report for one (S, D, region); see docs/STARLESS_METRICS2_20260923.md §二."""
    m1._check_rgb(source, "source")
    m1._check_rgb(estimate, "estimate")
    if estimate.shape != source.shape:
        raise ValueError("estimate must match the source shape")
    h, w = source.shape[:2]
    if not isinstance(region, np.ndarray) or region.shape != (h, w) or region.dtype != np.bool_:
        raise ValueError("region must be a same-size bool array")
    block = params["block"]
    L_S = m1.luminance16(source)
    L_D = m1.luminance16(estimate)
    bg_S = m1.block_median(L_S, block)
    bg_D = m1.block_median(L_D, block)
    hp_S = L_S - bg_S
    sigma0 = m1.noise_sigma(L_S, block)
    exclusion = m1.local_maxima(hp_S) & (hp_S >= params["exclude_k"] * sigma0)
    # Correction 1 (doc §六): σ from a star-masked MAD. σ_S (source) sets detection thresholds;
    # σ_D (estimate, same mask) scales the fill statistics and the noise expectation of E⁻.
    mask = m1.dilate(exclusion, params["sigma_mask_radius"])
    sigma = noise_sigma_masked(L_S, block, mask)
    sigma_D = noise_sigma_masked(L_D, block, mask)
    exclusion = m1.local_maxima(hp_S) & (hp_S >= params["exclude_k"] * sigma)
    strata = stratify(exclusion, region, params)
    ids, nblocks = strata["ids"], strata["nblocks"]
    report = {"version": VERSION, "params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in params.items()},
              "shape": list(source.shape), "region_pixels": int(region.sum()),
              "blocks": {"grid": list(strata["shape"]), "stratified": int((strata["block_stratum"] >= 0).sum())},
              "strata": {}}
    # --- per-star (dense-capable) statistics
    ys, xs, _ = dense_candidates(L_S, sigma, bg_S, region, source, params)
    chroma_D = [estimate[..., c].astype(np.float32) / np.float32(65535) - L_D for c in range(3)]
    planes = {"L_S": L_S, "L_D": L_D, "bg_S": bg_S, "bg_D": bg_D, "chroma_D": chroma_D,
              "chroma_ref_D": [p.box_mean(c, params["chroma_reference"]) for c in chroma_D]}
    star_stratum = strata["block_stratum"][ids[ys, xs]] if ys.size else np.zeros(0, np.int8)
    # --- block energies (no detection needed). High-pass = L − own block-median background
    # (correction 2: a box-mean high-pass carves negative moats around every star). E⁻ is
    # normalised by the noise expectation Σ σ²/2 of the same image (correction 3: the starry
    # source's inter-star gaps sit below its star-raised median, so E⁻(D)/E⁻(S) cannot read as
    # "new negative structure"); the D/S ratio of the two normalised values is still reported.
    r_lp = params["lowpass"]
    hp4_D = L_D - bg_D
    pos_S = block_sums(np.maximum(hp_S, 0) ** 2, ids, nblocks, region)
    pos_D = block_sums(np.maximum(hp4_D, 0) ** 2, ids, nblocks, region)
    neg_S = block_sums(np.minimum(hp_S, 0) ** 2, ids, nblocks, region)
    neg_D = block_sums(np.minimum(hp4_D, 0) ** 2, ids, nblocks, region)
    noise_S = block_sums(sigma.astype(np.float64) ** 2 / 2, ids, nblocks, region)
    noise_D = block_sums(sigma_D.astype(np.float64) ** 2 / 2, ids, nblocks, region)
    del hp4_D
    lp_S = block_sums(p.box_mean(L_S, r_lp), ids, nblocks, region)
    lp_D = block_sums(p.box_mean(L_D, r_lp), ids, nblocks, region)
    with np.errstate(divide="ignore", invalid="ignore"):
        pos_ratio = np.where(pos_S > 0, pos_D / pos_S, np.nan)
        neg_over_noise_S = np.where(noise_S > 0, neg_S / noise_S, np.nan)
        neg_over_noise_D = np.where(noise_D > 0, neg_D / noise_D, np.nan)
        neg_ratio = np.where(neg_over_noise_S > 0, neg_over_noise_D / neg_over_noise_S, np.nan)
        lp_ratio = np.where(lp_S > 0, lp_D / lp_S, np.nan)
    sigma_sq_S = block_sums(sigma.astype(np.float64), ids, nblocks, region)
    sigma_sq_D = block_sums(sigma_D.astype(np.float64), ids, nblocks, region)
    total_region = max(int(region.sum()), 1)
    for index, name in enumerate(STRATA):
        blocks = strata["block_stratum"] == index
        pixels = int(strata["region_px"][blocks].sum())
        entry = {"pixels": pixels, "region_fraction": pixels / total_region, "blocks": int(blocks.sum()),
                 "density_mean": float(np.nanmean(strata["density"][blocks])) if blocks.any() else None}
        sel = np.nonzero(star_stratum == index)[0] if ys.size else np.zeros(0, np.int64)
        if sel.size > params["max_stars"]:
            sel = sel[::-(-sel.size // params["max_stars"])]
        stats = _star_statistics(ys[sel], xs[sel], planes, sigma_D, params) if sel.size else None
        if stats is None:
            entry["stars"] = {"candidates": 0, "snr_kept": 0}
        else:
            keep = stats["snr"] >= params["snr_min"]
            entry["stars"] = {"candidates": int(sel.size), "snr_kept": int(keep.sum()),
                              "residual_ratio": m1._summary(stats["residual_ratio"][keep]),
                              "luminance_offset": m1._summary(stats["luminance_offset"][keep]),
                              "colour_shift": m1._summary(stats["colour_shift"][keep])}
        entry["energy"] = {"positive_hp_ratio": m1._summary(pos_ratio[blocks]),
                           "negative_hp_over_noise": m1._summary(neg_over_noise_D[blocks]),
                           "negative_hp_over_noise_source": m1._summary(neg_over_noise_S[blocks]),
                           "negative_hp_ratio": m1._summary(neg_ratio[blocks]),
                           "lowpass_ratio": m1._summary(lp_ratio[blocks])}
        with np.errstate(divide="ignore", invalid="ignore"):
            entry["sigma"] = {"source_mean": float(sigma_sq_S[blocks].sum() / max(pixels, 1)) if pixels else None,
                              "estimate_mean": float(sigma_sq_D[blocks].sum() / max(pixels, 1)) if pixels else None}
        report["strata"][name] = entry
    report["candidates_total"] = int(ys.size)
    return report


def report_digest(report):
    return m1.report_digest(report)
