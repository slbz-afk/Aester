"""The one display transform: scene-linear working-gamut RGB → display-referred sRGB, 16 bit.

Design (docs/COLOR_CONTRACT_PACKAGE_20260922.md §3.2), every step explicit and parameterised:

1. optional look hook (a callable supplied by the caller, e.g. the Z8 DCP HueSatMap) on the linear pixels;
2. saturation: ``c ← Y + s·(c − Y)`` with ``Y`` the working gamut's luminance;
2b. shadow chroma (stage 1b, §9 of the package doc): the chroma ``c − Y`` is box-smoothed with a fixed radius and blended
   in by normalised luminance ``x`` (fully smoothed at ``x ≤ shadow_smooth_start``, untouched at ``x ≥ shadow_smooth_end``);
   luminance is unchanged pixel by pixel.  Optionally ``shadow_chroma_end`` fades chroma to neutral below that luminance;
3. tone on luminance only: ``x = (Y − black)/(white − black)``, ``t = MTF(asinh(k·x)/asinh(k), m)``, then all three
   channels are multiplied by the same ratio ``t/x`` (ratio preservation = hue preservation);
4. highlight desaturation ``d = smoothstep(t; t0, t1)`` towards the neutral of the same luminance, and camera-saturated
   pixels (mask supplied by the caller from the master's white level) fully neutral;
5. gamut mapping to linear sRGB: negative channels pulled to 0 and channels above 1 pulled to 1 by desaturating towards
   the pixel's own luminance (luminance kept whenever it is itself within range);
6. sRGB OETF, ×65535, round → uint16.  The JPEG is ``round(v16/257)`` of the same array (see ``photo_render``).

No per-channel curves, no per-channel black, no maxRGB-driven tone, no scene fitting.  ``anchor`` and ``noise_sigma``
are numbers the caller measured and records, never something this module estimates from the image.  An optional
``wb_matrix`` (3×3, applied first) re-balances a master to another recorded white balance without re-exporting it.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math

import numpy as np

from . import icc_profiles as icc
from .whole_color import midtone_for_anchor

VERSION = "display-transform-2"
ROWS = 128


def _finite(name, value, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} 必须为有限数值")
    v = float(value)
    if not math.isfinite(v) or not lo <= v <= hi:
        raise ValueError(f"{name} 必须在 {lo} 与 {hi} 之间")
    return v


@dataclass(frozen=True)
class Params:
    """Explicit, JSON-serialisable display parameters (all floats recorded as given; nothing re-estimated on replay)."""
    gamut: str = "rec2020"
    black: float = 0.0                 # display black in master units (0 = physical zero after the master's black subtraction)
    white: float = 1.0                 # master units: sensor white after WB on G
    anchor: float = 0.02               # measured luminance (master units) that lands on ``target``
    target: float = 0.22               # encoded sRGB value the anchor maps to
    asinh_strength: float = 12.0
    saturation: float = 1.0
    highlight_desat_start: float = 0.8
    highlight_desat_end: float = 0.95
    look: str = "none"                 # "none", "dcp_hsm" (Z8 profile supplied by the caller) or "camera_hsm" (camera_looks)
    look_id: str | None = None         # camera_hsm: the camera_looks table; recorded only for that look
    look_strength: float = 1.0         # camera_hsm: table strength in (0, 1]
    wb_mode: str = "master"         # "master" (as exported), "daylight" or "as_shot"; the matrix below realises it
    wb_matrix: tuple | None = None     # 3×3 (nested tuples) applied to the master pixels first; None = identity
    wb_gains_rgb: tuple | None = None  # recorded gains the matrix realises (information only)
    noise_sigma: float | None = None   # measured robust luminance noise of the master (master units), recorded
    shadow_smooth_radius: int = 3      # box radius in pixels for the chroma smoothing
    shadow_smooth_start: float = 0.0   # x ≤ start: fully smoothed chroma (0/0 = smoothing off)
    shadow_smooth_end: float = 0.0     # x ≥ end: untouched
    shadow_chroma_end: float = 0.0     # optional: chroma fades to neutral below this luminance (0 = off)

    def validate(self) -> "Params":
        icc.Gamut(self.gamut)
        black = _finite("black", self.black, -1e6, 1e6)
        white = _finite("white", self.white, -1e6, 1e6)
        if white - black < 1e-9:
            raise ValueError("white 必须大于 black")
        anchor = _finite("anchor", self.anchor, black, white)
        if anchor <= black:
            raise ValueError("anchor 必须大于 black")
        _finite("target", self.target, 0.01, 0.95)
        _finite("asinh_strength", self.asinh_strength, 0.0, 100.0)
        _finite("saturation", self.saturation, 0.0, 2.0)
        t0 = _finite("highlight_desat_start", self.highlight_desat_start, 0.0, 1.0)
        t1 = _finite("highlight_desat_end", self.highlight_desat_end, 0.0, 1.0)
        if t1 <= t0:
            raise ValueError("highlight_desat_end 必须大于 highlight_desat_start")
        if self.look not in ("none", "dcp_hsm", "camera_hsm"):
            raise ValueError("look 只能是 none、dcp_hsm 或 camera_hsm")
        if self.look == "camera_hsm":
            from . import camera_looks
            if self.look_id not in camera_looks.LOOKS:
                raise ValueError("camera_hsm 观感需要已登记的 look_id")
            camera_looks.check_strength(self.look_strength)
        elif self.look_id is not None or self.look_strength != 1.0:
            raise ValueError("look_id / look_strength 只用于 camera_hsm 观感")
        if self.wb_mode not in ("master", "daylight", "as_shot"):
            raise ValueError("wb_mode 只能是 master、daylight 或 as_shot")
        if self.wb_matrix is not None:
            m = np.asarray(self.wb_matrix, dtype=np.float64)
            if m.shape != (3, 3) or not np.isfinite(m).all():
                raise ValueError("wb_matrix 需为有限 3×3")
        if self.wb_gains_rgb is not None and (len(self.wb_gains_rgb) != 3 or any(not (v > 0) for v in self.wb_gains_rgb)):
            raise ValueError("wb_gains_rgb 需为三个正数")
        if self.noise_sigma is not None:
            _finite("noise_sigma", self.noise_sigma, 0.0, 1e6)
        if type(self.shadow_smooth_radius) is not int or not 0 <= self.shadow_smooth_radius <= 16:
            raise ValueError("shadow_smooth_radius 需为 0–16 的整数")
        s0 = _finite("shadow_smooth_start", self.shadow_smooth_start, 0.0, 1e6)
        s1 = _finite("shadow_smooth_end", self.shadow_smooth_end, 0.0, 1e6)
        if (s0 > 0 or s1 > 0) and not (s1 > s0 and self.shadow_smooth_radius > 0):
            raise ValueError("shadow_smooth_end 必须大于 shadow_smooth_start，且半径 > 0")
        _finite("shadow_chroma_end", self.shadow_chroma_end, 0.0, 1e6)
        return self

    @property
    def shadow_smoothing(self) -> bool:
        return self.shadow_smooth_end > 0.0

    def midtone(self) -> float:
        x = (self.anchor - self.black) / (self.white - self.black)
        x = min(x, 1.0)
        k = self.asinh_strength
        a = math.asinh(k * x) / math.asinh(k) if k > 0 else x
        return midtone_for_anchor(a, float(icc.srgb_decode(self.target)))

    def to_json(self) -> dict:
        d = asdict(self)
        if self.look != "camera_hsm":          # records of the other looks keep their earlier form
            del d["look_id"], d["look_strength"]
        if d["wb_matrix"] is not None:
            d["wb_matrix"] = [list(map(float, row)) for row in d["wb_matrix"]]
        if d["wb_gains_rgb"] is not None:
            d["wb_gains_rgb"] = [float(v) for v in d["wb_gains_rgb"]]
        d["version"] = VERSION
        d["midtone"] = self.midtone()
        d["method"] = "luminance asinh+MTF, ratio-preserving RGB, highlight desaturation, luminance-preserving gamut mapping, sRGB OETF"
        return d

    @classmethod
    def from_json(cls, data: dict) -> "Params":
        if not isinstance(data, dict) or data.get("version") != VERSION:
            raise ValueError("显示变换参数版本不受支持")
        fields = {k: data[k] for k in cls.__dataclass_fields__ if k in data}
        if fields.get("wb_matrix") is not None:
            fields["wb_matrix"] = tuple(tuple(float(v) for v in row) for row in fields["wb_matrix"])
        if fields.get("wb_gains_rgb") is not None:
            fields["wb_gains_rgb"] = tuple(float(v) for v in fields["wb_gains_rgb"])
        p = cls(**fields).validate()
        if "midtone" in data and not math.isclose(float(data["midtone"]), p.midtone(), rel_tol=1e-12, abs_tol=1e-14):
            raise ValueError("记录的 midtone 与锚点/目标不一致")
        return p


def tone_curve(x: np.ndarray, params: Params) -> np.ndarray:
    """Display-linear luminance t for normalised scene luminance x (x clipped to [0, 1]); t(0)=0, t(1)=1, monotone."""
    k, m = params.asinh_strength, params.midtone()
    q = np.clip(x, 0.0, 1.0)
    a = np.arcsinh(k * q) / math.asinh(k) if k > 0 else q
    return ((m - 1.0) * a) / ((2.0 * m - 1.0) * a - m)


def smoothstep(t: np.ndarray, t0: float, t1: float) -> np.ndarray:
    u = np.clip((t - t0) / (t1 - t0), 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def _luma_rows(linear, gamut: str, rows, wb_matrix=None) -> np.ndarray:
    g = icc.Gamut(gamut)
    block = np.asarray(rows, dtype=np.float64)
    if wb_matrix is not None:
        block = block @ np.asarray(wb_matrix, dtype=np.float64).T
    return block @ g.luma


def measure_anchor(linear: np.ndarray, gamut: str, *, mask: np.ndarray | None = None, stride: int = 8,
                   wb_matrix=None) -> float:
    """Median luminance of the (optionally masked) image on a stride grid — the caller records the number."""
    sample = linear[::stride, ::stride]
    y = _luma_rows(linear, gamut, sample, wb_matrix)
    if mask is not None:
        m = np.asarray(mask)[::stride, ::stride]
        if m.shape != y.shape:
            raise ValueError("锚点蒙版尺寸不符")
        y = y[m > 0]
    if y.size == 0:
        raise ValueError("锚点样本为空")
    return float(np.median(y))


def measure_noise_sigma(linear: np.ndarray, gamut: str, *, stride: int = 8, darkest_fraction: float = 0.1,
                        wb_matrix=None) -> float:
    """Robust luminance noise of the master (master units): MAD×1.4826 of the horizontal neighbour difference /√2 over
    the darkest ``darkest_fraction`` of the sampled pixels (rows every ``stride``).  A measurement that the caller
    records; the shadow-chroma thresholds are fixed multiples of it."""
    rows = linear[::stride]
    y = _luma_rows(linear, gamut, rows, wb_matrix)
    d = (y[:, 1:] - y[:, :-1]) / math.sqrt(2.0)
    level = 0.5 * (y[:, 1:] + y[:, :-1])
    valid = np.isfinite(d) & (level > 0)
    if valid.sum() < 100:
        raise ValueError("噪声样本不足")
    cut = np.quantile(level[valid], darkest_fraction)
    sel = d[valid & (level <= cut)]
    if sel.size < 100:
        raise ValueError("最暗分位样本不足")
    mad = np.median(np.abs(sel - np.median(sel)))
    return float(1.4826 * mad)


def box_mean(a: np.ndarray, radius: int) -> np.ndarray:
    """(2r+1)² box mean over the first two axes with edge replication; exact via cumulative sums (float64)."""
    if radius <= 0:
        return np.asarray(a, dtype=np.float64)
    r = int(radius)
    pad = [(r, r), (r, r)] + [(0, 0)] * (a.ndim - 2)
    p = np.pad(np.asarray(a, dtype=np.float64), pad, mode="edge")
    c = np.cumsum(p, axis=0)
    c = np.concatenate([np.zeros_like(c[:1]), c], axis=0)
    rows = c[2 * r + 1:] - c[:-(2 * r + 1)]
    c2 = np.cumsum(rows, axis=1)
    c2 = np.concatenate([np.zeros_like(c2[:, :1]), c2], axis=1)
    cols = c2[:, 2 * r + 1:] - c2[:, :-(2 * r + 1)]
    return cols / float((2 * r + 1) ** 2)


FLAG_CAMERA_SATURATED = 1
FLAG_HIGHLIGHT_DESATURATED = 2
FLAG_GAMUT_NEGATIVE = 4
FLAG_GAMUT_OVER_WHITE = 8
FLAG_AT_OR_BELOW_BLACK = 16
FLAG_ROUNDED_TO_ZERO = 32          # some channel positive in float but below half a code → 0 after rounding


def render(linear: np.ndarray, params: Params, *, camera_saturated: np.ndarray | None = None,
           look=None, region: np.ndarray | None = None, flags: np.ndarray | None = None, rows: int = ROWS,
           workers: int = 1):
    """Bounded parallel rendering with unchanged row blocks and halo arithmetic.

    Arbitrary callbacks can keep mutable counters; they deliberately stay serial.  A look that declares
    ``thread_safe = True`` (the HueSatMap looks, whose counters are locked) runs in the parallel ranges too.  Workers
    write disjoint output rows; statistics merge in deterministic range order.
    """
    from .performance import MAX_RENDER_WORKERS, run_row_ranges
    if type(workers) is not int or not 1 <= workers <= MAX_RENDER_WORKERS:
        raise ValueError(f'workers 必须为 1–{MAX_RENDER_WORKERS} 的整数')
    if type(rows) is not int or rows < 1:
        raise ValueError('rows 必须为正整数')
    if not isinstance(linear, np.ndarray) or linear.ndim != 3 or linear.shape[2] != 3 or linear.dtype.kind != 'f':
        raise ValueError('需要浮点 HWC RGB 数组')
    if look is None and params.look == 'camera_hsm':
        # Self-contained look: every caller rendering these params (previews, candidates) gets the same pixels.
        from .camera_looks import make_look
        look, _ = make_look(params.look_id, params.look_strength, params.gamut)
    parallel_look = look is None or getattr(look, 'thread_safe', False) is True
    if workers == 1 or not parallel_look or linear.shape[0] <= rows:
        return _render_serial(linear, params, camera_saturated=camera_saturated, look=look,
                              region=region, flags=flags, rows=rows)
    h, w = linear.shape[:2]
    out = np.empty((h, w, 3), dtype=np.uint16)
    def run(bound):
        return _render_serial(linear, params, camera_saturated=camera_saturated, look=look, region=region,
                              flags=flags, rows=rows, _out=out, _bounds=bound)[1]
    parts = run_row_ranges(run, h, rows, workers)
    stats = parts[0]
    for part in parts[1:]:
        for key, value in part.items():
            if isinstance(value, dict):
                for label, n in value.items():
                    stats[key][label] = stats[key].get(label, 0) + n
            else:
                stats[key] += value
    return out, stats


def _render_serial(linear: np.ndarray, params: Params, *, camera_saturated: np.ndarray | None = None,
           look=None, region: np.ndarray | None = None, flags: np.ndarray | None = None, rows: int = ROWS,
           _out=None, _bounds=None):
    """Return (uint16 HxWx3 sRGB, stats).  ``linear`` is float32/float64 HxWx3 in ``params.gamut`` (TOP-DOWN or not:
    this function is row-order agnostic).  ``camera_saturated`` is an optional HxW bool mask.  ``look`` is an optional
    callable ``f(block_float64) -> block_float64`` applied first.  ``region`` (optional HxW int8/uint8: 0 = ground,
    1 = sky, or any labels) only splits the gamut-mapping counts in ``stats``.  ``flags`` (optional HxW uint8, written
    in place) receives the FLAG_* bits per pixel so that an acceptance script can attribute every zero or white."""
    params = params.validate()
    if not isinstance(linear, np.ndarray) or linear.ndim != 3 or linear.shape[2] != 3 or linear.dtype.kind != "f":
        raise ValueError("需要浮点 HWC RGB 数组")
    h, w = linear.shape[:2]
    if camera_saturated is not None and (camera_saturated.shape != (h, w)):
        raise ValueError("相机饱和蒙版尺寸不符")
    if region is not None and region.shape != (h, w):
        raise ValueError("区域标签尺寸不符")
    if flags is not None and (flags.shape != (h, w) or flags.dtype != np.uint8):
        raise ValueError("标志数组需为 HxW uint8")
    g = icc.Gamut(params.gamut)
    to_srgb = g.matrix_to(icc.SRGB)
    luma_w = g.luma
    luma_s = icc.SRGB.luma
    scale = params.white - params.black
    out = np.empty((h, w, 3), dtype=np.uint16) if _out is None else _out
    first, last = (0, h) if _bounds is None else _bounds
    stats = {"pixels": int((last-first) * w), "camera_saturated": 0, "highlight_desaturated": 0, "luminance_at_or_below_black": 0,
             "gamut_negative_mapped": 0, "gamut_over_white_mapped": 0, "luminance_over_white": 0, "rounded_to_zero": 0,
             "gamut_negative_mapped_by_region": {}, "gamut_over_white_mapped_by_region": {}}
    labels = None if region is None else np.unique(region)
    if labels is not None:
        for lab in labels:
            stats["gamut_negative_mapped_by_region"][str(int(lab))] = 0
            stats["gamut_over_white_mapped_by_region"][str(int(lab))] = 0
    wb = None if params.wb_matrix is None else np.asarray(params.wb_matrix, dtype=np.float64)
    halo = params.shadow_smooth_radius if params.shadow_smoothing else 0
    stats["shadow_chroma_smoothed"] = 0
    stats["shadow_chroma_faded"] = 0
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        for y0 in range(first, last, rows):
            y1 = min(y0 + rows, h)
            ya, yb = max(0, y0 - halo), min(h, y1 + halo)          # rows read with the smoothing halo
            block = np.asarray(linear[ya:yb], dtype=np.float64)
            if not np.isfinite(block).all():
                raise ValueError("输入包含非有限值")
            if wb is not None:
                block = block @ wb.T
            if look is not None:
                block = np.asarray(look(block), dtype=np.float64)
                if block.shape != (yb - ya, w, 3) or not np.isfinite(block).all():
                    raise ValueError("观感函数返回了无效数组")
            # 2. saturation about the working-gamut luminance
            Y = block @ luma_w
            if params.saturation != 1.0:
                block = Y[..., None] + params.saturation * (block - Y[..., None])
            # 2b. shadow chroma: smoothed chroma blended in by luminance, then optional fade to neutral
            if halo:
                chroma = block - Y[..., None]
                smoothed = box_mean(chroma, params.shadow_smooth_radius)
                xs = (Y - params.black) / scale
                wgt = 1.0 - smoothstep(xs, params.shadow_smooth_start, params.shadow_smooth_end)
                block = Y[..., None] + chroma + wgt[..., None] * (smoothed - chroma)
                stats["shadow_chroma_smoothed"] += int((wgt[y0 - ya:y0 - ya + (y1 - y0)] > 0).sum())
            block = block[y0 - ya:y0 - ya + (y1 - y0)]
            Y = Y[y0 - ya:y0 - ya + (y1 - y0)]
            if params.shadow_chroma_end > 0.0:
                xs = (Y - params.black) / scale
                keep = smoothstep(xs, 0.0, params.shadow_chroma_end)
                block = Y[..., None] + keep[..., None] * (block - Y[..., None])
                stats["shadow_chroma_faded"] += int((keep < 1.0).sum())
            # 3. tone on luminance, ratio-preserving
            u = (block - params.black) / scale
            x = (Y - params.black) / scale
            positive = x > 0
            stats["luminance_at_or_below_black"] += int((~positive).sum())
            fl = np.zeros(x.shape, dtype=np.uint8)
            fl[~positive] |= FLAG_AT_OR_BELOW_BLACK
            stats["luminance_over_white"] += int((x > 1.0).sum())
            t = tone_curve(x, params)
            gain = np.zeros_like(x)
            np.divide(t, x, out=gain, where=positive)
            ct = u * gain[..., None]
            ct[~positive] = 0.0
            # 4. highlight desaturation towards neutral of the same display luminance
            d = smoothstep(t, params.highlight_desat_start, params.highlight_desat_end)
            if camera_saturated is not None:
                sat = np.asarray(camera_saturated[y0:y1], dtype=bool)
                d = np.where(sat, 1.0, d)
                stats["camera_saturated"] += int(sat.sum())
                fl[sat] |= FLAG_CAMERA_SATURATED
            stats["highlight_desaturated"] += int((d > 0).sum())
            fl[d > 0] |= FLAG_HIGHLIGHT_DESATURATED
            ct = ct + d[..., None] * (t[..., None] - ct)
            # 5. gamut mapping to linear sRGB, keeping luminance where it is itself representable
            v = ct @ to_srgb.T
            Yv = v @ luma_s
            lo = v.min(axis=2)
            neg = lo < -1e-12
            if neg.any():
                fl[neg] |= FLAG_GAMUT_NEGATIVE
                f = np.zeros_like(Yv)
                ok = neg & (Yv > 0.0)
                np.divide(Yv, Yv - lo, out=f, where=ok)
                v = np.where(neg[..., None], Yv[..., None] + (v - Yv[..., None]) * f[..., None], v)
                v[neg & ~(Yv > 0.0)] = 0.0
                stats["gamut_negative_mapped"] += int(neg.sum())
                if labels is not None:
                    r = region[y0:y1]
                    for lab in labels:
                        stats["gamut_negative_mapped_by_region"][str(int(lab))] += int((neg & (r == lab)).sum())
            hi = v.max(axis=2)
            over = hi > 1.0 + 1e-12
            if over.any():
                fl[over] |= FLAG_GAMUT_OVER_WHITE
                f = np.zeros_like(Yv)
                ok = over & (Yv < 1.0)
                np.divide(1.0 - Yv, hi - Yv, out=f, where=ok)
                v = np.where(over[..., None], Yv[..., None] + (v - Yv[..., None]) * f[..., None], v)
                v[over & ~(Yv < 1.0)] = 1.0
                stats["gamut_over_white_mapped"] += int(over.sum())
                if labels is not None:
                    r = region[y0:y1]
                    for lab in labels:
                        stats["gamut_over_white_mapped_by_region"][str(int(lab))] += int((over & (r == lab)).sum())
            v = np.clip(v, 0.0, 1.0)     # float rounding guard only; the mapping above already bounds the values
            # 6. encode
            encoded = icc.srgb_encode(v) * 65535.0
            rounded_zero = ((encoded > 0.0) & (encoded < 0.5)).any(axis=2)
            fl[rounded_zero] |= FLAG_ROUNDED_TO_ZERO
            stats["rounded_to_zero"] += int(rounded_zero.sum())
            out[y0:y1] = np.rint(encoded).astype(np.uint16)
            if flags is not None:
                flags[y0:y1] = fl
    return out, stats


def to_8bit(rgb16: np.ndarray) -> np.ndarray:
    """The JPEG array: the same 16-bit render rounded to 8 bits (65535/257 = 255 exactly)."""
    if rgb16.dtype != np.uint16:
        raise ValueError("需要 uint16 数组")
    # integer form of round(v/257): v/257 is never exactly half-way for integer v, so (v + 128) // 257 is exact
    return ((rgb16.astype(np.uint32) + 128) // 257).astype(np.uint8)


# ---- OKLab helpers (Björn Ottosson, 2020), used by the acceptance tests to measure hue ------------------------------
_OK_M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                   [0.2119034982, 0.6806995451, 0.1073969566],
                   [0.0883024619, 0.2817188376, 0.6299787005]])
_OK_M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                   [1.9779984951, -2.4285922050, 0.4505937099],
                   [0.0259040371, 0.7827717662, -0.8086757660]])


def oklab_from_linear_srgb(rgb: np.ndarray) -> np.ndarray:
    lms = np.asarray(rgb, dtype=np.float64) @ _OK_M1.T
    return np.cbrt(lms) @ _OK_M2.T


def oklab_hue_degrees(rgb_linear_srgb: np.ndarray) -> np.ndarray:
    lab = oklab_from_linear_srgb(rgb_linear_srgb)
    return np.degrees(np.arctan2(lab[..., 2], lab[..., 1]))


def hue_difference_degrees(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = (a - b + 180.0) % 360.0 - 180.0
    return np.abs(d)
