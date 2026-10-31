"""How does Siril 1.4.4 stacking treat pixels that are exactly zero — as "no data" or as values? (S3, RUN_LOG §24)

The sky stack is contaminated above the horizon by the ground of frames far
from the reference (the ground is static on the sensor, the sky is
registered, so the ground drifts into the sky region of the reference grid;
where it is the majority the winsorized rejection cannot remove it).  The
transferable fix is to keep each frame's ground out of the sky stack
altogether: mask it *before* export.  Whether that can be done by zeroing the
masked pixels depends on one Siril fact this proof establishes on synthetic
frames with known contents:

* ``Z_min``  zeros in a minority of frames (2/8)  — rejection alone could hide them
* ``Z_maj``  zeros in a majority (6/8)            — decisive: only "no data" handling keeps the sky value
* ``Z_one``  zeros in all but one frame (7/8)     — does a single valid frame survive
* ``Z_all``  zeros in every frame                 — what the master holds where nothing was observed
* ``D_min``  a dark ground value in a minority (3/8) — what the current pipeline does with real ground
* ``D_maj``  a dark ground value in a majority (6/8) — the mechanism of the observed band

Every frame has its own sky level (0.30 + 0.01·k) so ``-norm=addscale`` is
exercised too: the clean-sky level of the master says which frame the
normalisation targets and whether zeros disturb it.  ``applyreg`` variants
first run ``seqapplyreg`` with translation matrices we write (the product's
own-registration path) so the zero regions move in the output and pass
through Siril's interpolation before stacking.

Nothing here is a product change; the product decision is taken from the
verdicts (``zeros_excluded`` / ``zeros_included`` per variant).
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
from pathlib import Path

import numpy as np

from . import fits_crop, siril_sequence
from . import registration_adapter as adapter
from .synthetic_starfield import write_fits_rgb16

NODATA_PROOF_VERSION = "1.1.0"
WIDTH, HEIGHT, FRAMES = 640, 480, 8
FULL = 65535.0
SKY_BASE, SKY_STEP, NOISE = 0.30, 0.01, 0.004
GROUND = 0.05
REGIONS = {
    # name: (x0, x1, y0, y1, kind, frames affected 1-based)
    "Z_min": (40, 200, 40, 140, "zero", [1, 2]),
    "Z_maj": (240, 400, 40, 140, "zero", [1, 2, 3, 4, 5, 6]),
    "Z_one": (440, 600, 40, 140, "zero", [1, 2, 3, 4, 5, 6, 7]),
    "Z_all": (440, 600, 200, 300, "zero", [1, 2, 3, 4, 5, 6, 7, 8]),
    "D_min": (40, 200, 200, 300, "dark", [1, 2, 3]),
    "D_maj": (240, 400, 200, 300, "dark", [1, 2, 3, 4, 5, 6]),
}
SKY_CONTROL = (40, 600, 340, 440)
# translation of frame k -> reference (frame 1) for the applyreg variants, top-down px
SHIFT_PX = (2.5, -1.5)
STACK_COMMAND = "stack {seq} rej w 3 3 -norm=addscale -32b -out=stack.fit"
VARIANTS = {
    "rej_addscale": ["stack light rej w 3 3 -norm=addscale -32b -out=stack.fit"],          # the product's stack command
    "rej_nonorm": ["stack light rej w 3 3 -nonorm -32b -out=stack.fit"],
    "mean_nonorm": ["stack light mean n -nonorm -32b -out=stack.fit"],                     # plain mean, no rejection ("n")
    "applyreg_rej_addscale": ["seqapplyreg light", "stack r_light rej w 3 3 -norm=addscale -32b -out=stack.fit"],
    # RUN_LOG §27 (quality gate): frame DESELECTED_FRAME is deselected in the sequence we write (I line 0, zero matrix);
    # does Siril leave it out of the export and of the stack — with -filter-included, and by default?
    "applyreg_deselected_filtered": ["seqapplyreg light -filter-included", "stack r_light rej w 3 3 -norm=addscale -32b -filter-included -out=stack.fit"],
    "applyreg_deselected_default": ["seqapplyreg light", "stack r_light rej w 3 3 -norm=addscale -32b -out=stack.fit"],
}
APPLYREG_VARIANTS = {"applyreg_rej_addscale", "applyreg_deselected_filtered", "applyreg_deselected_default"}
DESELECTED_VARIANTS = {"applyreg_deselected_filtered", "applyreg_deselected_default"}
DESELECTED_FRAME = 3


def sky_level(frame: int) -> float:
    return SKY_BASE + SKY_STEP * (frame - 1)


def frame_shift(frame: int) -> tuple[float, float]:
    return (SHIFT_PX[0] * (frame - 1), SHIFT_PX[1] * (frame - 1))


def render_frame(frame: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed + frame)
    image = rng.normal(sky_level(frame), NOISE, size=(HEIGHT, WIDTH))
    for name, (x0, x1, y0, y1, kind, frames) in REGIONS.items():
        if frame in frames:
            if kind == "zero":
                image[y0:y1, x0:x1] = 0.0
            else:
                image[y0:y1, x0:x1] = rng.normal(GROUND, NOISE, size=(y1 - y0, x1 - x0))
    return np.clip(np.round(image * FULL), 0, FULL).astype(np.uint16)


def generate_dataset(work: Path) -> dict:
    work = Path(work)
    work.mkdir(parents=True, exist_ok=False)
    frames = []
    for k in range(1, FRAMES + 1):
        image = render_frame(k)
        digest = write_fits_rgb16(work / f"light_{k:05d}.fit", image, comments=[f"synthetic no-data proof frame {k}: sky {sky_level(k):.3f}"])
        frames.append({"frame": k, "sky_level": sky_level(k), "sha256": digest, "shift_px": list(frame_shift(k))})
    truth = {"proof_version": NODATA_PROOF_VERSION, "width": WIDTH, "height": HEIGHT, "frames": frames, "noise": NOISE, "ground": GROUND,
             "regions": {n: {"x0": r[0], "x1": r[1], "y0": r[2], "y1": r[3], "kind": r[4], "frames": r[5]} for n, r in REGIONS.items()},
             "sky_control": list(SKY_CONTROL), "shift_px_per_frame": list(SHIFT_PX), "reference_frame": 1,
             "hypothesis": adapter.DEFAULT_HYPOTHESIS}
    (work / "truth.json").write_text(json.dumps(truth, indent=1), encoding="utf-8")
    return truth


def translation_matrices(truth: dict) -> dict:
    """Position (0-based) -> 3x3 matrix in Siril's convention: pure translations frame -> reference (frame 1)."""
    hyp = adapter.HYPOTHESES[truth.get("hypothesis", adapter.DEFAULT_HYPOTHESIS)]
    height = truth["height"]
    out = {}
    for i, f in enumerate(truth["frames"]):
        dx, dy = f["shift_px"]
        h_td = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]])
        out[i] = np.asarray(adapter.to_siril_matrix(hyp, h_td, height, height))
    return out


def write_translation_sequence(target: Path, truth: dict, *, deselect: int | None = None) -> dict:
    """Put our translation matrices into ``light_.seq``; ``deselect`` (1-based frame) is written unselected with a zero matrix.

    If Siril already created the sequence file (the run driver asks it to
    with a harmless ``seqheader``), only the ``R1`` lines are replaced with
    ``siril_sequence.write_registration`` — the product's path.  Otherwise the
    whole file is written by hand in the layout Siril 1.4.4 writes.
    """
    target = Path(target)
    matrices = translation_matrices(truth)
    n = len(truth["frames"])
    selected = None
    if deselect is not None:
        matrices.pop(deselect - 1, None)
        selected = {i: (i != deselect - 1) for i in range(n)}
    seq = target / "light_.seq"
    if seq.exists():
        text = seq.read_text(encoding="utf-8")
        how = "rewrote_siril_sequence"
        lines = text.splitlines()
        # a sequence Siril created without loading pixels says "reference -1" and "L -1" (unknown layers); our parser
        # and seqapplyreg need a reference inside the list and the layer count of the R lines
        for i, line in enumerate(lines):
            tokens = line.split()
            if tokens[:1] == ["S"] and len(tokens) >= 7 and tokens[6] == "-1":
                tokens[6] = "0"
                lines[i] = " ".join(tokens)
                how = "fixed_reference_in_siril_sequence"
            elif tokens[:2] == ["L", "-1"]:
                lines[i] = "L 3"
        if not any(line.split(None, 1)[0] == "R1" for line in lines if line.strip() and not line.startswith("#")):
            # Siril's fresh sequence has no registration lines yet: add placeholders after the image list, then rewrite them
            last_i = max(i for i, line in enumerate(lines) if line.strip().startswith("I "))
            placeholders = ["R1 3.5 3.5 0.8 0 0.02 100 H 1 0 0 0 1 0 0 0 1"] * n
            lines[last_i + 1:last_i + 1] = placeholders
            how = "added_r_lines_to_siril_sequence"
        seq.write_text("\n".join(lines) + "\n", encoding="utf-8")
        parsed = siril_sequence.write_registration(seq, matrices, layer=1, reference_position=0, selected=selected)
        return {"how": how, "images": len(parsed["images"]), "deselected": deselect, "sha256": hashlib.sha256(seq.read_bytes()).hexdigest()}
    lines = ["#Siril sequence file (written by starstack.siril_nodata_proof: translations only, reference = frame 1)",
             "#S 'sequence_name' start_index nb_images nb_selected fixed_len reference_image version variable_size fz_flag drizzle",
             f"S 'light_' 1 {n} {n} 5 0 6 0 0 0", "L 3"] + [f"I {i + 1} 1" for i in range(n)]
    if deselect is not None:
        lines = [(f"I {deselect} 0" if line == f"I {deselect} 1" else line) for line in lines]
        lines[2] = f"S 'light_' 1 {n} {n - 1} 5 0 6 0 0 0"
    for i in range(n):
        if i in matrices:
            lines.append("R1 3.5 3.5 0.8 0 0.02 100 H %s" % " ".join(f"{v:.9g}" for v in matrices[i].reshape(-1)))
        else:
            lines.append("R1 0 0 0 0 0 0 H 0 0 0 0 0 0 0 0 0")
    text = "\n".join(lines) + "\n"
    seq.write_text(text, encoding="utf-8")
    return {"how": "written_by_hand", "images": n, "deselected": deselect, "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}


def _plane_top_down(path: Path) -> np.ndarray:
    view, header, roworder = fits_crop.plane_memmap(path, 0)
    data = np.asarray(view, dtype=np.float64)
    if int(header["BITPIX"]) == 16:
        data = data + 32768.0
    if roworder == "BOTTOM-UP":
        data = data[::-1, :]
    return data


def _region_stats(plane: np.ndarray, box, shift=(0.0, 0.0), inset: int = 8) -> dict:
    x0, x1, y0, y1 = box
    dx, dy = int(round(shift[0])), int(round(shift[1]))
    ys, ye, xs, xe = y0 + dy + inset, y1 + dy - inset, x0 + dx + inset, x1 + dx - inset
    ys, xs = max(ys, 0), max(xs, 0)
    ye, xe = min(ye, plane.shape[0]), min(xe, plane.shape[1])
    sub = plane[ys:ye, xs:xe]
    finite = np.isfinite(sub)
    return {"median": float(np.median(sub[finite])) if finite.any() else None, "nan_fraction": float(1 - finite.mean()),
            "min": float(np.nanmin(sub)) if finite.any() else None, "max": float(np.nanmax(sub)) if finite.any() else None,
            "pixels": int(sub.size)}


def analyze_variant(truth: dict, target: Path, *, variant: str) -> dict:
    """Classify Siril's zero handling from the master's region medians (values as fractions of full scale)."""
    stack = target / "stack.fit"
    if not stack.exists():
        return {"status": "no_stack_output", "files": sorted(p.name for p in target.iterdir())}
    plane = _plane_top_down(stack)
    header, _ = fits_crop.read_header(stack)
    scale = FULL if int(header["BITPIX"]) == 16 else 1.0
    plane = plane / scale
    # the master is in the reference frame's grid: for the applyreg variants the regions of frame k sit at +shift_k, so
    # a region's "affected" area in the master is the intersection over the affected frames' shifted copies — for the
    # decisive majority/all regions that intersection still holds the region core (shifts are a few px, inset 8 px)
    applyreg = variant in APPLYREG_VARIANTS
    n = len(truth["frames"])
    max_shift = frame_shift(n) if applyreg else (0.0, 0.0)
    inset = 8 + int(np.ceil(max(abs(max_shift[0]), abs(max_shift[1]))))
    sky = _region_stats(plane, truth["sky_control"], inset=inset)
    out = {"status": "analyzed", "variant": variant, "sky_control": sky, "regions": {}, "verdicts": {}}
    ref_level = truth["frames"][0]["sky_level"]
    mean_level = float(np.mean([f["sky_level"] for f in truth["frames"]]))
    out["normalisation"] = {"sky_median": sky["median"], "reference_level": ref_level, "mean_level": mean_level,
                            "target": ("reference" if sky["median"] is not None and abs(sky["median"] - ref_level) < 0.004
                                       else "mean" if sky["median"] is not None and abs(sky["median"] - mean_level) < 0.004 else "other")}
    for name, r in truth["regions"].items():
        box = (r["x0"], r["x1"], r["y0"], r["y1"])
        stats = _region_stats(plane, box, inset=inset)
        valid = [f for f in truth["frames"] if f["frame"] not in r["frames"]]
        expected_if_excluded = None
        if valid:
            # with addscale to the reference every valid frame contributes the reference level; without normalisation the
            # mean of the valid frames' own levels (rejection-free approximation)
            expected_if_excluded = ref_level if out["normalisation"]["target"] == "reference" else float(np.mean([f["sky_level"] for f in valid]))
        ratio = (stats["median"] / sky["median"]) if (stats["median"] is not None and sky["median"]) else None
        out["regions"][name] = {**stats, "kind": r["kind"], "affected_frames": len(r["frames"]), "valid_frames": len(valid),
                                "expected_if_excluded": expected_if_excluded, "ratio_to_sky": ratio}
    # verdicts
    def ratio(name):
        return out["regions"][name]["ratio_to_sky"]
    z_maj, z_one, z_all, z_min = ratio("Z_maj"), ratio("Z_one"), ratio("Z_all"), ratio("Z_min")
    if z_maj is None:
        out["verdicts"]["zeros"] = "undetermined"
    elif z_maj > 0.9:
        out["verdicts"]["zeros"] = "zeros_excluded"          # a 6/8 zero majority left the sky value intact
    elif z_maj < 0.35:
        out["verdicts"]["zeros"] = "zeros_included"          # the majority of zeros dragged the master to ~0
    else:
        out["verdicts"]["zeros"] = "partial"
    out["verdicts"]["single_valid_frame_survives"] = (z_one is not None and z_one > 0.9)
    out["verdicts"]["minority_zeros_hidden"] = (z_min is not None and z_min > 0.95)
    out["verdicts"]["all_zero_region"] = out["regions"]["Z_all"]
    d_min, d_maj = ratio("D_min"), ratio("D_maj")
    out["verdicts"]["dark_minority_bias"] = None if d_min is None else float(1 - d_min)   # 0 = rejection removed the ground fully
    out["verdicts"]["dark_majority_ratio"] = d_maj
    if variant in DESELECTED_VARIANTS:
        exported = sorted(p.name for p in target.glob("r_light_*.fit"))
        header_cards, _ = fits_crop.read_header(stack)
        stackcnt = header_cards.get("STACKCNT")
        n = len(truth["frames"])
        out["deselected"] = {"frame": DESELECTED_FRAME, "exported_frames": len(exported), "deselected_exported": f"r_light_{DESELECTED_FRAME:05d}.fit" in exported,
                             "stackcnt": int(float(stackcnt)) if stackcnt is not None else None}
        skipped = (not out["deselected"]["deselected_exported"]) and out["deselected"]["exported_frames"] == n - 1 and out["deselected"]["stackcnt"] in (None, n - 1)
        out["verdicts"]["deselected_frame_skipped"] = bool(skipped)
    if applyreg:
        # interpolation rim: profile across the left edge of Z_maj in the exported frame 2 (shift (2.5, -1.5))
        exported = target / "r_light_00002.fit"
        if exported.exists():
            p2 = _plane_top_down(exported)
            r = truth["regions"]["Z_maj"]
            dx, dy = frame_shift(2)
            row = int((r["y0"] + r["y1"]) / 2 + dy)
            x_edge = r["x0"] + dx
            xs = np.arange(int(x_edge) - 6, int(x_edge) + 7)
            prof = p2[row, xs] / (FULL if int(fits_crop.read_header(exported)[0]["BITPIX"]) == 16 else 1.0)
            level = float(np.median(p2[row, xs[0] - 40:xs[0] - 10])) / (FULL if int(fits_crop.read_header(exported)[0]["BITPIX"]) == 16 else 1.0)
            out["interpolation_rim"] = {"row": row, "x": xs.tolist(), "values_over_level": [float(v / level) if level else None for v in prof],
                                        "min_over_level": float(prof.min() / level) if level else None,
                                        "negative_pixels_in_frame": int(np.sum(p2 < 0)),
                                        "rim_width_px_below_0.98": int(np.sum((prof / level < 0.98) & (prof > 0))) if level else None}
    return out


def render(results: dict) -> str:
    lines = [f"# Siril 零值/无数据处理实证（{results['run']}）", "",
             f"生成：{results['environment']['timestamp']} · 帧 {FRAMES} 张 {WIDTH}×{HEIGHT}，各帧天空 {SKY_BASE:.2f}+{SKY_STEP:.2f}·(k−1)，噪声 {NOISE}，地景值 {GROUND}",
             f"区域：Z_min 零值 2/8 帧，Z_maj 零值 6/8，Z_one 零值 7/8，Z_all 零值 8/8，D_min 地景值 3/8，D_maj 地景值 6/8；applyreg 变体先按我们写的平移矩阵 seqapplyreg 再堆栈。", ""]
    for name, v in results["variants"].items():
        lines.append(f"## {name}")
        lines.append("")
        if v.get("status") != "analyzed":
            lines.append(f"状态：{v.get('status')} {v.get('error', '')}")
            lines.append("")
            continue
        nz = v["normalisation"]
        lines.append(f"结论：**{v['verdicts']['zeros']}** · 单一有效帧存活 {v['verdicts']['single_valid_frame_survives']} · 少数零值被藏起 {v['verdicts']['minority_zeros_hidden']} · "
                     f"地景少数(3/8)剩余偏差 {v['verdicts']['dark_minority_bias']:.3f} · 地景多数(6/8)比值 {v['verdicts']['dark_majority_ratio']:.3f} · "
                     f"清洁天空 {nz['sky_median']:.4f}（参考帧 {nz['reference_level']:.3f}，均值 {nz['mean_level']:.3f} → 归一目标 {nz['target']}）")
        lines.append("")
        lines.append("| 区域 | 类型 | 受影响帧 | 中位 | 相对天空 | NaN 比例 | 若排除应为 |")
        lines.append("|---|---|---|---|---|---|---|")
        for rname, r in v["regions"].items():
            med = "None" if r["median"] is None else f"{r['median']:.4f}"
            rat = "None" if r["ratio_to_sky"] is None else f"{r['ratio_to_sky']:.3f}"
            exp = "—" if r["expected_if_excluded"] is None else f"{r['expected_if_excluded']:.3f}"
            lines.append(f"| {rname} | {r['kind']} | {r['affected_frames']}/{FRAMES} | {med} | {rat} | {r['nan_fraction']:.2f} | {exp} |")
        lines.append("")
        if "deselected" in v:
            ds = v["deselected"]
            lines.append(f"取消选择的帧 {ds['frame']}：导出帧数 {ds['exported_frames']}，该帧被导出 {ds['deselected_exported']}，STACKCNT {ds['stackcnt']} → **{'跳过' if v['verdicts'].get('deselected_frame_skipped') else '未跳过'}**")
            lines.append("")
        if "interpolation_rim" in v:
            rim = v["interpolation_rim"]
            vals = ", ".join("None" if x is None else f"{x:.3f}" for x in rim["values_over_level"])
            lines.append(f"导出帧 2 零区左缘剖面（相对天空）：{vals} · 最低 {rim['min_over_level']} · 负像素 {rim['negative_pixels_in_frame']} · 0.98 以下的过渡宽度 {rim['rim_width_px_below_0.98']} px")
            lines.append("")
    return "\n".join(lines) + "\n"


def now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()
