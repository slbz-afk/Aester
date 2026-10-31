"""Starless evaluation driver (docs/STARLESS_PACKAGE_20260923.md, step 1).

Subcommands:
  cases    run the frozen case list (nox cache + compact baseline) → BASELINE.json / BASELINE.md
  measure  one (source, estimate[, masks][, truth]) pair → metrics JSON
  inject   add known-flux synthetic stars to a display TIFF → injected TIFF + truth.json
  compare  |A − B| in codes with optional tile-seam split (runtime ↔ nox consistency, R8)

Read-only on inputs; every input is SHA-256 verified against the case list before use.
Numbers are baselines, not certificates; visual defect counts stay a human entry.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from starstack import core, starless_metrics as sm  # noqa: E402
from starstack import starless_metrics2 as sm2  # noqa: E402
from starstack import starless_metrics3 as sm3  # noqa: E402
from starstack import photographic_job as pj  # noqa: E402

VERSION = "starless-eval-1"
KEY_COLUMNS = (
    ("残余比 med", lambda r: r["stars"]["residual_ratio"]["median"]),
    ("残余比 p90", lambda r: r["stars"]["residual_ratio"]["p90"]),
    ("饱和残余 med", lambda r: r["saturated_stars"]["residual_ratio"]["median"]),
    ("非星扰动 p95", lambda r: r["nonstar"]["p95"]),
    ("填充噪声比 med", lambda r: r["stars"]["fill_noise_ratio"]["median"]),
    ("填充亮度 p10", lambda r: r["stars"]["luminance_offset"]["p10"]),
    ("填充色度 p90", lambda r: r["stars"]["colour_shift"]["p90"]),
    ("细/中/宽频带比", lambda r: "/".join(_fmt(r["bands"][b]["ratio"] if r["bands"][b] else None, 2) for b in sm.BAND_NAMES)),
    ("饱和足迹亮度", lambda r: r["saturation_footprint"].get("luminance_offset")),
    ("饱和足迹色度", lambda r: r["saturation_footprint"].get("colour_shift")),
    ("星数", lambda r: r["stars"]["snr_kept"]),
)


KEY_COLUMNS2 = (      # metrics-2 (docs/STARLESS_METRICS2_20260923.md), one row per density stratum
    ("区占比", lambda st: st["region_fraction"]),
    ("密度/kpx", lambda st: st["density_mean"]),
    ("星数", lambda st: st["stars"]["snr_kept"]),
    ("残余比 med", lambda st: st["stars"].get("residual_ratio", {}).get("median")),
    ("残余比 p90", lambda st: st["stars"].get("residual_ratio", {}).get("p90")),
    ("亮度偏差 p10", lambda st: st["stars"].get("luminance_offset", {}).get("p10")),
    ("色度 p90", lambda st: st["stars"].get("colour_shift", {}).get("p90")),
    ("E⁺比 med", lambda st: st["energy"]["positive_hp_ratio"]["median"]),
    ("E⁻/噪 med", lambda st: st["energy"]["negative_hp_over_noise"]["median"]),
    ("低通比 med", lambda st: st["energy"]["lowpass_ratio"]["median"]),
)
METRICS = {1: (sm.VERSION, lambda S, D, region, truth=None: sm.evaluate(S, D, region, truth=truth)),
           2: (sm2.VERSION, lambda S, D, region, truth=None: sm2.evaluate2(S, D, region)),
           3: (sm3.VERSION, lambda S, D, region, truth=None: sm3.evaluate3(S, D, region, truth=truth))}


def _fmt(value, digits=3):
    if value is None:
        return "—"
    if isinstance(value, str):
        return value
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.{digits}f}"


def resolve(root: Path, relative: str) -> Path:
    return root / Path(relative.replace("\\", "/"))


def verified(path: Path, expected: str | None) -> dict:
    if not Path(path).is_file():
        return {"path": str(path), "sha256": None, "expected": expected, "ok": False, "error": "missing"}
    digest = core._sha(Path(path))
    return {"path": str(path), "sha256": digest, "expected": expected, "ok": expected is None or digest == expected}


def read_region(shape, sky_mask, protected_mask):
    if sky_mask is None and protected_mask is None:
        return np.ones(shape, bool), "whole image (no masks given)"
    region = np.ones(shape, bool)
    if sky_mask is not None:
        region &= pj._mask(sky_mask, shape) == 1
    if protected_mask is not None:
        region &= ~pj._mask(protected_mask, shape, binary=True)
    return region, "sky == 1 and not protected"


def read_rgb(path: Path):
    image = pj._read_display(path)
    if image.dtype != np.uint16 or image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"需要 uint16 RGB TIFF：{path}")
    return image


def _stable_backend(identity):
    if isinstance(identity, dict):
        return {k: _stable_backend(v) for k, v in identity.items() if k not in ('session_load_seconds', 'model_path')}
    if isinstance(identity, list):
        return [_stable_backend(v) for v in identity]
    return identity


def _bound_pairing(source, estimate, base, base_estimate, arrays, region, estimate_record, base_estimate_record):
    """Fail closed on swapped/wrong-output inputs, backend settings, or masks."""
    if not estimate_record or not base_estimate_record:
        raise ValueError('paired-response requires --estimate-record and --base-estimate-record')
    identities = []
    record_hashes = []
    for src, dst, record_path in ((source, estimate, estimate_record), (base, base_estimate, base_estimate_record)):
        if dst is None:
            raise ValueError('paired-response requires --base-estimate')
        record_path = Path(record_path)
        record = json.loads(record_path.read_text(encoding='utf-8'))
        processing_source = record.get('processing_base', record.get('source'))
        source_hash = record.get('source_hashes', {}).get(processing_source)
        output_hash = record.get('artifacts', {}).get('diffuse_srgb16.tif', {}).get('sha256')
        if record.get('state') != 'succeeded' or record.get('domain') != 'display-sRGB':
            raise ValueError('pairing record must be successful display-sRGB separation')
        if not record.get('version') or not record.get('separation_scope') or record.get('shape') != list(arrays[0].shape):
            raise ValueError('pairing record lacks processing version/scope or has wrong shape')
        if not source_hash or source_hash != core._sha(Path(src)) or not output_hash or output_hash != core._sha(Path(dst)):
            raise ValueError('pairing record input/output file hash mismatch')
        backend = _stable_backend(record.get('backend'))
        masks = record.get('masks', {})
        mask_hashes = {name: record.get('source_hashes', {}).get(masks.get(name)) for name in ('sky', 'protected')}
        if not backend or not all(mask_hashes.values()):
            raise ValueError('pairing record lacks backend/mask identity')
        identities.append(dict(backend=backend, masks=mask_hashes, separation_version=record['version'],
                               separation_scope=record['separation_scope'], domain=record['domain']))
        record_hashes.append(core._sha(record_path))
    if identities[0] != identities[1]:
        raise ValueError('pairing backend/configuration or mask identity differs')
    keys = ('source_array_sha256', 'estimate_array_sha256', 'base_array_sha256', 'base_estimate_array_sha256')
    return dict(version='starless-pairing-3', backend=identities[0], records_sha256=record_hashes,
                region_array_sha256=sm3.p.array_identity(region),
                **{key: sm3.p.array_identity(array) for key, array in zip(keys, arrays)})


def measure_pair(source: Path, estimate: Path, *, sky_mask=None, protected_mask=None, truth=None, extra=None, metrics=1,
                 base=None, base_estimate=None, estimate_record=None, base_estimate_record=None) -> dict:
    started = time.perf_counter()
    version, evaluate = METRICS[metrics]
    truth_doc = json.loads(Path(truth).read_text(encoding="utf-8")) if truth else None
    if (base or base_estimate) and (metrics != 3 or truth_doc is None):
        raise ValueError("--base/--base-estimate require --metrics 3 and --truth")
    if metrics == 3 and truth_doc is not None:
        if base is None:
            raise ValueError("metrics-3 truth requires explicit --base")
        for path, key in ((source, 'injected_sha256'), (Path(base), 'source_sha256')):
            expected = truth_doc.get(key)
            if not expected or not verified(Path(path), expected)['ok']:
                raise ValueError('truth file identity mismatch: ' + key)
    arrays = []
    try:
        S = read_rgb(source); arrays.append(S)
        D = read_rgb(estimate); arrays.append(D)
        region, rule = read_region(S.shape[:2], sky_mask, protected_mask)
        if metrics == 3:
            B = read_rgb(Path(base)) if base else None
            if B is not None: arrays.append(B)
            BD = read_rgb(Path(base_estimate)) if base_estimate else None
            if BD is not None: arrays.append(BD)
            pairing = None
            if truth_doc is not None and truth_doc.get('base_kind') == 'paired-response':
                pairing = _bound_pairing(source, estimate, base, base_estimate, (S, D, B, BD), region,
                                         estimate_record, base_estimate_record)
            result = sm3.evaluate3(S, D, region, truth=truth_doc, base=B, base_estimate=BD, pairing=pairing)
        else:
            result = evaluate(S, D, region, truth=truth_doc)
    finally:
        for a in arrays:
            pj._close_map(a)
    report = {"version": VERSION, "metrics_version": version, "numpy": np.__version__,
              "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
              "inputs": {"source": verified(source, None), "estimate": verified(estimate, None),
                         "sky_mask": verified(Path(sky_mask), None) if sky_mask else None,
                         "protected_mask": verified(Path(protected_mask), None) if protected_mask else None,
                         "truth": str(truth) if truth else None},
              "region_rule": rule, "metrics": result, "seconds": time.perf_counter() - started}
    if metrics == 3:
        report['inputs'].update(base=verified(Path(base), None) if base else None,
                                base_estimate=verified(Path(base_estimate), None) if base_estimate else None,
                                truth_record=verified(Path(truth), None) if truth else None,
                                estimate_record=verified(Path(estimate_record), None) if estimate_record else None,
                                base_estimate_record=verified(Path(base_estimate_record), None) if base_estimate_record else None)
    if extra:
        report.update(extra)
    report["digest"] = sm.report_digest(result)
    return report


def run_cases(args) -> int:
    doc = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    root = Path(args.root) if args.root else Path(doc["root"])
    output = Path(args.output).resolve()
    wanted_cases = set(args.case or doc["cases"].keys())
    for name, case in doc["cases"].items():
        status = (case.get("lifecycle") or {}).get("status", "active")
        if name in wanted_cases and status != "active":
            raise ValueError(f"Case {name} is {status}; historical metrics cannot substitute for its new baseline")
    output.mkdir(parents=True, exist_ok=False)
    wanted_backends = set(args.backend or ["nox", "compact"])
    version, evaluate = METRICS[args.metrics]
    baseline = {"version": VERSION, "metrics_version": version, "cases_file": str(Path(args.cases).resolve()),
                "cases_sha256": core._sha(Path(args.cases)), "root": str(root), "numpy": np.__version__,
                "started": dt.datetime.now(dt.timezone.utc).isoformat(), "results": {}}
    for name, case in doc["cases"].items():
        if name not in wanted_cases:
            continue
        for backend, spec in case["backends"].items():
            if backend not in wanted_backends:
                continue
            key = f"{name}-{backend}"
            print(f"[{key}] 校验输入…", flush=True)
            checks = {"source": verified(resolve(root, case["source"]["path"]), case["source"]["sha256"]),
                      "sky_mask": verified(resolve(root, case["sky_mask"]["path"]), case["sky_mask"]["sha256"]),
                      "protected_mask": verified(resolve(root, case["protected_mask"]["path"]), case["protected_mask"]["sha256"]),
                      "estimate": verified(resolve(root, spec["estimate"]["path"]), spec["estimate"]["sha256"])}
            entry = {"case": name, "track": case["track"], "backend": backend, "inputs": checks,
                     "runtime": {"seconds_model": spec.get("seconds_model"), "seconds_total": spec.get("seconds_total"),
                                 "megapixels": case["shape"][0] * case["shape"][1] / 1e6,
                                 "parameters": spec.get("parameters"), "model": spec.get("model")}}
            if not all(c["ok"] for c in checks.values()):
                entry["status"] = "hash_mismatch"
                print(f"[{key}] 哈希不符，跳过：{[k for k, c in checks.items() if not c['ok']]}", flush=True)
            else:
                print(f"[{key}] 评测…", flush=True)
                started = time.perf_counter()
                S = read_rgb(Path(checks["source"]["path"]))
                D = read_rgb(Path(checks["estimate"]["path"]))
                region, _ = read_region(S.shape[:2], checks["sky_mask"]["path"], checks["protected_mask"]["path"])
                entry["metrics"] = evaluate(S, D, region)
                pj._close_map(S)
                pj._close_map(D)
                entry["status"] = "measured"
                entry["seconds_evaluation"] = time.perf_counter() - started
                entry["digest"] = sm.report_digest(entry["metrics"])
                seconds = spec.get("seconds_model") or spec.get("seconds_total")
                entry["runtime"]["megapixels_per_second"] = (entry["runtime"]["megapixels"] / seconds) if seconds else None
                print(f"[{key}] 完成 {entry['seconds_evaluation']:.1f} s", flush=True)
            core.atomic_json(output / f"{key}.json", entry)
            baseline["results"][key] = entry
    baseline["finished"] = dt.datetime.now(dt.timezone.utc).isoformat()
    core.atomic_json(output / "BASELINE.json", baseline)
    (output / "BASELINE.md").write_text(({1: markdown, 2: markdown2, 3: markdown3}[args.metrics])(baseline), encoding="utf-8")
    print(f"基线写入 {output / 'BASELINE.json'}")
    return 0


def markdown3(baseline: dict) -> str:
    lines = ["# 无星图 metrics-3 护栏", "", "源固定样本；不替换冻结基线。完整分层、逐饱和区与样本身份见 JSON。", "",
             "| 例 | 后端 | 状态 | 星数 | 绝对残余 p90 | 负残余 p90 | 亮度绝对 p90 | 色度 p90 | 非星色移 p95 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in baseline['results'].values():
        if r['status'] != 'measured':
            lines.append(f"| {r['case']} | {r['backend']} | {r['status']} | — | — | — | — | — | — |")
            continue
        g = r['metrics']['guards']; s = g['isolated']
        cells = [s['count']] + [s[k]['p90'] for k in ('residual_absolute', 'residual_negative', 'luminance_absolute', 'colour_shift')]
        cells.append(g['nonstar']['chroma']['p95'])
        lines.append(f"| {r['case']} | {r['backend']} | measured | " + " | ".join(_fmt(c) for c in cells) + " |")
    return '\n'.join(lines) + '\n'


def markdown2(baseline: dict) -> str:
    lines = [f"# 无星图基线补充：密星场指标 metrics-2（{baseline['started'][:10]}）", "",
             f"指标 {baseline['metrics_version']}，案例清单 SHA-256 `{baseline['cases_sha256'][:12]}…`。按星密度分层（稀 < 3、中 3–10、密 ≥ 10 颗/千像素）；"
             "数值是描述，不是通过/不通过。E⁺比 = 紧致正结构剩余，E⁻/噪 = 估计图紧致负结构相对噪声期望（1 为噪声水平，> 1 暗坑/过减），低通比在密区会因去掉未分辨星而低于 1。", "",
             "| 例 | 后端 | 层 | " + " | ".join(c[0] for c in KEY_COLUMNS2) + " |",
             "|---|---|---|" + "---|" * len(KEY_COLUMNS2)]
    for key, entry in baseline["results"].items():
        if entry["status"] != "measured":
            lines.append(f"| {entry['case']} | {entry['backend']} | {entry['status']} |" + " — |" * len(KEY_COLUMNS2))
            continue
        for stratum, st in entry["metrics"]["strata"].items():
            if not st["blocks"]:
                continue
            cells = [_fmt(f(st)) for _, f in KEY_COLUMNS2]
            lines.append(f"| {entry['case']} | {entry['backend']} | {stratum} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def markdown(baseline: dict) -> str:
    lines = [f"# 无星图基线（{baseline['started'][:10]}）", "",
             f"指标 {baseline['metrics_version']}，案例清单 SHA-256 `{baseline['cases_sha256'][:12]}…`。数值是基线，不是通过/不通过；六类缺陷目视计数见各 JSON 的 `visual`。", "",
             "| 例 | 后端 | 状态 | " + " | ".join(c[0] for c in KEY_COLUMNS) + " | MP/s |",
             "|---|---|---|" + "---|" * (len(KEY_COLUMNS) + 1)]
    for key, entry in baseline["results"].items():
        if entry["status"] != "measured":
            lines.append(f"| {entry['case']} | {entry['backend']} | {entry['status']} |" + " — |" * (len(KEY_COLUMNS) + 1))
            continue
        r = entry["metrics"]
        cells = [_fmt(f(r)) for _, f in KEY_COLUMNS]
        lines.append(f"| {entry['case']} | {entry['backend']} | measured | " + " | ".join(cells)
                     + f" | {_fmt(entry['runtime'].get('megapixels_per_second'), 3)} |")
    return "\n".join(lines) + "\n"


def run_measure(args) -> int:
    report = measure_pair(Path(args.source), Path(args.estimate), sky_mask=args.sky_mask, protected_mask=args.protected_mask,
                          truth=args.truth, metrics=args.metrics, base=args.base, base_estimate=args.base_estimate,
                          estimate_record=args.estimate_record, base_estimate_record=args.base_estimate_record)
    core.atomic_json(Path(args.output), report)
    print(json.dumps({"output": args.output, "digest": report["digest"], "seconds": report["seconds"]}, ensure_ascii=False))
    return 0


def run_guard(args) -> int:
    reports = [json.loads(Path(f).read_text(encoding='utf-8')) for f in (args.reference, args.candidate)]
    result = sm3.e1_guard(*(r.get('metrics', r) for r in reports))
    core.atomic_json(Path(args.output), result)
    print(json.dumps(result, ensure_ascii=False))
    return 0


def run_inject(args) -> int:
    import tifffile
    source = Path(args.source).resolve(strict=True)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    shape, icc, metadata = pj._display_source(source)
    S = read_rgb(source)
    rng = np.random.default_rng(args.seed)
    if args.metrics == 3:
        region, _ = read_region(S.shape[:2], args.sky_mask, args.protected_mask)
        injected, truth = sm3.inject_stars3(S, rng, count=args.count, saturated=args.saturated,
                                          min_separation=args.min_separation, region=region, base_kind=args.base_kind)
    else:
        injected, truth = sm.inject_stars(np.ascontiguousarray(S), rng, count=args.count, saturated=args.saturated,
                                        min_separation=args.min_separation)
    pj._close_map(S)
    metadata = {**metadata, "injected_stars_for_sha256": core._sha(source), "injection_seed": args.seed}
    target = output / "injected_srgb16.tif"
    tifffile.imwrite(target, injected, photometric="rgb", metadata=metadata, extratags=[(34675, "B", len(icc), icc, False)])
    truth.update({"source": str(source), "source_sha256": metadata["injected_stars_for_sha256"], "seed": args.seed,
                  "injected": str(target), "injected_sha256": core._sha(target)})
    core.atomic_json(output / "truth.json", truth)
    print(json.dumps({"injected": str(target), "truth": str(output / "truth.json"), "stars": len(truth["stars"])}, ensure_ascii=False))
    return 0


def compare_arrays(A, B, region, *, tile=None, stride=None) -> dict:
    """|A − B| in codes over the region; optional split on tile-boundary rows/columns."""
    h, w = A.shape[:2]
    if B.shape != A.shape:
        raise ValueError("两幅图尺寸不同")
    counts = np.zeros(65536, np.int64)
    per_channel = np.zeros(3, np.float64)
    total = 0
    seam = np.zeros((h, w), bool)
    if tile and stride:
        for k in range(1, max(h, w) // stride + 2):
            edge = k * stride
            for offset in (-1, 0):
                if 0 <= edge + offset < h:
                    seam[edge + offset, :] = True
                if 0 <= edge + offset < w:
                    seam[:, edge + offset] = True
    seam_sum = seam_count = other_sum = other_count = 0
    maximum = 0
    for y in range(0, h, 256):
        a = A[y:y + 256].astype(np.int32)
        b = B[y:y + 256].astype(np.int32)
        d = np.abs(a - b)
        r = region[y:y + 256]
        dm = d.max(-1)[r]
        counts += np.bincount(dm, minlength=65536)
        total += dm.size
        per_channel += d[r].sum(0)
        if dm.size:
            maximum = max(maximum, int(dm.max()))
        s = seam[y:y + 256] & r
        o = ~seam[y:y + 256] & r
        seam_sum += float(d.max(-1)[s].sum())
        seam_count += int(s.sum())
        other_sum += float(d.max(-1)[o].sum())
        other_count += int(o.sum())
    cumulative = np.cumsum(counts)

    def percentile(q):
        return int(np.searchsorted(cumulative, q * total, side="left")) if total else None
    return {"pixels": total, "p50": percentile(.5), "p99": percentile(.99), "p999": percentile(.999), "max": maximum,
            "mean_per_channel": (per_channel / total).tolist() if total else None,
            "fraction_nonzero": float(1 - counts[0] / total) if total else None,
            "seam": {"tile": tile, "stride": stride, "seam_pixels": seam_count,
                     "seam_mean": (seam_sum / seam_count) if seam_count else None,
                     "other_mean": (other_sum / other_count) if other_count else None}}


def run_compare(args) -> int:
    A = read_rgb(Path(args.a))
    B = read_rgb(Path(args.b))
    if args.region_from:
        record = json.loads(Path(args.region_from).read_text(encoding="utf-8"))
        masks = record["masks"]
        sky_mask, protected_mask = masks["sky"], masks["protected"]
    else:
        sky_mask, protected_mask = args.sky_mask, args.protected_mask
    region, rule = read_region(A.shape[:2], sky_mask, protected_mask)
    started = time.perf_counter()
    result = compare_arrays(A, B, region, tile=args.tile, stride=args.stride)
    pj._close_map(A)
    pj._close_map(B)
    report = {"version": VERSION, "a": verified(Path(args.a), None), "b": verified(Path(args.b), None), "region_rule": rule,
              "difference_codes": result, "seconds": time.perf_counter() - started,
              "criterion_r8": {"p99_le_4": result["p99"] is not None and result["p99"] <= 4,
                               "max_le_64": result["max"] <= 64, "consistent": result["p99"] is not None and result["p99"] <= 4 and result["max"] <= 64}}
    core.atomic_json(Path(args.output), report)
    print(json.dumps({"output": args.output, "difference_codes": result, "criterion_r8": report["criterion_r8"]}, ensure_ascii=False))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    cases = sub.add_parser("cases", help="按案例清单评测已缓存的 nox / 轻量结果并冻结基线")
    cases.add_argument("--cases", required=True)
    cases.add_argument("--output", required=True, help="新目录（已存在则拒绝）")
    cases.add_argument("--root", help="覆盖清单里的 root（默认清单值）")
    cases.add_argument("--case", action="append", help="只跑这些例（可重复）")
    cases.add_argument("--backend", action="append", choices=("nox", "compact"))
    cases.add_argument("--metrics", type=int, choices=(1, 2, 3), default=1, help="1 = 冻结定义；2 = 密星场；3 = 固定源护栏")
    cases.set_defaults(func=run_cases)
    measure = sub.add_parser("measure", help="评测一对源／估计")
    measure.add_argument("--source", required=True)
    measure.add_argument("--estimate", required=True)
    measure.add_argument("--sky-mask")
    measure.add_argument("--protected-mask")
    measure.add_argument("--truth", help="inject 写出的 truth.json（metrics 3 须提供绑定基底）")
    measure.add_argument("--metrics", type=int, choices=(1, 2, 3), default=1)
    measure.add_argument("--base", help="metrics-3 注入前基底")
    measure.add_argument("--base-estimate", help="paired-response 中同后端处理基底的结果")
    measure.add_argument("--estimate-record", help="paired-response 的注入图处理 result.json")
    measure.add_argument("--base-estimate-record", help="paired-response 的未注入基底处理 result.json")
    measure.add_argument("--output", required=True)
    measure.set_defaults(func=run_measure)
    inject = sub.add_parser("inject", help="向显示域 TIFF 注入已知通量的合成星")
    inject.add_argument("--source", required=True)
    inject.add_argument("--output", required=True, help="新目录")
    inject.add_argument("--count", type=int, default=300)
    inject.add_argument("--saturated", type=int, default=0)
    inject.add_argument("--seed", type=int, default=20260923)
    inject.add_argument("--min-separation", type=int, default=30)
    inject.add_argument("--metrics", type=int, choices=(1, 3), default=1)
    inject.add_argument("--base-kind", choices=('known-background', 'paired-response'))
    inject.add_argument("--sky-mask")
    inject.add_argument("--protected-mask")
    inject.set_defaults(func=run_inject)
    guard = sub.add_parser('guard', help='比较两份 metrics-3 报告的 E1 护栏；不自动切默认')
    guard.add_argument('--reference', required=True)
    guard.add_argument('--candidate', required=True)
    guard.add_argument('--output', required=True)
    guard.set_defaults(func=run_guard)
    compare = sub.add_parser("compare", help="两幅 16 位 RGB TIFF 的 |差| 分布（code），可按分块接缝拆分")
    compare.add_argument("--a", required=True)
    compare.add_argument("--b", required=True)
    compare.add_argument("--region-from", help="separation result.json：用其天空／保护蒙版定义评测区")
    compare.add_argument("--sky-mask")
    compare.add_argument("--protected-mask")
    compare.add_argument("--tile", type=int)
    compare.add_argument("--stride", type=int)
    compare.add_argument("--output", required=True)
    compare.set_defaults(func=run_compare)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
