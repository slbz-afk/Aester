#!/usr/bin/env python3
"""Read-only support-stage replay of a completed strict job (no Siril/RAW IO).

Example: python scripts/replay_support.py --job <job-directory> --repeat 2
This times dependency support, not a complete photo job or image quality.

``--workers N`` (efficiency P1c F-2 / G-4t, prereg §6.2 (b)): the product's paths on the frame pool instead of the
reference kernel -- every selected frame's support map by the buffered kernel (one per worker, as prepare_exports runs
it), compared with the recorded count and, where the job retained it, with the digest of the .npy the job wrote; then
the native map by registered_support.native_mask (quick route) from the job's stack map, compared with the recorded
native map. Without --frames it selects every export frame. Still read-only: nothing of the job is written.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
sys.path.insert(0, str(SOURCE_ROOT))
from starstack import core, registered_support as rs, siril_support_map as sm  # noqa: E402
from starstack.records import atomic_json  # noqa: E402
from starstack.siril_sequence import parse_seq_text  # noqa: E402

NOTICE = "仅支持阶段回放，非端到端出片，非画质验收"


def positive(value):
    try:
        result = int(value)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if str(result) != str(value) or result < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def _file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _array_hash(array):
    # C-order uint8/bool bytes; no full-size tobytes() copy.
    return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast("B")).hexdigest()


def _manifest(paths):
    result = {}
    for path in sorted(set(paths), key=str):
        try:
            result[str(path)] = _file_hash(path)
        except OSError:
            result[str(path)] = None
    return result


def _code_manifest():
    return _manifest([*SOURCE_ROOT.rglob("*.py"), Path(__file__).resolve()])


def _read_bound(path, before):
    path = Path(path)
    raw = path.read_bytes()
    before[str(path)] = hashlib.sha256(raw).hexdigest()
    return raw


def _json_bound(path, before):
    return json.loads(_read_bound(path, before))


def _expect_file(path, expected, before):
    actual = _file_hash(path)
    before[str(path)] = actual
    if not isinstance(expected, str) or actual != expected:
        raise ValueError(f"job artifact identity mismatch: {path.name}")


def _local(directory, *parts):
    path = directory.joinpath(*parts).resolve()
    if not path.is_relative_to(directory):
        raise ValueError("job artifact must remain inside the selected job")
    return path


def _select_frames(requested, transforms, reference):
    available = list(transforms)
    wanted = requested if requested is not None else [available[0], reference, available[-1]]
    if not wanted:
        raise ValueError("at least one frame is required")
    chosen = list(dict.fromkeys(wanted))
    missing = [stem for stem in chosen if stem not in transforms]
    if missing:
        raise ValueError(f"frames are not selected export stems: {', '.join(missing)}")
    return chosen


def _report_text(report):
    lines = ["# 插值支持阶段回放", "", f"**{NOTICE}。**", "",
             f"状态：{report['state']}；来源：`{report['job']}`。",
             "", "不解码 RAW，不调用 Siril，不写旧作业或成片。",
             "只与旧记录的 supported_pixels 比较数量；旧作业没有逐帧支持图，不能证明逐像素一致。",
             "初始化冷／热指本进程的 SIP 网格与掩膜缓存，未清系统文件缓存；不是磁盘冷启动基准。", ""]
    if report.get("pool"):
        pool = report["pool"]
        lines += [f"帧池回放（--workers {pool['workers']}）：产品缓冲核按帧并行；有保留逐帧图时按其 .npy 摘要逐图比较；"
                  f"原生图与作业记录比较={pool.get('native_matches_record')}。",
                  f"各次帧段秒数={[round(r['frames_seconds'], 3) for r in pool['runs']]}；"
                  f"原生图秒数={round(pool['native_seconds'], 3) if 'native_seconds' in pool else '无'}。", ""]
    for frame in report["frames"]:
        runs = frame.get("runs")
        lines.append(f"- `{frame['stem']}`：{frame.get('supported_pixels', '未完成')} 个支持像素；"
                     f"旧数量一致={frame.get('matches_recorded_count', '未完成')}；"
                     + (f"支持计算秒数={[round(r['support_seconds'], 6) for r in runs]}。" if runs is not None
                        else f"与保留逐帧图一致={frame.get('matches_retained_map')}。"))
    if report.get("error"):
        lines.extend(["", f"失败：{report['error']['type']}: {report['error']['message']}"])
    if report.get("integrity_error"):
        lines.extend(["", f"完整性检查失败：{report['integrity_error']}"])
    lines.extend(["", "精确 SHA、逐次时间、初始化及源码／源工件前后哈希见 `replay.json`。",
                  "资格不是拒绝后的生效曝光，计数相同也不能说明支持集合相同。", ""])
    return "\n".join(lines)


def _npy_digest(array):
    """sha256 of the bytes registered_support.save_array writes for ``array`` (np.save, no pickle)."""
    stream = io.BytesIO()
    np.save(stream, array, allow_pickle=False)
    return hashlib.sha256(stream.getbuffer()).hexdigest()


def _replay_pool(report, work, record, source, transforms, reference_h, shape, wcs_header, chosen, prior, before,
                 workers, repeat):
    """--workers: the frames' maps by the buffered kernel on the frame pool, then the native map (P1c F-2 / G-4t)."""
    from starstack import frame_pool, support_kernel as sk
    from starstack.registration_distortion import RadialDistortion
    from starstack.support_anchor import anchor_grid_from_validity
    height, width = shape
    report["support_implementation"] = rs.SUPPORT_IMPLEMENTATION
    pool = report["pool"] = {"workers": workers, "runs": []}
    cache = {"key": None, "anchors": None}

    def prepare(i, stem, check):
        valid = source.for_frame(stem)
        if hasattr(source, "_cache"):
            source._cache.clear()
        valid_sha = _array_hash(valid)
        if valid_sha != prior[stem].get("source_valid_sha256"):
            raise ValueError(f"source validity differs from old job: {stem}")
        if cache["key"] != (valid.shape, valid_sha):
            cache["anchors"] = None
            cache["anchors"] = anchor_grid_from_validity(valid, check=check)
            cache["key"] = valid.shape, valid_sha
        params = sm.make_export_mapping(transforms[stem], reference_h, width, height, wcs_header).export_parameters
        return stem, valid, cache["anchors"], params

    def unit(payload, u):
        stem, valid, anchors, params = payload
        u.state["kernel"] = sk.kernel_for(u.state["kernel"], params)
        supported = sk.registered_mask_kernel(valid, transforms[stem], params=params, anchors=anchors,
                                              kernel=u.state["kernel"], check=u.check)
        explicit = prior[stem].get("explicit_validity")
        retained = None if not explicit else _npy_digest(supported) == explicit.get("sha256")
        return stem, _array_hash(supported), int(np.count_nonzero(supported)), retained

    for iteration in range(repeat):
        entries = []

        def deliver(i, value):
            stem, digest, count, retained = value
            entries.append({"stem": stem, "sha256": digest, "supported_pixels": count,
                            "matches_recorded_count": count == prior[stem]["supported_pixels"],
                            "matches_retained_map": retained})
        pool_stats = {}
        mark = time.perf_counter()
        try:
            frame_pool.run_ordered(unit, chosen, workers=workers, name="replay_support", prepare=prepare,
                                   on_result=deliver, init=lambda: {"kernel": None},
                                   finalize=lambda state: state.update(kernel=None), inflight=1, stats=pool_stats)
        except BaseException:
            sm.release_distortion_grid()
            raise
        if iteration == repeat - 1:
            sm.release_distortion_grid()              # kept between repeats, as the reference replay keeps it
        pool["runs"].append({"iteration": iteration + 1, "frames_seconds": time.perf_counter() - mark,
                             "pool": {k: pool_stats.get(k) for k in ("workers", "serial", "reason", "max_running",
                                                                     "unit_cpu_seconds", "max_unit_seconds")}})
        if iteration == 0:
            report["frames"] = entries
        elif [(e["stem"], e["sha256"]) for e in entries] != [(e["stem"], e["sha256"]) for e in report["frames"]]:
            raise ValueError("repeated support changed")
        for entry in entries:
            if not entry["matches_recorded_count"]:
                raise ValueError(f"support count differs from old job: {entry['stem']}")
            if entry["matches_retained_map"] is False:
                raise ValueError(f"support map differs from the job's retained map: {entry['stem']}")
    pool["retained_maps_compared"] = sum(e["matches_retained_map"] is not None for e in report["frames"])
    if (record.get("state") == "complete" and record.get("model") is not None
            and record.get("implementation") == rs.SUPPORT_IMPLEMENTATION):
        arrays = {}
        for key in ("stack_valid", "native_valid"):
            descriptor = record[key]
            raw = _read_bound(work / descriptor["file"], before)
            if before[str(work / descriptor["file"])] != descriptor["sha256"]:
                raise ValueError(f"job artifact identity mismatch: {descriptor['file']}")
            arrays[key] = np.load(io.BytesIO(raw), allow_pickle=False)
        native_stats = {}
        mark = time.perf_counter()
        native = rs.native_mask(arrays["stack_valid"], RadialDistortion(**record["model"]), quick=True,
                                workers=workers, pool_stats=native_stats)
        pool["native_seconds"] = time.perf_counter() - mark
        pool["native_pool"] = {k: native_stats.get(k) for k in ("workers", "serial", "reason", "units")}
        pool["native_matches_record"] = bool(np.array_equal(native, arrays["native_valid"]))
        if not pool["native_matches_record"]:
            raise ValueError("native support differs from the job's native map")
    else:
        pool["native_matches_record"] = None


def replay(job, *, output_root=ROOT / "outputs/support-replay", frames=None, repeat=1, rows=None, workers=None):
    """Create a new evidence directory; return a succeeded/failed report.

    Only one anchor grid and the current output map are retained. Per-frame
    mask caches are emptied after retrieval, so memory does not grow with
    selected-frame count. The production single-entry SIP cache is reused.
    ``workers``: the frame-pool replay of the product's paths (module docstring).
    """
    directory, output_root = Path(job).resolve(), Path(output_root).resolve()
    if output_root.is_relative_to(directory):
        raise ValueError("output root must be outside the old job")
    if type(repeat) is not int or repeat < 1 or (rows is not None and (type(rows) is not int or rows < 1)):
        raise ValueError("repeat and rows must be positive integers")
    if workers is not None and (type(workers) is not int or workers < 1 or rows is not None):
        raise ValueError("workers must be a positive integer, without rows (the product kernel's own blocks)")
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / (dt.datetime.now(dt.timezone.utc).strftime("replay-%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:10])
    destination.mkdir()
    started = time.perf_counter()
    report = {"schema": "starstack.support_replay.v1", "state": "failed", "notice": NOTICE,
              "job": str(directory), "output": str(destination), "repeat": repeat, "rows": rows,
              "support_version": rs.SUPPORT_VERSION,
              # The replay times and hashes the reference kernel (invalid_integral, build_anchor_grid,
              # registered_mask), not the P1a product kernel: report the label of the path it runs.
              "support_implementation": rs.REFERENCE_IMPLEMENTATION,
              "support_sha_encoding": "bool/uint8, C order, TOP-DOWN; shape recorded separately",
              "legacy_comparison": "counts_only_no_old_per_frame_map_pixel_equivalence_claim",
              "frames": [], "initialization": {}, "source_before": {}, "code_before": {}}
    before = report["source_before"]
    try:
        report["code_before"] = _code_manifest()
        job_record = _json_bound(_local(directory, "job.json"), before)
        if job_record.get("state") != "succeeded" or job_record.get("support_mode") != rs.MODE:
            raise ValueError("--job must be a succeeded strict-support job")
        work = _local(directory, "work")
        record = _json_bound(work / rs.RECORD, before)
        validity = _json_bound(work / "sky_validity.json", before)
        if (record.get("state") != "complete" or record.get("mode") != rs.MODE
                or record.get("version") != rs.SUPPORT_VERSION
                or job_record.get("support_version") != rs.SUPPORT_VERSION):
            raise ValueError("completed support record/version required")
        shape = record.get("shape")
        if (not isinstance(shape, list) or len(shape) != 2
                or any(type(v) is not int or not 2 <= v < 32767 for v in shape)):
            raise ValueError("invalid recorded image shape")
        height, width = shape
        dilation = validity.get("dilation_px")
        if type(dilation) is not int or dilation < 0:
            raise ValueError("invalid recorded sky-validity dilation")
        seq_path = work / rs.EXPORT_SEQUENCE
        seq_raw = _read_bound(seq_path, before)
        if before[str(seq_path)] != record.get("sequence_sha256"):
            raise ValueError("pre-export sequence identity mismatch")
        sequence = parse_seq_text(seq_raw.decode("utf-8"))
        transforms = rs.export_transforms(seq_path)
        position = sequence["header"]["reference_image"]
        reference = sequence["images"][position]["stem"]
        reference_h = sequence["regdata"][1][position]["H"]
        chosen = (list(transforms) if workers is not None and frames is None
                  else _select_frames(frames, transforms, reference))
        report.update(shape=shape, reference=reference, selected_frames=chosen,
                      dilation_px=dilation, selected_export_count=len(transforms))
        prior = {entry["stem"]: entry for entry in record["frames"]}
        if len(prior) != len(record["frames"]) or set(prior) != set(transforms):
            raise ValueError("recorded/exported frame identities differ")
        distortion_lines = [line.strip() for line in seq_raw.decode("utf-8").splitlines()
                            if line.strip().startswith("D")]
        has_sip = record.get("model") is not None
        if distortion_lines != (["D1 2 distortion.wcs"] if has_sip else []):
            raise ValueError("unsupported pre-export distortion contract")
        wcs_header = None
        if has_sip:
            from astropy.io import fits
            raw = _read_bound(work / "distortion.wcs", before)
            if before[str(work / "distortion.wcs")] != record.get("wcs_sha256"):
                raise ValueError("WCS identity mismatch")
            wcs_header = fits.getheader(io.BytesIO(raw))
        elif record.get("wcs_sha256") is not None:
            raise ValueError("unexpected WCS identity without distortion")
        mask = _local(directory, "sky_mask.png")
        _expect_file(mask, job_record["mask"]["sha256"], before)
        if before[str(mask)] != validity.get("mask_sha256") or before[str(mask)] != record.get("mask_sha256"):
            raise ValueError("sky mask differs from validity/support record")
        own_masks = (job_record.get("frame_masks") or {}).get("files") or {}
        for stem in chosen:
            if stem in own_masks:
                _expect_file(_local(directory, "frame_masks", f"{stem}.png"), own_masks[stem]["sha256"], before)
            if not np.array_equal(transforms[stem], np.asarray(prior[stem]["H_stored"])):
                raise ValueError(f"recorded export transform differs: {stem}")
        engine = job_record["engine"]
        rs.require_engine(engine)
        executable = Path(engine["executable"]).resolve()
        _expect_file(executable, rs.SIRIL_SHA256, before)
        for name, expected in rs.INTERPOLATION_DLLS.items():
            _expect_file(executable.parent / name, expected, before)
        if (record.get("engine_sha256") != rs.SIRIL_SHA256
                or record.get("interpolation_dll_sha256") != rs.INTERPOLATION_DLLS):
            raise ValueError("old support record engine identity mismatch")
        report["initialization"]["validation_and_initial_hash_seconds"] = time.perf_counter() - started
        mark = time.perf_counter()
        source = core._validity_source(directory, job_record, width, height, dilation)
        report["initialization"]["source_mask_load_seconds"] = time.perf_counter() - mark
        # Precompute the same lazy grid explicitly so first/repeat support
        # timings do not hide a large one-off SIP allocation.
        sm._distortion_grid.cache_clear()
        report["initialization"]["sip_grid_cold_seconds"] = None
        report["initialization"]["sip_grid_hot_lookup_seconds"] = None
        if wcs_header is not None:
            spec = sm._sip_spec(wcs_header)
            mark = time.perf_counter()
            sm._distortion_grid(width, height, spec)
            report["initialization"]["sip_grid_cold_seconds"] = time.perf_counter() - mark
            mark = time.perf_counter()
            sm._distortion_grid(width, height, spec)
            report["initialization"]["sip_grid_hot_lookup_seconds"] = time.perf_counter() - mark
        if workers is not None:
            _replay_pool(report, work, record, source, transforms, reference_h, (height, width), wcs_header, chosen,
                         prior, before, workers, repeat)
        cached_key, anchors = None, None
        for stem in (chosen if workers is None else ()):  # the reference replay (without --workers)
            mark = time.perf_counter()
            valid = source.for_frame(stem)
            if hasattr(source, "_cache"):
                source._cache.clear()
            valid_load_seconds = time.perf_counter() - mark
            mark = time.perf_counter()
            valid_sha = _array_hash(valid)
            valid_hash_seconds = time.perf_counter() - mark
            if valid_sha != prior[stem].get("source_valid_sha256"):
                raise ValueError(f"source validity differs from old job: {stem}")
            cache_hit = cached_key == (valid.shape, valid_sha)
            mark = time.perf_counter()
            if not cache_hit:
                anchors = None
                integral = rs.invalid_integral(valid)
                anchors = rs.build_anchor_grid(integral)
                del integral
                cached_key = valid.shape, valid_sha
            anchor_seconds = time.perf_counter() - mark
            frame = {"stem": stem, "source_valid_sha256": valid_sha,
                     "recorded_supported_pixels": prior[stem]["supported_pixels"],
                     "initialization": {"source_validity_seconds": valid_load_seconds,
                                        "validity_hash_seconds": valid_hash_seconds,
                                        "anchor_cache": "hot" if cache_hit else "cold",
                                        "anchor_seconds": anchor_seconds}, "runs": []}
            report["frames"].append(frame)
            for iteration in range(repeat):
                mark = time.perf_counter()
                mapping = sm.make_export_mapping(transforms[stem], reference_h, width, height, wcs_header)
                mapping_seconds = time.perf_counter() - mark
                mark = time.perf_counter()
                supported = rs.registered_mask(valid, transforms[stem], mapping=mapping, anchors=anchors, rows=rows)
                support_seconds = time.perf_counter() - mark
                mark = time.perf_counter()
                digest, count = _array_hash(supported), int(np.count_nonzero(supported))
                del supported
                summary_seconds = time.perf_counter() - mark
                frame["runs"].append({"iteration": iteration + 1, "phase": "first" if iteration == 0 else "repeat",
                                      "mapping_setup_seconds": mapping_seconds, "support_seconds": support_seconds,
                                      "hash_and_count_seconds": summary_seconds,
                                      "sha256": digest, "supported_pixels": count})
                if iteration == 0:
                    frame.update(sha256=digest, supported_pixels=count,
                                 matches_recorded_count=count == prior[stem]["supported_pixels"])
                elif digest != frame["sha256"] or count != frame["supported_pixels"]:
                    raise ValueError(f"repeated support changed: {stem}")
                if not frame["matches_recorded_count"]:
                    raise ValueError(f"support count differs from old job: {stem}")
            del valid
        report["state"] = "succeeded"
    except (Exception, KeyboardInterrupt) as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc) or "cancelled"}
    finally:
        mark = time.perf_counter()
        report["source_after"] = _manifest(Path(path) for path in before)
        report["code_after"] = _code_manifest()
        report["source_unchanged"] = report["source_after"] == before
        report["code_unchanged"] = report["code_after"] == report["code_before"]
        if not report["source_unchanged"] or not report["code_unchanged"]:
            report["state"] = "failed"
            report["integrity_error"] = "source artifact or source code changed during replay"
        report["final_hash_seconds"] = time.perf_counter() - mark
        report["total_seconds"] = time.perf_counter() - started
        atomic_json(destination / "replay.json", report)
        (destination / "REPORT.md").write_text(_report_text(report), encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=NOTICE)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/support-replay")
    parser.add_argument("--frames", nargs="+", help="selected export stems, e.g. light_00001")
    parser.add_argument("--repeat", type=positive, default=1)
    parser.add_argument("--rows", type=positive, help="positive tile rows; omitted = production automatic")
    parser.add_argument("--workers", type=positive,
                        help="replay the product's buffered kernel and native mask on the frame pool with this many "
                             "threads (P1c F-2 / G-4t); without --frames: every export frame")
    args = parser.parse_args(argv)
    try:
        report = replay(args.job, output_root=args.output_root, frames=args.frames, repeat=args.repeat, rows=args.rows,
                        workers=args.workers)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({"state": report["state"], "output": report["output"], "notice": NOTICE}, ensure_ascii=False))
    return 0 if report["state"] == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
