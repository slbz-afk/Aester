"""Small IO-contract tests; no RAWs, real Siril, or performance workloads."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest


_PATH = Path(__file__).resolve().parents[1] / "scripts/replay_support.py"
_SPEC = importlib.util.spec_from_file_location("support_replay_tool", _PATH)
tool = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(tool)


def _write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def job_factory(tmp_path, monkeypatch):
    code = tmp_path / "fake_src"
    code.mkdir()
    (code / "sample.py").write_text("pass\n", encoding="utf-8")
    monkeypatch.setattr(tool, "SOURCE_ROOT", code)
    engine = tmp_path / "engine.exe"
    engine.write_bytes(b"test engine (not executable)")
    monkeypatch.setattr(tool.rs, "SIRIL_SHA256", _hash(engine))
    monkeypatch.setattr(tool.rs, "INTERPOLATION_DLLS", {})
    engine_checks = []
    monkeypatch.setattr(tool.rs, "require_engine", lambda value: engine_checks.append(value))

    def create(count=3, sip=False):
        directory = tmp_path / f"job-{count}-{int(sip)}"
        work = directory / "work"
        work.mkdir(parents=True)
        Image.fromarray(np.full((24, 32), 255, dtype=np.uint8)).save(directory / "sky_mask.png")
        mask_sha = _hash(directory / "sky_mask.png")
        stems = [f"light_{i:05d}" for i in range(1, count + 1)]
        identity = np.eye(3).tolist()
        sequence = [f"S 'light_' 1 {count} {count} 5 {count//2} 6 0 0 0", "L 3"]
        sequence += [f"I {i} 1" for i in range(1, count + 1)]
        sequence += ["R1 4 4 0.7 0 0.02 20 H 1 0 0 0 1 0 0 0 1"] * count
        if sip:
            sequence.append("D1 2 distortion.wcs")
            from astropy.io import fits
            header = fits.Header({"CTYPE1": "RA---TAN-SIP", "CTYPE2": "DEC--TAN-SIP",
                                  "CRPIX1": 16., "CRPIX2": 12., "AP_ORDER": 0, "BP_ORDER": 0})
            fits.PrimaryHDU(header=header).writeto(work / "distortion.wcs")
        (work / tool.rs.EXPORT_SEQUENCE).write_text("\n".join(sequence) + "\n", encoding="utf-8")
        expected = np.zeros((24, 32), dtype=bool)
        expected[5:18, 5:26] = True  # full identity footprint, then output cross
        old = {"state": "complete", "mode": tool.rs.MODE, "version": tool.rs.SUPPORT_VERSION,
               "shape": [24, 32], "model": {} if sip else None,
               "wcs_sha256": _hash(work / "distortion.wcs") if sip else None,
               "sequence_sha256": _hash(work / tool.rs.EXPORT_SEQUENCE), "mask_sha256": mask_sha,
               "engine_sha256": _hash(engine), "interpolation_dll_sha256": {},
               "frames": [{"stem": stem, "H_stored": identity,
                           "source_valid_sha256": tool._array_hash(np.ones((24, 32), dtype=bool)),
                           "supported_pixels": int(expected.sum())} for stem in stems]}
        _write_json(work / tool.rs.RECORD, old)
        _write_json(work / "sky_validity.json", {"dilation_px": 3, "mask_sha256": mask_sha})
        _write_json(directory / "job.json", {"state": "succeeded", "support_mode": tool.rs.MODE,
                    "support_version": tool.rs.SUPPORT_VERSION,
                    "engine": {"executable": str(engine), "version": "1.4.4"},
                    "mask": {"sha256": mask_sha, "path": "absent-original-mask.png"}, "frame_masks": None})
        return directory, expected, engine_checks
    return create


def test_default_replay_readonly_unique_and_repeat(job_factory, tmp_path, monkeypatch):
    directory, expected, checks = job_factory()
    before = {str(p): _hash(p) for p in directory.rglob("*") if p.is_file()}
    anchor_calls = []
    original = tool.rs.build_anchor_grid

    def anchor(*args, **kwargs):
        anchor_calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(tool.rs, "build_anchor_grid", anchor)
    report = tool.replay(directory, output_root=tmp_path / "out", repeat=2, rows=2)
    assert report["state"] == "succeeded", report
    assert report["support_implementation"] == tool.rs.REFERENCE_IMPLEMENTATION   # the path the replay runs
    assert report["selected_frames"] == ["light_00001", "light_00002", "light_00003"]
    assert len(checks) == 1 and len(anchor_calls) == 1
    for frame in report["frames"]:
        assert frame["sha256"] == tool._array_hash(expected)
        assert frame["supported_pixels"] == 273
        assert [r["phase"] for r in frame["runs"]] == ["first", "repeat"]
    assert [f["initialization"]["anchor_cache"] for f in report["frames"]] == ["cold", "hot", "hot"]
    assert report["source_unchanged"] and report["code_unchanged"]
    assert before == {str(p): _hash(p) for p in directory.rglob("*") if p.is_file()}
    second = tool.replay(directory, output_root=tmp_path / "out", frames=["light_00001"])
    assert second["output"] != report["output"]
    result_dir = Path(report["output"])
    assert {p.name for p in result_dir.iterdir()} == {"replay.json", "REPORT.md"}
    text = (result_dir / "REPORT.md").read_text(encoding="utf-8")
    assert tool.NOTICE in text and "不能证明逐像素一致" in text


def test_explicit_dedup_preserves_order(job_factory, tmp_path):
    directory, _, _ = job_factory()
    report = tool.replay(directory, output_root=tmp_path / "out",
                         frames=["light_00003", "light_00001", "light_00003"])
    assert report["state"] == "succeeded"
    assert report["selected_frames"] == ["light_00003", "light_00001"]


def test_single_frame_defaults_deduplicated(job_factory, tmp_path):
    directory, _, _ = job_factory(count=1)
    report = tool.replay(directory, output_root=tmp_path / "out")
    assert report["state"] == "succeeded"
    assert report["selected_frames"] == ["light_00001"]


@pytest.mark.parametrize("flag,value", [("--repeat", "0"), ("--repeat", "-1"),
                                         ("--repeat", "1.5"), ("--rows", "0")])
def test_cli_rejects_invalid_numbers_before_io(tmp_path, flag, value):
    out = tmp_path / "out"
    with pytest.raises(SystemExit) as caught:
        tool.main(["--job", str(tmp_path / "missing"), "--output-root", str(out), flag, value])
    assert caught.value.code == 2
    assert not out.exists()


@pytest.mark.parametrize("field,value", [("state", "failed"), ("support_mode", "legacy")])
def test_refuses_failed_or_legacy_job(job_factory, tmp_path, field, value):
    directory, _, checks = job_factory()
    job = json.loads((directory / "job.json").read_text())
    job[field] = value
    _write_json(directory / "job.json", job)
    report = tool.replay(directory, output_root=tmp_path / "out")
    assert report["state"] == "failed" and not checks
    assert "succeeded strict-support" in report["error"]["message"]
    assert Path(report["output"], "replay.json").is_file()


def test_unknown_stem_failed_report_and_cli_exit(job_factory, tmp_path):
    directory, _, checks = job_factory()
    assert tool.main(["--job", str(directory), "--output-root", str(tmp_path / "out"),
                      "--frames", "missing"]) == 1
    report = json.loads(next((tmp_path / "out").glob("*/replay.json")).read_text(encoding="utf-8"))
    assert report["state"] == "failed" and not checks
    assert "not selected export stems" in report["error"]["message"]


def test_output_cannot_write_inside_old_job(job_factory):
    directory, _, _ = job_factory()
    with pytest.raises(ValueError, match="outside the old job"):
        tool.replay(directory, output_root=directory / "replays")
    assert not (directory / "replays").exists()


@pytest.mark.parametrize("change", ["mask", "code"])
def test_source_or_code_changed_during_replay_fails(job_factory, tmp_path, monkeypatch, change):
    directory, _, _ = job_factory()
    original = tool.rs.registered_mask

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        path = directory / "sky_mask.png" if change == "mask" else tool.SOURCE_ROOT / "sample.py"
        with path.open("ab") as stream:
            stream.write(b"changed")
        return result
    monkeypatch.setattr(tool.rs, "registered_mask", mutate)
    report = tool.replay(directory, output_root=tmp_path / "out", frames=["light_00001"])
    assert report["state"] == "failed"
    assert report["integrity_error"]
    assert not report["source_unchanged" if change == "mask" else "code_unchanged"]


def test_cancelled_first_frame_still_writes_failure(job_factory, tmp_path, monkeypatch):
    directory, _, _ = job_factory()

    def cancel(*args, **kwargs):
        raise KeyboardInterrupt
    monkeypatch.setattr(tool.rs, "registered_mask", cancel)
    report = tool.replay(directory, output_root=tmp_path / "out")
    assert report["state"] == "failed" and report["error"]["type"] == "KeyboardInterrupt"
    assert Path(report["output"], "REPORT.md").is_file()


def test_selected_local_per_frame_mask_only(job_factory, tmp_path):
    directory, _, _ = job_factory()
    (directory / "frame_masks").mkdir()
    path = directory / "frame_masks/light_00001.png"
    Image.fromarray(np.zeros((24, 32), dtype=np.uint8)).save(path)
    job = json.loads((directory / "job.json").read_text())
    job["frame_masks"] = {"files": {"light_00001": {"sha256": _hash(path), "path": "missing-original.png"},
                                     "light_00002": {"sha256": "missing-unselected-mask"}}}
    _write_json(directory / "job.json", job)
    old = json.loads((directory / "work" / tool.rs.RECORD).read_text())
    old["frames"][0].update(source_valid_sha256=tool._array_hash(np.zeros((24, 32), dtype=bool)), supported_pixels=0)
    _write_json(directory / "work" / tool.rs.RECORD, old)
    report = tool.replay(directory, output_root=tmp_path / "out", frames=["light_00001"])
    assert report["state"] == "succeeded", report
    assert str(path) in report["source_before"]
    assert report["frames"][0]["supported_pixels"] == 0


def test_old_count_disagreement_is_not_silently_passed(job_factory, tmp_path):
    directory, _, _ = job_factory()
    path = directory / "work" / tool.rs.RECORD
    old = json.loads(path.read_text())
    old["frames"][0]["supported_pixels"] += 1
    _write_json(path, old)
    report = tool.replay(directory, output_root=tmp_path / "out", frames=["light_00001"])
    assert report["state"] == "failed"
    assert report["frames"][0]["matches_recorded_count"] is False


def test_sip_grid_initialization_separate_from_support(job_factory, tmp_path):
    directory, expected, _ = job_factory(count=1, sip=True)
    report = tool.replay(directory, output_root=tmp_path / "out", repeat=2)
    assert report["state"] == "succeeded", report
    assert report["initialization"]["sip_grid_cold_seconds"] >= 0
    assert report["initialization"]["sip_grid_hot_lookup_seconds"] >= 0
    assert str(directory / "work/distortion.wcs") in report["source_before"]
    assert report["frames"][0]["sha256"] == tool._array_hash(expected)


# --------------------------------------------------------------------------- --workers (efficiency P1c F-2 / G-4t)

@pytest.mark.parametrize("sip", [False, True])
def test_workers_replay_runs_the_product_kernel_on_the_frame_pool(job_factory, tmp_path, sip):
    directory, expected, checks = job_factory(count=4, sip=sip)
    old = json.loads((directory / "work" / tool.rs.RECORD).read_text())
    old["frames"][1]["explicit_validity"] = {"file": "frame_support_light_00002.npy",
                                             "sha256": tool._npy_digest(expected)}
    _write_json(directory / "work" / tool.rs.RECORD, old)
    before = {str(p): _hash(p) for p in directory.rglob("*") if p.is_file()}
    report = tool.replay(directory, output_root=tmp_path / "out", workers=3, repeat=2)
    assert report["state"] == "succeeded", report
    assert report["support_implementation"] == tool.rs.SUPPORT_IMPLEMENTATION
    assert report["selected_frames"] == [f"light_{i:05d}" for i in range(1, 5)]          # every export frame
    assert [f["sha256"] for f in report["frames"]] == [tool._array_hash(expected)] * 4
    assert [f["matches_retained_map"] for f in report["frames"]] == [None, True, None, None]
    pool = report["pool"]
    assert [run["pool"]["workers"] for run in pool["runs"]] == [3, 3] and pool["retained_maps_compared"] == 1
    assert pool["native_matches_record"] is None                     # the record names no quick native route
    assert len(checks) == 1 and report["source_unchanged"] and report["code_unchanged"]
    assert before == {str(p): _hash(p) for p in directory.rglob("*") if p.is_file()}
    assert "帧池回放" in Path(report["output"], "REPORT.md").read_text(encoding="utf-8")


def test_workers_replay_refuses_a_different_retained_map_and_rows(job_factory, tmp_path):
    directory, expected, _ = job_factory()
    old = json.loads((directory / "work" / tool.rs.RECORD).read_text())
    old["frames"][2]["explicit_validity"] = {"file": "frame_support_light_00003.npy", "sha256": "0" * 64}
    _write_json(directory / "work" / tool.rs.RECORD, old)
    report = tool.replay(directory, output_root=tmp_path / "out", workers=2)
    assert report["state"] == "failed" and "retained map" in report["error"]["message"]
    with pytest.raises(ValueError, match="without rows"):
        tool.replay(directory, output_root=tmp_path / "out", workers=2, rows=4)
    with pytest.raises(SystemExit):
        tool.main(["--job", str(directory), "--output-root", str(tmp_path / "out"), "--workers", "0"])
