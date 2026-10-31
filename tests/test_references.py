"""Reference geometry and failure contracts using controlled Siril products."""
from pathlib import Path
from types import SimpleNamespace
import json
import shutil

import numpy as np
import pytest
from astropy.io import fits
from PIL import Image

from starstack import core, references


def scene():
    top = np.full((13, 19, 3), .03, dtype=np.float32)
    top[0, :] = (.8, .05, .05)
    top[-1, :] = (.05, .05, .8)
    top[3, 5] = (.05, .9, .05)
    return top


def write_frame(path, top, roworder="BOTTOM-UP"):
    header = fits.Header({"ROWORDER": roworder, "INSTRUME": "Controlled camera",
                          "EXPTIME": 15., "ISOSPEED": 6400, "FOCALLEN": 24.})
    rows = top[::-1] if roworder == "BOTTOM-UP" else top
    fits.writeto(path, np.moveaxis(rows, -1, 0), header)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    source = tmp_path / "参考素材 with spaces"
    source.mkdir()
    for name in ["c.fit", "a.fit", "b.fit"]:
        write_frame(source / name, scene())
    output = tmp_path / "参考结果 with spaces"
    existing = output / "job-existing"
    existing.mkdir(parents=True)
    (existing / "job.json").write_text("keep existing job", encoding="utf-8")
    monkeypatch.setattr(core, "doctor", lambda path=None: {
        "available": True, "executable": "controlled-not-executed.exe", "version": "1.4.4"})
    return SimpleNamespace(source=source, output=output, existing=existing,
                           originals={p.name: p.read_bytes() for p in source.iterdir()})


class ControlledSiril:
    def __init__(self, monkeypatch, *, count=1, roworder="BOTTOM-UP", fail=False, pixels=None):
        self.calls = []
        self.count, self.roworder, self.fail = count, roworder, fail
        self.pixels = scene() if pixels is None else pixels
        monkeypatch.setattr(references.siril, "run_script", self.run)

    def run(self, executable, directory, name, commands, cancel_check):
        self.calls.append((executable, directory, name, commands))
        cancel_check()
        (directory / f"{name}.log").write_text("Controlled Siril log\n", encoding="utf-8")
        assert len(list((directory / "stage").iterdir())) == 1
        if self.fail:
            raise RuntimeError("controlled conversion failure")
        for index in range(self.count):
            write_frame(directory / "work" / f"ref_{index + 1:05d}.fit", self.pixels, self.roworder)


def failed_record(output):
    manifests = list(output.glob("reference-*/reference.json"))
    assert len(manifests) == 1
    record = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert record["state"] == "failed"
    assert record["error"]
    assert Path(record["log_path"]).is_file()
    return record


def test_reference_preserves_full_geometry_orientation_sources_and_existing_jobs(setup, monkeypatch):
    engine = ControlledSiril(monkeypatch)
    result = references.prepare_reference(str(setup.source), str(setup.output))
    assert result["state"] == "succeeded"
    assert result["source_path"] == str(setup.source / "a.fit")
    assert result["width"] == 19 and result["height"] == 13
    assert result["no_crop"] is True and result["no_scale"] is True
    assert result["linear_row_order"] == "BOTTOM-UP"
    assert result["preview_row_order"] == result["mask_row_order"] == "TOP-DOWN"
    with Image.open(result["preview_path"]) as preview:
        assert preview.size == (19, 13) and preview.mode == "RGB"
        assert preview.info["icc_profile"]
        pixels = np.asarray(preview)
    # Independent asymmetric landmarks detect double flips and axis swaps.
    assert np.all(pixels[0, :, 0] > pixels[0, :, 2])
    assert np.all(pixels[-1, :, 2] > pixels[-1, :, 0])
    assert pixels[3, 5, 1] > pixels[3, 5, 0]
    with Image.open(result["mask_template_path"]) as mask:
        assert mask.size == (19, 13) and mask.mode == "L"
        assert np.all(np.asarray(mask) == 255)
        assert "not a sky segmentation" in mask.info["Purpose"]
    assert result["metadata"]["mask"]["is_segmentation"] is False
    assert result["metadata"]["rendering"]["color_calibrated"] is False
    assert result["metadata"]["fits"]["EXPTIME"] == 15.
    assert result["metadata"]["fits"]["INSTRUME"] == "Controlled camera"
    assert result["source_sha256"] == result["staged_sha256"] == core._sha(setup.source / "a.fit")
    for path_key, hash_key in [("linear_path", "linear_sha256"), ("preview_path", "preview_sha256"),
                               ("mask_template_path", "mask_template_sha256")]:
        assert core._sha(Path(result[path_key])) == result[hash_key]
    persisted = json.loads((Path(result["reference_dir"]) / "reference.json").read_text(encoding="utf-8"))
    assert persisted == result
    assert engine.calls[0][3] == ["setext fit", "set32bits", "setcompress 0", "cd stage", "convert ref -out=../work"]
    staged = list((Path(result["reference_dir"]) / "stage").iterdir())
    assert len(staged) == 1 and staged[0].read_bytes() == setup.originals["a.fit"]
    assert {p.name: p.read_bytes() for p in setup.source.iterdir()} == setup.originals
    assert (setup.existing / "job.json").read_text(encoding="utf-8") == "keep existing job"


def test_each_reference_uses_a_new_directory(setup, monkeypatch):
    ControlledSiril(monkeypatch)
    first = references.prepare_reference(str(setup.source), str(setup.output))
    before = Path(first["preview_path"]).read_bytes()
    second = references.prepare_reference(str(setup.source), str(setup.output))
    assert first["reference_dir"] != second["reference_dir"]
    assert Path(first["preview_path"]).read_bytes() == before


@pytest.mark.parametrize("options", [{"count": 0}, {"count": 2}, {"roworder": "TOP-DOWN"}, {"fail": True}])
def test_invalid_conversion_keeps_failure_record_and_engine_log(setup, monkeypatch, options):
    ControlledSiril(monkeypatch, **options)
    with pytest.raises(RuntimeError, match="参考图准备失败"):
        references.prepare_reference(str(setup.source), str(setup.output))
    record = failed_record(setup.output)
    assert "preview_path" not in record and "mask_template_path" not in record
    assert (Path(record["reference_dir"]) / "01_convert.log").read_text(encoding="utf-8") == "Controlled Siril log\n"
    assert not list(Path(record["reference_dir"]).glob("*.png"))
    assert {p.name: p.read_bytes() for p in setup.source.iterdir()} == setup.originals


def test_nonfinite_converted_pixels_cannot_publish_reference(setup, monkeypatch):
    pixels = scene()
    pixels[2, 4, 1] = np.nan
    ControlledSiril(monkeypatch, pixels=pixels)
    with pytest.raises(RuntimeError, match="有限值"):
        references.prepare_reference(str(setup.source), str(setup.output))
    assert failed_record(setup.output)["stage"] == "validate"


def test_input_admission_still_requires_three_frames(setup, monkeypatch):
    (setup.source / "c.fit").unlink()
    engine = ControlledSiril(monkeypatch)
    with pytest.raises(RuntimeError, match="至少 3 张"):
        references.prepare_reference(str(setup.source), str(setup.output))
    assert failed_record(setup.output)["stage"] == "inspect"
    assert not engine.calls


def test_unavailable_engine_fails_without_copying_or_conversion(setup, monkeypatch):
    engine = ControlledSiril(monkeypatch)
    monkeypatch.setattr(core, "doctor", lambda *_: {"available": False, "message": "unsupported engine version"})
    with pytest.raises(RuntimeError, match="unsupported engine version"):
        references.prepare_reference(str(setup.source), str(setup.output), "requested.exe")
    assert failed_record(setup.output)["stage"] == "engine"
    assert not engine.calls


def test_copy_hash_mismatch_is_a_recorded_failure(setup, monkeypatch):
    engine = ControlledSiril(monkeypatch)
    original_copy = shutil.copyfile

    def damaged_copy(source, target):
        original_copy(source, target)
        with Path(target).open("ab") as stream:
            stream.write(b"controlled damage to copy only")

    monkeypatch.setattr(references.shutil, "copyfile", damaged_copy)
    with pytest.raises(RuntimeError, match="复制时发生变化"):
        references.prepare_reference(str(setup.source), str(setup.output))
    assert failed_record(setup.output)["stage"] == "copy"
    assert not engine.calls
    assert {p.name: p.read_bytes() for p in setup.source.iterdir()} == setup.originals


def test_mask_save_failure_does_not_publish_partial_preview_as_success(setup, monkeypatch):
    ControlledSiril(monkeypatch)
    original_save = Image.Image.save

    def fail_mask(self, path, *args, **kwargs):
        if Path(path).name == "sky_mask_template.png":
            raise OSError("controlled mask save failure")
        return original_save(self, path, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "save", fail_mask)
    with pytest.raises(RuntimeError, match="controlled mask save failure"):
        references.prepare_reference(str(setup.source), str(setup.output))
    record = failed_record(setup.output)
    assert record["stage"] == "render"
    assert "preview_path" not in record and "mask_template_path" not in record
    assert (Path(record["reference_dir"]) / "reference_preview.png").exists()


def test_success_manifest_write_failure_is_recorded_as_failure(setup, monkeypatch):
    ControlledSiril(monkeypatch)
    original_write = core.atomic_json

    def fail_success(path, record):
        if record["state"] == "succeeded":
            raise PermissionError("controlled manifest write failure")
        return original_write(path, record)

    monkeypatch.setattr(core, "atomic_json", fail_success)
    with pytest.raises(RuntimeError, match="controlled manifest write failure"):
        references.prepare_reference(str(setup.source), str(setup.output))
    assert failed_record(setup.output)["stage"] == "complete"


@pytest.mark.parametrize("metadata_available", [False, True])
def test_raw_uses_pipeline_decode_and_optional_metadata_without_color_application(
        tmp_path, monkeypatch, metadata_available):
    source = tmp_path / "RAW 素材"
    source.mkdir()
    for name in ["z.CR2", "a.CR2", "m.CR2"]:
        (source / name).write_bytes(b"Controlled source, not an actual camera RAW: " + name.encode())
    monkeypatch.setattr(core, "doctor", lambda *_: {"available": True, "version": "1.4.4", "executable": "fake"})
    engine = ControlledSiril(monkeypatch)
    read_paths = []

    def reader(path):
        read_paths.append(path)
        return {"available": True, "status": "partial", "fields": {"camera_whitebalance": {
            "value": [2., 1., 1.5, 1.], "valid": True}}, "white_balance_applied": False,
                "color_correction_applied": False}

    def module(*args, **kwargs):
        if not metadata_available:
            raise ModuleNotFoundError("optional rawpy is unavailable")
        return SimpleNamespace(read_raw_metadata=reader)

    monkeypatch.setattr(references, "import_module", module)
    result = references.prepare_reference(str(source), str(tmp_path / "output"))
    assert engine.calls[0][3][-1] == "convertraw ref -debayer -out=../work"
    assert result["source_path"] == str(source / "a.CR2")
    assert (result["width"], result["height"]) == (19, 13)  # decoded dimensions, not a thumbnail
    raw = result["metadata"]["raw"]
    assert raw["available"] is metadata_available
    if metadata_available:
        assert read_paths == [Path(result["staged_path"])]
        assert raw["fields"]["camera_whitebalance"]["value"] == [2., 1., 1.5, 1.]
    else:
        assert raw["status"] == "unknown"
    assert result["metadata"]["rendering"]["additional_camera_white_balance_applied"] is False
    with Image.open(result["preview_path"]) as preview:
        pixels = np.asarray(preview)
        # No camera gain: the identical R and B input at this neutral pixel
        # remains identical after the common-channel display function.
        assert pixels[5, 5, 0] == pixels[5, 5, 2]


def test_output_cannot_be_the_source_directory(setup):
    with pytest.raises(ValueError, match="必须分开"):
        references.prepare_reference(str(setup.source), str(setup.source))
    assert {p.name: p.read_bytes() for p in setup.source.iterdir()} == setup.originals


def test_optional_reader_unavailability_is_unknown_with_original_reason(monkeypatch):
    reported = {"available": False, "status": "rawpy_unavailable", "fields": {},
                "errors": ["optional library missing"]}
    monkeypatch.setattr(references, "import_module", lambda *args, **kwargs:
                        SimpleNamespace(read_raw_metadata=lambda path: dict(reported)))
    result = references._optional_raw_metadata(Path("unopened-controlled.CR2"), "raw")
    assert result["status"] == "unknown"
    assert result["reader_status"] == "rawpy_unavailable"
    assert result["errors"] == reported["errors"]
    assert reported["status"] == "rawpy_unavailable"


def test_single_frame_conversion_has_five_minute_deadline_and_records_failure(setup, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(references.time, "monotonic", lambda: clock[0])
    cleanup_observed = []

    def stalled_engine(executable, directory, name, commands, cancel_check):
        (directory / f"{name}.log").write_text("Controlled conversion is still running\n", encoding="utf-8")
        try:
            cancel_check()
            clock[0] = 1299.0
            cancel_check()  # A slow but still permitted conversion gets time.
            clock[0] = 1300.0
            cancel_check()
            pytest.fail("The engine must be interrupted at the five-minute deadline")
        finally:
            cleanup_observed.append(True)

    monkeypatch.setattr(references.siril, "run_script", stalled_engine)
    with pytest.raises(RuntimeError, match="超过 5 分钟") as error:
        references.prepare_reference(str(setup.source), str(setup.output))
    assert isinstance(error.value.__cause__, TimeoutError)
    assert cleanup_observed == [True]
    record = failed_record(setup.output)
    assert record["stage"] == "convert"
    assert record["conversion_timeout_seconds"] == 300
    assert "preview_path" not in record and "mask_template_path" not in record
    assert not list(Path(record["reference_dir"]).glob("*.png"))
    assert (Path(record["reference_dir"]) / "01_convert.log").is_file()
    assert {p.name: p.read_bytes() for p in setup.source.iterdir()} == setup.originals
