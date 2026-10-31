"""Two-pass registration in the product flow: commands and actual-reference reporting.

The Siril one-pass flow (``setref light 1`` + ``register``) detected the
reference at 16-bit and the other frames at 32-bit on the real Z8 job, so
long-span frames lost star pairs and smeared 1-2 px in the corners
(outputs/fable-s2-20260917/RUN_LOG.md §14).  The product now lets Siril run
two-pass registration and pick the reference; the job diagnostics must report
the frame Siril actually used instead of assuming the first input.
"""
from pathlib import Path
import importlib.util
import json
import shutil
import tempfile
import unittest
from types import SimpleNamespace

from starstack import core

HAVE_STACK = all(importlib.util.find_spec(name) for name in ("numpy", "astropy", "tifffile", "PIL"))


def _seq_text(count: int, reference_position: int, *, indices=None, with_regdata=True) -> str:
    """Minimal Siril 1.4 ``.seq`` text (version 6) for ``count`` images."""
    indices = list(indices) if indices is not None else list(range(1, count + 1))
    lines = ["#Siril sequence file. Contains list of images, selection, registration data and statistics",
             f"S 'light_' 1 {count} {count} 5 {reference_position} 6 0 0 0", "L 3"]
    lines += [f"I {index} 1" for index in indices]
    if with_regdata:
        for position in range(count):
            shift = 0.0 if position == reference_position else 1.5 * (position - reference_position)
            lines.append(f"R1 3.5 3.6 0.7 0 1500 1800 H 1 0 {shift} 0 1 {shift} 0 0 1")
    return "\n".join(lines) + "\n"

def admit_before_contract(case):
    """Jobs this controlled engine runs are admitted as before the EFF-D4 stack-settings contract (efficiency P1c
    §4.2): it models Siril's file contract before it (no settings lines, no stack log lines, no STACKCNT).  A
    TestCase calls it in setUp, so plain unittest runs get it too; contract jobs: tests/test_stack_settings.py."""
    from starstack import stack_settings
    saved = stack_settings.admission_field
    stack_settings.admission_field = lambda *args, **kwargs: None
    case.addCleanup(setattr, stack_settings, "admission_field", saved)


class StackScriptTest(unittest.TestCase):
    def test_probe_and_distortion_aware_scripts(self):
        self.assertEqual(core.PROBE_SCRIPT, ["cd work", "register light -2pass -transf=homography"])
        with_disto = core.stack_script(core.DISTORTION_WCS)
        self.assertEqual(with_disto[1], "register light -2pass -transf=homography -disto=file distortion.wcs")
        self.assertEqual(with_disto[2], "seqapplyreg light -filter-included")
        self.assertEqual(with_disto[-1], core.STACK_COMMAND)
        self.assertEqual(core.stack_script(None), core.STACK_SCRIPT)
        self.assertIn("-disto=file x.wcs", core.registration_commands("x.wcs")[0])
        # RUN_LOG §22: with our own matrices in the sequence file Siril only applies them
        self.assertEqual(core.OWN_REGISTRATION_STACK_SCRIPT, ["cd work", "seqapplyreg light -filter-included", core.STACK_COMMAND])
        self.assertEqual(core.registration_commands(core.DISTORTION_WCS, own_registration=True), ["seqapplyreg light -filter-included"])
        self.assertNotIn("register", " ".join(core.OWN_REGISTRATION_STACK_SCRIPT[1:2]))

    def test_script_uses_two_pass_then_applies_then_stacks(self):
        script = core.STACK_SCRIPT
        self.assertEqual(script[0], "cd work")
        self.assertEqual(script[1], "register light -2pass -transf=homography")
        self.assertEqual(script[2], "seqapplyreg light -filter-included")
        self.assertTrue(script[-1].startswith("stack r_light "))
        self.assertIn("-32b", script[-1])
        self.assertTrue(script[-1].endswith("-out=sky_master.fit"))
        self.assertFalse(any(cmd.startswith("setref") for cmd in script), "the reference is Siril's choice now")
        self.assertEqual(tuple(script[1:3]), core.REGISTRATION_COMMANDS)

    def test_registration_options_stay_within_documented_set(self):
        # ``-framing``/``-prefix`` belong to seqapplyreg; ``register`` keeps only the options RUN_LOG §14 verified.
        register, apply = core.REGISTRATION_COMMANDS
        self.assertEqual(sorted(tok for tok in register.split() if tok.startswith("-")), ["-2pass", "-transf=homography"])
        self.assertEqual(apply.split(), ["seqapplyreg", "light", "-filter-included"])  # default prefix r_, framing current (reference)

    def test_sequence_names_follow_siril_convention(self):
        self.assertEqual(core.SEQUENCE_FILE, "light_.seq")
        self.assertEqual(core.SEQUENCE_SNAPSHOT, "light_.initial.seq")


class SequenceReferenceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="s2-core-ref-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.inputs = [{"path": f"C:\\shots\\DSC_{2204 + i}.dng"} for i in range(16)]

    def _write(self, text: str, name: str = "light_.seq") -> Path:
        path = self.tmp / name
        path.write_text(text, encoding="utf-8")
        return path

    def test_reports_the_frame_siril_chose(self):
        path = self._write(_seq_text(16, 10))  # real two-pass run: S ... reference_image 10 -> light_00011
        report = core.sequence_reference(path, self.inputs)
        self.assertEqual(report["reference_selection"], core.REFERENCE_SELECTION)
        self.assertEqual(report["reference_position"], 10)
        self.assertEqual(report["reference_stem"], "light_00011")
        self.assertEqual(report["reference_input"], self.inputs[10]["path"])
        self.assertEqual(report["reference_sequence"], str(path))
        self.assertNotIn("reference_error", report)

    def test_first_frame_reference_still_maps_to_first_input(self):
        report = core.sequence_reference(self._write(_seq_text(3, 0)), self.inputs[:3])
        self.assertEqual(report["reference_input"], self.inputs[0]["path"])
        self.assertEqual(report["reference_stem"], "light_00001")

    def test_binding_uses_image_numbers_not_positions(self):
        # A sequence that lost image 2 binds position 1 to light_00003 -> inputs[2].
        report = core.sequence_reference(self._write(_seq_text(2, 1, indices=[1, 3])), self.inputs[:3])
        self.assertEqual(report["reference_stem"], "light_00003")
        self.assertEqual(report["reference_input"], self.inputs[2]["path"])

    def test_missing_sequence_is_reported_unknown_not_guessed(self):
        report = core.sequence_reference(self.tmp / "light_.seq", self.inputs)
        self.assertEqual(report["reference_selection"], "unknown")
        self.assertIsNone(report["reference_input"])
        self.assertIsNone(report["reference_stem"])
        self.assertIn("FileNotFoundError", report["reference_error"])

    def test_unreadable_or_unregistered_sequence_is_unknown(self):
        for text in ("garbage\n", "S 'light_' 1 2 2 5 -1 6 0 0 0\nL 3\nI 1 1\nI 2 1\n"):
            report = core.sequence_reference(self._write(text), self.inputs)
            self.assertEqual(report["reference_selection"], "unknown", text)
            self.assertIsNone(report["reference_input"])
            self.assertIn("reference_error", report)

    def test_image_number_beyond_inputs_is_unknown_with_context(self):
        report = core.sequence_reference(self._write(_seq_text(2, 1, indices=[1, 7])), self.inputs[:3])
        self.assertEqual(report["reference_selection"], "unknown")
        self.assertIsNone(report["reference_input"])
        self.assertEqual(report["reference_stem"], "light_00007")
        self.assertIn("no staged input", report["reference_error"])


@unittest.skipUnless(HAVE_STACK, "product imaging stack (numpy/astropy/tifffile/PIL) not installed")
class JobDiagnosticsTest(unittest.TestCase):
    """Run the real ``_run_locked`` flow with a controlled engine that writes Siril's sequence file."""

    def setUp(self):
        import numpy as np
        from starstack import imaging
        self.tmp = Path(tempfile.mkdtemp(prefix="s2-core-job-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        source = self.tmp / "素材 with spaces"
        source.mkdir()
        for number in range(3):
            data = np.full((18, 24, 3), .03 + number * .001, dtype=np.float32)
            data[3:6, 5:8] += .2
            imaging.write_linear(source / f"原片 {number}.fit", data, "Test fixture")
        self.source = source
        self.output = self.tmp / "results"
        self.engine = {"available": True, "executable": str(self.tmp / "fake engine.exe"),
                       "version": "1.4.4", "message": "Test engine"}
        self.saved = {name: getattr(core, name) for name in ("doctor", "_spawn_worker", "run_script")}
        admit_before_contract(self)
        self.saved_disk_usage = core.shutil.disk_usage
        core.doctor = lambda *_: dict(self.engine)
        core._spawn_worker = lambda directory: None
        core.shutil.disk_usage = lambda _: SimpleNamespace(free=10**12)
        self.scripts = []
        # The pre-P1b single L script; since P1b-3 it runs as an export process (all of it but the stack command)
        # followed by a stack process (core.LEGACY_STACK_SCRIPT).
        self.expected_stack_script = core.STACK_SCRIPT

        def run_script(executable, directory, name, commands, cancel_check):
            import numpy as np
            from starstack import imaging
            self.scripts.append((name, list(commands)))
            work = directory / "work"
            if name == "01_convert":
                for i, path in enumerate(sorted((directory / "stage").glob("*")), 1):
                    shutil.copyfile(path, work / f"light_{i:05d}.fit")
            elif name == "02_probe":
                self.assertEqual(list(commands), core.PROBE_SCRIPT)   # transforms-only probe; the controlled engine writes no sequence
            elif name == "02_export":
                self.assertEqual(self.expected_stack_script[-1], core.STACK_COMMAND)
                self.assertEqual(list(commands), list(self.expected_stack_script[:-1]))
                frames = sorted(work.glob("light_*.fit"))
                (work / core.SEQUENCE_FILE).write_text(_seq_text(len(frames), 1), encoding="utf-8")
            elif name == "02_stack":
                self.assertEqual(list(commands), list(core.LEGACY_STACK_SCRIPT))
                frames = sorted(work.glob("light_*.fit"))
                data = np.mean([imaging.read_rgb(p) for p in frames], axis=0).astype(np.float32)
                imaging.write_linear(work / "sky_master.fit", data, "Controlled engine, not Siril")
            else:
                raise AssertionError(name)

        core.run_script = run_script

    def tearDown(self):
        for name, value in self.saved.items():
            setattr(core, name, value)
        core.shutil.disk_usage = self.saved_disk_usage

    def test_diagnostics_record_the_reference_from_the_sequence_file(self):
        job = core.submit_job(str(self.source), str(self.output))
        core.run_job(job["job_dir"])
        result = core.load_job(job["job_dir"])
        self.assertEqual(result["state"], "succeeded", result.get("error"))
        self.assertEqual([name for name, _ in self.scripts], ["01_convert", "02_probe", "02_export", "02_stack"])
        self.assertEqual(result["checkpoints"], ["convert", "distortion", "stack", "transients"])
        diagnostics = result["diagnostics"]
        self.assertEqual(diagnostics["reference_selection"], core.REFERENCE_SELECTION)
        self.assertEqual(diagnostics["reference_stem"], "light_00002")
        self.assertEqual(diagnostics["reference_input"], result["inputs"][1]["path"])
        self.assertEqual(diagnostics["registration"], list(core.REGISTRATION_COMMANDS))
        self.assertEqual(diagnostics["stacked_count"], None)  # controlled engine writes no STACKCNT
        self.assertEqual(diagnostics["distortion"]["status"], "unavailable")   # the probe wrote no sequence file
        self.assertFalse(diagnostics["distortion"]["applied"])
        self.assertEqual(diagnostics["sky_geometry"], "native_direct")
        self.assertTrue(any("reference_input" in item for item in result["limitations"]))
        self.assertTrue((Path(job["job_dir"]) / "work" / "distortion.json").exists())

    @staticmethod
    def _fake_calibration(work, registered_frames, quality_excluded=(), count=3):
        """What ``calibrate`` returns for a fitted term (RUN_LOG §22 shape): a benign model, tiny distortion,
        magnification 1 (the warp-back is almost the identity), and one matrix per frame; ``quality_excluded`` frames
        carry the gate's status (RUN_LOG §27). ``count`` is the number of frames in the sequence (light_00001..)."""
        from starstack import registration_calibration as rc
        parameters = {"k1": 1e-4, "k2": 0.0, "cx": 11.5, "cy": 8.5, "scale": 15.0, "magnification": 1.0}
        (Path(work) / rc.DISTORTION_WCS).write_bytes(b"fake wcs")
        stems = [f"light_{i:05d}" for i in range(1, count + 1)]
        frames = {}
        for position, stem in enumerate(stems):
            if stem in quality_excluded:
                shift = 1.5 * (position - 1)
                frames[stem] = {"span": position - 1, "status": "excluded_quality", "reason": "quality gate: wfwhm 2.9x median", "pairs": 40,
                                "validation": {"overall_rms_px": 0.4, "corner_rms_px": 0.5, "count": 20},
                                "H_top_down_magnified": [[1.0, 0.0, shift], [0.0, 1.0, shift], [0.0, 0.0, 1.0]], "H_siril": [[1.0, 0.0, shift], [0.0, 1.0, shift], [0.0, 0.0, 1.0]]}
            elif stem in registered_frames:
                shift = 0.0 if position == 1 else 1.5 * (position - 1)
                h = [[1.0, 0.0, shift], [0.0, 1.0, shift], [0.0, 0.0, 1.0]]
                frames[stem] = {"span": position - 1, "status": "registered", "reason": "reference" if position == 1 else None,
                                "pairs": 40, "validation": {"overall_rms_px": 0.1, "corner_rms_px": 0.12, "count": 20},
                                "H_top_down_magnified": h, "H_siril": [[1.0, 0.0, shift], [0.0, 1.0, shift], [0.0, 0.0, 1.0]]}
            else:
                frames[stem] = {"span": position - 1, "status": "not_registered", "reason": "only 3 pairs at 3 px", "pairs": 3, "validation": None}
        summary = {"status": "fitted", "file": rc.DISTORTION_WCS, "parameters": parameters, "corner_shift_px": 0.02, "reference": "light_00002",
                   "magnification": 1.0, "validation_corner_rms_median_px": {"homography_only": 0.5, "with_distortion": 0.2},
                   "frames_dropped_by_siril": [], "reassociation": {"matched_fraction_median": 0.9, "corner_rms_median_px": 0.2,
                                                                    "recovered_dropped_frames": []},
                   "quality_gate": {"version": "1.0.0", "enabled": True, "applied": bool(quality_excluded), "min_frames": 8,
                                    "thresholds": {"wfwhm_ratio": 1.5, "fwhm_ratio": 1.3, "nstars_ratio": 0.5, "holdout_rms_ratio": 2.0}, "frames_with_quality": count,
                                    "medians": {"fwhm": 3.5, "wfwhm": 3.6, "nstars": 1800, "holdout_rms": 0.1},
                                    "frames": {st: {"ratios": {"wfwhm": 2.9 if st in quality_excluded else 1.0}, "flags": ["wfwhm 2.9x median"] if st in quality_excluded else []} for st in stems},
                                    "excluded": [{"stem": st, "reasons": ["wfwhm 2.9x median"]} for st in stems if st in quality_excluded],
                                    "excluded_frames": [st for st in stems if st in quality_excluded], "flagged_not_applied": [], "reference_flagged": False},
                   "registration": {"reference": "light_00002", "registered": len(registered_frames), "registered_frames": sorted(registered_frames),
                                    "unregistered_frames": [st for st in stems if st not in registered_frames], "frames": frames,
                                    "quality_excluded_frames": sorted(quality_excluded),
                                    "validation_overall_rms_median_px": 0.1, "validation_corner_rms_median_px": 0.12, "max_rms_px": 1.0, "min_pairs": 12},
                   "sequence_template": {"header": {"name": "light_", "start_index": 1, "nb_images": count, "nb_selected": count, "fixed_len": 5,
                                                    "reference_image": 1, "version": 6, "variable_size": 0, "fz_flag": 0, "drizzle": 0},
                                         "layers": 3, "images": [{"index": i + 1, "position": i, "stem": stems[i]} for i in range(count)],
                                         "quality": {str(i): [3.5, 3.6, 0.7, 0, 1500, 1800] for i in range(count)}}}
        (Path(work) / rc.DISTORTION_JSON).write_text(json.dumps(summary), encoding="utf-8")
        return summary

    def test_fitted_calibration_writes_its_own_registration_and_warps_back(self):
        from starstack import registration_calibration as rc
        from starstack import imaging, siril_sequence

        saved = rc.calibrate
        rc.calibrate = lambda work, **kwargs: self._fake_calibration(work, {"light_00001", "light_00002", "light_00003"})
        self.expected_stack_script = core.OWN_REGISTRATION_STACK_SCRIPT
        seen = {}

        original_run_script = core.run_script

        def run_script(executable, directory, name, commands, cancel_check):
            if name == "02_export":   # P1b-3: seqapplyreg runs in the export process
                # the sequence file Siril will apply must already carry our matrices and the D line
                text = (directory / "work" / core.SEQUENCE_FILE).read_text(encoding="utf-8")
                seen["seq_before_stack"] = text
                parsed = siril_sequence.parse_seq_text(text)
                seen["regdata"] = parsed["regdata"][1]
                seen["reference"] = parsed["header"]["reference_image"]
                original_run_script(executable, directory, name, commands, cancel_check)
                # Siril re-saves the file after seqapplyreg; the controlled engine wrote its own copy (same reference)
                return
            original_run_script(executable, directory, name, commands, cancel_check)

        core.run_script = run_script
        try:
            job = core.submit_job(str(self.source), str(self.output))
            core.run_job(job["job_dir"])
        finally:
            rc.calibrate = saved
        result = core.load_job(job["job_dir"])
        self.assertEqual(result["state"], "succeeded", result.get("error"))
        self.assertEqual([name for name, _ in self.scripts], ["01_convert", "02_probe", "02_export", "02_stack"])
        self.assertIn("D1 2 distortion.wcs", seen["seq_before_stack"].splitlines())
        self.assertEqual(seen["reference"], 1)
        self.assertEqual([r["H"][0][2] for r in seen["regdata"]], [-1.5, 0.0, 1.5])
        work = Path(job["job_dir"]) / "work"
        self.assertTrue((work / "sky_master_undistorted.fit").exists())
        self.assertTrue((work / "sky_master.fit").exists())
        diagnostics = result["diagnostics"]
        self.assertTrue(diagnostics["distortion"]["applied"])
        self.assertEqual(diagnostics["sky_geometry"], "native_via_undistorted_stack")
        self.assertEqual(diagnostics["registration"], ["seqapplyreg light -filter-included"])
        self.assertEqual(diagnostics["registration_source"], core.REGISTRATION_SOURCE_OWN)
        detail = diagnostics["registration_detail"]
        self.assertEqual(detail["registered"], 3)
        self.assertEqual(detail["sequence"]["registered"], 3)
        self.assertEqual(detail["frames"]["light_00003"]["validation"]["overall_rms_px"], 0.1)
        native = imaging.read_rgb(work / "sky_master.fit")
        undistorted = imaging.read_rgb(work / "sky_master_undistorted.fit")
        self.assertEqual(native.shape, undistorted.shape)
        inner = (slice(4, -4), slice(4, -4))
        self.assertLess(float(abs(native[inner] - undistorted[inner]).max()), 0.02)   # near-identity model
        self.assertEqual(result["checkpoints"], ["convert", "distortion", "stack", "transients"])
        records = {r["path"].replace("\\", "/").split("/")[-1] for r in result["checkpoint_files"]["stack"]}
        self.assertEqual(records, {"sky_master.fit", "sky_master_undistorted.fit"})

    def test_sky_mask_keeps_the_ground_out_of_the_stack_and_composes_by_coverage(self):
        """RUN_LOG §25: with a sky mask the ground master is taken first, every frame's ground (mask < 1, dilated 3 px) is
        zeroed before the probe, the stack's coverage map is written, and the compose falls back to the ground master
        exactly where no frame observed the sky (not where the sky master merely came from fewer frames)."""
        import numpy as np
        from PIL import Image
        from starstack import registration_calibration as rc
        from starstack import imaging, sky_validity as sv

        height, width = 18, 24
        mask = np.full((height, width), 255, dtype=np.uint8)
        mask[12, :] = 128                      # feathered edge
        mask[13:, :] = 0                       # ground (top-down rows 13..17)
        mask_png = self.tmp / "sky_mask.png"
        Image.fromarray(mask).save(mask_png)
        saved = rc.calibrate
        rc.calibrate = lambda work, **kwargs: self._fake_calibration(work, {"light_00001", "light_00002", "light_00003"})
        self.expected_stack_script = core.OWN_REGISTRATION_STACK_SCRIPT
        original_run_script = core.run_script
        seen = {}

        def run_script(executable, directory, name, commands, cancel_check):
            if name != "02_stack":
                return original_run_script(executable, directory, name, commands, cancel_check)
            # Siril semantics on the controlled engine: apply the (integer part of the) written translations, then
            # average over the frames whose pixel is not exactly zero (RUN_LOG §24 no-data proof)
            self.scripts.append((name, list(commands)))
            work = directory / "work"
            frames = sorted(work.glob("light_*.fit"))
            seen["masked_frames"] = [imaging.read_rgb(p) for p in frames]
            shifts = [-2, 0, 2]                # round(1.5 * (position - 1)), half to even, as sky_validity rounds
            stack = np.zeros_like(seen["masked_frames"][0]); count = np.zeros(stack.shape[:2], dtype=np.int32)
            for data, shift in zip(seen["masked_frames"], shifts):
                moved = np.zeros_like(data)
                # frame -> reference is +shift in x and in top-down y; the arrays are bottom-up, so the row shift flips sign
                src_rows = slice(max(0, shift), height - max(0, -shift))   # bottom-up: content moving down in top-down = up in storage
                dst_rows = slice(max(0, -shift), height - max(0, shift))
                src_cols = slice(max(0, -shift), width - max(0, shift))
                dst_cols = slice(max(0, shift), width - max(0, -shift))
                moved[dst_rows, dst_cols] = data[src_rows, src_cols]
                valid = moved[..., 1] != 0
                stack[valid] += moved[valid]; count[valid] += 1
            master = np.where(count[..., None] > 0, stack / np.maximum(count, 1)[..., None], 0.0).astype(np.float32)
            (work / core.SEQUENCE_FILE).write_text(_seq_text(len(frames), 1), encoding="utf-8")
            imaging.write_linear(work / "sky_master.fit", master, "Controlled engine: shifted, zero-excluded mean")

        core.run_script = run_script
        try:
            job = core.submit_job(str(self.source), str(self.output), sky_mask_path=str(mask_png))
            core.run_job(job["job_dir"])
        finally:
            rc.calibrate = saved
        result = core.load_job(job["job_dir"])
        self.assertEqual(result["state"], "succeeded", result.get("error"))
        self.assertEqual(result["checkpoints"], ["foreground", "convert", "distortion", "stack", "transients"])
        work = Path(job["job_dir"]) / "work"
        record = sv.read_record(work)
        self.assertEqual(record["dilation_px"], 3)
        self.assertEqual(record["frames_masked"], 3)
        self.assertAlmostEqual(record["invalid_fraction"], 9 / 18, places=6)      # rows 9..17 invalid (12 - 3 = 9)

        def td(a, row):                     # top-down row of a bottom-up array
            return a[height - 1 - row]

        # 1. every frame was zeroed in the invalid rows only
        for data in seen["masked_frames"]:
            self.assertEqual(float(np.abs(data[:9]).max()), 0.0)          # storage rows 0..8 = top-down rows 9..17
            self.assertGreater(float(data[9:].min()), 0.0)
        # 2. the ground master came from the unmasked frames (median of .030/.031/.032 in the ground rows)
        ground = imaging.read_rgb(work / "foreground_master.fit")
        self.assertAlmostEqual(float(td(ground, 15)[0, 0]), 0.031, places=4)
        # 3. coverage map next to the master, sampled every 4 px, counts at most the three frames
        cov = sv.read_coverage_png(work / sv.COVERAGE_PNG)
        self.assertEqual(cov.shape, (5, 6))
        self.assertLessEqual(int(cov.max()), 3)
        self.assertEqual(int(cov[1, 1]), 3)                                # open sky (row 4, col 4): all three frames
        summary = json.loads((work / "sky_coverage.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["frames"], 3)
        self.assertEqual(summary["registration_source"], core.REGISTRATION_SOURCE_OWN)
        # 4. compose: sky where observed, ground master where no frame observed the sky, never a zero-sky blend
        composite = imaging.read_rgb(next(Path(a["path"]) for a in result["artifacts"] if a["label"] == "合成线性母版"))
        sky = imaging.read_rgb(work / "sky_master.fit")
        inner = slice(4, -4)                                # the warp-back leaves the outermost columns empty (near-identity model)
        np.testing.assert_allclose(td(composite, 4)[inner], td(sky, 4)[inner], atol=1e-6)   # open sky
        np.testing.assert_allclose(td(composite, 15), td(ground, 15), atol=1e-6)            # ground
        # row 9 is invalid in the reference but frame 3 (its content 1.5 px lower) observed it: sky, not ground, not zero
        self.assertGreater(float(td(composite, 9)[inner, 1].min()), 0.0)
        np.testing.assert_allclose(td(composite, 9)[inner], td(sky, 9)[inner], atol=1e-6)
        # rows 11-12 were observed by no frame: the ground master, with no zero-sky blend even in the feathered row 12
        np.testing.assert_allclose(td(composite, 11), td(ground, 11), atol=1e-6)
        np.testing.assert_allclose(td(composite, 12), td(ground, 12), atol=1e-6)
        diagnostics = result["diagnostics"]
        self.assertTrue(diagnostics["sky_validity"]["applied"])
        self.assertEqual(diagnostics["sky_validity"]["source"], "job_mask_static")
        self.assertGreaterEqual(diagnostics["sky_validity"]["compose"]["no_coverage_pixels"], 2 * width)
        self.assertEqual(diagnostics["sky_validity"]["coverage"]["frames"], 3)
        self.assertTrue(any(a["label"].startswith("天空有效帧数图") for a in result["artifacts"]))

    def test_per_frame_masks_reach_masking_and_coverage(self):
        """RUN_LOG §26: a per-frame mask directory (<input stem>.png) removes that frame's occluded sky from the stack and
        from its coverage; frames without a file use the static mask."""
        import numpy as np
        from PIL import Image
        from starstack import registration_calibration as rc
        from starstack import imaging, sky_validity as sv

        height, width = 18, 24
        mask = np.full((height, width), 255, dtype=np.uint8); mask[13:, :] = 0
        mask_png = self.tmp / "sky_mask.png"; Image.fromarray(mask).save(mask_png)
        masks_dir = self.tmp / "frame_masks"; masks_dir.mkdir()
        own = np.full((height, width), 255, dtype=np.uint8); own[2:5, 8:14] = 0        # an occluder in input 1 only
        Image.fromarray(own).save(masks_dir / "原片 1.png")                              # input stems are "原片 0/1/2"
        saved = rc.calibrate
        rc.calibrate = lambda work, **kwargs: self._fake_calibration(work, {"light_00001", "light_00002", "light_00003"})
        self.expected_stack_script = core.OWN_REGISTRATION_STACK_SCRIPT
        original_run_script = core.run_script
        seen = {}

        def run_script(executable, directory, name, commands, cancel_check):
            if name == "02_stack":
                work = directory / "work"
                seen["frames"] = [imaging.read_rgb(p) for p in sorted(work.glob("light_*.fit"))]
            return original_run_script(executable, directory, name, commands, cancel_check)

        core.run_script = run_script
        try:
            job = core.submit_job(str(self.source), str(self.output), sky_mask_path=str(mask_png), frame_masks_dir=str(masks_dir))
            self.assertEqual(job["frame_masks"]["count"], 1)
            self.assertIn("light_00002", job["frame_masks"]["files"])
            core.run_job(job["job_dir"])
        finally:
            rc.calibrate = saved
        result = core.load_job(job["job_dir"])
        self.assertEqual(result["state"], "succeeded", result.get("error"))
        f1, f2, f3 = seen["frames"]

        def td(a, row):
            return a[height - 1 - row]

        self.assertEqual(float(np.abs(np.stack([td(f2, r) for r in range(0, 8)])[:, 5:17]).max()), 0.0)   # occluder + 3 px dilation, frame 2 only
        self.assertGreater(float(td(f1, 3)[10].min()), 0.0)
        self.assertGreater(float(td(f3, 3)[10].min()), 0.0)
        work = Path(job["job_dir"]) / "work"
        record = sv.read_record(work)
        self.assertEqual(record["source"], "job_mask_per_frame")
        self.assertEqual(record["own_mask_count"], 1)
        self.assertTrue((Path(job["job_dir"]) / "frame_masks" / "light_00002.png").is_file())
        cov = sv.read_coverage_png(work / sv.COVERAGE_PNG)
        self.assertEqual(int(cov[1, 3]), 2)          # (row 4, col 12): frame 2 does not observe it
        self.assertEqual(int(cov[1, 1]), 3)          # (row 4, col 4): all three
        self.assertEqual(result["diagnostics"]["sky_validity"]["source"], "job_mask_per_frame")
        self.assertEqual(result["diagnostics"]["sky_validity"]["own_mask_count"], 1)
        with self.assertRaises(ValueError):
            core.submit_job(str(self.source), str(self.output), frame_masks_dir=str(masks_dir))   # per-frame masks need the static one

    def test_quality_gate_exclusion_is_deselected_reported_and_counted(self):
        """RUN_LOG §27: a frame the gate excluded is deselected in the sequence the product writes (Siril skips it, proven),
        the job message names it instead of asking for the log, and the diagnostics carry the gate record."""
        import numpy as np
        from starstack import registration_calibration as rc
        from starstack import imaging, siril_sequence

        # a fourth frame: after the gate excludes light_00001, three frames stay registered (own registration needs >= 3)
        extra = np.full((18, 24, 3), .033, dtype=np.float32)
        extra[3:6, 5:8] += .2
        imaging.write_linear(self.source / "原片 3.fit", extra, "Test fixture (blurred frame stand-in)")
        saved = rc.calibrate
        rc.calibrate = lambda work, **kwargs: self._fake_calibration(work, {"light_00002", "light_00003", "light_00004"},
                                                                     quality_excluded=("light_00001",), count=4)
        self.expected_stack_script = core.OWN_REGISTRATION_STACK_SCRIPT
        original_run_script = core.run_script
        seen = {}

        def run_script(executable, directory, name, commands, cancel_check):
            work = directory / "work"
            if name == "02_export":   # P1b-3: the export process applies the sequence the product wrote
                self.assertEqual(list(commands), list(self.expected_stack_script[:-1]))
                parsed = siril_sequence.parse_seq_text((work / core.SEQUENCE_FILE).read_text(encoding="utf-8"))
                seen["selected"] = [im["selected"] for im in parsed["images"]]
                seen["h"] = [r["H"] for r in parsed["regdata"][1]]
                self.scripts.append((name, list(commands)))
                # Keep the selection applied by the product, as the real Siril sequence does.
                return
            if name == "02_stack":
                self.assertEqual(list(commands), list(core.LEGACY_STACK_SCRIPT))
                # the controlled engine honours the selection like Siril does (RUN_LOG §27 proof): stack the selected frames only
                self.scripts.append((name, list(commands)))
                frames = [p for p, sel in zip(sorted(work.glob("light_*.fit")), seen["selected"]) if sel]
                data = np.mean([imaging.read_rgb(p) for p in frames], axis=0).astype(np.float32)
                imaging.write_linear(work / "sky_master.fit", data, "Controlled engine: selected frames only", extra={"STACKCNT": len(frames)})
                return
            original_run_script(executable, directory, name, commands, cancel_check)

        core.run_script = run_script
        try:
            job = core.submit_job(str(self.source), str(self.output))
            core.run_job(job["job_dir"])
        finally:
            rc.calibrate = saved
        result = core.load_job(job["job_dir"])
        self.assertEqual(result["state"], "succeeded", result.get("error"))
        self.assertEqual(seen["selected"], [False, True, True, True])
        self.assertEqual(seen["h"][0], [[0.0, 0.0, 0.0]] * 3)
        diagnostics = result["diagnostics"]
        self.assertEqual(diagnostics["stacked_count"], 3)
        self.assertEqual(diagnostics["skipped_count"], 1)
        self.assertIn("像质门槛排除的 light_00001", result["message"])
        self.assertNotIn("请检查配准日志", result["message"])
        gate = diagnostics["quality_gate"]
        self.assertTrue(gate["in_effect"])
        self.assertEqual(gate["excluded_frames"], ["light_00001"])
        self.assertEqual(gate["frames"]["light_00001"]["flags"], ["wfwhm 2.9x median"])
        detail = diagnostics["registration_detail"]
        self.assertEqual(detail["frames"]["light_00001"]["status"], "excluded_quality")
        self.assertEqual(detail["registered"], 3)
        self.assertEqual(detail["quality_excluded_frames"], ["light_00001"])

    def test_too_few_registered_frames_fall_back_to_siril_two_pass(self):
        from starstack import registration_calibration as rc

        saved = rc.calibrate
        rc.calibrate = lambda work, **kwargs: self._fake_calibration(work, {"light_00002"})   # only the reference registered
        self.expected_stack_script = core.STACK_SCRIPT
        try:
            job = core.submit_job(str(self.source), str(self.output))
            core.run_job(job["job_dir"])
        finally:
            rc.calibrate = saved
        result = core.load_job(job["job_dir"])
        self.assertEqual(result["state"], "succeeded", result.get("error"))
        diagnostics = result["diagnostics"]
        self.assertFalse(diagnostics["distortion"]["applied"])
        self.assertEqual(diagnostics["registration_source"], core.REGISTRATION_SOURCE_SIRIL)
        self.assertEqual(diagnostics["registration"], list(core.REGISTRATION_COMMANDS))
        self.assertEqual(diagnostics["sky_geometry"], "native_direct")
        self.assertFalse((Path(job["job_dir"]) / "work" / "sky_master_undistorted.fit").exists())

    def test_missing_sequence_file_does_not_fail_the_job(self):
        original = core.run_script

        def run_script(executable, directory, name, commands, cancel_check):
            original(executable, directory, name, commands, cancel_check)
            if name == "02_stack":
                (directory / "work" / core.SEQUENCE_FILE).unlink()

        core.run_script = run_script
        job = core.submit_job(str(self.source), str(self.output))
        core.run_job(job["job_dir"])
        result = core.load_job(job["job_dir"])
        self.assertEqual(result["state"], "succeeded", result.get("error"))
        self.assertEqual(result["diagnostics"]["reference_selection"], "unknown")
        self.assertIsNone(result["diagnostics"]["reference_input"])


@unittest.skipUnless(HAVE_STACK, "product imaging stack (numpy/astropy/tifffile/PIL) not installed")
class TransientFlowTest(unittest.TestCase):
    """RUN_LOG §28: the bright-trail stage in the product flow — candidates from the exported frames against the master,
    provenance, suggestion, policy decision, the second composite, and the recompose entry points."""

    HEIGHT, WIDTH = 200, 320

    def setUp(self):
        import numpy as np
        from starstack import imaging
        self.tmp = Path(tempfile.mkdtemp(prefix="s4-core-job-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        source = self.tmp / "素材"
        source.mkdir()
        rng = np.random.default_rng(5)
        h, w = self.HEIGHT, self.WIDTH
        yy, xx = np.mgrid[:h, :w].astype(np.float64)
        stars = np.zeros((h, w))
        for _ in range(40):
            x, y, a = rng.uniform(5, w - 5), rng.uniform(5, h - 5), rng.uniform(0.05, 0.3)
            stars += a * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 1.4 ** 2))
        # frame 3 (top-down coordinates) carries a bright trail from (40, 150) to (260, 40) with a meteor light curve
        # (faint start, peak past the middle, fades out) so that the automatic suggestion is "meteor" (1.1.0 rule)
        ux, uy = 260 - 40, 40 - 150
        length = (ux ** 2 + uy ** 2) ** 0.5
        ux, uy = ux / length, uy / length
        along = (xx - 40) * ux + (yy - 150) * uy
        across = -(xx - 40) * uy + (yy - 150) * ux
        curve = 0.12 + 1.25 * np.exp(-((along - 0.55 * length) / (0.22 * length)) ** 2)
        self.trail = np.where((along >= 0) & (along <= length), 0.03 * curve * np.exp(-across ** 2 / (2 * 1.5 ** 2)), 0.0)
        self.trail_mask = self.trail > 1e-4
        for number in range(1, 6):
            top_down = np.stack([0.03 + stars * g for g in (1.0, 0.9, 0.8)], axis=-1)
            if number == 3:
                top_down = top_down + self.trail[..., None] * np.array([0.9, 1.0, 1.1])
            top_down = top_down + rng.normal(0, 0.002, top_down.shape)
            imaging.write_linear(source / f"原片 {number}.fit", np.ascontiguousarray(top_down[::-1]).astype(np.float32), "Test fixture")
        self.source = source
        self.output = self.tmp / "results"
        self.engine = {"available": True, "executable": str(self.tmp / "fake engine.exe"), "version": "1.4.4", "message": "Test engine"}
        self.saved = {name: getattr(core, name) for name in ("doctor", "_spawn_worker", "run_script")}
        admit_before_contract(self)
        self.saved_disk_usage = core.shutil.disk_usage
        core.doctor = lambda *_: dict(self.engine)
        core._spawn_worker = lambda directory: None
        core.shutil.disk_usage = lambda _: SimpleNamespace(free=10**12)
        from starstack import registration_calibration as rc
        self.saved_calibrate = rc.calibrate
        rc.calibrate = lambda work, **kwargs: JobDiagnosticsTest._fake_calibration(work, {f"light_{i:05d}" for i in range(1, 6)}, count=5)

        def run_script(executable, directory, name, commands, cancel_check):
            work = directory / "work"
            if name == "01_convert":
                for i, path in enumerate(sorted((directory / "stage").glob("*")), 1):
                    shutil.copyfile(path, work / f"light_{i:05d}.fit")
            elif name == "02_probe":
                pass
            elif name == "02_export":
                # the controlled engine: identity export (r_ frames = frames); P1b-3 runs it as its own process
                assert list(commands) == core.export_script(core.DISTORTION_WCS, own_registration=True), commands
                frames = sorted(work.glob("light_*.fit"))
                for path in frames:
                    shutil.copyfile(path, work / f"r_{path.name}")
                (work / core.SEQUENCE_FILE).write_text(_seq_text(len(frames), 1), encoding="utf-8")
            elif name == "02_stack":
                # and a per-pixel median stack of the exported frames (the trail vanishes)
                assert list(commands) == list(core.LEGACY_STACK_SCRIPT), commands
                frames = sorted(work.glob("r_light_*.fit"))
                master = np.median(np.stack([imaging.read_rgb(path) for path in frames]), axis=0).astype(np.float32)
                imaging.write_linear(work / "sky_master.fit", master, "Controlled engine: median", extra={"STACKCNT": len(frames)})
            else:
                raise AssertionError(name)

        core.run_script = run_script

    def tearDown(self):
        from starstack import registration_calibration as rc
        rc.calibrate = self.saved_calibrate
        for name, value in self.saved.items():
            setattr(core, name, value)
        core.shutil.disk_usage = self.saved_disk_usage

    def _composites(self, job):
        from starstack import imaging
        base = [a for a in job["artifacts"] if a["label"] == "合成线性母版"]
        with_ = [a for a in job["artifacts"] if a.get("group") == "transients" and a["kind"] == "fits"]
        return (imaging.read_rgb(base[0]["path"]) if base else None), (imaging.read_rgb(with_[0]["path"]) if with_ else None), with_

    def test_trail_is_detected_kept_and_composed_then_recomposed(self):
        import numpy as np
        from starstack import transient_stage as ts
        job = core.submit_job(str(self.source), str(self.output))
        core.run_job(job["job_dir"])
        result = core.load_job(job["job_dir"])
        self.assertEqual(result["state"], "succeeded", result.get("error"))
        self.assertEqual(result["checkpoints"][-1], "transients")
        work = Path(job["job_dir"]) / "work"
        record = ts.read_record(work)
        self.assertIsNotNone(record)
        self.assertEqual(record["frames_scanned"], 5)
        measured = [c for c in record["candidates"] if c["measurement"].get("status") == "measured"]
        self.assertEqual([c["stem"] for c in measured], ["light_00003"], [c["id"] for c in record["candidates"]])
        cand = measured[0]
        self.assertEqual(cand["suggestion"]["class"], "meteor", cand["suggestion"])
        self.assertEqual(cand["decision"], "keep")
        self.assertEqual(cand["provenance"]["frame_index"], 3)
        self.assertEqual(cand["provenance"]["input"]["path"], result["inputs"][2]["path"])
        self.assertTrue(Path(cand["review_png"]).is_file())
        self.assertIn("native_source", cand["geometry"])
        diag = result["diagnostics"]["transients"]
        self.assertEqual(diag["summary"]["kept"], [cand["id"]])
        self.assertIsNotNone(diag["composite"])
        self.assertIn("亮迹候选 1 条", result["message"])
        base, with_, items = self._composites(result)
        self.assertIsNotNone(with_)
        diff = np.abs(with_ - base)[::-1]                       # top-down
        self.assertGreater(float(diff[self.trail_mask].max()), 0.01)
        far = np.ones_like(self.trail_mask)
        ys, xs = np.nonzero(self.trail_mask)
        for y, x in zip(ys, xs):                              # anything within 12 px of the trail may be in the feathered footprint
            far[max(0, y - 12):y + 13, max(0, x - 12):x + 13] = False
        self.assertEqual(float(diff[far].max()), 0.0)          # bit-exact elsewhere
        self.assertTrue(any(i["label"].startswith("含亮迹") for i in items))
        # omit it -> the with-transients composite disappears, the base stays
        updated = core.recompose_transients(job["job_dir"], overrides={cand["id"]: "omit"})
        self.assertEqual(updated["diagnostics"]["transients"]["summary"]["kept"], [])
        self.assertFalse([a for a in updated["artifacts"] if a.get("group") == "transients"])
        self.assertTrue([a for a in updated["artifacts"] if a["label"] == "合成线性母版"])
        # a manual label + policy: the user photographs satellites
        updated = core.recompose_transients(job["job_dir"], overrides={cand["id"]: None}, labels={cand["id"]: "satellite"}, policy={"satellite": "keep"})
        record = ts.read_record(work)
        c = next(c for c in record["candidates"] if c["id"] == cand["id"])
        self.assertEqual(c["label"], "satellite")
        self.assertEqual(c["decision"], "keep")
        self.assertEqual(updated["diagnostics"]["transients"]["summary"]["kept"], [cand["id"]])
        base2, with2, _ = self._composites(updated)
        np.testing.assert_array_equal(base2, base)
        np.testing.assert_allclose(with2, with_, atol=1e-6)
        # a manual mark on a frame without a detected trail is measured and recorded as manual
        updated = core.recompose_transients(job["job_dir"], marks=[{"input": "原片 2.fit", "x0": 50, "y0": 100, "x1": 150, "y1": 120}])
        record = ts.read_record(work)
        manual = [c for c in record["candidates"] if c["source"] == "manual"]
        self.assertEqual(len(manual), 1)
        self.assertEqual(manual[0]["stem"], "light_00002")
        self.assertIn(manual[0]["detection"]["status"], ("mark_only", "detected"))
        self.assertEqual(manual[0]["decision"], "omit")     # nothing measurable there -> not measured -> omitted
        with self.assertRaises(ValueError):
            core.recompose_transients(job["job_dir"], labels={cand["id"]: "comet"})
        with self.assertRaises(ValueError):
            core.recompose_transients(job["job_dir"], marks=[{"input": "nope.fit", "x0": 0, "y0": 0, "x1": 1, "y1": 1}])


if __name__ == "__main__":
    unittest.main()
