"""Tests for the Siril-free parts of scripts/s2_register_only_long_span.py.

Run:  python -B -m unittest tests.test_s2_register_only   (from WORKSPACE, PYTHONPATH=src)
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from starstack import registration_distortion as rd  # noqa: E402


def _module():
    spec = importlib.util.spec_from_file_location("s2_register_only_long_span", ROOT / "scripts" / "s2_register_only_long_span.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SelectionTest(unittest.TestCase):
    def test_every_third_raw_in_name_order_and_links(self):
        m = _module()
        with tempfile.TemporaryDirectory() as d:
            raw = Path(d) / "dng"
            raw.mkdir()
            for i in range(10):
                (raw / f"DSC_{2204 + i}.dng").write_bytes(b"x" * (i + 1))
            (raw / "notes.txt").write_text("ignored")
            chosen = m.select_frames(raw, 3, None)
            self.assertEqual([p.name for p in chosen], ["DSC_2204.dng", "DSC_2207.dng", "DSC_2210.dng", "DSC_2213.dng"])
            self.assertEqual([p.name for p in m.select_frames(raw, 3, 2, start=1)], ["DSC_2205.dng", "DSC_2208.dng"])
            info = m.link_inputs(chosen, Path(d) / "inputs")
            self.assertEqual(info["count"], 4)
            self.assertEqual(sorted(p.name for p in (Path(d) / "inputs").iterdir() if p.suffix == ".dng"),
                             ["frame00001.dng", "frame00002.dng", "frame00003.dng", "frame00004.dng"])
            recorded = json.loads((Path(d) / "inputs" / "inputs.json").read_text(encoding="utf-8"))
            self.assertEqual(recorded["frames"][1]["source"], str(raw / "DSC_2207.dng"))
            self.assertEqual((Path(d) / "inputs" / "frame00002.dng").read_bytes(), b"x" * 4)


class RenderTest(unittest.TestCase):
    def test_render_from_synthetic_analysis(self):
        import test_registration_distortion as t
        m = _module()
        frames = t._synthetic_frames("radial", [-6, -3, 3, 6], k1=0.02)
        analysis = rd.analyse(frames, t.W, t.H, degrees=(2,))
        stems = sorted(frames)
        report = {"run": "register-only-test", "every": 3, "reference": "syn_00020",
                  "inputs": {"count": 4, "frames": [{"stem": s, "source": f"/raw/{s}.dng"} for s in stems]},
                  "siril_seconds": {"convert": 10.0, "register": 20.0},
                  "diagnostics": {"frames": {s: {"matched": 600, "frame_stars": 700, "corners_rms_px": 0.3} for s in stems}},
                  "distortion": analysis, "environment": {"timestamp": "now"}}
        text = m.render(report)
        self.assertIn("共享径向项", text)
        self.assertIn("syn_00014", text)
        self.assertIn("| syn_00026 | +6 |", text)
        self.assertIn("迁移测试", text)


if __name__ == "__main__":
    unittest.main()
