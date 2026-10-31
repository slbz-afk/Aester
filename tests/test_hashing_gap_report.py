"""scripts/bench_report.py, efficiency P1b-5c: ledger-miss bytes and the largest misses per worker."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_bench_report import GB, make_run, report  # noqa: E402


def _note(path, mode, miss_bytes=None, largest=None, refused=None):
    record = json.loads(path.read_text(encoding="utf-8"))
    timings = record["timings"] if "timings" in record else record["diagnostics"]["timings"]
    note = {"mode": mode, "shadow": {"checks": 0, "mismatches": 0, "concurrent": 0}}
    if miss_bytes is not None:
        note.update(version="aester-hashing-3", miss_bytes=miss_bytes, largest_misses=largest or [],
                    refused=refused or {"touched": 0, "seal": 0})
    timings["notes"] = {"hash_ledger": note}
    path.write_text(json.dumps(record), encoding="utf-8")


def test_miss_columns_and_largest_misses(tmp_path):
    new = make_run(tmp_path, "20261001T000000Z-L20-P1B", 60.0, label="P1B")
    records = new / "records"
    workflow_largest = [[int(0.548 * GB), "racy", "photo_render.source_master", "stack/master_scene_linear.tif"],
                        [int(0.274 * GB), "no_entry", "workflow._checkpoint", "masks-1/sky_weight.tiff"]] + \
                       [[1000 - i, "no_entry", "workflow._file", f"x/{i}.json"] for i in range(8)]
    _note(records / "workflow.json", "on",
          {"no_entry": int(0.4 * GB), "stale": 0, "racy": int(0.55 * GB), "busy": 0}, workflow_largest,
          {"touched": 1, "seal": 0})
    _note(records / "job.json", "on", {"no_entry": int(2.1 * GB), "stale": int(0.01 * GB), "racy": 0, "busy": 0},
          [[int(0.5 * GB), "no_entry", "core._sha", "work/r_light_00001.fit"]])
    old = make_run(tmp_path, "20260930T000000Z-L20-P1A", 60.0, label="P1A")
    _note(old / "records" / "workflow.json", "on")             # a P1a note: no miss bytes
    groups = report.build([new, old], group_by="run")
    runs = {g["runs"][0]["run_id"]: g["runs"][0]["hashing"] for g in groups}
    h, before = runs["20261001T000000Z-L20-P1B"], runs["20260930T000000Z-L20-P1A"]
    assert (h["miss_racy_gb"], h["miss_no_entry_gb"], h["miss_stale_gb"], h["miss_busy_gb"]) == (0.55, 2.5, 0.01, 0.0)
    assert (h["refused_touched"], h["refused_seal"]) == (1, 0)
    assert len(h["largest_misses"]["workflow"]) == report.LARGEST_SHOWN and len(h["largest_misses"]["job"]) == 1
    assert h["largest_misses"]["workflow"][0] == {"gb": 0.548, "status": "racy", "source": "photo_render.source_master",
                                                  "file": "stack/master_scene_linear.tif"}
    assert before["miss_racy_gb"] is None and before["largest_misses"] is None and before["refused_seal"] is None
    text = report.markdown(groups)
    assert "### Ledger misses (efficiency P1b-5c)" in text
    assert "| 20261001T000000Z-L20-P1B | 0.55 | 2.50 | 0.01 | 0.00 | 1 | 0 |" in text
    assert "| 20260930T000000Z-L20-P1A | – | – | – | – | – | – |" in text
    assert ("| 20261001T000000Z-L20-P1B | workflow | 0.548 | racy | photo_render.source_master | "
            "stack/master_scene_linear.tif |") in text
    assert text.count("| 20261001T000000Z-L20-P1B | workflow |") == report.LARGEST_SHOWN
