"""Whole-workflow equivalence verifier (efficiency plan P0 item 2, acceptance E0).

Synthetic workflow-like trees (FITS/TIFF/JPEG/PNG/NPY/JSON/text, real record structure) check that equal
runs pass across roots and ids, that a one-value change in every format is reported and located, and
that every exclusion hides only its listed field.  Reruns in one slot must reproduce the deterministic
ids literally; timing fields must never enter a signature/key preimage, and an excluded field present on
one side only fails outside the instrumentation's diagnostic slots; the directory set and the whole own
delivery parent are compared.  Two runs of the real workflow orchestration (stack and model stubbed
exactly as in test_workflow) check the normalisation against real records.
"""
import copy
import functools
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from starstack import workflow as wf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import verify_workflow_equivalence as wfeq  # noqa: E402
from test_workflow import setup, stub_model  # noqa: E402,F401  (fixture used by the real-run tests)
from test_trail_default_and_extras import build  # noqa: E402

BLOCK = 1024


def run(tmp_path, label, seed, variants=(), **kwargs):
    tree = wfeq.build_synthetic_workflow(tmp_path / label, seed=seed, variants=variants, inputs=tmp_path / "inputs",
                                         **kwargs)
    return wfeq.fingerprint(tree["workflow_dir"], workers=3, block_bytes=BLOCK), tree


def named(report, section, suffix):
    matches = [name for name in report[section] if name.endswith(suffix)]
    assert matches, (section, suffix, sorted(report[section]))
    return report[section][matches[0]] if isinstance(report[section], dict) else matches[0]


@pytest.fixture
def base(tmp_path):
    return run(tmp_path, "base", 1)


def test_equal_runs_in_different_roots_pass_and_list_what_was_hidden(tmp_path, base):
    other, tree = run(tmp_path, "other", 2)
    report = wfeq.compare(base[0], other)
    assert report["pass"], wfeq.format_report(report)
    assert not report["different"] and not report["only_in_a"] and not report["only_in_b"]
    assert report["counts"]["IDENTICAL_BYTES"] > 5 and report["counts"]["IDENTICAL_CONTENT"] > 5
    assert report["strict_ids"] == {"on": False, "mode": "auto", "same_base_roots": False,
                                    "kinds": list(wfeq.DETERMINISTIC_ID_KINDS), "differ": []}
    assert not report["dirs_only_in_a"] and not report["dirs_only_in_b"] and not report["one_sided_fields"]
    assert base[0]["delivery_parents"] == other["delivery_parents"] == {"{DELIV}": "whole"}
    rules = set(report["hidden_by_rule"])
    for expected in (wfeq.EXCLUDED_KEYS["timings"], wfeq.EXCLUDED_KEYS["worker_pid"], wfeq.EXCLUDED_KEYS["seconds"],
                     wfeq.EXCLUDED_KEYS["implementation"], wfeq.EXCLUDED_KEYS["free_bytes_at_admission"],
                     wfeq.EXCLUDED_KEYS["export_before_sha256"], wfeq.EXCLUDED_KEYS["signed_trail_source_sha256"],
                     wfeq.EXCLUDED_KEYS["workers"], wfeq.EXCLUDED_KEYS["render_workers"],
                     wfeq.FITS_DROPPED_CARDS["DATE"], wfeq.DERIVED_RULES["data_sha256"],
                     wfeq.DERIVED_RULES["review key"], wfeq.DERIVED_RULES["signature+request"],
                     wfeq.DICT_RULES["unretained file"], wfeq.DICT_RULES["record file size"]):
        assert expected in rules, expected
    # run ids map to the same placeholders on both sides
    for placeholder in ("<workflow#1>", "<job#1>", "<finite#1>", "<delivery#1>", "<delivery-trails-1kept#1>",
                        "<masks#1>", "<stack_snapshot#1>", "<milkyway-evidence#1>", "<milkyway-evidence#2>",
                        "<products-hex12#1>", "<scan-runs-hex32#1>", "<draft#1>", "<aester#1>",
                        "<workflow-signature#1>", "<job-signature#1>", "<review-key#1>", "<draft_revision#1>"):
        assert placeholder in report["aliases"]["a"] and placeholder in report["aliases"]["b"], placeholder
        assert report["aliases"]["a"][placeholder] != report["aliases"]["b"][placeholder], placeholder
    names = set(wfeq.Side(other, "b").by_name)
    assert "{DELIV}/delivery-<delivery#1>/final.jpg" in names
    assert "{WF}/stack/job-<job#1>/finite-<finite#1>/finite_master.fit" in names
    assert report["derived_digests"]["a"]["checked"] >= 4 and not report["derived_digests"]["b"]["failed"]
    assert report["bindings"]["a"]["verified"] > 10 and not report["bindings"]["b"]["mismatch"]


def test_fingerprint_then_delete_rerun_in_the_same_root_passes(tmp_path):
    first, tree = run(tmp_path, "bench", 1, idempotency_key="bench-l20-r1")
    shutil.rmtree(tmp_path / "bench")
    second, _ = run(tmp_path, "bench", 7, idempotency_key="bench-l20-r1")  # new uuids and timings, same roots
    assert first["roots"] == second["roots"]
    report = wfeq.compare(first, second)
    assert report["pass"], wfeq.format_report(report)
    assert report["strict_ids"]["on"] and report["strict_ids"]["same_base_roots"] and not report["strict_ids"]["differ"]
    for placeholder in ("<workflow#1>", "<workflow-signature#1>", "<job#1>", "<job-signature#1>", "<aester#1>"):
        assert report["aliases"]["a"][placeholder] == report["aliases"]["b"][placeholder], placeholder  # derived
    assert report["aliases"]["a"]["<finite#1>"] != report["aliases"]["b"]["<finite#1>"]  # uuid4 per run
    assert report["aliases"]["a"]["<review-key#1>"] != report["aliases"]["b"]["<review-key#1>"]  # uuid4 revision
    assert "strict ids: on (auto; base roots equal)" in wfeq.format_report(report)


@pytest.fixture
def slot(tmp_path):
    """A first run in a fixed slot (fixed roots and idempotency key); rerun(...) replaces it by a new run."""
    first, _ = run(tmp_path, "slot", 1, idempotency_key="aester-bench-l20")

    def rerun(seed, variants=()):
        shutil.rmtree(tmp_path / "slot")
        return run(tmp_path, "slot", seed, variants, idempotency_key="aester-bench-l20")[0]
    return first, rerun


def test_a_timing_field_in_the_request_fails_a_same_slot_rerun_on_its_deterministic_ids(slot):
    first, rerun = slot
    changed = rerun(2, ("request_timing",))
    report = wfeq.compare(first, changed)
    assert not report["pass"] and report["strict_ids"]["on"]  # {DELIV} moved with the signature: not a base root
    assert {d["placeholder"] for d in report["strict_ids"]["differ"]} == {"<workflow-signature#1>", "<job#1>",
                                                                          "<aester#1>"}
    leaked = report["excluded_keys_in_preimages"]["b"]
    assert [(e["path"], e["preimage"]) for e in leaked][0] == ("/request/timings", "workflow signature")
    assert not report["excluded_keys_in_preimages"]["a"] and not report["derived_digests"]["b"]["failed"]
    text = wfeq.format_report(report)
    assert "DETERMINISTIC IDS DIFFER" in text and "EXCLUDED KEYS INSIDE SIGNATURE/KEY PREIMAGES IN B" in text
    # without strict ids (e.g. other roots) the leaked key alone still fails
    off = wfeq.compare(first, changed, strict_ids=False)
    assert not off["pass"] and not off["strict_ids"]["on"] and off["excluded_keys_in_preimages"]["b"]


def test_a_changed_stack_job_signature_fails_on_ids_and_on_its_recomputation(slot):
    first, rerun = slot
    report = wfeq.compare(first, rerun(3, ("job_signature",)))
    assert not report["pass"]
    assert [d["placeholder"] for d in report["strict_ids"]["differ"]] == ["<job-signature#1>"]
    assert [(f["path"], f["rule"]) for f in report["derived_digests"]["b"]["failed"]] == [("/signature", "job signature")]
    assert not report["derived_digests"]["a"]["failed"]  # the synthetic job signature uses core's recipe
    off = wfeq.compare(first, rerun(4, ("job_signature",)), strict_ids=False)
    assert not off["pass"] and off["derived_digests"]["b"]["failed"]


def test_strict_ids_can_be_forced_across_roots(tmp_path, base):
    other, _ = run(tmp_path, "other", 2)
    report = wfeq.compare(base[0], other, strict_ids=True)
    assert not report["pass"] and report["strict_ids"]["mode"] == "forced"
    kinds = {d["kind"] for d in report["strict_ids"]["differ"]}
    assert {"workflow", "workflow-signature", "job", "job-signature", "aester"} <= kinds


def test_an_excluded_key_in_a_review_key_source_fails(base):
    a, b, _, doc_b = _record_pair(base[0], "/review_delivery.json")
    doc_b["source_revision"]["recorded_at"] = "2026-09-28T01:02:03Z"
    doc_b["key"] = wfeq._workflow_digest({"revision": doc_b["source_revision"], "signature": doc_b["base_signature"]})
    report = wfeq.compare(a, b)
    assert not report["pass"]
    assert [(e["path"], e["preimage"]) for e in report["excluded_keys_in_preimages"]["b"]] == [
        ("/source_revision/recorded_at", "review key")]


@pytest.mark.parametrize("variant,files,path", [
    ("checkpoint_timing", ["{WF}/workflow.json"], "/checkpoints/masks/data/timings"),
    ("receipt_timing", ["{WF}/publication-receipts/delivery-<delivery#1>.json",
                        "{WF}/publication-receipts/delivery-trails-1kept-<delivery-trails-1kept#1>.json"], "/timings"),
    ("description_timing", ["{DELIV}/delivery-<delivery#1>/final_srgb16.tif",
                            "{WF}/stack/job-<job#1>/products/<products-hex12#1>/natural_display_16bit.tif"],
     "<page0:ImageDescription>/encode_seconds"),
])
def test_an_excluded_field_on_one_side_only_fails_outside_the_diagnostic_slots(tmp_path, base, variant, files, path):
    other, _ = run(tmp_path, variant, 2, (variant,))
    report = wfeq.compare(base[0], other)
    assert not report["pass"]
    for name in files:
        assert path in [item["path"] for item in report["one_sided_fields"][name]], (name, report["one_sided_fields"])
        assert all(item["only_in"] == "b" for item in report["one_sided_fields"][name])
    assert "ONE-SIDED EXCLUDED FIELDS (fails" in wfeq.format_report(report)
    if variant == "checkpoint_timing":  # data_sha256 was recomputed over it: only the one-sided rule catches it
        assert not report["derived_digests"]["b"]["failed"]
        assert set(report["different"]) == {"{WF}/workflow.json"}
        allowed = wfeq.compare(base[0], other, allow_fields=[r"{WF}/workflow.json=/checkpoints/masks/data/timings"])
        assert allowed["pass"] and allowed["allow_fields"] == [[["{WF}/workflow.json"], "/checkpoints/masks/data/timings"]]
        assert allowed["one_sided_diagnostic_fields"][0]["slot"].startswith("--allow-field")


@pytest.mark.parametrize("key", sorted(wfeq.EXCLUDED_KEYS) + ["finished_at", "render_seconds"])
def test_each_excluded_key_added_on_one_side_fails(base, key):
    a, b, _, doc_b = _record_pair(base[0], "/job.json")
    doc_b["diagnostics"]["exposure"][key] = "2026-09-28T09:10:00Z" if key.endswith("_at") else 1
    report = wfeq.compare(a, b)
    assert not report["pass"]
    assert [item["path"] for item in named(report, "one_sided_fields", "/job.json")] == ["/diagnostics/exposure/" + key]


def test_one_sided_timing_receipts_in_the_diagnostic_slots_pass_and_are_listed(tmp_path, base):
    other, _ = run(tmp_path, "slots", 2, ("diagnostic_slots",))
    report = wfeq.compare(base[0], other)
    assert report["pass"], wfeq.format_report(report)
    seen = {(d["name"].rsplit("/", 1)[-1], d["path"], d["only_in"]) for d in report["one_sided_diagnostic_fields"]}
    assert seen == {("workflow.json", "/timings", "b"), ("workflow.json", "/timing_attempts", "b"),
                    ("workflow.json", "/review_deliveries[0]/started_at", "b"),
                    ("workflow.json", "/review_deliveries[0]/finished_at", "b"),
                    ("workflow.json", "/review_deliveries[0]/timings", "b"),
                    ("job.json", "/diagnostics/timings", "a"), ("job.json", "/diagnostics/timing_attempts", "a")}
    assert "ONE-SIDED FIELDS IN DIAGNOSTIC SLOTS (informational)" in wfeq.format_report(report)
    # a slot is bound to its file: the same key at the top of a publication receipt is not a slot
    receipt = next(r for r in base[0]["files"] if "/publication-receipts/delivery-" in r)
    a, b, _, doc_b = _record_pair(base[0], receipt)
    doc_b["timings"] = {"x": 1}
    report = wfeq.compare(a, b)
    assert not report["pass"] and [i["path"] for i in named(report, "one_sided_fields", ".json")] == ["/timings"]


def test_extra_or_missing_directories_fail_unless_allowed(tmp_path, base):
    other, _ = run(tmp_path, "dirs", 2, ("extra_dir",))
    report = wfeq.compare(base[0], other)
    assert not report["pass"] and report["dirs_only_in_b"] == ["{WF}/left-over/"] and not report["only_in_b"]
    assert "DIRECTORIES ONLY IN B (fails):" in wfeq.format_report(report)
    reverse = wfeq.compare(other, base[0])
    assert not reverse["pass"] and reverse["dirs_only_in_a"] == ["{WF}/left-over/"]
    allowed = wfeq.compare(base[0], other, allow=["{WF}/left-over"])
    assert allowed["pass"] and allowed["allowed"]["{WF}/left-over/"]["verdict"] == "ALLOWED_DIR_ONLY_IN_B"


@pytest.mark.parametrize("variant,file_suffix", [("extra_delivery", "/final.jpg"), ("staging_leftover", "/failure.json")])
def test_the_whole_own_delivery_parent_is_fingerprinted(tmp_path, base, variant, file_suffix):
    other, tree = run(tmp_path, variant, 2, (variant,))
    assert other["delivery_parents"] == {"{DELIV}": "whole"}
    assert [label for label, _ in other["walked"]] == ["{WF}", "{DELIV}"]
    report = wfeq.compare(base[0], other)
    assert not report["pass"]
    extra = [name for name in report["only_in_b"] if name.startswith("{DELIV}/") and name.endswith(file_suffix)]
    assert extra and any(extra[0].startswith(d) for d in report["dirs_only_in_b"]), report["only_in_b"]
    if variant == "staging_leftover":  # the run id of the version it staged maps like everywhere else
        assert extra[0].startswith("{DELIV}/.partial-delivery-trails-1kept-<delivery-trails-1kept#1>-<")
    record = json.loads((tree["workflow_dir"] / "workflow.json").read_text(encoding="utf-8"))
    assert wfeq.own_delivery_parent(record, tree["workflow_dir"]) == Path(other["roots"]["{DELIV}"])


def test_links_and_older_manifests_are_reported(base):
    a, b = copy.deepcopy(base[0]), copy.deepcopy(base[0])
    b["links"] = {"{WF}/extra_link.tif": b["roots"]["{WF}"] + "\\stack\\x.tif"}
    report = wfeq.compare(a, b)
    assert not report["pass"] and report["links_different"] == [
        {"name": "{WF}/extra_link.tif", "a": "<absent>", "b": "{WF}/stack/x.tif"}]
    del b["delivery_parents"]
    b["links"] = {}
    report = wfeq.compare(a, b)
    assert report["pass"] and any("walked differently" in w for w in report["warnings"])


@pytest.mark.parametrize("variant,suffix,section,index,max_abs", [
    ("fits_pixel", "work/sky_master.fit", "data", [1, 5, 6], 1.0),
    ("tiff_pixel", "products/<products-hex12#1>/natural_display_16bit.tif", "pixels", [3, 3, 0], 1.0),
    ("tiff_zlib_pixel", "products/<products-hex12#1>/master_scene_linear.tif", "pixels", [2, 2, 1], 0.5),
    ("png_pixel", "render_flags.png", "pixels", [0, 0], 255.0),
    ("npy_element", "frame_support_light_00001.npy", "data", [0, 0], 1.0),
    ("jpeg_pixel", "products/<products-hex12#1>/natural.jpg", "pixels", None, None),
])
def test_one_value_change_in_each_pixel_format_is_reported_and_located(tmp_path, base, variant, suffix, section,
                                                                       index, max_abs):
    other, _ = run(tmp_path, variant, 3, (variant,))
    report = wfeq.compare(base[0], other)
    assert not report["pass"]
    detail = named(report, "different", suffix)
    located = detail[section][0] if isinstance(detail[section], list) else detail[section]
    assert located["blocks_different"] >= 1 and "first_byte_range" in located and "block_starts_at_index" in located
    deep = wfeq.compare(base[0], other, deep=True)
    assert not deep["pass"]  # --deep adds statistics, never a pass
    arrays = named(deep, "different", suffix)["deep"]["arrays"]
    assert arrays and arrays[0]["values_different"] >= 1
    if index is not None:
        assert arrays[0]["values_different"] == 1 and arrays[0]["first_difference"]["index"] == index
        assert arrays[0]["max_abs_diff"] == pytest.approx(max_abs, rel=1e-6)
    text = wfeq.format_report(deep)
    assert "DIFFERENT (fails):" in text and suffix in text and "max |a-b|" in text


def test_a_changed_display_tiff_is_reported_in_the_product_and_its_delivery_copy(tmp_path, base):
    other, _ = run(tmp_path, "copy", 3, ("tiff_pixel",))
    report = wfeq.compare(base[0], other)
    named(report, "different", "products/<products-hex12#1>/natural_display_16bit.tif")
    named(report, "different", "{DELIV}/delivery-<delivery#1>/final_srgb16.tif")


@pytest.mark.parametrize("variant,suffix,section,needle", [
    ("fits_header", "work/sky_master.fit", "header", "EXPTIME"),
    ("tiff_compression", "natural_display_16bit.tif", "tags", "Compression"),
    ("tiff_description", "natural_display_16bit.tif", "tags", "rounded_to_zero"),
    ("json_semantic", "job-<job#1>/job.json", "json", "/json/diagnostics/support/zeroed"),
    ("json_new_field", "job-<job#1>/job.json", "json", "/json/diagnostics/support/new_field"),
    ("text_line", "work/light_.seq", "text", "light_"),
])
def test_metadata_record_and_text_changes_are_reported(tmp_path, base, variant, suffix, section, needle):
    other, _ = run(tmp_path, variant, 3, (variant,))
    report = wfeq.compare(base[0], other)
    assert not report["pass"]
    detail = named(report, "different", suffix)
    assert needle in json.dumps(detail[section]), detail
    if variant == "tiff_compression":
        assert "pixels" not in detail  # decoded pixels stay equal; only the encoding is reported


def _record_pair(base_manifest, doc_suffix):
    """Two copies of one manifest whose chosen JSON record is forced to be compared on content."""
    a, b = copy.deepcopy(base_manifest), copy.deepcopy(base_manifest)
    rel = next(r for r in b["files"] if r.endswith(doc_suffix))
    b["files"][rel]["sha256"] = "f" * 64
    return a, b, a["files"][rel]["content"]["doc"], b["files"][rel]["content"]["doc"]


@pytest.mark.parametrize("key", sorted(wfeq.EXCLUDED_KEYS) + ["created_at", "finished_at", "render_seconds",
                                                               "phases_seconds"])
def test_each_excluded_key_is_ignored_but_a_semantic_sibling_is_compared(base, key):
    a, b, doc_a, doc_b = _record_pair(base[0], "/job.json")
    timestamp = key.endswith("_at")
    doc_a["diagnostics"]["support"][key] = "2026-09-28T08:51:29Z" if timestamp else 1
    doc_b["diagnostics"]["support"][key] = "2026-09-28T09:10:00.5+00:00" if timestamp else {"differs": [2, 3]}
    report = wfeq.compare(a, b)
    assert report["pass"], wfeq.format_report(report)
    assert any(item["path"].endswith("/" + key) for item in named(report, "identical_content", "/job.json")
               ["hidden_differences"])
    doc_b["diagnostics"]["support"]["zeroed"] += 1
    assert not wfeq.compare(a, b)["pass"]


@pytest.mark.parametrize("key,value_a,value_b", [(k, 20.0, 30.0) for k in sorted(wfeq.KEPT_KEYS)] + [
    ("exptime", 20.0, 30.0), ("seconds_total", 1, 2), ("created", "2026-09-28", "2026-09-29"),
    ("clipped_at", {"level": 1}, {"level": 2}), ("look_at", "sky", "ground")])
def test_semantic_keys_that_look_like_timings_are_compared(base, key, value_a, value_b):
    a, b, doc_a, doc_b = _record_pair(base[0], "/job.json")
    doc_a["diagnostics"]["exposure"][key] = value_a
    doc_b["diagnostics"]["exposure"][key] = value_b
    report = wfeq.compare(a, b)
    assert not report["pass"] and key in json.dumps(named(report, "different", "/job.json")["json"])


@pytest.mark.parametrize("value_a,value_b", [("2026-09-28T08:51:29.095725+00:00", "2026-09-28T09:00:00Z"),
                                             (1790596868.5, 1790596999), (None, "2026-09-28T08:52:49Z")])
def test_timestamps_under_at_keys_are_hidden(base, value_a, value_b):
    a, b, doc_a, doc_b = _record_pair(base[0], "/job.json")
    doc_a["diagnostics"]["support"]["finished_at"], doc_b["diagnostics"]["support"]["finished_at"] = value_a, value_b
    assert wfeq.compare(a, b)["pass"]


def test_float_formatting_and_types_are_compared_exactly(base):
    a, b, doc_a, doc_b = _record_pair(base[0], "/job.json")
    for value_a, value_b in ((0.0, -0.0), (1, 1.0), (1, True)):
        doc_a["diagnostics"]["support"]["zeroed"], doc_b["diagnostics"]["support"]["zeroed"] = value_a, value_b
        assert not wfeq.compare(a, b)["pass"], (value_a, value_b)


def test_unretained_digest_is_masked_but_a_retained_file_digest_is_not(base):
    a, b, doc_a, doc_b = _record_pair(base[0], "/workflow.json")
    doc_b["staging"]["sha256"] = "1" * 64
    doc_b["staging"]["bytes"] = 1
    assert wfeq.compare(a, b)["pass"]
    doc_b["checkpoints"]["masks"]["files"][0]["sha256"] = "2" * 64
    report = wfeq.compare(a, b)
    assert not report["pass"]
    assert "/json/checkpoints/masks/files[0]/sha256" in json.dumps(named(report, "different", "/workflow.json"))


def test_derived_digests_are_recomputed_on_each_side(tmp_path, base):
    other, _ = run(tmp_path, "derived", 6, ("broken_derived",))
    report = wfeq.compare(base[0], other)
    assert not report["pass"] and report["derived_digests"]["b"]["failed"][0]["rule"] == "data_sha256"
    a, b, _, doc_b = _record_pair(base[0], "/review_delivery.json")
    doc_b["key"] = "3" * 64
    report = wfeq.compare(a, b)
    rules = {f["rule"] for f in report["derived_digests"]["b"]["failed"]}
    assert not report["pass"] and rules == {"review key", "review key names a verified receipt"}


def test_broken_binding_fails(tmp_path, base):
    other, _ = run(tmp_path, "binding", 7, ("broken_binding",))
    assert other["bindings"]["mismatch"]
    report = wfeq.compare(base[0], other)
    assert not report["pass"] and report["bindings"]["b"]["mismatch"]
    assert "BINDING MISMATCH IN B" in wfeq.format_report(report)


def test_logs_locks_and_worker_ledgers_are_listed_not_compared(tmp_path, base):
    other, _ = run(tmp_path, "ledger", 4, ("missing_ledger",))
    report = wfeq.compare(base[0], other)
    assert report["pass"]
    assert any(n.endswith("worker-001.json") for n in report["only_in_listed"]["a"])
    log = named(report, "excluded", "/01_convert.log")
    assert log["verdict"] == "EXCLUDED" and log["bytes_equal"] is False


def test_allow_globs_mark_intended_differences_and_are_reported_separately(tmp_path, base):
    other, _ = run(tmp_path, "extra", 5, ("extra_file",))
    report = wfeq.compare(base[0], other)
    assert not report["pass"] and named(report, "only_in_b", "/extra.txt")
    report = wfeq.compare(base[0], other, allow=["*/work/extra.txt"])
    assert report["pass"] and named(report, "allowed", "/extra.txt")["allowed_by"] == "*/work/extra.txt"
    changed, _ = run(tmp_path, "allowed-pixel", 3, ("tiff_pixel",))
    report = wfeq.compare(base[0], changed, allow=["*natural_display_16bit.tif", "*/final_srgb16.tif"])
    assert report["pass"] and not report["different"]
    assert named(report, "allowed", "natural_display_16bit.tif")["verdict"] == "ALLOWED_DIFFERENT"
    assert "ALLOWED (by --allow, not failing):" in wfeq.format_report(report)


def test_tolerance_is_explicit_numeric_and_limited_to_data(tmp_path, base):
    ulp, _ = run(tmp_path, "ulp", 8, ("fits_ulp",))
    plain = wfeq.compare(base[0], ulp, deep=True)
    assert not plain["pass"]
    arrays = named(plain, "different", "work/sky_master.fit")["deep"]["arrays"]
    assert arrays[0]["values_different"] == 1 and 0 < arrays[0]["max_abs_diff"] < 1e-7
    tolerated = wfeq.compare(base[0], ulp, tolerances=["*master.fit=1e-6"])
    assert tolerated["pass"] and named(tolerated, "tolerated", "work/sky_master.fit")["tolerance"] == 1e-6
    assert not wfeq.compare(base[0], ulp, tolerances=["*master.fit=1e-12"])["pass"]
    header, _ = run(tmp_path, "header", 9, ("fits_header",))
    assert not wfeq.compare(base[0], header, tolerances=["*.fit=1"])["pass"]  # never covers metadata
    with pytest.raises(ValueError):
        wfeq.compare(base[0], ulp, tolerances=["*.fit"])


def test_manifests_must_use_the_same_block_size(tmp_path, base):
    other = wfeq.fingerprint(base[1]["workflow_dir"], block_bytes=2 * BLOCK)
    with pytest.raises(ValueError, match="block"):
        wfeq.compare(base[0], other)


def test_fingerprint_is_independent_of_worker_count_and_memory_budget(base):
    single = wfeq.fingerprint(base[1]["workflow_dir"], workers=1, block_bytes=BLOCK, memory_bytes=1)
    strip = lambda m: {r: {k: v for k, v in e.items() if k != "birth_ns"} for r, e in m["files"].items()}
    assert strip(single) == strip(base[0])
    assert wfeq.compare(base[0], single)["counts"] == {"IDENTICAL_BYTES": len(single["files"]) - 8, "EXCLUDED": 8}


def test_long_history_card_with_a_path_is_split_at_root_dependent_places(tmp_path, base):
    other, _ = run(tmp_path, "a-much-longer-root-name-so-the-history-card-splits-elsewhere", 2)
    assert wfeq.compare(base[0], other)["pass"]


def test_self_test_passes():
    assert wfeq.self_test(verbose=False)


def test_cli_round_trip_and_refusals(tmp_path, base, capsys):
    tree = base[1]
    out = tmp_path / "manifests" / "a.json.gz"
    assert wfeq.main(["fingerprint", str(tree["workflow_dir"]), "-o", str(out), "--workers", "2", "--quiet"]) == 0
    assert wfeq.main(["fingerprint", str(tree["workflow_dir"]), "-o", str(out), "--quiet"]) == 2  # no overwrite
    inside = tree["workflow_dir"] / "manifest.json"
    assert wfeq.main(["fingerprint", str(tree["workflow_dir"]), "-o", str(inside), "--quiet"]) == 2
    assert not inside.exists()
    other, _ = run(tmp_path, "cli-other", 2)
    b = tmp_path / "manifests" / "b.json"
    wfeq._write_json_file(b, wfeq.fingerprint(other["workflow_dir"]))
    report = tmp_path / "manifests" / "report.json"
    capsys.readouterr()
    assert wfeq.main(["compare", str(out), str(b), "-o", str(report)]) == 0
    assert "Workflow equivalence: PASS" in capsys.readouterr().out
    assert json.loads(report.read_text(encoding="utf-8"))["pass"] is True
    changed, _ = run(tmp_path, "cli-changed", 3, ("npy_element",))
    c = tmp_path / "manifests" / "c.json"
    wfeq._write_json_file(c, wfeq.fingerprint(changed["workflow_dir"]))
    assert wfeq.main(["compare", str(out), str(c)]) == 1
    assert wfeq.main(["compare", str(out), str(c), "--allow", "*frame_support_light_00001.npy"]) == 0
    assert wfeq.main(["compare", str(out), str(c), "--tolerance", "nonsense"]) == 2
    assert wfeq.main(["compare", str(out), str(c), "--allow-field", "nonsense"]) == 2
    assert wfeq.main(["compare", str(out), str(c), "--allow-field", "*/job.json=(unclosed"]) == 2
    capsys.readouterr()
    assert wfeq.main(["compare", str(out), str(b), "--strict-ids"]) == 1  # other roots: every id differs
    assert "DETERMINISTIC IDS DIFFER" in capsys.readouterr().out
    assert wfeq.main(["compare", str(out), str(b), "--no-strict-ids"]) == 0
    with pytest.raises(SystemExit):
        wfeq.main(["compare", str(out), str(b), "--strict-ids", "--no-strict-ids"])
    assert wfeq.main(["rules"]) == 0
    printed = capsys.readouterr().out
    assert "free_bytes_at_admission" in printed and "diagnostic_slots" in printed and "deterministic_id_kinds" in printed


def test_workflow_record_is_required(tmp_path):
    with pytest.raises(ValueError, match="workflow.json"):
        wfeq.fingerprint(tmp_path)


# ---- two runs of the real workflow orchestration -------------------------------------------------
def _real_run(setup, monkeypatch, tmp_path, cache, works):
    record, _ = build(dict(setup, output=cache), monkeypatch, works=works, stars=True, save_starless=True,
                      save_star_layer=True)
    done = wf.run_workflow(record["workflow_dir"])
    assert done["state"] == "succeeded", done.get("error")
    assert done["base_delivery_dir"] != done["delivery_dir"]  # base plus the automatic meteor add-back
    return done, wfeq.fingerprint(done["workflow_dir"], workers=4)


def test_real_workflow_runs_in_different_roots_are_equivalent(setup, monkeypatch, tmp_path):
    first, a = _real_run(setup, monkeypatch, tmp_path, tmp_path / "cache1", tmp_path / "works1")
    second, b = _real_run(setup, monkeypatch, tmp_path, tmp_path / "cache2", tmp_path / "works2")
    assert first["workflow_dir"] != second["workflow_dir"] and first["signature"] != second["signature"]
    report = wfeq.compare(a, b)
    assert report["pass"], wfeq.format_report(report)
    assert not report["strict_ids"]["on"] and a["delivery_parents"] == {"{DELIV}": "whole"}
    assert report["derived_digests"]["a"]["checked"] >= 5 and not report["derived_digests"]["b"]["failed"]
    assert any(n.startswith("{DELIV}/delivery-trails-1kept-") for n in wfeq.Side(b, "b").by_name)
    # one flipped bit in a delivered master is found, located and quantified
    entry = next(e for r, e in b["files"].items() if r.endswith("master_scene_linear.tif") and r.startswith("{DELIV}"))
    page = entry["content"]["pages"][0]
    assert page.get("stream"), "product masters are uncompressed and streamed"
    with open(entry["path"], "r+b") as handle:
        handle.seek(page["stream"][0] + 4 * 1000)
        value = handle.read(1)
        handle.seek(-1, 1)
        handle.write(bytes([value[0] ^ 1]))
    changed = wfeq.fingerprint(second["workflow_dir"], workers=4)
    report = wfeq.compare(a, changed, deep=True)
    assert not report["pass"] and changed["bindings"]["mismatch"]
    detail = named(report, "different", "master_scene_linear.tif")
    assert detail["deep"]["arrays"][0]["values_different"] == 1
    assert detail["deep"]["arrays"][0]["first_difference"]["element"] == 1000


def test_real_workflow_rerun_with_a_fixed_idempotency_key_in_the_same_root(setup, monkeypatch, tmp_path):
    monkeypatch.setattr(wf, "submit_workflow", functools.partial(wf.submit_workflow, idempotency_key="bench-r1"))
    first, a = _real_run(setup, monkeypatch, tmp_path, tmp_path / "cache", tmp_path / "works")
    shutil.rmtree(tmp_path / "cache")
    shutil.rmtree(tmp_path / "works")
    second, b = _real_run(setup, monkeypatch, tmp_path, tmp_path / "cache", tmp_path / "works")
    assert first["workflow_dir"] == second["workflow_dir"]
    report = wfeq.compare(a, b)
    assert report["pass"], wfeq.format_report(report)
    assert report["strict_ids"]["on"] and not report["strict_ids"]["differ"]
    for placeholder in ("<workflow#1>", "<workflow-signature#1>", "<job#1>", "<aester#1>"):
        assert report["aliases"]["a"][placeholder] == report["aliases"]["b"][placeholder], placeholder
    assert first["signature"] == second["signature"] == report["aliases"]["b"]["<workflow-signature#1>"]
