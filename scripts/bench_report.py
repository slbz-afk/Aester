"""Stage tables for Aester benchmark runs (efficiency P0, criterion E3).

Reads the results/<run-id>/ folders written by scripts/bench_workflow.py and
prints Markdown (or JSON with --json):

  1. run overview   submit->base, submit->final, add-back, whole-tree CPU,
                    peak tree private memory, lowest system available memory,
                    hashed bytes, SHA-extension guess, simulation label
     hashing        (efficiency P1, criterion H4) real reads of the workers and of
                    the submitting process, reads saved by kind (ledger,
                    handover, write, batch), in-memory hashing, and the ledger
                    mode with its shadow checks / mismatches (timing notes);
                    ledger-miss bytes by status and each worker's largest
                    misses (efficiency P1b-5c, hash_ledger notes)
  2. status stages  from the 1 Hz timeline (whole process tree, Siril included):
                    wall, CPU seconds, mean busy cores, peak private / working
                    set, I/O read and write
  3. phases         from the product's own StageTimer records (workflow worker,
                    with the stack job's phases under stack_wait, and the
                    trail add-back): wall, own CPU, child-process CPU, I/O,
                    hashed bytes, rise of the process's peak private memory
  4. operations     overlapping job operations (native_remap, Siril scripts...)
  5. governor       (efficiency P1b-2, shadow only) A4: each Siril launch's
                    "today" prediction joined with the same receipt's measured
                    peak job commit and the direct child's peak working set;
                    A3: governor time and note sizes; A5: peak sampler rate and
                    CPU; A6: the what-if (laptop) recommendations
  6. O-1            (efficiency P1c §3.6) the trail scan beside the post-stack
                    work: the overlap.o1 note (eligible, reason, outcome, the
                    scan processes overlapped / serial, the coverage plan), the
                    window native_remap_support + diagnostics + transient_scan
                    (with O-1 on, transient_scan is only the join wait), the
                    scan itself (operation transient_detect) and the join wait
                    (operation transient_join; the note's join_wait_s)

Runs are chosen by result folder, or by suffix with --label (e.g. L48-R4-L1).
Several runs of the same slot are shown side by side in the overview and as
mean (min-max) in the stage tables.

    PY scripts\\bench_report.py --label L20-R1a --label L20-R1b
    PY scripts\\bench_report.py outputs\\perf-baseline-20260928\\results\\* --json out.json
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO / "outputs" / "perf-baseline-20260928"
GB = 1e9
SAVED_KINDS = ("ledger", "handover", "write", "batch")
STATUS_ORDER = ("queued", "preflight", "stack", "milkyway", "separation", "density_advice", "star_reduction",
                "layer_export", "publish", "trail_addback", "complete")


def load_run(folder: Path) -> dict:
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    records = folder / "records"
    workflow = _read(records / "workflow.json")
    job = _read(records / "job.json")
    timeline = []
    path = folder / "timeline.jsonl"
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                timeline.append(json.loads(line))
    return {"folder": folder, "summary": summary, "workflow": workflow, "job": job, "timeline": timeline}


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def slot_of(run) -> str:
    plan = run["summary"]["plan"]
    return plan.get("slot") or f"{plan['mode']}{plan['frames']}"


def overview(run) -> dict:
    s = run["summary"]
    wall = s.get("wall") or {}
    sampler = s.get("sampler") or {}
    timings = (run["workflow"] or {}).get("timings") or {}
    job_timings = ((run["job"] or {}).get("diagnostics") or {}).get("timings") or {}
    host = timings.get("host") or job_timings.get("host") or {}
    hashed = 0
    for t in (timings, job_timings):
        hashed += ((t.get("resources") or {}).get("total") or {}).get("hashed_bytes") or 0
    return {**_overview_times(run, s, wall, sampler, host, hashed), "hashing": hashing(run), "governor": governor(run),
            "overlap": overlap(run)}


# --------------------------------------------------------------------------- O-1 (efficiency P1c §3.6)

OVERLAP_NOTE = "overlap.o1"                         # starstack.core.O1_NOTE
OVERLAP_WINDOW_PHASES = ("native_remap_support", "diagnostics", "transient_scan")
OVERLAP_OPERATIONS = ("transient_detect", "transient_join")


def overlap(run) -> dict | None:
    """The stack job's O-1 view; None for runs made before O-1 (no overlap.o1 note).

    ``window_s`` is the sum of the phases the overlap spans (prereg §3.6: compare it between --overlap off and on);
    ``transient_detect_s`` the scan itself (the operation runs on the helper thread under O-1, on the main thread
    otherwise), ``join_wait_s`` the operation transient_join (the main thread waiting for the scan at its serial
    position; absent when O-1 did not run)."""
    job_timings = ((run["job"] or {}).get("diagnostics") or {}).get("timings") or {}
    note = (job_timings.get("notes") or {}).get(OVERLAP_NOTE)
    if isinstance(note, list):
        note = note[-1] if note else None
    note = note_view(note) if note is not None else None
    if not note:
        return None
    phases = job_timings.get("phases_seconds") or {}
    operations = job_timings.get("operations_seconds") or {}
    resources = (job_timings.get("resources") or {}).get("operations") or {}
    window = [phases.get(name) for name in OVERLAP_WINDOW_PHASES]
    detect = resources.get("transient_detect") or {}
    return {"eligible": note.get("eligible"), "reason": note.get("reason"), "outcome": note.get("outcome"),
            "n_o1": note.get("n_o1"), "n_serial": note.get("n_serial"), "cov_plan": note.get("cov_plan"),
            "crop_cap": note.get("crop_cap"), "b_done_at_join": note.get("b_done_at_join"),
            **{f"{name}_s": _r(phases.get(name)) for name in OVERLAP_WINDOW_PHASES},
            "window_s": _r(_sum(window)) if all(isinstance(v, (int, float)) for v in window) else None,
            "transient_detect_s": _r(operations.get("transient_detect")),
            "transient_detect_child_cpu_s": _r((detect.get("children") or {}).get("cpu_seconds")),
            "join_wait_s": _r(operations.get("transient_join", note.get("join_wait_s")))}


def overlap_markdown(groups) -> list[str]:
    runs = [o for g in groups for o in g["runs"] if o.get("overlap")]
    if not runs:
        return []
    out = ["", "## O-1: trail scan beside the post-stack work (efficiency P1c §3.6)", "",
           "window = native_remap_support + diagnostics + transient_scan (compare --overlap off with on); with O-1 "
           "on, transient_scan is only the join wait and the scan itself is the transient_detect operation.", "",
           "| run | eligible | reason | outcome | scan n (O-1 / serial) | coverage plan | crop cap | "
           "native_remap_support s | diagnostics s | transient_scan s | window s | transient_detect s | "
           "scan child CPU s | join wait s |",
           "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for o in runs:
        v = o["overlap"]
        out.append(f"| {o['run_id']} | {v['eligible']} | {v['reason'] or '–'} | {v['outcome'] or '–'} | "
                   f"{fmt(v['n_o1'], 0)} / {fmt(v['n_serial'], 0)} | {fmt(v['cov_plan'], 0)} | "
                   f"{fmt(v['crop_cap'], 0)} | {fmt(v['native_remap_support_s'])} | {fmt(v['diagnostics_s'])} | "
                   f"{fmt(v['transient_scan_s'])} | {fmt(v['window_s'])} | {fmt(v['transient_detect_s'])} | "
                   f"{fmt(v['transient_detect_child_cpu_s'])} | {fmt(v['join_wait_s'], 2)} |")
    return out


def _total(timings, key):
    return ((timings.get("resources") or {}).get("total") or {}).get(key)


def _sum(values):
    values = [v for v in values if isinstance(v, (int, float))]
    return sum(values) if values else None


def hashing(run) -> dict:
    """Efficiency P1 (H4): real reads and saved reads of the workers plus the submitting process.

    The workflow worker's timings include its in-process trail add-back; the stack job runs in its
    own process.  The submitter (bench_workflow's own process calling submit_workflow, a GUI/MCP
    process in production) is in no record: bench_workflow.py stores it as summary.submitter_hashing
    (absent in runs made before P1: then the totals exclude it and say so).
    """
    timings = (run["workflow"] or {}).get("timings") or {}
    job_timings = ((run["job"] or {}).get("diagnostics") or {}).get("timings") or {}
    workers = (timings, job_timings)
    submitter = run["summary"].get("submitter_hashing")
    submitter = submitter if isinstance(submitter, dict) else {}
    real = _sum(_total(t, "hashed_bytes") for t in workers)
    out = {"workers_real_gb": _gb(real), "submitter_real_gb": _gb(submitter.get("hashed_bytes")),
           "submitter_included": "hashed_bytes" in submitter,
           "total_real_gb": _gb(_sum([real, submitter.get("hashed_bytes")])),
           "saved_gb": _gb(_sum([*(_total(t, "hash_saved_bytes") for t in workers),
                                 submitter.get("hash_saved_bytes")])),
           "memory_gb": _gb(_sum([*(_total(t, "hashed_memory_bytes") for t in workers),
                                  submitter.get("hashed_memory_bytes")]))}
    for kind in SAVED_KINDS:
        key = f"hash_saved_{kind}_bytes"
        out[f"saved_{kind}_gb"] = _gb(_sum([*(_total(t, key) for t in workers), submitter.get(key)]))
    ledgers = [(scope, (t.get("notes") or {}).get("hash_ledger")) for scope, t in (("workflow", timings),
                                                                                   ("job", job_timings))]
    ledgers = [(scope, note) for scope, note in ledgers if isinstance(note, dict)]
    out["ledger_modes"] = {scope: note.get("mode") for scope, note in ledgers} or None
    for key in ("checks", "mismatches", "concurrent"):
        out[f"shadow_{key}"] = _sum((note.get("shadow") or {}).get(key) for _, note in ledgers) if ledgers else None
    out.update(_miss_columns(ledgers))
    return out


LARGEST_SHOWN = 5
MISS_STATUSES = ("no_entry", "stale", "racy", "busy")


def _miss_columns(ledgers) -> dict:
    """Efficiency P1b-5c: ledger-miss bytes per status (both workers) and each worker's largest misses.

    From ``miss_bytes`` / ``largest_misses`` of the ``hash_ledger`` notes (aester-hashing-3); runs made
    before P1b have neither and show None.  ``refused_touched`` / ``refused_seal``: seals and records a
    guard refused (expected 0 in product runs).
    """
    noted = [(scope, note) for scope, note in ledgers if isinstance(note.get("miss_bytes"), dict)]
    out = {f"miss_{status}_gb": _gb(_sum((note["miss_bytes"]).get(status) for _, note in noted)) if noted else None
           for status in MISS_STATUSES}
    for key in ("touched", "seal"):
        out[f"refused_{key}"] = (_sum((note.get("refused") or {}).get(key) for _, note in noted) if noted else None)
    out["largest_misses"] = {scope: [_miss_row(item) for item in (note.get("largest_misses") or [])[:LARGEST_SHOWN]]
                             for scope, note in noted} or None
    return out


def _miss_row(item):
    """``[bytes, status, source, "folder/name"]`` of a note as a dict (GB rounded to 3 places)."""
    size, status, source, name = (list(item) + [None] * 4)[:4]
    return {"gb": None if size is None else round(size / GB, 3), "status": status, "source": source, "file": name}


# --------------------------------------------------------------------------- governor (efficiency P1b-2)

A4_CENTRAL_BAND = (-0.10, 0.15)     # convert / probe: central prediction vs measured peak job commit
A4_HIGH_RATIO = 1.7                 # every other part: measured <= high <= 1.7 x measured
A4_ACCURACY_PARTS = {"convert", "probe"}
A6_GOVERNABLE = {"convert", "probe", "export", "stack_finite"}
# starstack.resources.POOL_TRIM_DEFAULTS: the values a note over its size cap drops from its pool entries
# ("pools_trimmed", efficiency P1c-0); an absent key in such an entry means this value
POOL_TRIM_DEFAULTS = {"calls": 1, "over_budget": 0, "planned_action": "run"}


def note_view(value) -> dict:
    """A decision's today / rec / whatif view or a pool entry as a dict. A note over its size cap writes them as
    one-line JSON text ("compact": true, starstack.resources.note_view); unreadable values are {}."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            loaded = json.loads(value)
        except ValueError:
            return {}
        return loaded if isinstance(loaded, dict) else {}
    return {}


def pool_view(entry, trimmed=False) -> dict:
    """A pool entry as a new dict; in a note marked ``pools_trimmed`` the dropped defaults (POOL_TRIM_DEFAULTS) are
    filled back in (starstack.resources.pool_view)."""
    view = dict(note_view(entry))
    if trimmed and view:
        for key, value in POOL_TRIM_DEFAULTS.items():
            view.setdefault(key, value)
    return view


def _governor_notes(run) -> dict:
    timings = (run["workflow"] or {}).get("timings") or {}
    job_timings = ((run["job"] or {}).get("diagnostics") or {}).get("timings") or {}
    out = {}
    for scope, t in (("workflow", timings), ("job", job_timings)):
        note = (t.get("notes") or {}).get("governor")
        if isinstance(note, dict):
            out[scope] = (note, t)
    return out


def governor(run) -> dict | None:
    """A3-A6 views of one run; None for runs made before P1b (no governor note)."""
    notes = _governor_notes(run)
    if not notes:
        return None
    out = {"modes": {scope: note.get("mode") for scope, (note, _) in notes.items()},
           "footprints": {scope: note.get("footprints") for scope, (note, _) in notes.items()},
           "note_bytes": {scope: note.get("bytes") for scope, (note, _) in notes.items()},
           "seconds": {scope: note.get("seconds") for scope, (note, _) in notes.items()},
           "operation_seconds": {scope: (t.get("operations_seconds") or {}).get("governor")
                                 for scope, (_, t) in notes.items()},
           "calls": {scope: note.get("calls") for scope, (note, _) in notes.items()},
           "dropped": {scope: note.get("dropped") for scope, (note, _) in notes.items()},
           "errors": [f"{scope}: {e}" for scope, (note, _) in notes.items() for e in note.get("errors") or []],
           "sampler": {scope: _sampler_view(t) for scope, (_, t) in notes.items()},
           "launches": [], "whatif": [], "pools": {}}
    for scope, (note, t) in notes.items():
        out["pools"].update({f"{scope}:{stage}": pool_view(entry, note.get("pools_trimmed"))
                             for stage, entry in (note.get("pools") or {}).items()})
    if "job" in notes:
        note, timings = notes["job"]
        out["launches"] = a4_launches(note, timings)
        out["whatif"] = a6_whatif(note)
    return out


def _sampler_view(timings) -> dict:
    res = timings.get("resources") or {}
    stats = res.get("peak_sampler") or {}
    phases = res.get("phases") or {}
    missing = sorted(name for name, view in phases.items() if view.get("local_peak_private_bytes") is None)
    return {"effective_hz": stats.get("effective_hz"), "cpu_share": stats.get("cpu_share"),
            "seconds": stats.get("seconds"), "phases_without_local_peak": missing}


def a4_launches(note, timings) -> list[dict]:
    """Siril decisions joined by script name with the receipt's operation script:<name> (max over same-named
    launches on both sides). The category follows the parts of the actual launch (X16)."""
    operations = (timings.get("resources") or {}).get("operations") or {}
    by_name = {}
    for decision in note.get("decisions") or []:
        if decision.get("kind") != "siril":
            continue
        today = note_view(decision.get("today"))
        row = by_name.setdefault(decision.get("name"), {"name": decision.get("name"), "parts": set(), "launches": 0,
                                                         "central_gb": None, "high_gb": None, "ws_high_gb": None,
                                                         "threads": today.get("t"), "pool": today.get("p"),
                                                         "allowance_gb": today.get("A_gb")})
        row["launches"] += 1
        row["parts"].update(str(decision.get("parts") or "").split("+"))
        for key, source in (("central_gb", "c_gb"), ("high_gb", "ch_gb"), ("ws_high_gb", "wh_gb")):
            value = today.get(source)
            if isinstance(value, (int, float)):
                row[key] = value if row[key] is None else max(row[key], value)
    rows = []
    for name, row in by_name.items():
        child = (operations.get(f"script:{name}") or {}).get("children") or {}
        measured = _gb(child.get("peak_job_memory_bytes"))
        measured_ws = _gb(child.get("child_peak_working_set_bytes"))
        parts = sorted(p for p in row["parts"] if p)
        accuracy = bool(parts) and set(parts) <= A4_ACCURACY_PARTS
        row.update(parts="+".join(parts), measured_gb=measured, measured_ws_gb=measured_ws,
                   category="a" if accuracy else "b")
        if measured is None or row["high_gb"] is None:
            row["verdict"] = "no data"
        elif accuracy:
            error = row["central_gb"] / measured - 1
            row["central_error"] = round(error, 3)
            row["verdict"] = "pass" if A4_CENTRAL_BAND[0] <= error <= A4_CENTRAL_BAND[1] else "miss"
        else:
            ratio = row["high_gb"] / measured
            row["high_ratio"] = round(ratio, 3)
            row["verdict"] = "pass" if 1.0 <= ratio <= A4_HIGH_RATIO else "miss"
        if measured_ws is not None and row["ws_high_gb"] is not None:
            row["ws_verdict"] = "pass" if measured_ws <= row["ws_high_gb"] else "miss"
        rows.append(row)
    return rows


def _cores_used(mask_hex, lps, cores):
    """(LPs in the mask, distinct cores they use, the mask's bit width) on the what-if machine, whose core i holds
    LPs i*per .. i*per+per-1 (resources.parse_whatif); cores used is None without lps/cores."""
    try:
        mask = int(mask_hex, 16)
    except (TypeError, ValueError):
        return None, None, None
    bits = [b for b in range(mask.bit_length()) if mask >> b & 1]
    if not lps or not cores:
        return len(bits), None, mask.bit_length()
    per = max(1, lps // cores)
    return len(bits), len({b // per for b in bits}), mask.bit_length()


_WHATIF_ITEM = re.compile(r"(?:^|,)\s*(lps|cores)\s*=\s*([0-9]+)\s*(?=,|$)", re.IGNORECASE)


def whatif_cpu(note) -> tuple:
    """(logical processors, physical cores) of the what-if machine: ``whatif_facts.cpu`` (the governor writes it), else
    the lps=/cores= items of the what-if text (resources.parse_whatif's format, older notes); None where unknown."""
    facts = note.get("whatif_facts") or {}
    cpu = facts.get("cpu") if isinstance(facts.get("cpu"), dict) else {}
    lps, cores = cpu.get("logical_total"), cpu.get("physical_cores")
    items = {key.lower(): int(value) for key, value in _WHATIF_ITEM.findall(str(facts.get("text") or ""))}
    return (lps if lps is not None else items.get("lps")), (cores if cores is not None else items.get("cores"))


A6_NEEDED = ("route", "c_gb", "w_gb", "ram_gb", "commit_gb", "neutral", "over", "action")


def a6_whatif(note) -> list[dict]:
    """The what-if (laptop) decisions with the A6 rule checked per launch. A decision whose what-if lacks a key the
    rule needs (a note trimmed before the review fix) is 'no data', never a failure; the mask rule still applies."""
    lps, cores = whatif_cpu(note)
    rows = []
    for decision in note.get("decisions") or []:
        whatif = note_view(decision.get("whatif"))
        if decision.get("kind") != "siril" or not whatif:
            continue
        parts = set(str(decision.get("parts") or "").split("+"))
        missing = [key for key in A6_NEEDED if key not in whatif]
        fits = None
        if None not in (whatif.get("c_gb"), whatif.get("w_gb"), whatif.get("commit_gb"), whatif.get("ram_gb")):
            fits = whatif["c_gb"] <= whatif["commit_gb"] and whatif["w_gb"] <= whatif["ram_gb"]
        bits, used, width = _cores_used(whatif.get("mask"), lps, cores)
        problems = []
        if missing:
            pass
        elif parts <= A6_GOVERNABLE:
            if not (fits or (whatif.get("over") and whatif.get("action"))):
                problems.append("neither fits nor over_budget with an action")
            if whatif.get("route") == "A" and whatif.get("neutral") is not True:
                problems.append("route A plan not record-neutral")
        elif parts & {"stack_rej", "export_stack"}:
            if whatif.get("route") != "today":
                problems.append("stock rejection stack changed")
            if whatif.get("over") and whatif.get("action") != "needs_U6":
                problems.append("overrun without needs_U6")
        if bits is not None and (bits > 64 or width > 64 or (used is not None and used < min(bits, cores))):
            problems.append("mask packs SMT siblings or exceeds 64")
        rows.append({"name": decision.get("name"), "parts": decision.get("parts"), "route": whatif.get("route"),
                     "threads": whatif.get("t"), "mask": whatif.get("mask"),
                     "commit_gb": whatif.get("c_gb"), "ws_gb": whatif.get("w_gb"),
                     "ram_allowance_gb": whatif.get("ram_gb"), "commit_allowance_gb": whatif.get("commit_gb"),
                     "over_budget": whatif.get("over"), "planned_action": whatif.get("action"),
                     "record_neutral": whatif.get("neutral"), "fits": fits,
                     "verdict": "; ".join(problems) if problems else
                     (f"no data (missing {', '.join(missing)})" if missing else "pass")})
    return rows


def governor_markdown(groups) -> list[str]:
    runs = [o for g in groups for o in g["runs"] if o.get("governor")]
    if not runs:
        return []
    out = ["", "## Governor (efficiency P1b-2, shadow only)", "",
           "A3/A5: governor time, note size (bytes as stored at the note's deepest copy: indent 2 at its nesting "
           "depth, CRLF; the job note in workflow.json checkpoints, X15 cap 8 KiB), peak sampler. Footprints table "
           "version per scope; a refit changes it (X16: an unchanged version is an out-of-sample check).", "",
           "| run | modes | footprints | job s | workflow s | job op s | job bytes | workflow bytes | dropped | "
           "job Hz | job CPU share | phases w/o peak | errors |",
           "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- |"]
    for o in runs:
        g = o["governor"]
        job_sampler = g["sampler"].get("job") or {}
        missing = sorted({p for view in g["sampler"].values() for p in view.get("phases_without_local_peak") or []})
        out.append(f"| {o['run_id']} | {', '.join(f'{k} {v}' for k, v in g['modes'].items())} | "
                   f"{', '.join(sorted({str(v) for v in g['footprints'].values()}))} | "
                   f"{fmt(g['seconds'].get('job'), 3)} | {fmt(g['seconds'].get('workflow'), 3)} | "
                   f"{fmt(g['operation_seconds'].get('job'), 3)} | {fmt(g['note_bytes'].get('job'), 0)} | "
                   f"{fmt(g['note_bytes'].get('workflow'), 0)} | {g['dropped']} | "
                   f"{fmt(job_sampler.get('effective_hz'))} | {fmt(job_sampler.get('cpu_share'), 4)} | "
                   f"{', '.join(missing) or '–'} | {len(g['errors'])} |")
    out += ["", "### A4: Siril launches, today's prediction vs measured (job receipt)", "",
            "(a) convert/probe: central within [-10 %, +15 %]; (b) other parts: measured ≤ high ≤ 1.7 × measured; "
            "(c) child peak WS ≤ predicted WS high.", "",
            "| run | launch | parts | n | t | p | A GB | measured GB | central GB | high GB | err / ratio | verdict | "
            "child WS GB | WS high GB | WS verdict |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |"]
    for o in runs:
        for r in o["governor"]["launches"]:
            score = r.get("central_error", r.get("high_ratio"))
            out.append(f"| {o['run_id']} | {r['name']} | {r['parts']} | {r['launches']} | {fmt(r['threads'], 0)} | "
                       f"{fmt(r['pool'], 0)} | {fmt(r['allowance_gb'])} | {fmt(r['measured_gb'], 2)} | "
                       f"{fmt(r['central_gb'], 2)} | {fmt(r['high_gb'], 2)} | {fmt(score, 3)} | {r['verdict']} | "
                       f"{fmt(r['measured_ws_gb'], 2)} | {fmt(r['ws_high_gb'], 2)} | {r.get('ws_verdict', '–')} |")
    whatif = [(o, r) for o in runs for r in o["governor"]["whatif"]]
    if whatif:
        out += ["", "### A6: what-if (laptop) recommendations", "",
                "| run | launch | parts | route | t | mask | commit GB | WS GB | RAM allow | commit allow | over | "
                "action | neutral | verdict |",
                "| --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |"]
        for o, r in whatif:
            out.append(f"| {o['run_id']} | {r['name']} | {r['parts']} | {r['route']} | {fmt(r['threads'], 0)} | "
                       f"{r['mask'] or '–'} | {fmt(r['commit_gb'], 2)} | {fmt(r['ws_gb'], 2)} | "
                       f"{fmt(r['ram_allowance_gb'], 2)} | {fmt(r['commit_allowance_gb'], 2)} | {r['over_budget']} | "
                       f"{r['planned_action']} | {r['record_neutral']} | {r['verdict']} |")
    pools = [(o, k, v) for o in runs for k, v in o["governor"]["pools"].items()]
    if pools:
        out += ["", "### A4 (d): Python pools, actual vs recommended (predictions are [est] rows; not gated)", "",
                "| run | scope:stage | calls | actual | recommended | bound by | predicted commit GB | action |",
                "| --- | --- | ---: | --- | --- | --- | ---: | --- |"]
        for o, k, v in pools:
            out.append(f"| {o['run_id']} | {k} | {fmt(v.get('calls'))} | {v.get('actual')} | {v.get('recommended')} | "
                       f"{fmt(v.get('bound_by'))} | {fmt(v.get('predicted_commit_gb'), 2)} | "
                       f"{fmt(v.get('planned_action'))} |")
    return out


def _overview_times(run, s, wall, sampler, host, hashed) -> dict:
    return {
        "run_id": s["run_id"], "slot": slot_of(run), "label": s["plan"].get("label"),
        "state": s.get("state"), "fingerprint": (s.get("fingerprint") or {}).get("status"),
        "simulation": (s.get("simulation") or {}).get("label"),
        "pins": s["plan"].get("pins"),
        "submit_to_base_s": wall.get("submit_to_base_s"), "submit_to_final_s": wall.get("submit_to_final_s"),
        "addback_s": wall.get("addback_s"),
        "tree_cpu_s": sampler.get("tree_cpu_s"),
        "peak_tree_private_gb": _gb(sampler.get("peak_tree_private_bytes")),
        "peak_tree_ws_gb": _gb(sampler.get("peak_tree_ws_bytes")),
        "min_sys_avail_gb": _gb(sampler.get("min_sys_avail_bytes")),
        "peak_job_commit_gb": _gb((s.get("job_accounting") or {}).get("peak_job_commit_bytes")),
        "hashed_gb": _gb(hashed),
        "sha_rate_gbps": host.get("sha256_rate_gbps"), "sha_ext_guess": host.get("sha_ext_guess"),
        "logical_processors": host.get("logical_processors"),
        "per_exe_cpu_s": {k: v.get("cpu_s") for k, v in (sampler.get("per_exe") or {}).items()},
        "kept_meteors": (s.get("trail_review") or {}).get("kept_count"),
    }


def _gb(value):
    return None if value is None else round(value / GB, 2)


def status_stages(run) -> dict:
    """Whole-tree view per workflow status stage, from the 1 Hz samples."""
    out, current, first, prev = {}, None, None, None
    rows = [r for r in run["timeline"] if r.get("stage")]

    def close(stage, a, b):
        if a is None or b is None:
            return
        name = stage.split("/", 1)[-1]
        entry = out.setdefault(name, {"wall_s": 0.0, "cpu_s": 0.0, "io_read_gb": 0.0, "io_write_gb": 0.0,
                                      "peak_private_gb": 0.0, "peak_ws_gb": 0.0, "min_sys_avail_gb": None})
        entry["wall_s"] += b["t"] - a["t"]
        entry["cpu_s"] += (b.get("tree_cpu_s") or 0) - (a.get("tree_cpu_s") or 0)
        entry["io_read_gb"] += ((b.get("io_read") or 0) - (a.get("io_read") or 0)) / GB
        entry["io_write_gb"] += ((b.get("io_write") or 0) - (a.get("io_write") or 0)) / GB

    for row in rows:
        if row["stage"] != current:
            close(current, first, row) if current else None
            current, first = row["stage"], row
        name = row["stage"].split("/", 1)[-1]
        entry = out.setdefault(name, {"wall_s": 0.0, "cpu_s": 0.0, "io_read_gb": 0.0, "io_write_gb": 0.0,
                                      "peak_private_gb": 0.0, "peak_ws_gb": 0.0, "min_sys_avail_gb": None})
        entry["peak_private_gb"] = max(entry["peak_private_gb"], (row.get("tree_private") or 0) / GB)
        entry["peak_ws_gb"] = max(entry["peak_ws_gb"], (row.get("tree_ws") or 0) / GB)
        avail = (row.get("sys_avail") or 0) / GB
        entry["min_sys_avail_gb"] = avail if entry["min_sys_avail_gb"] is None else min(entry["min_sys_avail_gb"], avail)
        prev = row
    if current and prev is not None and prev is not first:
        close(current, first, prev)
    for entry in out.values():
        entry["cores"] = round(entry["cpu_s"] / entry["wall_s"], 2) if entry["wall_s"] > 0 else None
        for key, value in list(entry.items()):
            if isinstance(value, float):
                entry[key] = round(value, 2)
    return dict(sorted(out.items(), key=lambda kv: STATUS_ORDER.index(kv[0]) if kv[0] in STATUS_ORDER else 99))


def phases(run) -> list[dict]:
    """Instrumented phases in record order; the stack job's phases follow stack_wait, indented."""
    rows = []
    timings = (run["workflow"] or {}).get("timings")
    job_timings = ((run["job"] or {}).get("diagnostics") or {}).get("timings")
    if not timings:
        return rows
    for name, row in _phase_rows(timings, "workflow"):
        rows.append(row)
        if name == "stack_wait" and job_timings:
            rows.extend(r for _, r in _phase_rows(job_timings, "job", indent=1))
    return rows


def _phase_rows(timings, scope, indent=0):
    res = (timings.get("resources") or {}).get("phases") or {}
    for name, wall in (timings.get("phases_seconds") or {}).items():
        view = res.get(name) or {}
        child = view.get("children") or {}
        depth = indent + name.count(".")
        yield name, {"scope": scope, "name": name, "depth": depth, "wall_s": round(wall, 2),
                     "cpu_s": _r(view.get("cpu_seconds")), "child_cpu_s": _r(child.get("cpu_seconds")),
                     "child_processes": child.get("processes"),
                     "io_read_gb": _gb(view.get("io_read_bytes")), "io_write_gb": _gb(view.get("io_write_bytes")),
                     "child_io_read_gb": _gb(child.get("io_read_bytes")),
                     "child_io_write_gb": _gb(child.get("io_write_bytes")),
                     "hashed_gb": _gb(view.get("hashed_bytes")), "saved_gb": _gb(view.get("hash_saved_bytes")),
                     "memory_gb": _gb(view.get("hashed_memory_bytes")),
                     "peak_private_rise_gb": _gb(view.get("peak_private_rise_bytes")),
                     "child_peak_gb": _gb(child.get("peak_job_memory_bytes"))}


def operations(run) -> list[dict]:
    job_timings = ((run["job"] or {}).get("diagnostics") or {}).get("timings") or {}
    res = (job_timings.get("resources") or {}).get("operations") or {}
    counts = job_timings.get("operation_counts") or {}
    out = []
    for name, wall in (job_timings.get("operations_seconds") or {}).items():
        view = res.get(name) or {}
        child = view.get("children") or {}
        out.append({"name": name, "count": counts.get(name), "wall_s": round(wall, 2),
                    "cpu_s": _r(view.get("cpu_seconds")), "child_cpu_s": _r(child.get("cpu_seconds")),
                    "hashed_gb": _gb(view.get("hashed_bytes")), "saved_gb": _gb(view.get("hash_saved_bytes"))})
    return out


def _r(value):
    return None if value is None else round(value, 2)


# --------------------------------------------------------------------------- aggregation

def combine(values):
    values = [v for v in values if isinstance(v, (int, float))]
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return {"mean": round(statistics.fmean(values), 2), "min": min(values), "max": max(values)}


def fmt(value, digits=1):
    if value is None:
        return "–"
    if isinstance(value, dict):
        return f"{value['mean']:.{digits}f} ({value['min']:.{digits}f}–{value['max']:.{digits}f})"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def group_report(runs) -> dict:
    slot = slot_of(runs[0])
    stage_names, phase_keys, op_names = [], [], []
    per_run_stages = [status_stages(r) for r in runs]
    per_run_phases = [{(p["scope"], p["name"]): p for p in phases(r)} for r in runs]
    per_run_ops = [{o["name"]: o for o in operations(r)} for r in runs]
    for stages in per_run_stages:
        stage_names += [n for n in stages if n not in stage_names]
    for run_phases in per_run_phases:
        phase_keys += [k for k in run_phases if k not in phase_keys]
    for ops in per_run_ops:
        op_names += [n for n in ops if n not in op_names]
    stage_table = [{"stage": n, **{k: combine([s.get(n, {}).get(k) for s in per_run_stages])
                                   for k in ("wall_s", "cpu_s", "cores", "peak_private_gb", "peak_ws_gb",
                                             "io_read_gb", "io_write_gb", "min_sys_avail_gb")}}
                   for n in stage_names]
    phase_table = []
    for key in phase_keys:
        sample = next(p[key] for p in per_run_phases if key in p)
        row = {"scope": key[0], "name": key[1], "depth": sample["depth"]}
        for k in ("wall_s", "cpu_s", "child_cpu_s", "io_read_gb", "io_write_gb", "child_io_read_gb",
                  "child_io_write_gb", "hashed_gb", "saved_gb", "memory_gb", "peak_private_rise_gb", "child_peak_gb"):
            row[k] = combine([p.get(key, {}).get(k) for p in per_run_phases])
        phase_table.append(row)
    op_table = [{"name": n, **{k: combine([o.get(n, {}).get(k) for o in per_run_ops])
                              for k in ("count", "wall_s", "cpu_s", "child_cpu_s", "hashed_gb", "saved_gb")}}
                for n in op_names]
    return {"slot": slot, "variant": variant(runs[0]), "runs": [overview(r) for r in runs], "stages": stage_table,
            "phases": phase_table, "operations": op_table}


def variant(run) -> str:
    """How a group differs from a product-default run: simulation, pins, other code."""
    plan, parts = run["summary"]["plan"], []
    simulation = (run["summary"].get("simulation") or {}).get("label")
    pins = plan.get("pins") or {}
    if simulation:
        parts.append(simulation)
    else:
        if pins.get("siril"):
            siril = pins["siril"]
            parts.append("Siril pinned " + (",".join(f"{k}={v}" for k, v in siril.items())
                                            if isinstance(siril, dict) else str(siril)))
        if pins.get("ort_intra_op_threads"):
            parts.append(f"nox {pins['ort_intra_op_threads']} threads")
    code = Path(plan.get("code_src") or "")
    if code and code.resolve() != (REPO / "src").resolve():
        parts.append(f"code {code.parent.name}")
    return "; ".join(parts) or "product defaults"


def markdown(groups) -> str:
    out = ["# Benchmark stage tables", ""]
    out += ["## Overview", "",
            "| run | slot | sim | state | fp | →base s | →final s | add-back s | tree CPU s | peak private GB | "
            "min sys avail GB | hashed GB | SHA GB/s |",
            "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for g in groups:
        for o in g["runs"]:
            out.append(f"| {o['run_id']} | {o['slot']} | {o['simulation'] or '–'} | {o['state']} | {o['fingerprint']} | "
                       f"{fmt(o['submit_to_base_s'])} | {fmt(o['submit_to_final_s'])} | {fmt(o['addback_s'])} | "
                       f"{fmt(o['tree_cpu_s'], 0)} | {fmt(o['peak_tree_private_gb'])} | {fmt(o['min_sys_avail_gb'])} | "
                       f"{fmt(o['hashed_gb'])} | {fmt(o['sha_rate_gbps'], 2)} |")
    out += ["", "## Hashing (efficiency P1, H4)", "",
            "Real reads (GB) of the workflow and job workers and of the submitting process; saved reads by kind "
            "(ledger/handover/write: answered from the run ledger; batch: a repeat inside one batch); in-memory "
            "hashing; ledger mode and shadow checks from the workers' timing notes. A run without submitter "
            "counts (made before P1) shows – under submit real, and its total real excludes the submitter.", "",
            "| run | workers real | submit real | total real | saved | ledger | handover | write | batch | "
            "memory | mode | shadow checks | mismatches | concurrent |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |"]
    for g in groups:
        for o in g["runs"]:
            h = o["hashing"]
            modes = h["ledger_modes"]
            mode = "–" if not modes else ", ".join(f"{k} {v}" for k, v in modes.items())
            out.append(f"| {o['run_id']} | {fmt(h['workers_real_gb'])} | {fmt(h['submitter_real_gb'])} | "
                       f"{fmt(h['total_real_gb'])} | {fmt(h['saved_gb'])} | {fmt(h['saved_ledger_gb'])} | "
                       f"{fmt(h['saved_handover_gb'])} | {fmt(h['saved_write_gb'])} | {fmt(h['saved_batch_gb'])} | "
                       f"{fmt(h['memory_gb'])} | {mode} | {fmt(h['shadow_checks'], 0)} | "
                       f"{fmt(h['shadow_mismatches'], 0)} | {fmt(h['shadow_concurrent'], 0)} |")
    out += ["", "### Ledger misses (efficiency P1b-5c)", "",
            "Bytes the workers read because the run ledger missed, by miss status (no-entry: never hashed at "
            "that path in this run; racy: recorded too close to the file's modification time), and the guard "
            "refusals of seals and records (expected 0). Runs made before P1b show –.", "",
            "| run | racy miss GB | no-entry miss GB | stale miss GB | busy miss GB | refused touched | refused seal |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for g in groups:
        for o in g["runs"]:
            h = o["hashing"]
            out.append(f"| {o['run_id']} | {fmt(h['miss_racy_gb'], 2)} | {fmt(h['miss_no_entry_gb'], 2)} | "
                       f"{fmt(h['miss_stale_gb'], 2)} | {fmt(h['miss_busy_gb'], 2)} | "
                       f"{fmt(h['refused_touched'], 0)} | {fmt(h['refused_seal'], 0)} |")
    largest = [(o["run_id"], scope, row) for g in groups for o in g["runs"]
               for scope, rows in (o["hashing"]["largest_misses"] or {}).items() for row in rows]
    if largest:
        out += ["", f"Largest ledger misses (top {LARGEST_SHOWN} per worker):", "",
                "| run | worker | GB | status | source | file |", "| --- | --- | ---: | --- | --- | --- |"]
        for run_id, scope, row in largest:
            out.append(f"| {run_id} | {scope} | {fmt(row['gb'], 3)} | {row['status']} | {row['source']} | "
                       f"{row['file']} |")
    for g in groups:
        n = len(g["runs"])
        title = f"{g['slot']} · {g['variant']}"
        out += ["", f"## {title} ({n} run{'s' if n > 1 else ''}{'; mean (min–max)' if n > 1 else ''})", "",
                "### Status stages (whole process tree, 1 Hz)", "",
                "| stage | wall s | CPU s | cores | peak private GB | peak WS GB | read GB | write GB |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for r in g["stages"]:
            out.append(f"| {r['stage']} | {fmt(r['wall_s'])} | {fmt(r['cpu_s'], 0)} | {fmt(r['cores'])} | "
                       f"{fmt(r['peak_private_gb'])} | {fmt(r['peak_ws_gb'])} | {fmt(r['io_read_gb'])} | {fmt(r['io_write_gb'])} |")
        out += ["", "### Instrumented phases (disjoint wall time per process; job phases under stack_wait)", "",
                "| phase | wall s | own CPU s | child CPU s | read GB | write GB | child read GB | child write GB | "
                "hashed GB | saved GB | memory GB | peak rise GB | child peak GB |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for r in g["phases"]:
            name = ("&nbsp;&nbsp;" * r["depth"]) + (f"job:{r['name']}" if r["scope"] == "job" else r["name"])
            out.append(f"| {name} | {fmt(r['wall_s'])} | {fmt(r['cpu_s'])} | {fmt(r['child_cpu_s'])} | "
                       f"{fmt(r['io_read_gb'])} | {fmt(r['io_write_gb'])} | {fmt(r['child_io_read_gb'])} | "
                       f"{fmt(r['child_io_write_gb'])} | {fmt(r['hashed_gb'])} | {fmt(r['saved_gb'])} | "
                       f"{fmt(r['memory_gb'])} | {fmt(r['peak_private_rise_gb'])} | {fmt(r['child_peak_gb'])} |")
        if g["operations"]:
            out += ["", "### Stack job operations (overlap phases; do not add)", "",
                    "| operation | count | wall s | own CPU s | child CPU s | hashed GB | saved GB |",
                    "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
            for r in g["operations"]:
                out.append(f"| {r['name']} | {fmt(r['count'], 0)} | {fmt(r['wall_s'])} | {fmt(r['cpu_s'])} | "
                           f"{fmt(r['child_cpu_s'])} | {fmt(r['hashed_gb'])} | {fmt(r['saved_gb'])} |")
    out += governor_markdown(groups)
    out += overlap_markdown(groups)
    return "\n".join(out) + "\n"


def resolve(args) -> list[Path]:
    folders = [Path(p) for p in args.folders]
    root = Path(args.bench_root) / "results"
    for label in args.label or []:
        matches = sorted(root.glob(f"*-{label}"))
        if not matches:
            raise SystemExit(f"no result folder ends with -{label} under {root}")
        folders.append(matches[-1])
    folders = [f for f in folders if (f / "summary.json").is_file()]
    if not folders:
        raise SystemExit("no benchmark result folders given")
    return folders


def build(folders, group_by="slot+sim") -> list[dict]:
    groups: dict[tuple, list] = {}
    for folder in folders:
        run = load_run(folder)
        if run["summary"].get("state") != "succeeded":
            print(f"skip {folder.name}: state {run['summary'].get('state')}", file=sys.stderr)
            continue
        key = (slot_of(run), (run["summary"].get("simulation") or {}).get("label"),
               json.dumps(run["summary"]["plan"].get("pins"), sort_keys=True),
               run["summary"]["plan"].get("code_src")) if group_by == "slot+sim" else (folder.name,)
        groups.setdefault(key, []).append(run)
    return [group_report(runs) for runs in groups.values()]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folders", nargs="*", help="results/<run-id> folders")
    parser.add_argument("--label", action="append", help="suffix of a result folder, e.g. L48-R4-L1 (repeatable)")
    parser.add_argument("--bench-root", default=str(DEFAULT_ROOT))
    parser.add_argument("--separate", action="store_true", help="one table per run instead of grouping repeats")
    parser.add_argument("--json", help="also write the tables as JSON here")
    parser.add_argument("-o", "--out", help="write the Markdown here instead of stdout")
    args = parser.parse_args(argv)
    groups = build(resolve(args), "run" if args.separate else "slot+sim")
    text = markdown(groups)
    if args.out:
        Path(args.out).write_bytes(text.encode("utf-8"))
    else:
        sys.stdout.write(text)
    if args.json:
        Path(args.json).write_bytes(json.dumps(groups, ensure_ascii=False, indent=1, default=str).encode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
