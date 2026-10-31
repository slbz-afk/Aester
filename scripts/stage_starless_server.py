"""Copy the starless evaluation set (and the nox model) to a network share, verifying every byte.

Reads docs/testdata/STARLESS_EVAL_CASES_CURRENT_20260925.json, copies per case the source, sky mask,
protected mask into ``<dest>/eval/<case>/``, the raw nox output (``model_output``) and the
product-locked nox diffuse (``estimate``) into ``<dest>/eval/<case>/nox/`` (``compact/`` for the
optional compact baseline), plus ``noxGeneratorColor.pb`` into ``<dest>/models/nox/``. Each file is hashed while it is read
(checked against the case list) and read back from the destination after the copy (checked
again), so a silent SMB corruption cannot pass. Already-present files with the right hash are
skipped, so the script can be re-run. It writes ``<dest>/eval/MANIFEST.json`` and a
server-side case list ``<dest>/eval/STARLESS_EVAL_CASES_server.json`` (root
``/tank/data/starless/eval``) for ``scripts/starless_eval.py cases`` on the server.

Usage (Windows, the share mapped as a drive, for example Z:):
  python scripts\\stage_starless_server.py --dest Z:\\data\\starless [--dry-run] [--no-estimate]
        [--with-compact] [--with-input] [--case z8 ...] [--no-verify-dest]
Nothing is deleted or modified on the workstation side.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
import time
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ROOT / "docs" / "testdata" / "STARLESS_EVAL_CASES_CURRENT_20260925.json"
SERVER_ROOT = "/tank/data/starless/eval"
NOX_MODEL = {"path": r"outputs\milkyway-x2-20260921\backend-nox\noxGeneratorColor.pb",
             "sha256": "d54bdca728d1d6db0b3eef41d4187d327909d1ec5cd2a71485bfa9d7924ba546"}
CHUNK = 8 << 20


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_hashed(src: Path, dst: Path) -> tuple[str, int]:
    """Copy src → dst through a .part file; returns (sha256 of the bytes read, bytes)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_name(dst.name + ".part")
    h = hashlib.sha256()
    n = 0
    with src.open("rb") as f, part.open("wb") as g:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
            g.write(chunk)
            n += len(chunk)
        g.flush()
        os.fsync(g.fileno())
    part.replace(dst)
    return h.hexdigest(), n


def plan_items(doc: dict, cases: set[str], roles: list[str]) -> list[dict]:
    items = []
    for name, case in doc["cases"].items():
        if name not in cases:
            continue
        status = (case.get("lifecycle") or {}).get("status", "active")
        if status != "active":
            raise ValueError(f"Case {name} is {status}; complete its new baseline before staging, or select other cases explicitly")
        for role in ("source", "sky_mask", "protected_mask"):
            items.append({"case": name, "role": role, **case[role]})
        for backend, spec in case["backends"].items():
            if backend == "nox":
                if "model_output" in roles:
                    items.append({"case": name, "role": "nox_model_output", **spec["model_output"]})
                if "estimate" in roles:
                    items.append({"case": name, "role": "nox_estimate", **spec["estimate"]})
                if "input" in roles and "model_input" in spec:
                    items.append({"case": name, "role": "nox_model_input", **spec["model_input"]})
            elif backend == "compact" and "compact" in roles:
                items.append({"case": name, "role": "compact_estimate", **spec["estimate"]})
    return items


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", default=str(DEFAULT_CASES))
    ap.add_argument("--dest", required=True, help=r"共享上的目标根，例如 Z:\data\starless")
    ap.add_argument("--case", action="append", help="只复制这些例（可重复）")
    ap.add_argument("--no-estimate", action="store_true", help="不复制产品锁定后的 nox 弥散 diffuse_srgb16.tif（服务器上就不能跑 cases 评测）")
    ap.add_argument("--with-compact", action="store_true", help="另复制轻量对照的弥散")
    ap.add_argument("--with-input", action="store_true", help="另复制 nox 实际读入的 nox_input_srgb16.tif")
    ap.add_argument("--no-model", action="store_true", help="不复制 noxGeneratorColor.pb")
    ap.add_argument("--no-verify-dest", action="store_true", help="复制后不重读目标校验（不建议）")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    doc = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    root = Path(doc["root"])
    dest = Path(args.dest)
    wanted = set(args.case or doc["cases"].keys())
    roles = ["model_output"] + ([] if args.no_estimate else ["estimate"]) + (["compact"] if args.with_compact else []) \
        + (["input"] if args.with_input else [])
    items = plan_items(doc, wanted, roles)
    if not args.no_model:
        items.append({"case": None, "role": "model", **NOX_MODEL})

    manifest = {"version": "stage-starless-server-1", "created": dt.datetime.now(dt.timezone.utc).isoformat(),
                "cases_file": str(Path(args.cases).resolve()), "cases_sha256": sha256_of(Path(args.cases)),
                "workstation_root": str(root), "dest": str(dest), "server_root": SERVER_ROOT, "files": []}
    total = copied = skipped = failed = 0
    started = time.perf_counter()
    for item in items:
        src = root / Path(item["path"].replace("\\", "/"))
        if item["role"] == "model":
            rel = PurePosixPath("models/nox") / src.name
        elif item["role"].startswith("nox_"):
            rel = PurePosixPath("eval") / item["case"] / "nox" / src.name       # nox and compact share basenames
        elif item["role"].startswith("compact_"):
            rel = PurePosixPath("eval") / item["case"] / "compact" / src.name
        else:
            rel = PurePosixPath("eval") / item["case"] / src.name
        dst = dest / Path(*rel.parts)
        entry = {"case": item["case"], "role": item["role"], "source": str(src), "dest": str(dst),
                 "server_path": f"/tank/data/starless/{rel}", "expected_sha256": item["sha256"]}
        if not src.is_file():
            entry["status"] = "missing_source"
            failed += 1
            print(f"[缺失] {src}")
            manifest["files"].append(entry)
            continue
        size = src.stat().st_size
        total += size
        if dst.is_file() and dst.stat().st_size == size and (args.no_verify_dest or sha256_of(dst) == item["sha256"]):
            entry.update(status="already_present", bytes=size, sha256=item["sha256"])
            skipped += 1
            print(f"[已有] {rel}  {size / 1e6:.0f} MB")
            manifest["files"].append(entry)
            continue
        if args.dry_run:
            entry.update(status="planned", bytes=size)
            print(f"[计划] {rel}  {size / 1e6:.0f} MB")
            manifest["files"].append(entry)
            continue
        t0 = time.perf_counter()
        read_hash, n = copy_hashed(src, dst)
        if read_hash != item["sha256"]:
            entry.update(status="source_hash_mismatch", bytes=n, sha256=read_hash)
            failed += 1
            print(f"[源哈希不符] {rel}  期望 {item['sha256'][:12]} 实际 {read_hash[:12]}")
            manifest["files"].append(entry)
            continue
        if not args.no_verify_dest:
            back = sha256_of(dst)
            if back != item["sha256"]:
                entry.update(status="dest_hash_mismatch", bytes=n, sha256=back)
                failed += 1
                print(f"[目标回读不符] {rel}")
                manifest["files"].append(entry)
                continue
        seconds = time.perf_counter() - t0
        entry.update(status="copied", bytes=n, sha256=item["sha256"], seconds=seconds)
        copied += 1
        print(f"[复制] {rel}  {n / 1e6:.0f} MB  {n / 1e6 / seconds:.0f} MB/s")
        manifest["files"].append(entry)

    manifest.update(summary={"files": len(items), "copied": copied, "already_present": skipped, "failed": failed,
                             "bytes_total": total, "seconds": time.perf_counter() - started})
    if not args.dry_run:
        (dest / "eval").mkdir(parents=True, exist_ok=True)
        (dest / "eval" / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
        server_doc = server_case_list(doc, manifest)
        (dest / "eval" / "STARLESS_EVAL_CASES_server.json").write_text(json.dumps(server_doc, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(manifest["summary"], ensure_ascii=False))
    print("在服务器上评测：python scripts/starless_eval.py cases --cases /tank/data/starless/eval/STARLESS_EVAL_CASES_server.json"
          " --root /tank/data/starless/eval --output /tank/data/starless/runs/baseline-<日期>")
    return 1 if failed else 0


def server_case_list(doc: dict, manifest: dict) -> dict:
    """The case list re-rooted to the server; keeps only files that are present there."""
    present = {(f["case"], f["role"]): f for f in manifest["files"] if f.get("status") in ("copied", "already_present")}

    def ref(case, role, expected):
        f = present.get((case, role))
        if f is None:
            return None
        return {"path": f["server_path"][len(SERVER_ROOT) + 1:], "sha256": expected["sha256"]}
    out = {"version": doc["version"] + "+server", "created": manifest["created"], "package": doc.get("package"),
           "root": SERVER_ROOT, "note": "由 stage_starless_server.py 生成：路径相对 root，只含已复制到服务器的文件；哈希与工作站清单相同。",
           "region_rule": doc["region_rule"], "tracks": doc["tracks"], "cases": {}}
    for name, case in doc["cases"].items():
        refs = {role: ref(name, role, case[role]) for role in ("source", "sky_mask", "protected_mask")}
        if any(v is None for v in refs.values()):
            continue
        entry = {"track": case["track"], "label": case["label"], "shape": case["shape"], **refs, "backends": {}}
        nox = case["backends"].get("nox")
        if nox:
            b = {k: v for k, v in nox.items() if k not in ("estimate", "model_output", "model_input", "residual", "record")}
            for role, key in (("nox_estimate", "estimate"), ("nox_model_output", "model_output"), ("nox_model_input", "model_input")):
                r = ref(name, role, nox[key]) if key in nox else None
                if r is not None:
                    b[key] = r
            if "estimate" in b:
                entry["backends"]["nox"] = b          # usable by `starless_eval.py cases`
            else:
                entry.setdefault("references", {})["nox"] = b   # raw outputs only (compare / R8)
        compact = case["backends"].get("compact")
        if compact:
            r = ref(name, "compact_estimate", compact["estimate"])
            if r is not None:
                entry["backends"]["compact"] = {**{k: v for k, v in compact.items() if k not in ("estimate", "residual", "record")}, "estimate": r}
        out["cases"][name] = entry
    return out


if __name__ == "__main__":
    sys.exit(main())
