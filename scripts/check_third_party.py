"""Reconcile ``requirements-lock.txt`` with the PyPI table of ``THIRD_PARTY.md`` (hardening package H1).

Exit 0 only when every locked package has exactly one table row with the locked version and no row names an unlocked
package.  In an installed environment the declared licence is also compared with each wheel's metadata
(``License-Expression`` / ``License`` / licence classifiers) by licence family; a mismatch is reported as a warning
(``metadata_mismatch``) because the table, not the metadata, is what ships — but it must be looked at.  ``--files``
lists the licence files each installed wheel carries (``License-File``), the evidence for "wheel 内许可证文件".

Usage: python scripts/check_third_party.py [--root DIR] [--files] [--json]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

VERSION = "check-third-party-1"
PYPI_SECTION = "## PyPI 组件"
NEXT_SECTION = "## 非 PyPI 组件"
FAMILY_ALIASES = {"MIT": "MIT", "BSD": "BSD", "APACHE": "APACHE", "MPL": "MPL", "PSF": "PSF", "LGPL": "LGPL", "GPL": "GPL",
                  "PYTHON": "PSF", "MOZILLA": "MPL", "HPND": "MIT"}


def normalise(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name.strip().lower())


def read_lock(path: Path) -> dict[str, str]:
    packages = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        if "==" not in line:
            raise ValueError(f"锁文件行不是固定版本：{line!r}")
        name, version = line.split("==", 1)
        packages[normalise(name)] = version.strip()
    return packages


def read_table(path: Path) -> list[dict]:
    """Rows of the PyPI table: name, version, licence (the first three columns)."""
    text = path.read_text(encoding="utf-8")
    start = text.find(PYPI_SECTION)
    end = text.find(NEXT_SECTION)
    if start < 0 or end < 0 or end < start:
        raise ValueError("THIRD_PARTY.md 缺少 PyPI 组件表或非 PyPI 组件表")
    rows = []
    for line in text[start:end].splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 3 or cells[0] in ("组件", "") or set(cells[0]) <= {"-"}:
            continue
        rows.append({"name": normalise(cells[0]), "raw_name": cells[0], "version": cells[1], "license": cells[2],
                     "distribution": cells[4] if len(cells) > 4 else ""})
    return rows


def families(text: str) -> set[str]:
    found = set()
    for token in re.split(r"[^A-Za-z]+", text.upper()):
        for prefix, family in FAMILY_ALIASES.items():
            if token.startswith(prefix):
                found.add(family)
    return found


def installed_metadata(name: str) -> dict | None:
    import importlib.metadata as md
    try:
        dist = md.distribution(name)
    except md.PackageNotFoundError:
        return None
    meta = dist.metadata
    classifiers = [c for c in (meta.get_all("Classifier") or []) if c.startswith("License")]
    licence_field = (meta.get("License") or "").strip()
    return {"version": dist.version, "license_expression": meta.get("License-Expression") or "",
            "license_field": licence_field.splitlines()[0][:120] if licence_field else "",
            "classifiers": classifiers, "license_files": list(meta.get_all("License-File") or [])}


def reconcile(root: Path, *, with_files: bool = False) -> dict:
    lock = read_lock(root / "requirements-lock.txt")
    rows = read_table(root / "THIRD_PARTY.md")
    seen: dict[str, list[dict]] = {}
    for row in rows:
        seen.setdefault(row["name"], []).append(row)
    missing = sorted(name for name in lock if name not in seen)
    extra = sorted(name for name in seen if name not in lock)
    duplicated = sorted(name for name, items in seen.items() if len(items) > 1)
    version_mismatch = sorted(f"{name}: 表 {seen[name][0]['version']} ≠ 锁 {lock[name]}" for name in lock
                              if name in seen and seen[name][0]["version"] != lock[name])
    metadata = {}
    metadata_mismatch = []
    for name in sorted(lock):
        if name not in seen:
            continue
        info = installed_metadata(name)
        if info is None:
            continue
        declared = seen[name][0]["license"]
        observed = " ".join([info["license_expression"], info["license_field"], *info["classifiers"]])
        declared_families, observed_families = families(declared), families(observed)
        entry = {"installed_version": info["version"], "declared": declared, "license_expression": info["license_expression"],
                 "license_field": info["license_field"], "classifiers": info["classifiers"],
                 "family_match": bool(declared_families & observed_families) if observed_families else None}
        if with_files:
            entry["license_files"] = info["license_files"]
        metadata[name] = entry
        if observed_families and not (declared_families & observed_families):
            metadata_mismatch.append(f"{name}: 表 {declared!r} / 元数据 {observed.strip()!r}")
    ok = not (missing or extra or duplicated or version_mismatch)
    return {"version": VERSION, "ok": ok, "locked": len(lock), "rows": len(rows), "missing_from_table": missing,
            "not_in_lock": extra, "duplicated_rows": duplicated, "version_mismatch": version_mismatch,
            "installed_checked": len(metadata), "metadata_mismatch": metadata_mismatch, "metadata": metadata}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--files", action="store_true", help="另列已安装 wheel 的许可证文件")
    parser.add_argument("--json", action="store_true", help="只打印 JSON")
    args = parser.parse_args(argv)
    result = reconcile(Path(args.root), with_files=args.files)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"锁文件 {result['locked']} 个包，表 {result['rows']} 行；缺项 {len(result['missing_from_table'])}，多项 {len(result['not_in_lock'])}，"
              f"重复 {len(result['duplicated_rows'])}，版本不符 {len(result['version_mismatch'])}；"
              f"已安装并对照元数据 {result['installed_checked']} 个，家族不符 {len(result['metadata_mismatch'])}")
        for key in ("missing_from_table", "not_in_lock", "duplicated_rows", "version_mismatch", "metadata_mismatch"):
            for item in result[key]:
                print(f"  {key}: {item}")
        if args.files:
            for name, entry in result["metadata"].items():
                print(f"  {name} {entry['installed_version']}: {', '.join(entry['license_files']) or '（无 License-File 元数据）'}")
        print("OK" if result["ok"] else "FAIL")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
