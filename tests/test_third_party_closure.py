"""Hardening package H1: the third-party table reconciles with the lock file, and the checker notices drift."""
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import check_third_party as ctp  # noqa: E402


def test_lock_file_and_third_party_table_reconcile():
    result = ctp.reconcile(ROOT)
    assert result["ok"], result
    assert result["locked"] == result["rows"] == 49
    assert result["missing_from_table"] == [] and result["not_in_lock"] == [] and result["duplicated_rows"] == []
    assert result["version_mismatch"] == [] and result["metadata_mismatch"] == []
    assert (ROOT / "NOTICE.md").is_file() and "THIRD_PARTY.md" in (ROOT / "NOTICE.md").read_text(encoding="utf-8")
    licence = (ROOT / "LICENSE").read_text(encoding="utf-8")            # L1: Apache-2.0, the canonical text
    assert "Apache License" in licence and "Version 2.0, January 2004" in licence and len(licence.encode()) == 11358
    assert "Apache License, Version 2.0" in (ROOT / "NOTICE.md").read_text(encoding="utf-8")


def test_checker_reports_missing_extra_and_version_drift(tmp_path):
    root = tmp_path / "copy"
    root.mkdir()
    shutil.copyfile(ROOT / "requirements-lock.txt", root / "requirements-lock.txt")
    text = (ROOT / "THIRD_PARTY.md").read_text(encoding="utf-8")
    text = text.replace("| numpy | 2.5.3 |", "| numpy | 2.5.2 |").replace("| tifffile | 2026.9.9 |", "| some-other-lib | 1.0 |")
    (root / "THIRD_PARTY.md").write_text(text, encoding="utf-8")
    result = ctp.reconcile(root)
    assert not result["ok"]
    assert result["missing_from_table"] == ["tifffile"] and result["not_in_lock"] == ["some-other-lib"]
    assert result["version_mismatch"] == ["numpy: 表 2.5.2 ≠ 锁 2.5.3"]
    assert ctp.main(["--root", str(root), "--json"]) == 1 and ctp.main(["--root", str(ROOT)]) == 0


def test_licence_family_matching_tolerates_classifier_wording():
    assert ctp.families("BSD-3-Clause") == {"BSD"} == ctp.families("License :: OSI Approved :: BSD License")
    assert ctp.families("LGPL-3.0-only OR GPL-2.0-only OR GPL-3.0-only") == {"LGPL", "GPL"}
    assert ctp.families("MIT-CMU") == {"MIT"} and ctp.families("Python Software Foundation License") == {"PSF"}
    assert ctp.families("Apache-2.0 OR BSD-3-Clause") == {"APACHE", "BSD"} and ctp.families("") == set()
