"""The wheel-closure builder still matches the installer manifest (ported from the retired trial-packaging test)."""
from pathlib import Path
import importlib.util
import json

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("download_manifest_builder", ROOT / "scripts/build_download_manifest.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_builder_roots_match_the_installer_manifest_and_out_is_required():
    pytest.importorskip("packaging.markers")               # the builder imports the PyPA 'packaging' library at module level
    builder = _load()
    manifest = json.loads((ROOT / "packaging/aester_p2/download-manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["roots"]) == set(builder.ROOTS)
    assert "rawpy" in builder.ROOTS and "pyside6-essentials" in builder.ROOTS and "pytest" not in builder.ROOTS
    assert builder.default_release_id("0.2.0rc1", "20260923") == "StarStack-Trial-0.2.0rc1-20260923-windows-x64"
    with pytest.raises(SystemExit):
        builder.main(["--release-id", "x"])                 # --out is required; nothing is fetched
