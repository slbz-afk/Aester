"""Independent regressions for v3 final-verification publication boundaries.

Tiny genuine assets and bundle checks; process/product boundaries are explicit
test doubles. These tests neither invoke GIMP nor certify renderer pixels.
"""
import json
from pathlib import Path
import sys

import numpy as np
from PIL import ImageCms
import pytest
import tifffile

from starstack import finishing_backend as backend, finishing_candidates as candidates
from starstack import finishing_layers as layers, finishing_session as sessions
from starstack.finishing_assets import recipe_source, sha256_file


@pytest.fixture
def layer_case(tmp_path):
    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    pixels = np.full((8, 12, 3), 10000, np.uint16)
    for name in ("source", "background", "diffuse"):
        tifffile.imwrite(tmp_path / f"{name}.tif", pixels, photometric="rgb", metadata=None,
                         extratags=[(34675, "B", len(icc), icc, False)])
    value = sessions.create_session(tmp_path / "source.tif", tmp_path, intent="publication boundary")
    support = tmp_path / "support.tif"
    tifffile.imwrite(support, np.full((8, 12), 255, np.uint8), photometric="minisblack", metadata=None)
    for name in ("pure", "support"):
        value = sessions.register_mask(value["directory"], support, mask_id=name,
                                       source_sha256=value["source"]["sha256"], origin="agent")
    evidence = tmp_path / "evidence.txt"
    evidence.write_text("caller-reported synthetic S=B=D", encoding="utf-8")
    value = layers.register_layers(value["directory"], bundle_id="pair",
        background_path=tmp_path / "background.tif", diffuse_path=tmp_path / "diffuse.tif",
        pure_sky_mask_id="pure", transient_support_mask_id="support",
        source_sha256=value["source"]["sha256"], provenance={
            "kind": "caller_reported_layer_chain", "description": "Synthetic test fixture only",
            "evidence": [{"path": str(evidence), "sha256": sha256_file(evidence)}]})
    bundle = value["layer_bundles"]["pair"]
    recipe = {"schema": "aester-finishing-recipe-3", "source": recipe_source(value["source"]),
              "intent": "Disabled operation still verifies dependencies",
              "operations": [{"id": "layers", "type": "astrophotographic_layers", "enabled": False,
                  "mask_id": None, "params": {"bundle_id": "pair", "bundle_sha256": bundle["bundle_sha256"],
                      "star_strength": 0, "milkyway_amount": 0, "milkyway_mask_id": None,
                      "meteor_core_extra": 0, "meteor_glow_amount": 0}}]}
    return sessions.revise_recipe(value["directory"], recipe, expected_revision=0), recipe, bundle


def test_cancel_arriving_during_last_bundle_check_cannot_publish(layer_case, monkeypatch):
    value, _, _ = layer_case
    state = {"cancelled": False, "checks": 0}
    original = layers.recipe_bundles

    def check_then_cancel(*args):
        result = original(*args)
        state["checks"] += 1
        if state["checks"] == 2:
            # Cancellation arrives after render and during final dependency verification.
            state["cancelled"] = True
        return result

    monkeypatch.setattr(layers, "recipe_bundles", check_then_cancel)
    monkeypatch.setattr(candidates, "render_recipe", lambda *a, **k: {
        "status": "validated", "fixture": "mock render returned before cancellation"})
    with pytest.raises(backend.RenderCancelled):
        candidates.render_candidate(value["directory"], expected_revision=1,
            gimp_path="not-launched", cancel=lambda: state["cancelled"])
    current = sessions.read_session(value["directory"])
    assert state == {"cancelled": True, "checks": 2}
    assert len(current["candidates"]) == 1
    assert current["candidates"][0]["status"] == "cancelled"
    assert "execution" not in current["candidates"][0]
    assert current["revisions"] == value["revisions"]


def test_deadline_expiring_during_product_validation_cannot_publish(layer_case, tmp_path, monkeypatch):
    value, recipe, bundle = layer_case
    clock = {"now": 0.0}

    class CompletedProcess:
        returncode = 0
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def poll(self): return 0
        def kill(self): pytest.fail("already-completed process should not need killing")

    def validate_then_expire(*args):
        # Explicit clock injection, not a sleep or a claim of real GIMP completion.
        clock["now"] = 10.0
        return {"receipt": {"host_stage": None}, "fixture": "validation crosses the one-second deadline"}

    work = tmp_path / "private-render"
    work.mkdir()
    monkeypatch.setattr(backend, "guard_kind", lambda: backend.GUARD_JOB_OBJECT)
    monkeypatch.setattr(backend, "guarded_process", lambda *a, **k: CompletedProcess())
    monkeypatch.setattr(backend, "verify_products", validate_then_expire)
    monkeypatch.setattr(backend.time, "monotonic", lambda: clock["now"])
    with pytest.raises(TimeoutError):
        backend.render_recipe(value["source"], recipe, {}, work, gimp_path=sys.executable,
                              timeout=1, bundles={"pair": bundle})
    execution = json.loads((work / "execution.json").read_text(encoding="utf-8"))
    assert execution["status"] == "failed"
    assert execution["seconds"] == 10.0
    assert "products" not in execution
