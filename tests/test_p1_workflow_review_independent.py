"""Independent adversarial delivery review; fixtures stub only raw stacking/model."""
from pathlib import Path
import copy
import json

import numpy as np
import pytest

from starstack import workflow as wf, workflow_edit as edit
from test_workflow import setup
from test_workflow_review_lifecycle import calibrated_workflow


def test_corrupted_master_copy_cannot_be_published_as_unchanged(setup, monkeypatch):
    done, session, state = calibrated_workflow(setup, monkeypatch)
    # The copy hashes the bytes it reads (P1 hashing); the destination is then read back
    # for real. Corrupt the written master after its bytes reach the disk, before that read-back.
    original = edit.hashing.copy_with_digest
    def broken_copy(src, target, **kwargs):
        target = Path(target)
        if target.name == "master_scene_linear.tif" and target.parent.name.startswith(".partial-reviewed-"):
            size, corrupted = Path(src).stat().st_size, []
            def check():
                if not corrupted and target.is_file() and target.stat().st_size == size:
                    with target.open("r+b") as handle:
                        handle.seek(-1, 2)
                        value = handle.read(1)
                        handle.seek(-1, 2)
                        handle.write(bytes([value[0] ^ 1]))
                    corrupted.append(True)
            kwargs["check"] = check
        return original(src, target, **kwargs)
    monkeypatch.setattr(edit.hashing, "copy_with_digest", broken_copy)
    with pytest.raises(ValueError, match="复核成片复制校验失败"):
        edit.apply_review(done["workflow_dir"], session, expected_draft_revision="token")
    assert wf.workflow_status(done["workflow_dir"])["delivery_dir"] == done["delivery_dir"]


def test_top_level_finishing_cannot_substitute_unfrozen_pixels(setup, monkeypatch):
    done, session, state = calibrated_workflow(setup, monkeypatch)
    root = Path(done["workflow_dir"])
    replacement = root / "unfrozen-alternate"
    replacement.mkdir()
    for source_key, name in (("tiff", "star_reduced_srgb16.tif"), ("jpeg", "star_reduced.jpg")):
        edit.shutil.copyfile(done["checkpoints"]["stack"]["data"]["colour"]["photo"][source_key], replacement/name)
    record = json.loads((root/"workflow.json").read_text(encoding="utf8"))
    record["finishing"]["reduction"]["processing_directory"] = str(replacement)
    wf.atomic_json(root/"workflow.json", record)
    with pytest.raises(ValueError):
        edit.apply_review(root, session, expected_draft_revision="token")
    assert state["applies"] == 0  # stale finishing rejected before candidate mutation
    assert wf.workflow_status(root)["delivery_dir"] == done["delivery_dir"]


def test_scalar_pair_oracle_preserves_rgb_direction_and_code_residual():
    # These include mixed RGB signs, endpoint channels, and headroom limits.
    cases = [([100, 200, 300], [170, 150, 310], [300, 500, 900]),
             ([65500, 600, 500], [65525, 550, 800], [65535, 900, 1600]),
             ([20, 65000, 50], [8, 65010, 40], [0, 10000, 0]),
             ([100, 200, 300], [0, 250, 280], [500, 600, 900]),
             ([100, 200, 300], [100, 200, 300], [0, 65535, 0])]
    b = np.array([c[0] for c in cases], np.uint16)[None, ...]
    s = np.array([c[1] for c in cases], np.uint16)[None, ...]
    proposed = np.array([c[2] for c in cases], np.uint16)[None, ...]
    out, q = np.empty_like(b), np.empty_like(b)
    edit.add_signed_trails(s, b, proposed, out, q)
    for index, (base, source, goal) in enumerate(cases):
        delta = [source[c]-base[c] for c in range(3)]
        assert [int(out[0,index,c])-int(q[0,index,c]) for c in range(3)] == delta
        if not any(delta):
            assert q[0,index].tolist() == goal
            continue
        limits = [1.]
        if any(value in (0, 65535) for value in [*base, *source]):
            limits.append(0.)
        for c in range(3):
            change = goal[c]-base[c]
            if change > 0:
                limits.append((65534-max(base[c],source[c]))/change)
            elif change < 0:
                limits.append((min(base[c],source[c])-1)/(-change))
        alpha = max(0., min(limits))
        expected = [base[c]+int((goal[c]-base[c])*alpha) for c in range(3)]
        assert q[0,index].tolist() == expected


def test_changed_draft_token_passed_unchanged_before_any_render(setup, monkeypatch):
    from starstack import trail_review_session as review
    done, session, state = calibrated_workflow(setup, monkeypatch)
    received = []
    def stale(path, **kwargs):
        received.append(kwargs)
        raise ValueError("draft changed")
    monkeypatch.setattr(review, "apply_review", stale)
    monkeypatch.setattr(edit, "_render_fixed", lambda *a, **k: pytest.fail("rendered stale review"))
    with pytest.raises(ValueError, match="draft changed"):
        edit.apply_review(done["workflow_dir"], session, expected_revision={"id": "known"}, expected_draft_revision="user-approved-token")
    assert received == [{"expected_revision": {"id": "known"}, "expected_draft_revision": "user-approved-token"}]
    assert wf.workflow_status(done["workflow_dir"])["delivery_dir"] == done["delivery_dir"]


def test_milkyway_mapping_reads_frozen_transform_not_later_work_files(setup, monkeypatch):
    from test_workflow_regions import bind_child_mask, region_project
    bind_child_mask(setup, monkeypatch)
    matrix = np.array([[1., 0., 9.], [0., 1., 0.], [0., 0., 1.]])
    monkeypatch.setattr(wf.core, "registration_transforms", lambda *a: ({"light_00001": matrix}, None))
    path, files = region_project(setup)
    record = wf.submit_workflow(setup["inputs"], setup["output"], regions_path=path,
                                 input_files=files, milkyway_enabled=True)
    original = wf._stack
    def mutate_after_stack(*args, **kwargs):
        stack = original(*args, **kwargs)
        work = Path(stack["job"]["job_dir"])/"work"
        work.mkdir(exist_ok=True)
        (work/"distortion.json").write_text('{"status":"fitted","parameters":{"changed":true}}')
        (work/"light_.seq").write_text("changed after frozen stack")
        monkeypatch.setattr(wf.core, "registration_transforms", lambda *a: pytest.fail("read mutable post-stack geometry"))
        return stack
    monkeypatch.setattr(wf, "_stack", mutate_after_stack)
    done = wf.run_workflow(record["workflow_dir"])
    assert done["state"] == "succeeded", done.get("error")
    assert done["checkpoints"]["milkyway_region"]["data"]["first_frame_to_reference"] == matrix.tolist()
