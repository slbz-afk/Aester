"""EFF-NOX (P1c contract K1): KN-1 unit tests (docs/EFFICIENCY_P1C_PREREG_20261005.md §4.1).

A new workflow request that runs nox freezes an explicit ORT intra-op thread count (8) when it names none; requests
without nox are unchanged (golden digests computed on 41ce6f3, before K1); a request frozen before P1c (null) still runs
ORT-auto on resume. Synthetic inputs, stub models and hand-encoded ONNX fixtures only: no RAW, Siril or nox weights.
"""
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from starstack import cli, frame_pool, starless_runtime as rt, workflow as wf
from starstack.mcp_server import WorkspaceTools
from test_starless_runtime import _bytes, _float_tensor, _int, _node, _str, _value_info, onnx_model, ort_or_skip
from test_workflow import setup  # noqa: F401  (existing synthetic engine/product fixture)

# The installer's nox adapter (packaging/aester_p2/download-manifest.json): it names no thread count.
NOX_ADAPTER = {"layout": "nhwc", "channel_order": "bgr", "value_range": "-1..1",
               "tile": 512, "stride": 256, "batch": 1, "stitch": "crop"}
SMALL = {"tile": 64, "stride": 32}        # the synthetic 144 x 160 products of the setup fixture

# Golden values of the unchanged code (41ce6f3), paths normalised to <tmp>: requests without nox keep their digests.
GOLDEN_DEFAULT_OPTIONS = "3bf960b892dbef59af8bf9c5889997d8fec6e428a0e6a2fb1c80ea3493717e0d"
GOLDEN_REQUESTS = {"defaults": "18e9b494f68a0a96fcce6be72937a5801e374d5f389a57efff1a635845ff10a8",
                   "model-off": "6b0930b6ed43551f9a09ca0e6a9bd49ee0d708b1875bb18b8f9771042f573872",
                   "null-off": "18e9b494f68a0a96fcce6be72937a5801e374d5f389a57efff1a635845ff10a8",
                   "int-off": "09db51465e002733f500a7534b133b1901d7a5a7bdd7374383883fb500e1ae9b"}
GUI_ADAPTER = {"channel_order": "bgr", "value_range": "-1..1", "layout": "nhwc", "stitch": "crop"}


def admit(s, **kwargs):
    return wf.submit_workflow(s["inputs"], s["output"], **kwargs)


def nox(s, **kwargs):
    return {"sky_mask_path": s["mask"], "star_model_path": s["model"], "stars_enabled": True, **kwargs}


def pre_p1c_admit(s, monkeypatch, **kwargs):
    """A nox workflow admitted by the rule before K1 (1c3097b): the thread count stays null in the frozen request."""
    current = wf._options
    with monkeypatch.context() as m:
        m.setattr(wf, "_options", lambda options, *, stars_enabled: current(options, stars_enabled=False))
        record = admit(s, **kwargs)
    assert record["request"]["backend_options"]["intra_op_threads"] is None
    assert record["signature"] == wf._digest(record["request"])
    return record


def recording_model(monkeypatch, seen):
    class Model:
        def __init__(self, path, **kw):
            seen.append(kw)
            self.identity = {"model_sha256": wf.core._sha(Path(path)), "backend": "test-scale", **kw}

        def __call__(self, x):
            return np.ascontiguousarray(x * .8, np.float32)
    monkeypatch.setattr(rt, "OnnxModel", Model)


def _normalized(value, root):
    if isinstance(value, dict):
        return {key: _normalized(item, root) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalized(item, root) for item in value]
    if isinstance(value, str):
        return value.replace(root, "<tmp>").replace("\\", "/")
    return value


def conv_model(path, *, hidden=32, seed=3):
    """NCHW 3 -> hidden -> 3 two-layer 3x3 Conv/Relu/Conv graph, same spatial size (pads 1).

    Unlike an element-wise Mul, ORT splits these Convs across its intra-op pool (MLAS), and every output is a float
    sum over hidden x 3 x 3 products. A probe on the development PC (2026-10-05) ran the test sizes 2-3x faster with
    8 threads than with 1, and bit-identical for 1, 2, 3, 4, 8, 16, 64 threads and ORT-auto. So the graph can show a
    spin mode that changes the work done, not a change of reduction order: nox's own thread sensitivity (ORT-auto vs
    8 differ, P1b §0) is only exercised by the real-graph test.
    """
    rng = np.random.default_rng(seed)
    w1 = (rng.normal(0, 1, (hidden, 3, 3, 3)) / np.sqrt(27)).astype(np.float32)
    w2 = rng.uniform(0, 1, (3, hidden, 3, 3)).astype(np.float32)
    w2 /= w2.reshape(3, -1).sum(1)[:, None, None, None]      # keeps the estimate inside 0..1

    def pads(node):                                          # AttributeProto: name, ints, type INTS (7)
        return _bytes(1, node + _bytes(5, _str(1, "pads") + b"".join(_int(8, 1) for _ in range(4)) + _int(20, 7)))
    dims = ["N", 3, "H", "W"]
    graph = (pads(_node("Conv", ["inputs", "w1"], ["hidden"])) + _bytes(1, _node("Relu", ["hidden"], ["relu"]))
             + pads(_node("Conv", ["relu", "w2"], ["Identity"])) + _str(2, "conv-fixture")
             + _bytes(5, _float_tensor("w1", list(w1.shape), w1.ravel()))
             + _bytes(5, _float_tensor("w2", list(w2.shape), w2.ravel()))
             + _bytes(11, _value_info("inputs", dims)) + _bytes(12, _value_info("Identity", dims)))
    model = _int(1, 8) + _str(2, "starstack-test") + _bytes(7, graph) + _bytes(8, _str(1, "") + _int(2, 17))
    Path(path).write_bytes(model)
    return Path(path)


# ---- admission ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("stars", [False, True])
@pytest.mark.parametrize("given", ["absent", "null", 1, 3, 8, 16])
def test_options_matrix(stars, given):
    options = dict(GUI_ADAPTER) if given == "absent" else {**GUI_ADAPTER, "intra_op_threads": None if given == "null" else given}
    sent = dict(options)
    out = wf._options(options, stars_enabled=stars)
    if given in ("absent", "null"):
        expected = rt.NOX_INTRA_OP_THREADS if stars else None
    else:
        expected = given
    assert out["intra_op_threads"] == expected and type(out["intra_op_threads"]) is type(expected)
    assert {k: v for k, v in out.items() if k != "intra_op_threads"} == \
        {k: v for k, v in {**wf.BACKEND_DEFAULTS, **GUI_ADAPTER}.items() if k != "intra_op_threads"}
    assert options == sent                      # the caller's dict is not changed


def test_options_defaults_and_validation():
    assert rt.NOX_INTRA_OP_THREADS == 8
    assert wf.BACKEND_DEFAULTS["intra_op_threads"] is None
    assert wf._options(None, stars_enabled=False) == wf.BACKEND_DEFAULTS
    assert wf._options(None, stars_enabled=True) == {**wf.BACKEND_DEFAULTS, "intra_op_threads": 8}
    assert wf._digest(wf._options(None, stars_enabled=False)) == GOLDEN_DEFAULT_OPTIONS
    with pytest.raises(TypeError):
        wf._options(None)                       # every caller states whether nox runs
    for bad in (0, 257, 2.0, True):
        with pytest.raises(ValueError, match="intra_op_threads"):
            wf._options({"intra_op_threads": bad}, stars_enabled=True)


def test_requests_without_nox_keep_their_golden_digests(setup, tmp_path):
    root = str(Path(tmp_path).resolve())
    for name, kwargs in (("defaults", {}),
                         ("model-off", {"star_model_path": setup["model"], "backend_options": GUI_ADAPTER}),
                         ("null-off", {"backend_options": {"intra_op_threads": None}}),
                         ("int-off", {"backend_options": {"intra_op_threads": 3}})):
        record = admit(setup, idempotency_key="golden-" + name, **kwargs)
        assert record["signature"] == wf._digest(record["request"])
        assert wf._digest(_normalized(record["request"], root)) == GOLDEN_REQUESTS[name], name
        assert record["request"]["backend_options"]["intra_op_threads"] == (3 if name == "int-off" else None)


@pytest.mark.parametrize("backend", [None, {}, {"intra_op_threads": None}, {"intra_op_threads": 8},
                                     NOX_ADAPTER, {**NOX_ADAPTER, "intra_op_threads": None}])
def test_nox_request_freezes_8_and_equals_a_pinned_request(setup, backend):
    pinned = admit(setup, idempotency_key="pinned", **nox(setup, backend_options={**(backend or {}), "intra_op_threads": 8}))
    record = admit(setup, idempotency_key="default", **nox(setup, backend_options=backend))
    for frozen in (record["request"]["backend_options"], record["request"]["config"]["backend_options"],
                   record["config"]["backend_options"]):
        assert frozen["intra_op_threads"] == 8
    assert record["signature"] == wf._digest(record["request"])
    # bench --pin-ort 8 and the new default are one contract: byte-identical requests, key order included
    assert json.dumps(record["request"]) == json.dumps(pinned["request"])
    assert record["signature"] == pinned["signature"]


def test_explicit_count_is_kept_for_nox(setup):
    record = admit(setup, **nox(setup, backend_options={"intra_op_threads": 3}))
    assert record["request"]["backend_options"]["intra_op_threads"] == 3


def test_same_key_resubmission_of_a_pre_p1c_nox_workflow_is_refused(setup, monkeypatch):
    old = pre_p1c_admit(setup, monkeypatch, idempotency_key="old-nox", **nox(setup, backend_options=SMALL))
    before = (Path(old["workflow_dir"]) / "workflow.json").read_bytes()
    with pytest.raises(ValueError, match="相同提交编号对应不同"):
        admit(setup, idempotency_key="old-nox", **nox(setup, backend_options=SMALL))
    assert (Path(old["workflow_dir"]) / "workflow.json").read_bytes() == before
    assert setup["calls"]["submits"] == 0
    wf._verify(wf._read(Path(old["workflow_dir"])))          # the old record itself is intact


# ---- runtime -----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("era", ["new", "pre-p1c"])
def test_the_frozen_value_reaches_the_model(setup, monkeypatch, era):
    seen = []
    recording_model(monkeypatch, seen)
    kwargs = nox(setup, backend_options=SMALL, background_strength=.4)
    record = pre_p1c_admit(setup, monkeypatch, **kwargs) if era == "pre-p1c" else admit(setup, **kwargs)
    done = wf.run_workflow(record["workflow_dir"])
    assert done["state"] == "succeeded", done.get("error")
    assert [kw["intra_op_threads"] for kw in seen] == [8 if era == "new" else None]


def test_pre_p1c_nox_workflow_cancelled_in_separation_resumes_with_ort_auto(setup, monkeypatch):
    record = pre_p1c_admit(setup, monkeypatch, **nox(setup, backend_options=SMALL, background_strength=.4))
    d = Path(record["workflow_dir"])
    seen = []

    class Cancel:
        def __init__(self, path, **kw):
            seen.append(kw)
            self.identity = {"model_sha256": wf.core._sha(Path(path))}

        def __call__(self, x):
            wf.cancel_workflow(d)
            return x
    monkeypatch.setattr(rt, "OnnxModel", Cancel)
    cancelled = wf.run_workflow(d)
    assert cancelled["state"] == "cancelled" and not cancelled["artifacts"]
    assert "separation" not in cancelled["checkpoints"] and "stack" in cancelled["checkpoints"]
    recording_model(monkeypatch, seen)
    assert wf.resume_workflow(d)["attempt"] == 2
    done = wf.run_workflow(d)
    assert done["state"] == "succeeded", done.get("error")
    assert [kw["intra_op_threads"] for kw in seen] == [None, None]
    assert done["request"] == record["request"] and done["signature"] == record["signature"]
    assert setup["calls"]["submits"] == 1


@pytest.mark.parametrize("era", ["new", "pre-p1c"])
def test_real_onnx_identity_auto_for_pre_p1c_and_explicit_8_for_new(setup, monkeypatch, era):
    ort_or_skip()
    onnx_model(setup["model"], op="Mul", scale=0.8)          # a real ONNX graph in place of the stub weights
    kwargs = nox(setup, backend_options=SMALL, background_strength=.4)
    record = pre_p1c_admit(setup, monkeypatch, **kwargs) if era == "pre-p1c" else admit(setup, **kwargs)
    done = wf.run_workflow(record["workflow_dir"])
    assert done["state"] == "succeeded", done.get("error")
    backend = json.loads(Path(done["separation_result"]).read_text(encoding="utf-8"))["backend"]
    if era == "new":
        assert backend["intra_op_threads_requested"] == 8 and backend["intra_op_threads"] == 8
        assert backend["intra_op_threads_mode"] == "explicit"
    else:
        assert backend["intra_op_threads_requested"] is None and backend["intra_op_threads"] is None
        assert backend["intra_op_threads_mode"] == "onnxruntime-auto"
    assert not any("spinning" in key for key in backend)
    note = done["timings"]["notes"]["nox_threads"]
    assert note["intra_op_threads"] == (8 if era == "new" else "ort_auto")
    assert note["usable_cpus"] == frame_pool.usable_cpus()
    assert note["allow_spinning"] == ("0" if era == "new" and frame_pool.usable_cpus() < 8 else "default")
    outside = json.dumps({k: v for k, v in done.items() if k not in ("timings", "timing_attempts")})
    assert "nox_threads" not in outside and "allow_spinning" not in outside


@pytest.mark.parametrize("threads,cpus,spin_off", [(8, 4, True), (8, 7, True), (8, 8, False), (8, 192, False),
                                                   (4, 4, False), (None, 1, False)])
def test_spinning_is_off_only_when_threads_exceed_usable_cpus(tmp_path, monkeypatch, threads, cpus, spin_off):
    ort_or_skip()
    path = onnx_model(tmp_path / "identity.onnx")
    notes = []
    monkeypatch.setattr(frame_pool, "usable_cpus", lambda: cpus)
    monkeypatch.setattr(rt.profiling, "note", lambda key, value: notes.append((key, value)))
    model = rt.OnnxModel(path, intra_op_threads=threads)
    session = model.session.get_session_options()
    if spin_off:
        assert session.get_session_config_entry(rt.ALLOW_SPINNING) == "0"
    else:
        with pytest.raises(Exception):
            session.get_session_config_entry(rt.ALLOW_SPINNING)
    assert notes == [("nox_threads", {"intra_op_threads": "ort_auto" if threads is None else threads,
                                      "usable_cpus": cpus, "allow_spinning": "0" if spin_off else "default"})]
    # how idle threads wait never enters the identity
    monkeypatch.setattr(frame_pool, "usable_cpus", lambda: 1024)
    other = rt.OnnxModel(path, intra_op_threads=threads)
    strip = lambda identity: {k: v for k, v in identity.items() if k != "session_load_seconds"}  # noqa: E731
    assert strip(model.identity) == strip(other.identity)


def test_spin_option_path_smoke_on_a_threaded_conv(tmp_path, monkeypatch):
    """Smoke test of the session-option path only: usable_cpus is patched, the process keeps every processor.

    A session built with spinning off (patched 4 usable processors) and one with ORT's default (patched 64) give the
    same estimate on a graph ORT runs across its 8 intra-op threads. The real processor restriction is
    test_threaded_conv_is_bit_identical_with_spinning_off_under_affinity_4 (asset-free) and, on the nox graph itself,
    test_real_nox_crop_is_bit_identical_with_spinning_off_under_affinity_4 (needs the main checkout's assets).
    """
    ort_or_skip()
    from test_starless_runtime import image
    path = conv_model(tmp_path / "conv.onnx")
    src = image((300, 420), seed=7)
    out = {}
    for cpus, spin in ((4, "0"), (64, None)):
        monkeypatch.setattr(frame_pool, "usable_cpus", lambda: cpus)
        model = rt.OnnxModel(path, layout="nchw", intra_op_threads=8)
        options = model.session.get_session_options()
        if spin is None:
            with pytest.raises(Exception):
                options.get_session_config_entry(rt.ALLOW_SPINNING)
        else:
            assert options.get_session_config_entry(rt.ALLOW_SPINNING) == spin
        estimate = np.empty_like(src)
        rt.run_tiled(src, estimate, model, tile=128, stride=64, stitch="crop")
        out[cpus] = estimate
    assert out[4].tobytes() == out[64].tobytes()
    assert np.any(out[4] != src)                       # the graph did act on the image


# ---- entries -----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("argv,threads", [([], 8), (["--threads", "3"], 3), (["--threads", "8"], 8)])
def test_cli_starless_defaults_to_8(monkeypatch, capsys, argv, threads):
    calls = []
    monkeypatch.setattr(rt, "starless", lambda *a, **k: calls.append(k) or {"state": "succeeded"})
    monkeypatch.setattr(sys, "argv", ["starstack", "starless", "--source", "s.tif", "--sky-mask", "sky.tif",
                                      "--protected-mask", "p.png", "--model", "m.onnx", *argv])
    assert cli.main() == 0
    assert [k["intra_op_threads"] for k in calls] == [threads]
    assert json.loads(capsys.readouterr().out)["state"] == "succeeded"


def test_library_defaults_are_unchanged():
    assert inspect.signature(rt.starless).parameters["intra_op_threads"].default is None
    assert inspect.signature(rt.OnnxModel).parameters["intra_op_threads"].default is None


# ---- the real nox graph on a real display crop (assets of the main checkout; skipped when absent) -----------------

REPO = Path(__file__).resolve().parents[1]
NOX_MODEL_REL = Path("outputs") / "starless-acceptance-20260923" / "models" / "nox_color.onnx"
F48_REL = Path("outputs") / "perf-baseline-20260928" / "runs" / "F48"
CHILD = r"""
import ctypes, sys
import numpy as np
k32 = ctypes.WinDLL("kernel32", use_last_error=True)
k32.GetCurrentProcess.restype = ctypes.c_void_p
k32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
k32.SetProcessAffinityMask.restype = ctypes.c_int
if not k32.SetProcessAffinityMask(k32.GetCurrentProcess(), 0xF):
    raise SystemExit("SetProcessAffinityMask failed %d" % ctypes.get_last_error())
import json
from starstack import frame_pool, starless_runtime as rt
assert frame_pool.usable_cpus() == 4, frame_pool.usable_cpus()
adapter, tiling = json.loads(sys.argv[2]), json.loads(sys.argv[3])
model = rt.OnnxModel(sys.argv[1], intra_op_threads=8, **adapter)
assert model.session.get_session_options().get_session_config_entry(rt.ALLOW_SPINNING) == "0"
source = np.load(sys.argv[4])
estimate = np.empty_like(source)
rt.run_tiled(source, estimate, model, stitch="crop", **tiling)
np.save(sys.argv[5], estimate)
"""
NOX_GRAPH = {"layout": "nhwc", "channel_order": "bgr", "value_range": "-1..1"}


def _main_checkout(rel):
    roots = [REPO / rel]
    try:                                   # a git worktree: untracked assets live in the main checkout
        common = subprocess.run(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"], cwd=REPO,
                                capture_output=True, text=True, timeout=30).stdout.strip()
        if common:
            roots.append(Path(common).parent / rel)
    except (OSError, subprocess.SubprocessError):
        pass
    return roots


def _host_and_affinity_4(tmp_path, model, adapter, tiling, source):
    """The estimate of 8 intra-op threads on the unrestricted host (spinning default) and in a child process held to
    4 processors by SetProcessAffinityMask (spinning off)."""
    host = rt.OnnxModel(model, intra_op_threads=8, **adapter)
    with pytest.raises(Exception):
        host.session.get_session_options().get_session_config_entry(rt.ALLOW_SPINNING)
    expected = np.empty_like(source)
    rt.run_tiled(source, expected, host, stitch="crop", **tiling)
    np.save(tmp_path / "source.npy", source)
    env = dict(os.environ, PYTHONPATH=str(REPO / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""))
    done = subprocess.run([sys.executable, "-X", "utf8", "-c", CHILD, str(model), json.dumps(adapter), json.dumps(tiling),
                           str(tmp_path / "source.npy"), str(tmp_path / "affinity4.npy")],
                          env=env, capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stderr[-2000:]
    return expected, np.load(tmp_path / "affinity4.npy")


def _affinity_preconditions():
    if os.name != "nt":
        pytest.skip("Windows processor-affinity check")
    ort_or_skip()
    if frame_pool.usable_cpus() <= rt.NOX_INTRA_OP_THREADS:
        pytest.skip("the host side must keep spinning (more than 8 usable processors)")


def test_threaded_conv_is_bit_identical_with_spinning_off_under_affinity_4(tmp_path):
    """Asset-free: 8 threads really held to 4 processors (spinning off) vs the host, on the threaded Conv graph.

    It shows that spinning off on an oversubscribed processor set leaves the estimate unchanged for a graph ORT splits
    across its pool; it cannot show nox's own sensitivity (see conv_model), which only the real-graph test can.
    """
    _affinity_preconditions()
    from test_starless_runtime import image
    model = conv_model(tmp_path / "conv.onnx", hidden=64)
    source = image((640, 896), seed=11)
    expected, restricted = _host_and_affinity_4(tmp_path, model, {"layout": "nchw"},
                                                {"tile": 256, "stride": 128}, source)
    assert restricted.tobytes() == expected.tobytes()
    assert np.any(expected != source)                  # the graph did act on the image


def test_real_nox_crop_is_bit_identical_with_spinning_off_under_affinity_4(tmp_path):
    """KN-4 in small: 8 threads on 4 usable processors (spinning off) vs the unrestricted host (spinning default).

    The only nox-graph evidence among the unit tests: it skips where the main checkout's untracked assets are absent.
    """
    _affinity_preconditions()
    model = next((p for p in _main_checkout(NOX_MODEL_REL) if p.is_file()), None)
    sources = [s for root in _main_checkout(F48_REL) if root.is_dir()
               for s in sorted(root.glob("c/workflow-*/finishing/milkyway/milkyway-*/milkyway_srgb16.tif"))]
    if model is None or not sources:
        pytest.skip(f"untracked assets {NOX_MODEL_REL} / {F48_REL} are absent")
    import tifffile
    with tifffile.TiffFile(sources[0]) as tf:          # read-only fixture
        full = tf.pages[0].asarray()
    h, w = full.shape[:2]
    crop = np.ascontiguousarray(full[h // 5:h // 5 + 768, w // 2 - 384:w // 2 + 384])
    del full
    assert crop.dtype == np.uint16 and crop.shape == (768, 768, 3)
    expected, restricted = _host_and_affinity_4(tmp_path, model, NOX_GRAPH, {"tile": 512, "stride": 256}, crop)
    assert restricted.tobytes() == expected.tobytes()
    assert np.any(expected != crop)                    # the graph did act on the crop


@pytest.mark.parametrize("stars", [True, False])
def test_mcp_submission_gets_8_only_with_nox(setup, stars):
    setup["output"].mkdir(parents=True, exist_ok=True)
    mask = setup["output"] / "sky.png"
    mask.write_bytes(setup["mask"].read_bytes())
    service = WorkspaceTools(setup["inputs"], setup["output"], setup["engine"],
                             star_model_path=setup["model"], star_backend_options=NOX_ADAPTER)
    files = [str(setup["inputs"] / f"{i}.NEF") for i in range(3)]
    result = service.submit_workflow(input_files=files, sky_mask_path=str(mask), stars_enabled=stars)
    assert result["request"]["stars_enabled"] is stars
    assert result["request"]["backend_options"]["intra_op_threads"] == (8 if stars else None)
    if stars:
        assert {k: result["request"]["backend_options"][k] for k in NOX_ADAPTER} == NOX_ADAPTER
