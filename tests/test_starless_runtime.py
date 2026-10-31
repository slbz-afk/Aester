"""Starless package step 2 (docs/STARLESS_PACKAGE_20260923.md §八): tiled runtime R1–R7.

ONNX fixtures are hand-encoded protobuf (no ``onnx`` package needed); onnxruntime cases
skip when the optional dependency is absent — the Windows venv run is the acceptance.
"""
import json
import struct
import sys
from pathlib import Path
import numpy as np
import pytest
import tifffile
from PIL import Image, ImageCms
from starstack import starless_runtime as rt
from starstack.star_separation import separate_stars


# ---- minimal ONNX protobuf writer -------------------------------------------------------------
def _varint(n):
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _key(field, wire):
    return _varint((field << 3) | wire)


def _bytes(field, data):
    return _key(field, 2) + _varint(len(data)) + data


def _int(field, value):
    return _key(field, 0) + _varint(value)


def _str(field, text):
    return _bytes(field, text.encode())


def _value_info(name, dims):
    shape = b"".join(_bytes(1, _int(1, d) if isinstance(d, int) else _str(2, d)) for d in dims)
    tensor = _int(1, 1) + _bytes(2, shape)            # elem_type FLOAT, shape
    return _str(1, name) + _bytes(2, _bytes(1, tensor))


def _node(op, inputs, outputs):
    return b"".join(_str(1, i) for i in inputs) + b"".join(_str(2, o) for o in outputs) + _str(4, op)


def _float_tensor(name, dims, values):
    return (b"".join(_int(1, d) for d in dims) + _int(2, 1) + _str(8, name)
            + _bytes(9, np.asarray(values, np.float32).tobytes()))


def onnx_model(path, *, op="Identity", input_dims=("N", "H", "W", 3), scale=None, opset=17):
    initializers = []
    inputs = ["inputs"]
    if scale is not None:
        initializers.append(_float_tensor("scale", [], [scale]))
        inputs.append("scale")
    graph = (_bytes(1, _node(op, inputs, ["Identity"])) + _str(2, "starless-fixture")
             + b"".join(_bytes(5, t) for t in initializers)
             + _bytes(11, _value_info("inputs", list(input_dims))) + _bytes(12, _value_info("Identity", list(input_dims))))
    model = _int(1, 8) + _str(2, "starstack-test") + _bytes(7, graph) + _bytes(8, _str(1, "") + _int(2, opset))
    Path(path).write_bytes(model)
    return Path(path)


# ---- fixtures --------------------------------------------------------------------------------
def image(shape, seed=1):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[:shape[0], :shape[1]]
    base = 6000 + 30 * x + 12 * y
    a = np.stack([base, base + 1500, base + 2600], -1) + rng.integers(0, 400, shape + (3,))
    a[shape[0] // 3:shape[0] // 3 + 3, shape[1] // 2:shape[1] // 2 + 3] += 40000
    return np.clip(a, 0, 65535).astype(np.uint16)


def identity_callable(x):
    return np.ascontiguousarray(x, np.float32)


identity_callable.identity = {"backend": "python-identity", "model_sha256": "0" * 64}


def half_callable(x):
    return np.ascontiguousarray(x * np.float32(0.5))


def ramp_callable(x):
    ramp = np.linspace(0, 0.02, x.shape[2], dtype=np.float32)[None, None, :, None]
    return np.ascontiguousarray(x + ramp)


def ort_or_skip():
    return pytest.importorskip("onnxruntime")


# ---- R1 --------------------------------------------------------------------------------------
@pytest.mark.parametrize("shape", [(512, 512), (700, 1100), (200, 300)])
@pytest.mark.parametrize("stitch", ["blend", "crop"])
def test_r1_identity_roundtrip_python(shape, stitch):
    src = image(shape)
    out = np.empty_like(src)
    receipt = rt.run_tiled(src, out, identity_callable, tile=512, stride=256, stitch=stitch)
    assert np.array_equal(out, src)
    assert receipt["tiles"] >= 1 and receipt["shape"] == [*shape, 3] and receipt["stitch"] == stitch


@pytest.mark.parametrize("shape", [(512, 512), (700, 1100), (200, 300)])
@pytest.mark.parametrize("stitch", ["blend", "crop"])
def test_r1_identity_roundtrip_onnx(tmp_path, shape, stitch):
    ort_or_skip()
    model = rt.OnnxModel(onnx_model(tmp_path / "identity.onnx"))
    src = image(shape)
    out = np.empty_like(src)
    receipt = rt.run_tiled(src, out, model, tile=512, stride=256, stitch=stitch)
    assert np.array_equal(out, src)
    assert receipt["model"]["backend"] == "onnx-tiled-1" and receipt["model"]["uses_learned_model"] is True
    assert len(receipt["model"]["model_sha256"]) == 64 and receipt["model"]["expects_tile"] is None


# ---- R2 --------------------------------------------------------------------------------------
def test_r2_half_model_no_seams():
    src = image((700, 1100), seed=4)
    expected = np.rint(src.astype(np.float64) * 0.5).astype(np.uint16)
    crop = np.empty_like(src)
    rt.run_tiled(src, crop, half_callable, tile=256, stride=128, stitch="crop")
    assert np.array_equal(crop, expected)
    blend = np.empty_like(src)
    rt.run_tiled(src, blend, half_callable, tile=256, stride=128, stitch="blend", batch=3)
    assert np.abs(blend.astype(np.int32) - expected.astype(np.int32)).max() <= 1
    whole = np.rint(np.clip(half_callable(src[None].astype(np.float32) / 65535)[0], 0, 1) * 65535).astype(np.uint16)
    assert np.abs(blend.astype(np.int32) - whole.astype(np.int32)).max() <= 1


def test_r2_half_model_onnx(tmp_path):
    ort_or_skip()
    src = image((700, 1100), seed=4)
    expected = np.rint(src.astype(np.float64) * 0.5).astype(np.uint16)
    model = rt.OnnxModel(onnx_model(tmp_path / "half.onnx", op="Mul", scale=0.5))
    via_onnx = np.empty_like(src)
    rt.run_tiled(src, via_onnx, model, tile=256, stride=128, stitch="crop")
    assert np.array_equal(via_onnx, expected)
    blend = np.empty_like(src)
    rt.run_tiled(src, blend, model, tile=256, stride=128, stitch="blend", batch=4)
    assert np.abs(blend.astype(np.int32) - expected.astype(np.int32)).max() <= 1


# ---- R3 --------------------------------------------------------------------------------------
def test_r3_blend_reduces_tile_seams():
    src = np.full((600, 900, 3), 20000, np.uint16)
    jumps = {}
    for stitch in ("crop", "blend"):
        out = np.empty_like(src)
        rt.run_tiled(src, out, ramp_callable, tile=256, stride=128, stitch=stitch)
        columns = [c for c in range(128, 900, 128)]
        jumps[stitch] = max(int(np.abs(out[:, c].astype(np.int32) - out[:, c - 1].astype(np.int32)).max()) for c in columns)
    assert jumps["crop"] > 0 and jumps["blend"] <= 0.5 * jumps["crop"]


# ---- R4 --------------------------------------------------------------------------------------
@pytest.mark.parametrize("layout,order,dims", [("nhwc", "bgr", ("N", "H", "W", 3)), ("nchw", "rgb", ("N", 3, "H", "W")), ("nchw", "bgr", ("N", 3, "H", "W"))])
def test_r4_layout_and_channel_order_adapters(tmp_path, layout, order, dims):
    ort_or_skip()
    model = rt.OnnxModel(onnx_model(tmp_path / f"{layout}-{order}.onnx", input_dims=dims), layout=layout, channel_order=order)
    src = image((300, 420), seed=9)
    out = np.empty_like(src)
    rt.run_tiled(src, out, model, tile=128, stride=64, stitch="blend")
    assert np.array_equal(out, src)
    assert model.identity["layout"] == layout and model.identity["channel_order"] == order
    half = rt.OnnxModel(onnx_model(tmp_path / f"half-{layout}-{order}.onnx", op="Mul", scale=0.5, input_dims=dims), layout=layout, channel_order=order)
    batch = np.linspace(0, 1, 2 * 8 * 8 * 3, dtype=np.float32).reshape(2, 8, 8, 3)
    assert np.allclose(half(batch), batch * 0.5)


# ---- R5 --------------------------------------------------------------------------------------
def display_fixture(tmp_path):
    a = image((96, 120), seed=2)
    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    src = tmp_path / "source.tif"
    tifffile.imwrite(src, a, photometric="rgb", metadata={"scene_linear": False, "display_encoding_applied": True,
                     "color_space": "encoded_sRGB", "roworder": "TOP-DOWN"}, extratags=[(34675, "B", len(icc), icc, False)])
    sky = np.ones(a.shape[:2], np.float32)
    sky[80:] = 0
    protect = np.zeros(a.shape[:2], np.uint8)
    protect[:, 3] = 255
    tifffile.imwrite(tmp_path / "sky.tif", sky)
    Image.fromarray(protect).save(tmp_path / "protect.png")
    return src, tmp_path / "sky.tif", tmp_path / "protect.png", a, (sky != 1) | (protect > 0)


def test_r5_provider_contract_with_separate_stars(tmp_path):
    src, sky, protect, a, locked = display_fixture(tmp_path)
    estimate, identity = rt.provider(half_callable_with_identity(), tile=64, stride=32, stitch="crop", batch=2)
    record = separate_stars(src, tmp_path / "jobs", sky, protect, provider=estimate, provider_identity=identity)
    backend = record["backend"]
    for key in ("backend", "model_sha256", "uses_learned_model", "tile", "stride", "stitch", "batch", "layout", "channel_order", "input_scale"):
        assert key in backend, key
    assert backend["tile"] == 64 and backend["stitch"] == "crop" and record["learned_star_separation"] is True
    receipt = record["provider_receipt"]
    for key in ("seconds", "megapixels", "megapixels_per_second", "tiles", "batch", "padded_shape"):
        assert key in receipt, key
    d = tifffile.imread(record["artifacts"]["diffuse_srgb16.tif"]["path"])
    assert np.array_equal(d[locked], a[locked])
    assert np.array_equal(d[~locked], np.rint(a[~locked].astype(np.float64) * 0.5).astype(np.uint16))
    assert np.array_equal(tifffile.imread(src), a)


def half_callable_with_identity():
    def model(x):
        return half_callable(x)
    model.identity = {"backend": "onnx-tiled-1", "model_sha256": "f" * 64, "uses_learned_model": True, "layout": "nhwc",
                      "channel_order": "rgb", "input_scale": rt.DOMAIN["input_scale"]}
    return model


# ---- R6 / R7 ---------------------------------------------------------------------------------
def test_r6_validation_and_missing_runtime(tmp_path, monkeypatch):
    src = image((64, 64))
    out = np.empty_like(src)
    with pytest.raises(ValueError, match="stride"):
        rt.run_tiled(src, out, identity_callable, tile=64, stride=96)
    with pytest.raises(ValueError, match="偶数"):
        rt.run_tiled(src, out, identity_callable, tile=64, stride=31)
    with pytest.raises(ValueError, match="stitch"):
        rt.run_tiled(src, out, identity_callable, tile=64, stride=32, stitch="average")
    with pytest.raises(ValueError, match="aliases"):
        rt.run_tiled(src, src, identity_callable, tile=64, stride=32)
    with pytest.raises(ValueError, match="model_sha256"):
        rt.provider(half_callable, tile=64, stride=32)
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    with pytest.raises(RuntimeError, match="onnxruntime 未安装"):
        rt.OnnxModel(tmp_path / "none.onnx")


def test_r6_cli_end_to_end(tmp_path, capsys):
    ort_or_skip()
    from starstack.cli import main
    src, sky, protect, a, locked = display_fixture(tmp_path)
    model = onnx_model(tmp_path / "half.onnx", op="Mul", scale=0.5)
    argv = ["starless", "--source", str(src), "--sky-mask", str(sky), "--protected-mask", str(protect), "--model", str(model),
            "--output", str(tmp_path / "out"), "--tile", "64", "--stride", "32", "--stitch", "crop", "--channel-order", "bgr"]
    capsys.readouterr()
    monkey = pytest.MonkeyPatch()
    monkey.setattr(sys, "argv", ["starstack"] + argv)
    try:
        assert main() == 0
    finally:
        monkey.undo()
    result = json.loads(capsys.readouterr().out)
    assert result["state"] == "succeeded" and result["backend"]["channel_order"] == "bgr" and result["backend"]["stitch"] == "crop"
    d = tifffile.imread(result["artifacts"]["diffuse_srgb16.tif"]["path"])
    assert np.array_equal(d[~locked], np.rint(a[~locked].astype(np.float64) * 0.5).astype(np.uint16))
    monkey.setattr(sys, "argv", ["starstack"] + argv[:-2] + ["--stride", "33"])
    try:
        assert main() == 1
    finally:
        monkey.undo()
    assert "偶数" in json.loads(capsys.readouterr().out)["error"]


def test_r7_fixed_input_size_is_enforced(tmp_path):
    ort_or_skip()
    model = rt.OnnxModel(onnx_model(tmp_path / "fixed.onnx", input_dims=("N", 128, 128, 3)))
    assert model.expects_tile == 128 and model.identity["expects_tile"] == 128
    src = image((300, 300))
    out = np.empty_like(src)
    with pytest.raises(ValueError, match="128"):
        rt.run_tiled(src, out, model, tile=256, stride=128)
    receipt = rt.run_tiled(src, out, model, tile=128, stride=64)
    assert np.array_equal(out, src) and receipt["tile"] == 128 and receipt["megapixels_per_second"] > 0


# ---- value range / fixed batch (added before R8; nox.cpp feeds x*2-1 and maps (y+1)/2 back) ------
def test_value_range_adapter_roundtrips_and_records(tmp_path):
    ort_or_skip()
    src = image((300, 420), seed=11)
    model = rt.OnnxModel(onnx_model(tmp_path / "identity.onnx"), value_range="-1..1", channel_order="bgr")
    out = np.empty_like(src)
    receipt = rt.run_tiled(src, out, model, tile=128, stride=64, stitch="crop")
    assert np.abs(out.astype(np.int32) - src.astype(np.int32)).max() <= 1      # float32 (2x-1+1)/2 rounding only
    assert receipt["model"]["value_range"] == "-1..1" and model.identity["expects_batch"] is None
    half = rt.OnnxModel(onnx_model(tmp_path / "half.onnx", op="Mul", scale=0.5), value_range="-1..1")
    batch = np.linspace(0, 1, 2 * 8 * 8 * 3, dtype=np.float32).reshape(2, 8, 8, 3)
    assert np.allclose(half(batch), ((batch * 2 - 1) * 0.5 + 1) / 2, atol=1e-6)   # the graph sees [-1, 1]
    with pytest.raises(ValueError, match="value_range"):
        rt.OnnxModel(tmp_path / "identity.onnx", value_range="0..255")


def test_fixed_batch_dimension_is_enforced(tmp_path):
    ort_or_skip()
    model = rt.OnnxModel(onnx_model(tmp_path / "fixed-batch.onnx", input_dims=(1, 128, 128, 3)))
    assert model.expects_batch == 1 and model.expects_tile == 128
    src = image((300, 300))
    out = np.empty_like(src)
    with pytest.raises(ValueError, match="批维固定为 1"):
        rt.run_tiled(src, out, model, tile=128, stride=64, batch=2)
    rt.run_tiled(src, out, model, tile=128, stride=64, batch=1)
    assert np.array_equal(out, src)


@pytest.mark.parametrize("shape", [(96, 96), (8, 12), (8, 13)])
@pytest.mark.parametrize("stitch", ["crop", "blend"])
def test_fixed_batch_tail_is_padded_and_not_placed(tmp_path, shape, stitch):
    ort_or_skip()
    model = rt.OnnxModel(onnx_model(tmp_path / "fixed-two.onnx", input_dims=(2, 64, 64, 3)))
    batches = []

    def checked(x):
        batches.append(x.copy())
        return model(x)

    checked.expects_batch, checked.expects_tile = 2, 64
    src = image(shape)
    out = np.full_like(src, 65535)
    receipt = rt.run_tiled(src, out, checked, tile=64, stride=32, batch=2, stitch=stitch)
    assert all(x.shape == (2, 64, 64, 3) for x in batches)
    # Odd small widths pad to tile+1, giving two genuine tiles, not a tail.
    if shape != (8, 13):
        assert any(np.array_equal(x[0], x[1]) for x in batches)
    assert np.array_equal(out, src)
    assert receipt["tiles"] == {(96, 96): 9, (8, 12): 1, (8, 13): 2}[shape]
    assert receipt["model_calls"] == (6 if shape == (96, 96) else 1)


@pytest.mark.parametrize("exception,state", [(KeyboardInterrupt, "cancelled"), (ValueError, "failed")])
def test_interrupted_provider_records_terminal_partial(tmp_path, exception, state):
    src, sky, protect, original, _ = display_fixture(tmp_path)

    def interrupted(source, output, stage):
        output[:] = 17
        raise exception("fixture interruption")

    root = tmp_path / "jobs"
    with pytest.raises(exception):
        separate_stars(src, root, sky, protect, provider=interrupted, provider_identity={"backend": "fixture"})
    partials = list(root.glob(".partial-separation-*"))
    assert len(partials) == 1 and not list(root.glob("separation-*"))
    record = json.loads((partials[0] / "result.json").read_text())
    assert record["state"] == state and record["artifacts"] == {}
    assert "roundtrip_exact" not in record
    if state == "cancelled":
        assert record["cancel_reason"] == "keyboard_interrupt"
    assert np.array_equal(tifffile.imread(src), original)
    # On Windows this also verifies that finally released the writable mapping.
    output = partials[0] / "diffuse_srgb16.tif"
    moved = output.with_name("closed-map.tif")
    output.rename(moved)
    moved.rename(output)


@pytest.mark.parametrize("options,message", [
    ({"tile": 64, "stride": 65}, "stride"),
    ({"stitch": "bad"}, "stitch"),
    ({"tile": 4096, "stride": 2, "batch": 64}, "memory"),
    ({"intra_op_threads": 0}, "intra_op_threads"),
    ({"layout": "bad"}, "layout"),
    ({"value_range": "0..255"}, "value_range"),
])
def test_invalid_options_rejected_before_model_loading(monkeypatch, options, message):
    def unexpected(*args, **kwargs):
        pytest.fail("model was loaded before preflight")

    monkeypatch.setattr(rt, "OnnxModel", unexpected)
    with pytest.raises(ValueError, match=message):
        rt.starless("absent.tif", "out", "absent-mask", "absent-protect", model_path="absent.onnx", **options)


def test_source_and_full_shape_resources_checked_before_model(tmp_path, monkeypatch):
    from starstack import photographic_job as job
    src, sky, protect, _, _ = display_fixture(tmp_path)
    monkeypatch.setattr(rt, "OnnxModel", lambda *a, **k: pytest.fail("unexpected model load"))
    with pytest.raises(FileNotFoundError):
        rt.starless(tmp_path / "missing.tif", tmp_path / "out", sky, protect, model_path="unused.onnx")
    monkeypatch.setattr(job, "_display_source", lambda *a: ((100000, 100000, 3), b"icc", {}))
    with pytest.raises(ValueError, match="memory"):
        rt.starless(src, tmp_path / "out", sky, protect, model_path="unused.onnx")
    plan = rt.resource_plan(shape=(5504, 8256, 3))
    assert plan["estimated_runtime_working_bytes"] < rt.MAX_RUNTIME_WORKING_BYTES
    assert plan["padded_source_bytes"] > 5504 * 8256 * 3 * 2


@pytest.mark.parametrize("dims,message", [
    (("N", 64, 64), "rank-4"),
    (("N", 64, 64, 4), "3 channels"),
    (("N", 64, 128, 3), "正方形"),
])
def test_graph_shape_rejected_before_inference(tmp_path, dims, message):
    ort_or_skip()
    with pytest.raises(ValueError, match=message):
        rt.OnnxModel(onnx_model(tmp_path / "bad-shape.onnx", input_dims=dims))


@pytest.mark.parametrize("threads", [None, 4])
def test_thread_configuration_is_recorded(tmp_path, threads):
    ort_or_skip()
    model = rt.OnnxModel(onnx_model(tmp_path / "identity.onnx"), intra_op_threads=threads)
    assert model.identity["intra_op_threads_requested"] == threads
    assert model.identity["intra_op_threads"] == (0 if threads is None else threads)
    assert model.identity["intra_op_threads_mode"] == ("onnxruntime-auto" if threads is None else "explicit")
    assert model.identity["session_load_seconds"] >= 0
    assert model.identity["output"]["type"] == "tensor(float)"


def test_cli_negative_range_equals_and_threads(tmp_path, monkeypatch, capsys):
    ort_or_skip()
    from starstack.cli import main
    src, sky, protect, original, _ = display_fixture(tmp_path)
    model = onnx_model(tmp_path / "identity.onnx")
    monkeypatch.setattr(sys, "argv", ["starstack", "starless", "--source", str(src), "--sky-mask", str(sky),
                        "--protected-mask", str(protect), "--model", str(model), "--output", str(tmp_path / "out"),
                        "--tile", "64", "--stride", "32", "--value-range=-1..1", "--threads", "4"])
    assert main() == 0
    record = json.loads(capsys.readouterr().out)
    assert record["backend"]["value_range"] == "-1..1"
    assert record["backend"]["intra_op_threads"] == 4
    output = tifffile.imread(record["artifacts"]["diffuse_srgb16.tif"]["path"])
    assert np.abs(output.astype(np.int32) - original.astype(np.int32)).max() <= 1
