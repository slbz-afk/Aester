"""In-product tiled inference runtime for starless (diffuse) estimation — starless package step 2.

The model is a pluggable callable on float32 NHWC RGB batches in [0, 1]; ``OnnxModel``
wraps an onnxruntime session (imported lazily, optional dependency) and adapts layout
(nhwc / nchw), channel order (rgb / bgr) and the graph's value range (0..1 / -1..1).
nox's frozen graph was trained on RGB in [-1, 1] (nox.py); nox.exe feeds it OpenCV BGR
after x*2-1 and maps (y+1)/2 back, so reproducing the cached nox outputs takes
channel_order="bgr", value_range="-1..1". Tiling follows the input-domain contract of
docs/STARLESS_PACKAGE_20260923.md §二: symmetric reflection padding by
(tile − stride) / 2, the last row/column of tiles shifted inwards, and either nox's
crop-and-place stitching (``crop``) or Hann-weighted overlap blending (``blend``).

Nothing here judges quality: the signed residual and the locked regions stay with
``star_separation.separate_stars``; this module only produces the uint16 estimate.
"""
from __future__ import annotations
from pathlib import Path
import time
import numpy as np
from . import frame_pool, hashing, photographic as p, profiling

VERSION = "starless-runtime-1.1"
BACKEND = "onnx-tiled-1"
# EFF-NOX (docs/EFFICIENCY_P1C_PREREG_20261005.md §4.1, contract K1): a workflow request that runs nox freezes this
# explicit intra-op thread count when it names none; ORT's automatic choice (about 85 cores on the development PC) is
# not bit-identical to it. A frozen None stays the old ORT-auto contract; the explicit value is its own version.
NOX_INTRA_OP_THREADS = 8
# With more intra-op threads than usable processors, idle ORT threads wait instead of spinning (no identity change).
ALLOW_SPINNING = "session.intra_op.allow_spinning"
STITCHES = ("blend", "crop")
LAYOUTS = ("nhwc", "nchw")
CHANNEL_ORDERS = ("rgb", "bgr")
VALUE_RANGES = ("0..1", "-1..1")      # what the graph itself expects; nox's frozen graph takes x*2-1 and returns (y+1)/2
# A guard on this runtime's host buffers, not a promise about ORT/model/VRAM use.
MAX_RUNTIME_WORKING_BYTES = 4 * 1024**3
DOMAIN = {"domain": "display-sRGB", "dtype": "uint16", "channels": "RGB interleaved (adapter may present BGR to the model)",
          "input_scale": "code / 65535 -> float32 [0, 1]", "output": "float32 [0, 1] -> rint(clip * 65535) uint16",
          "value_range": "adapter maps [0, 1] to the graph's declared range (0..1 or -1..1) and back",
          "padding": "symmetric reflection by (tile - stride) / 2 (OpenCV BORDER_REFLECT)",
          "last_tile": "shifted inwards to fit the image", "stitch": {"crop": "central stride x stride region, later tiles overwrite",
                                                                      "blend": "2-D Hann weights (> 0 everywhere), normalised sum"}}


def hann_window(tile):
    """Periodic Hann evaluated at pixel centres: strictly positive, sums to a constant at 50 % overlap."""
    i = (np.arange(tile, dtype=np.float64) + 0.5) / tile
    w = 0.5 - 0.5 * np.cos(2 * np.pi * i)
    return (w[:, None] * w[None, :]).astype(np.float32)


def tile_starts(length, tile, stride):
    """Tile origins along one padded axis: 0, stride, … while origin + tile < length, then length − tile."""
    if length < tile:
        raise ValueError("padded axis shorter than the tile")
    starts = []
    origin = 0
    while origin + tile < length:
        starts.append(origin)
        origin += stride
    starts.append(length - tile)
    return starts


def validate_geometry(tile, stride, batch):
    p.integer_parameter(tile, "tile", 64, 4096)
    p.integer_parameter(stride, "stride", 1, 4096)
    p.integer_parameter(batch, "batch", 1, 64)
    if stride > tile:
        raise ValueError("stride 不能大于 tile")
    if (tile - stride) % 2:
        raise ValueError("tile 与 stride 之差须为偶数（边距 = 差 / 2）")


def resource_plan(*, tile=512, stride=256, stitch="blend", batch=1, shape=None):
    """Preflight known host buffers without allocating image or model tensors.

    The conservative batch allowance covers conversion, adapters and output.
    Model weights/activations, ORT arenas, source/mask storage and VRAM are extra.
    """
    validate_geometry(tile, stride, batch)
    if stitch not in STITCHES:
        raise ValueError("stitch must be one of " + ", ".join(STITCHES))
    batch_bytes = batch * tile * tile * 3 * (5 * 4 + 2)
    padded_bytes = ring_bytes = 0
    window_bytes = tile * tile * 4 if stitch == "blend" else 0
    if shape is not None:
        if len(shape) != 3 or shape[2] != 3 or min(shape[:2]) < 8:
            raise ValueError("source must be uint16 H,W,3 at least 8x8")
        h, w = map(int, shape[:2])
        margin = (tile - stride) // 2
        py, px = max(margin, (tile-h+1)//2), max(margin, (tile-w+1)//2)
        padded_bytes = (h + 2*py) * (w + 2*px) * 3 * 2
        if stitch == "blend":
            ring_bytes = tile * (w + 2*px) * 4 * 4
    total = batch_bytes + padded_bytes + ring_bytes + window_bytes
    if total > MAX_RUNTIME_WORKING_BYTES:
        raise ValueError(f"runtime temporary memory estimate {total} bytes exceeds "
                         f"{MAX_RUNTIME_WORKING_BYTES} bytes; reduce tile/batch or image size")
    return {"estimated_runtime_working_bytes": total, "limit_bytes": MAX_RUNTIME_WORKING_BYTES,
            "batch_buffers_bytes": batch_bytes, "padded_source_bytes": padded_bytes,
            "blend_buffers_bytes": ring_bytes + window_bytes,
            "scope": "runtime host buffers only; excludes model/ORT, source/masks and VRAM"}


def _validate_adapter_options(layout, channel_order, value_range, intra_op_threads):
    if layout not in LAYOUTS or channel_order not in CHANNEL_ORDERS or value_range not in VALUE_RANGES:
        raise ValueError("layout must be nhwc/nchw, channel_order rgb/bgr, value_range 0..1/-1..1")
    if intra_op_threads is not None:
        p.integer_parameter(intra_op_threads, "intra_op_threads", 1, 256)


def run_tiled(source, estimate, model, *, tile=512, stride=256, stitch="blend", batch=1, progress=None):
    """Fill ``estimate`` (writable uint16 H,W,3) from ``source`` (uint16 H,W,3) tile by tile.

    ``model(batch_nhwc_float32) -> nhwc float32`` on [0, 1] RGB tiles. Returns a receipt.
    """
    if not isinstance(source, np.ndarray) or source.dtype != np.uint16 or source.ndim != 3 or source.shape[-1] != 3 or min(source.shape[:2]) < 8:
        raise ValueError("source must be uint16 H,W,3 at least 8x8")
    if (not isinstance(estimate, np.ndarray) or estimate.shape != source.shape or estimate.dtype != np.uint16
            or not estimate.flags.writeable or not estimate.flags.c_contiguous):
        raise ValueError("writable contiguous uint16 estimate of the source shape required")
    if np.shares_memory(source, estimate):
        raise ValueError("estimate aliases source")
    if not callable(model):
        raise ValueError("model must be callable")
    resources = resource_plan(tile=tile, stride=stride, stitch=stitch, batch=batch, shape=source.shape)
    expects = getattr(model, "expects_tile", None)
    if expects is not None and expects != tile:
        raise ValueError(f"模型输入尺寸固定为 {expects}，与 tile {tile} 不符")
    expects_batch = getattr(model, "expects_batch", None)
    if expects_batch is not None and batch != expects_batch:
        raise ValueError(f"模型批维固定为 {expects_batch}，请用 batch={expects_batch}")
    started = time.perf_counter()
    h, w = source.shape[:2]
    margin = (tile - stride) // 2
    pad_y = max(margin, -(-(tile - h) // 2))     # small images: pad up to at least one tile
    pad_x = max(margin, -(-(tile - w) // 2))
    padded = np.pad(source, ((pad_y, pad_y), (pad_x, pad_x), (0, 0)), mode="symmetric")
    hp, wp = padded.shape[:2]
    ys, xs = tile_starts(hp, tile, stride), tile_starts(wp, tile, stride)
    window = hann_window(tile) if stitch == "blend" else None
    tiles = 0
    calls = 0
    if stitch == "crop":
        for py in ys:
            for x0 in range(0, len(xs), batch):
                block = [padded[py:py + tile, px:px + tile] for px in xs[x0:x0 + batch]]
                out = _infer(model, block, tile)
                calls += 1
                for k, px in enumerate(xs[x0:x0 + batch]):
                    tiles += 1
                    centre = out[k, margin:tile - margin, margin:tile - margin]
                    y_dst, x_dst = py + margin - pad_y, px + margin - pad_x       # unpadded coordinates
                    _place(estimate, centre, y_dst, x_dst)
            if progress:
                progress(tiles, len(ys) * len(xs))
    else:
        acc = np.zeros((tile, wp, 3), np.float32)
        wacc = np.zeros((tile, wp), np.float32)
        row0 = ys[0]
        for py in ys:
            shift = py - row0
            if shift > 0:
                _emit(estimate, acc[:shift], wacc[:shift], row0, pad_y, pad_x, h, w)
                acc[:tile - shift] = acc[shift:]
                acc[tile - shift:] = 0
                wacc[:tile - shift] = wacc[shift:]
                wacc[tile - shift:] = 0
                row0 = py
            for x0 in range(0, len(xs), batch):
                block = [padded[py:py + tile, px:px + tile] for px in xs[x0:x0 + batch]]
                out = _infer(model, block, tile)
                calls += 1
                for k, px in enumerate(xs[x0:x0 + batch]):
                    tiles += 1
                    acc[:, px:px + tile] += out[k] * window[..., None]
                    wacc[:, px:px + tile] += window
            if progress:
                progress(tiles, len(ys) * len(xs))
        _emit(estimate, acc, wacc, row0, pad_y, pad_x, h, w)
    seconds = time.perf_counter() - started
    megapixels = h * w / 1e6
    receipt = {"version": VERSION, "tile": tile, "stride": stride, "margin": margin, "stitch": stitch, "batch": batch,
               "shape": [h, w, 3], "padded_shape": [hp, wp, 3], "padding": [pad_y, pad_x], "tiles": tiles, "model_calls": calls,
               "seconds": seconds, "megapixels": megapixels, "megapixels_per_second": (megapixels / seconds) if seconds > 0 else None,
               "domain": DOMAIN["domain"], "input_scale": DOMAIN["input_scale"],
               "resources": resources, "fixed_batch_tail_policy": "repeat-last-tile-and-discard-padding"}
    identity = getattr(model, "identity", None)
    if isinstance(identity, dict):
        receipt["model"] = identity
    return receipt


def _infer(model, block, tile):
    valid_count = len(block)
    fixed_batch = getattr(model, "expects_batch", None)
    if fixed_batch is not None and valid_count < fixed_batch:
        block = block + [block[-1]] * (fixed_batch - valid_count)
    x = np.stack(block).astype(np.float32) / np.float32(65535)
    out = model(x)
    if not isinstance(out, np.ndarray) or out.shape != x.shape or out.dtype != np.float32:
        raise ValueError("model must return float32 NHWC of the input shape")
    if not np.isfinite(out).all():
        raise ValueError("model produced non-finite values")
    return out[:valid_count]


def _place(estimate, centre, y_dst, x_dst):
    h, w = estimate.shape[:2]
    y0, x0 = max(0, y_dst), max(0, x_dst)
    y1, x1 = min(h, y_dst + centre.shape[0]), min(w, x_dst + centre.shape[1])
    if y1 <= y0 or x1 <= x0:
        return
    piece = centre[y0 - y_dst:y1 - y_dst, x0 - x_dst:x1 - x_dst]
    estimate[y0:y1, x0:x1] = np.rint(np.clip(piece, 0, 1) * 65535).astype(np.uint16)


def _emit(estimate, acc, wacc, row0, pad_y, pad_x, h, w):
    """Finalise accumulated padded rows [row0, row0 + len(acc)) into the unpadded estimate."""
    for k in range(acc.shape[0]):
        y = row0 + k - pad_y
        if 0 <= y < h:
            values = acc[k, pad_x:pad_x + w] / wacc[k, pad_x:pad_x + w][:, None]
            estimate[y] = np.rint(np.clip(values, 0, 1) * 65535).astype(np.uint16)


class OnnxModel:
    """onnxruntime adapter: NHWC RGB float32 in and out, whatever the graph's own layout / order."""

    def __init__(self, path, *, layout="nhwc", channel_order="rgb", value_range="0..1", providers=None, input_name=None,
                 output_name=None, intra_op_threads=None):
        _validate_adapter_options(layout, channel_order, value_range, intra_op_threads)
        try:
            import onnxruntime as ort
        except ImportError as exc:          # optional dependency, see THIRD_PARTY.md 可选组件
            raise RuntimeError("onnxruntime 未安装：pip install onnxruntime（可选组件，不在试用包内）") from exc
        self.path = Path(path).resolve(strict=True)
        model_sha256 = hashing.sha256_file(self.path, source="starless_runtime.model")
        options = ort.SessionOptions()
        spinning = "default"
        from .frame_pool import usable_cpus
        cpus = usable_cpus()
        if intra_op_threads is not None:
            options.intra_op_num_threads = intra_op_threads
            if intra_op_threads > cpus:
                # How idle threads wait, not what they compute: kept out of the identity below.
                options.add_session_config_entry(ALLOW_SPINNING, "0")
                spinning = "0"
        # Timings notes only (never a record, an identity or a digest).
        profiling.note("nox_threads", {"intra_op_threads": "ort_auto" if intra_op_threads is None else intra_op_threads,
                                       "usable_cpus": cpus, "allow_spinning": spinning})
        load_started = time.perf_counter()
        self.session = ort.InferenceSession(str(self.path), options, providers=list(providers) if providers else None)
        load_seconds = time.perf_counter() - load_started
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        names_in, names_out = [i.name for i in inputs], [o.name for o in outputs]
        if len(inputs) != 1:
            raise ValueError("model must have exactly one required image input; got: " + ", ".join(names_in))
        if input_name is None:
            input_name = names_in[0]
        if output_name is None:
            if len(outputs) != 1:
                raise ValueError("模型有多个输出，请用 output_name 指定：" + ", ".join(names_out))
            output_name = names_out[0]
        if input_name not in names_in or output_name not in names_out:
            raise ValueError("input_name / output_name 不在模型里")
        self.input_name, self.output_name = input_name, output_name
        self.layout, self.channel_order, self.value_range = layout, channel_order, value_range
        input_meta = inputs[0]
        output_meta = next(o for o in outputs if o.name == output_name)
        shape = self._image_shape(input_meta, layout, "input")
        output_shape = self._image_shape(output_meta, layout, "output")
        for a, b in zip(shape, output_shape):
            if isinstance(a, int) and isinstance(b, int) and a != b:
                raise ValueError(f"model input/output fixed dimensions differ: {shape} vs {output_shape}")
        spatial = shape[1:3] if layout == "nhwc" else shape[2:4]
        fixed = [d for d in spatial if isinstance(d, int)]
        if fixed and len(set(fixed)) > 1:
            raise ValueError(f"模型固定输入不是正方形：{shape}")
        self.expects_tile = fixed[0] if fixed else None
        self.expects_batch = shape[0] if shape and isinstance(shape[0], int) else None
        self.identity = {"backend": BACKEND, "version": VERSION, "uses_learned_model": True, "model_path": str(self.path),
                         "model_sha256": model_sha256, "model_bytes": self.path.stat().st_size,
                         "onnxruntime": ort.__version__, "providers": list(self.session.get_providers()),
                         "intra_op_threads_requested": intra_op_threads,
                         "intra_op_threads": options.intra_op_num_threads,
                         "intra_op_threads_mode": "onnxruntime-auto" if options.intra_op_num_threads == 0 else "explicit",
                         "inter_op_threads": options.inter_op_num_threads,
                         "session_load_seconds": load_seconds,
                         "input": {"name": input_name, "shape": [d if isinstance(d, int) else str(d) for d in shape],
                                   "type": input_meta.type},
                         "output": {"name": output_name, "shape": [d if isinstance(d, int) else str(d) for d in
                                                                    output_shape], "type": output_meta.type},
                         "layout": layout, "channel_order": channel_order, "input_scale": DOMAIN["input_scale"],
                         "value_range": value_range, "expects_tile": self.expects_tile, "expects_batch": self.expects_batch}

    @staticmethod
    def _image_shape(metadata, layout, label):
        shape = metadata.shape
        if metadata.type != "tensor(float)" or not isinstance(shape, (list, tuple)) or len(shape) != 4:
            raise ValueError(f"model {label} must be a rank-4 float32 image tensor")
        if any(isinstance(d, int) and d <= 0 for d in shape):
            raise ValueError(f"model {label} has a non-positive fixed dimension: {shape}")
        channel = shape[3 if layout == "nhwc" else 1]
        if isinstance(channel, int) and channel != 3:
            raise ValueError(f"model {label} must have 3 channels in declared layout {layout}: {shape}")
        return shape

    def __call__(self, x):
        y = x[..., ::-1] if self.channel_order == "bgr" else x
        if self.value_range == "-1..1":
            y = y * np.float32(2) - np.float32(1)
        if self.layout == "nchw":
            y = np.transpose(y, (0, 3, 1, 2))
        out = self.session.run([self.output_name], {self.input_name: np.ascontiguousarray(y, np.float32)})[0]
        if not isinstance(out, np.ndarray) or out.shape != y.shape or out.dtype != np.float32:
            raise ValueError("model must return a float32 tensor matching the input shape/layout")
        if self.layout == "nchw":
            out = np.transpose(out, (0, 2, 3, 1))
        if self.value_range == "-1..1":
            out = (out + np.float32(1)) * np.float32(0.5)
        if self.channel_order == "bgr":
            out = out[..., ::-1]
        return np.ascontiguousarray(out)


def provider(model, *, tile=512, stride=256, stitch="blend", batch=1):
    """(callable, identity) for star_separation.separate_stars."""
    resource_plan(tile=tile, stride=stride, stitch=stitch, batch=batch)
    base = getattr(model, "identity", None)
    if not isinstance(base, dict) or not base.get("model_sha256"):
        raise ValueError("model must carry an identity with model_sha256")
    identity = {**base, "tile": tile, "stride": stride, "stitch": stitch, "batch": batch}

    def estimate(source_view, output, stage_dir):
        # P1c §2.2: while the ORT threads run, a frame pool started anywhere in this process runs serially
        with frame_pool.exclusive("nox"):
            return run_tiled(source_view, output, model, tile=tile, stride=stride, stitch=stitch, batch=batch)
    return estimate, identity


def starless(source, output_root, sky_mask, protected_mask, *, model_path, tile=512, stride=256, stitch="blend", batch=1,
             layout="nhwc", channel_order="rgb", value_range="0..1", providers=None, transient_background=None, intra_op_threads=None):
    """CLI entry: run an ONNX starless model through separate_stars into a new separation directory."""
    from .star_separation import separate_stars
    from . import photographic_job as job
    resource_plan(tile=tile, stride=stride, stitch=stitch, batch=batch)
    _validate_adapter_options(layout, channel_order, value_range, intra_op_threads)
    # Cheap path/header checks precede loading weights. separate_stars still
    # validates masks, binding and full input hashes immediately before use.
    shape, _, _ = job._display_source(Path(source).resolve(strict=True))
    resource_plan(tile=tile, stride=stride, stitch=stitch, batch=batch, shape=shape)
    for path in (sky_mask, protected_mask, transient_background):
        if path is not None:
            Path(path).resolve(strict=True)
    model = OnnxModel(model_path, layout=layout, channel_order=channel_order, value_range=value_range, providers=providers,
                      intra_op_threads=intra_op_threads)
    estimate, identity = provider(model, tile=tile, stride=stride, stitch=stitch, batch=batch)
    return separate_stars(source, output_root, sky_mask, protected_mask, provider=estimate, provider_identity=identity,
                          transient_background=transient_background)
