"""Host-side validation and lifecycle for fixed, optional GIMP execution."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

from .finishing_assets import (ordinary_path, inspect_source, verify_source,
                               recipe_source, inspect_mask, sha256_file)
from .finishing_recipe import normalize_recipe, recipe_digest, recipe_mask_ids
from .process_guard import guarded_process, guard_kind, GUARD_JOB_OBJECT
from .records import atomic_json


class RenderCancelled(RuntimeError):
    pass


def _format_supported(path):
    """Match the first fixed driver's decoder domain before starting GIMP.

    Registration is backend-independent and can admit more formats. Render
    rejection is explicit; there is no silent re-encoding of coverage weights.
    """
    import tifffile
    path = ordinary_path(path)
    if path.suffix.lower() in {".tif", ".tiff"}:
        with tifffile.TiffFile(path) as tf:
            page = tf.pages[0]
            if (tf.is_bigtiff or page.is_tiled or int(page.compression) != 1
                    or int(page.fillorder) != 1):
                raise ValueError("GIMP adapter v1 requires classic uncompressed strip TIFF with FillOrder=1")
    elif path.suffix.lower() == ".png":
        with path.open("rb") as handle:
            header = handle.read(29)
        if len(header) != 29 or header[28] != 0:
            raise ValueError("GIMP adapter v1 requires non-interlaced grayscale PNG masks")


def _inputs(source, recipe, masks, bundles=None):
    normalized = normalize_recipe(recipe)
    verify_source(source)
    if normalized["source"] != recipe_source(source):
        raise ValueError("Render recipe belongs to a different source")
    _format_supported(source["path"])
    used = recipe_mask_ids(normalized)
    if type(masks) is not dict or set(masks) != used:
        raise ValueError("Render request must contain exactly its registered masks")
    for record in masks.values():
        actual = inspect_mask(record["path"], source, source_sha256=record["source_sha256"])
        if actual != record:
            raise ValueError("Render mask changed after registration")
        _format_supported(record["path"])
    layer_ops = [op for op in normalized["operations"] if op["type"] == "astrophotographic_layers"]
    bundles = {} if bundles is None else bundles
    if type(bundles) is not dict or set(bundles) != {op["params"]["bundle_id"] for op in layer_ops}:
        raise ValueError("Render request must contain exactly its registered layer bundles")
    if layer_ops:
        from .finishing_layers import verify_render_bundle
        for op in layer_ops:
            bundle = bundles[op["params"]["bundle_id"]]
            if bundle.get("bundle_sha256") != op["params"]["bundle_sha256"]:
                raise ValueError("Layer bundle digest differs from recipe")
            verify_render_bundle(source, bundle)
    return normalized


def _product_file(directory, name):
    path = ordinary_path(directory / name)
    if path.parent != directory or path.stat().st_size == 0:
        raise ValueError("Renderer returned an invalid product")
    return path


def verify_products(directory, source, recipe) -> dict:
    """Validate complete products independently of the GIMP exit status."""
    import numpy as np
    import tifffile
    from PIL import Image
    from .photographic_job import _validate_srgb

    directory = ordinary_path(directory, directory=True)
    receipt_path = _product_file(directory, "receipt.json")
    if receipt_path.stat().st_size > 1024 * 1024:
        raise ValueError("GIMP receipt is too large")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    expected = {"schema": "aester-finishing-gimp-receipt-1", "ok": True,
                "source_sha256": source["sha256"], "recipe_sha256": recipe_digest(recipe),
                "width": source["width"], "height": source["height"],
                "applied_operations": [op["id"] for op in recipe["operations"] if op["enabled"]]}
    if not isinstance(receipt, dict) or any(receipt.get(key) != item for key, item in expected.items()):
        raise ValueError("GIMP receipt does not match the requested render")
    if receipt["ok"] is not True or not isinstance(receipt.get("gimp_version"), str):
        raise ValueError("GIMP receipt is missing execution identity")
    tiff = _product_file(directory, "result.tif")
    jpeg = _product_file(directory, "result.jpg")
    xcf = _product_file(directory, "result.xcf")
    replay = _product_file(directory, "replay.tif")
    rendered = inspect_source(tiff, encoding="srgb16")
    if (rendered["width"], rendered["height"]) != (source["width"], source["height"]):
        raise ValueError("Rendered TIFF dimensions changed")
    if rendered["icc_sha256"] != source["icc_sha256"]:
        raise ValueError("Rendered TIFF ICC changed")
    # Decode, not just parse headers: incomplete/corrupt compressed pixels fail.
    pixels = tifffile.imread(tiff)
    if not expected["applied_operations"] and not np.array_equal(pixels, tifffile.imread(source["path"])):
        raise ValueError("Empty recipe changed source pixels")
    replay_record = inspect_source(replay, encoding="srgb16")
    if replay_record["icc_sha256"] != source["icc_sha256"]:
        raise ValueError("XCF replay ICC changed")
    if not np.array_equal(pixels, tifffile.imread(replay)):
        raise ValueError("XCF replay changed rendered pixels")
    pixel_hash = hashlib.sha256(memoryview(np.ascontiguousarray(pixels)).cast("B")).hexdigest()
    del pixels
    with Image.open(jpeg) as im:
        if im.format != "JPEG" or im.mode != "RGB" or im.size != (source["width"], source["height"]):
            raise ValueError("Rendered JPEG format or dimensions changed")
        _validate_srgb(im.info.get("icc_profile", b""))
        im.load()
    with xcf.open("rb") as stream:
        if not stream.read(14).startswith(b"gimp xcf "):
            raise ValueError("Renderer did not return an XCF file")
    return {"receipt": receipt, "tiff": rendered, "pixel_sha256": pixel_hash,
            "replay_tiff": replay_record,
            "checks": {"tiff_decoded": True, "xcf_replay_pixels_equal": True},
            "jpeg": {"path": str(jpeg), "sha256": sha256_file(jpeg), "bytes": jpeg.stat().st_size},
            "xcf": {"path": str(xcf), "sha256": sha256_file(xcf), "bytes": xcf.stat().st_size}}


def render_recipe(source, recipe, masks, work_dir, *, gimp_path, timeout=600, cancel=None, bundles=None) -> dict:
    """Render into a fresh private directory. This does not publish a candidate.

    The executable is host configuration, never a recipe argument in public
    Agent tools. Windows job ownership covers launcher, GIMP and plugin children.
    """
    if type(timeout) not in {int, float} or not math.isfinite(timeout) or not 0 < timeout <= 3600:
        raise ValueError("Render timeout must be between zero and one hour")
    normalized = _inputs(source, recipe, masks, bundles)
    executable = ordinary_path(gimp_path)
    work = ordinary_path(work_dir, directory=True)
    if any(work.iterdir()):
        raise ValueError("Render working directory must be empty")
    if guard_kind() != GUARD_JOB_OBJECT:
        raise RuntimeError("NL-1 GIMP process lifecycle is currently validated for Windows only")
    if cancel is not None and cancel():
        raise RenderCancelled("Render cancelled before launch")
    driver = Path(__file__).with_name("finishing_gimp_script.py")
    if not driver.is_file():
        raise RuntimeError("The fixed GIMP driver has not been installed")
    artifacts = work / "artifacts"
    artifacts.mkdir()
    started = time.monotonic()
    try:
        host_stage = None
        layer_ops = [op for op in normalized["operations"] if op["type"] == "astrophotographic_layers"]
        if layer_ops and layer_ops[0]["enabled"]:
            from .finishing_layer_engine import render_layer_stage, StageCancelled
            import numpy as np
            import tifffile
            stage_dir = work / "stages"
            stage_dir.mkdir()
            stage_path = stage_dir / "layers.tif"
            op = layer_ops[0]
            try:
                details = render_layer_stage(source, bundles[op["params"]["bundle_id"]],
                    op["params"], masks, stage_path, cancel=cancel, deadline=started + timeout)
            except StageCancelled as exc:
                raise RenderCancelled(str(exc)) from exc
            stage_image = inspect_source(stage_path, encoding="srgb16")
            if any(stage_image[k] != source[k] for k in ("width", "height", "encoding", "icc_sha256")):
                raise ValueError("Host stage changed geometry or color encoding")
            pixels = tifffile.memmap(stage_path, mode="r")
            digest = hashlib.sha256()
            try:
                for y in range(0, source["height"], 128):
                    if cancel is not None and cancel():
                        raise RenderCancelled("Cancelled during host-stage verification")
                    digest.update(np.ascontiguousarray(pixels[y:y+128], dtype="<u2").tobytes())
            finally:
                pixels._mmap.close()
            host_stage = {"source_sha256": source["sha256"], "recipe_sha256": recipe_digest(normalized),
                "bundle_sha256": op["params"]["bundle_sha256"], "implementation": details["implementation"],
                "image": stage_image, "pixel_sha256": digest.hexdigest(), "details": details}
            atomic_json(stage_dir / "stage.json", host_stage)
        _inputs(source, normalized, masks, bundles)
        if cancel is not None and cancel():
            raise RenderCancelled("Render cancelled before GIMP launch")
        if time.monotonic() >= started + timeout:
            raise TimeoutError("Host-stage preparation exceeded render timeout")
        request = {"schema": "aester-finishing-render-2" if layer_ops else "aester-finishing-render-1",
                   "source": source, "recipe": normalized, "masks": masks, "output_dir": str(artifacts)}
        if layer_ops:
            request["host_stage"] = host_stage
        request_path = work / "request.json"
        atomic_json(request_path, request)
        with (work / "stdout.log").open("wb") as stdout, (work / "stderr.log").open("wb") as stderr:
            command = [sys.executable, "-I", str(Path(__file__).with_name("finishing_launch.py")),
                       str(executable), str(request_path)]
            with guarded_process(command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                 cwd=str(work), creationflags=subprocess.CREATE_NO_WINDOW) as process:
                while process.poll() is None:
                    if cancel is not None and cancel():
                        process.kill()
                        raise RenderCancelled("Render cancelled")
                    if time.monotonic() - started >= timeout:
                        process.kill()
                        raise TimeoutError("GIMP render exceeded its timeout")
                    time.sleep(0.1)
                if process.returncode != 0:
                    raise RuntimeError(f"GIMP launcher exited with status {process.returncode}")
        if cancel is not None and cancel():
            raise RenderCancelled("Render cancelled before validation")
        _inputs(source, normalized, masks, bundles)
        if host_stage is not None:
            verify_source(host_stage["image"])
        products = verify_products(artifacts, source, normalized)
        if layer_ops:
            reported = products["receipt"].get("host_stage")
            expected_stage = None if host_stage is None else {key: host_stage[key] for key in
                ("source_sha256", "recipe_sha256", "bundle_sha256", "implementation", "pixel_sha256")}
            if reported != expected_stage:
                raise ValueError("GIMP did not verify the exact host stage")
            enabled = [op for op in normalized["operations"] if op["enabled"]]
            if len(enabled) == 1 and enabled[0]["type"] == "astrophotographic_layers":
                if products["pixel_sha256"] != host_stage["pixel_sha256"]:
                    raise ValueError("GIMP stage import changed pixels")
        if cancel is not None and cancel():
            raise RenderCancelled("Render cancelled during product validation")
        if time.monotonic() >= started + timeout:
            raise TimeoutError("Render validation exceeded its timeout")
        result = {"schema": "aester-finishing-execution-1", "status": "validated",
                  "seconds": time.monotonic() - started, "products": products}
        if layer_ops:
            result.update(host_stage=host_stage, layer_bundles=bundles,
                          editability="Cumulative snapshots; parameter rerun requires recipe and retained layer bundle")
        atomic_json(work / "execution.json", result)
        return result
    except Exception as exc:
        atomic_json(work / "execution.json", {"schema": "aester-finishing-execution-1",
                    "status": "cancelled" if isinstance(exc, RenderCancelled) else "failed",
                    "seconds": time.monotonic() - started, "error": str(exc)})
        raise
