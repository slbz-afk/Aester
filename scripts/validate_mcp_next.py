"""Exercise reference preparation and display rendering through real MCP stdio.

Uses the existing numerical FITS fixture and a completed synthetic job. This
checks integration, dimensions and preservation, not camera RAW image quality.
Only new reference/variation artifacts and docs/diagnostics/mcp-next.json are
written. The Windows validation subtree is owned by a bounded Job Object.
"""
from __future__ import annotations

import argparse
import asyncio
import ctypes
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "docs" / "diagnostics" / "mcp-next.json"
sys.path.insert(0, str(ROOT / "src"))


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(report: dict) -> None:
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    RECORD.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def in_windows_job() -> bool:
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    kernel.IsProcessInJob.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    kernel.IsProcessInJob.restype = ctypes.c_int
    answer = ctypes.c_int()
    if not kernel.IsProcessInJob(kernel.GetCurrentProcess(), None, ctypes.byref(answer)):
        raise ctypes.WinError(ctypes.get_last_error())
    return bool(answer.value)


async def exercise(report: dict) -> None:
    # Load numerical libraries before the SDK starts its stdio worker threads.
    import numpy as np
    from PIL import Image
    import tifffile
    from astropy.io import fits
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    inputs = ROOT / "data" / "demo" / "synthetic-fbd621c4"
    directory = ROOT / "outputs" / "job-795bd89ece753f024bce52a2"
    job_path = directory / "job.json"
    job = json.loads(job_path.read_text(encoding="utf-8"))
    assert job["state"] == "succeeded"
    files = sorted(inputs.glob("*.fit"))
    assert len(files) == 8
    artifacts = []
    for item in job["artifacts"]:
        path = Path(item["path"])
        if not path.is_absolute():
            path = directory / path
        assert path.resolve().is_relative_to(directory.resolve())
        artifacts.append(path)
    masters = [path for path in artifacts if path.name == "master_linear.fit"]
    assert len(masters) == 1
    master = masters[0]
    originals = {str(path): sha(path) for path in [*files, inputs / "sky_mask.png", job_path, *artifacts]}
    report.update(input_dir=str(inputs), job_dir=str(directory), originals_sha256=originals)
    report["client_in_owned_windows_job"] = in_windows_job()
    assert report["client_in_owned_windows_job"]
    save(report)
    env = dict(os.environ)
    env.update(PYTHONPATH=str(ROOT / "src"), STARSTACK_INPUT_ROOT=str(ROOT / "data"),
               STARSTACK_OUTPUT_ROOT=str(ROOT / "outputs"),
               STARSTACK_SIRIL=str(ROOT / "tools" / "siril" / "bin" / "siril-cli.exe"))
    parameters = StdioServerParameters(command=sys.executable, args=["-m", "starstack.mcp_server"],
                                       env=env, cwd=str(ROOT))
    async with stdio_client(parameters) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            async def call(name: str, arguments: dict) -> dict:
                result = await session.call_tool(name, arguments)
                if result.isError or result.structuredContent is None:
                    raise RuntimeError(f"{name}: {result.content}")
                return result.structuredContent

            initialized = await session.initialize()
            report["protocol_version"] = initialized.protocolVersion
            tools = {tool.name: tool for tool in (await session.list_tools()).tools}
            report["tools"] = list(tools)
            assert {"prepare_reference", "render_job_preview"} <= tools.keys()
            assert set(tools["prepare_reference"].inputSchema["properties"]) == {"input_dir"}
            assert set(tools["render_job_preview"].inputSchema["properties"]) == {
                "job_dir", "red_gain", "blue_gain", "exposure_ev"}
            report["checks"].append("official SDK initialize/list_tools includes reference and preview tools; roots and Siril are host-only")

            began = time.monotonic()
            reference = await call("prepare_reference", {"input_dir": "demo/synthetic-fbd621c4"})
            report.update(reference=reference, reference_elapsed_seconds=round(time.monotonic() - began, 3))
            save(report)
            ref_dir = Path(reference["reference_dir"])
            assert ref_dir.parent == ROOT / "outputs" and ref_dir.name.startswith("reference-")
            assert reference["state"] == "succeeded" and reference["width"] == 768 and reference["height"] == 512
            assert reference["no_crop"] and reference["no_scale"]
            assert reference["linear_row_order"] == "BOTTOM-UP" and reference["preview_row_order"] == "TOP-DOWN"
            assert Path(reference["source_path"]) == files[0]
            for key, hash_key in (("source_path", "source_sha256"), ("staged_path", "staged_sha256"),
                                  ("linear_path", "linear_sha256"), ("preview_path", "preview_sha256"),
                                  ("mask_template_path", "mask_template_sha256")):
                path = Path(reference[key])
                assert key == "source_path" or path.resolve().is_relative_to(ref_dir.resolve())
                assert sha(path) == reference[hash_key]
            assert reference["source_sha256"] == reference["staged_sha256"] == originals[str(files[0])]
            assert len(list((ref_dir / "stage").iterdir())) == 1
            with Image.open(reference["preview_path"]) as picture:
                assert picture.format == "PNG" and picture.size == (768, 512) and picture.info.get("icc_profile")
            with Image.open(reference["mask_template_path"]) as picture:
                assert picture.size == (768, 512) and picture.mode == "L" and picture.getextrema() == (255, 255)
            assert reference["metadata"]["mask"]["is_segmentation"] is False
            with fits.open(reference["linear_path"], memmap=False) as hdus:
                assert hdus[0].data.shape == (3, 512, 768)
                # Siril may omit this optional header; the established FITS
                # contract uses BOTTOM-UP when ROWORDER is absent.
                report["reference_fits_roworder_header"] = hdus[0].header.get("ROWORDER")
                assert hdus[0].header.get("ROWORDER", "BOTTOM-UP") == "BOTTOM-UP"
            report["checks"].append("real Siril first-frame FITS conversion: full 768x512 PNG, all-white editing template, hashes and geometry")

            gains = {"red_gain": 1.2, "blue_gain": .9, "exposure_ev": .35}
            began = time.monotonic()
            rendered = await call("render_job_preview", {"job_dir": directory.name, **gains})
            report.update(rendered=rendered, render_elapsed_seconds=round(time.monotonic() - began, 3))
            save(report)
            output = Path(rendered["directory"])
            assert output.parent == directory / "preview-variations"
            assert rendered["state"] == "succeeded" and rendered["parameters"] == gains
            assert rendered["source_sha256"] == originals[str(master)]
            assert rendered["job_sha256"] == originals[str(job_path)]
            assert rendered["color_calibrated"] is False and rendered["linear_master_modified"] is False
            assert json.loads(Path(rendered["recipe_path"]).read_text(encoding="utf-8")) == rendered
            items = rendered["artifacts"]
            assert len(items) == 4 and sum(item["kind"] == "jpeg" for item in items) == 3
            for item in items:
                path = Path(item["path"])
                assert path.parent == output and sha(path) == item["sha256"]
                if item["kind"] == "jpeg":
                    with Image.open(path) as picture:
                        assert picture.format == "JPEG" and picture.size == (768, 512)
                else:
                    pixels = tifffile.imread(path)
                    assert pixels.shape == (512, 768, 3) and pixels.dtype == np.uint16
            report["checks"].append("real renderer through SDK: independent recipe, three full-size JPEGs and uint16 display TIFF")

            variants_before = sorted(path.name for path in output.parent.iterdir())
            for bad_value in (True, "1.25", .1, 5):
                rejected = await session.call_tool("render_job_preview", {"job_dir": directory.name, "red_gain": bad_value})
                assert rejected.isError
            rejected = await session.call_tool("prepare_reference", {"input_dir": "../outside"})
            assert rejected.isError
            assert variants_before == sorted(path.name for path in output.parent.iterdir())
            report["checks"].append("SDK rejects bool/string/out-of-range gains and input traversal without creating another variation")

    report["originals_unchanged"] = all(sha(Path(path)) == digest for path, digest in originals.items())
    assert report["originals_unchanged"]
    report["checks"].append("eight FITS inputs, mask, original job.json and every original artifact SHA256 unchanged")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--inside-host-job", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if os.name != "nt" or not 10 <= args.timeout <= 300:
        parser.error("Requires Windows; timeout must be 10 to 300 seconds")
    if not args.inside_host_job:
        from starstack.process_guard import guarded_process

        command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--inside-host-job"]
        try:
            with guarded_process(command, stdin=subprocess.DEVNULL, stdout=sys.stdout, stderr=sys.stderr,
                                 cwd=str(ROOT), creationflags=subprocess.CREATE_NO_WINDOW) as child:
                return child.wait(timeout=args.timeout + 40)
        except BaseException as error:
            previous = json.loads(RECORD.read_text(encoding="utf-8")) if RECORD.exists() else {}
            previous.update(acceptance_passed=False, validation_host_error=repr(error),
                            cleanup="closed only the owned validation process subtree")
            save(previous)
            raise

    report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "sdk": f"mcp {version('mcp')}",
              "material": "eight numerical synthetic RGB FITS; not camera RAW or a photographic quality comparison",
              "checks": [], "scope": "real MCP stdio reference preparation and display rendering, normal completion",
              "source_sha256": {name: sha(ROOT / name) for name in (
                  "src/starstack/mcp_server.py", "src/starstack/references.py", "src/starstack/preview.py",
                  "src/starstack/imaging.py", "scripts/validate_mcp_next.py")}}
    began, passed = time.monotonic(), False
    try:
        asyncio.run(asyncio.wait_for(exercise(report), timeout=args.timeout))
        passed = True
        return 0
    except BaseException as error:
        report["run_error"] = repr(error)
        return 1
    finally:
        report.update(acceptance_passed=passed, elapsed_seconds=round(time.monotonic() - began, 3))
        save(report)
        print(json.dumps({"acceptance_passed": passed, "record": str(RECORD),
                          "run_error": report.get("run_error")}, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
