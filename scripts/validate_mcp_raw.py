"""Bounded real MCP stdio acceptance for RAW audit and manual trace evidence.

Reads the existing numerical DNG and keeps a new host-owned output directory.
No photo or production source is changed. An owned Windows Job Object bounds
the complete client/server subtree, including cancellation that stalls in DLLs.
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
import uuid

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "docs" / "diagnostics" / "mcp-raw-20260916.json"
INPUT = ROOT / "outputs" / "raw-foundation-20260916"
SOURCE = INPUT / "synthetic-rational.dng"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from development_assets import INPUT_DIR


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
    import numpy as np
    # Keep numerical DLL initialization before SDK stdio starts reader threads.
    import rawpy
    from astropy.io import fits
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    output = INPUT / f"mcp-{uuid.uuid4().hex}"
    output.mkdir()
    assert SOURCE.is_file()
    originals = {str(SOURCE): sha(SOURCE)}
    report.update(host_input_root=str(INPUT), host_output_root=str(output), source_path=str(SOURCE),
                  original_sha256=originals, client_in_owned_windows_job=in_windows_job(),
                  rawpy_version=rawpy.__version__, libraw_version=list(rawpy.libraw_version))
    assert report["client_in_owned_windows_job"]
    known = (np.arange(96 * 128).reshape(96, 128) % 1024).astype(np.uint16)
    expected_signal = known.astype(np.float32) - np.tile(np.array([[10.25, 9.75], [10.75, 10]], dtype=np.float32), (48, 64))
    report["known_fixture"] = {"shape": [96, 128], "codes": "row-major arange(96*128) modulo 1024, uint16",
                               "black_tile": [[10.25, 9.75], [10.75, 10.0]], "white_level": 1000}
    env = dict(os.environ)
    env.update(PYTHONPATH=str(ROOT / "src"), STARSTACK_INPUT_ROOT=str(INPUT), STARSTACK_OUTPUT_ROOT=str(output))
    params = StdioServerParameters(command=sys.executable, args=["-m", "starstack.mcp_server"], env=env, cwd=str(ROOT))
    save(report)
    async with stdio_client(params) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            async def call(name: str, arguments: dict) -> dict:
                result = await session.call_tool(name, arguments)
                if result.isError or result.structuredContent is None:
                    raise RuntimeError(f"{name}: {result.content}")
                return result.structuredContent

            initialized = await session.initialize()
            report["protocol_version"] = initialized.protocolVersion
            available = {item.name: item for item in (await session.list_tools()).tools}
            report["tools"] = list(available)
            assert len(available) == 10 and {"inspect_raw", "protect_trace"} <= available.keys()
            properties = {name: sorted(tool.inputSchema.get("properties", {})) for name, tool in available.items()}
            report["tool_argument_names"] = properties
            assert properties["inspect_raw"] == ["source_path"]
            assert properties["protect_trace"] == ["audit_dir", "classification", "note", "rectangle"]
            assert not any({"input_root", "output_root", "siril_path"} & set(names) for names in properties.values())
            report["checks"].append("official SDK stdio handshake and ten tools; root/executable settings absent from caller schemas")
            report["stage"] = "inspect_raw"
            save(report)

            began = time.monotonic()
            audit = await call("inspect_raw", {"source_path": SOURCE.name})
            report.update(audit=audit, audit_elapsed_seconds=round(time.monotonic() - began, 3))
            save(report)
            assert audit["state"] == "succeeded", audit
            directory = Path(audit["audit_dir"])
            assert directory.parent == output and directory.name.startswith("raw-audit-")
            assert audit["source"]["path"] == str(SOURCE) and audit["source"]["sha256"] == originals[str(SOURCE)]
            audit_report = directory / "audit.json"
            assert json.loads(audit_report.read_text(encoding="utf-8")) == audit
            immutable = {str(audit_report): sha(audit_report)}
            for name, expected in (("raw_codes", known), ("black_subtracted_cfa", expected_signal)):
                artifact = audit["artifacts"][name]
                path = Path(artifact["path"])
                assert path.parent == directory and sha(path) == artifact["sha256"]
                actual = np.load(path, allow_pickle=False)
                assert actual.dtype == expected.dtype
                np.testing.assert_array_equal(actual, expected)
                immutable[str(path)] = sha(path)
            report["checks"].append("real LibRaw through MCP: all 12288 known CFA codes exact; rational black subtraction exact with negative/over-white samples retained")

            rectangle = [0, 0, 16, 16]
            report["stage"] = "protect_trace"
            save(report)
            candidate = await call("protect_trace", {"audit_dir": directory.name, "rectangle": rectangle,
                "classification": "unknown", "note": "MCP 数值测试区域；非真实流星或自动识别结果。"})
            report["candidate"] = candidate
            save(report)
            assert candidate["state"] == "succeeded", candidate
            candidate_dir = Path(candidate["candidate_dir"])
            assert candidate_dir.parent == output and candidate_dir != directory
            assert candidate["rectangle"] == rectangle and candidate["classification"] == "unknown"
            assert candidate["classification_source"] == "manual"
            assert candidate["geometry"]["coordinate_basis"] == "decoder_full_raw_mosaic"
            assert json.loads((candidate_dir / "candidate.json").read_text(encoding="utf-8")) == candidate
            assert json.loads((candidate_dir / "audit_snapshot.json").read_text(encoding="utf-8")) == audit
            artifacts = candidate["artifacts"]
            items = artifacts.values() if isinstance(artifacts, dict) else artifacts
            crop_checks = []
            for artifact in items:
                path = Path(artifact["path"])
                assert path.resolve().is_relative_to(candidate_dir.resolve())
                assert sha(path) == artifact["sha256"]
                if path.suffix.lower() == ".npy":
                    crop = np.load(path, allow_pickle=False)
                    if crop.dtype == np.uint16:
                        np.testing.assert_array_equal(crop, known[:16, :16])
                        crop_checks.append("uint16 raw crop exact")
                    elif crop.dtype == np.float32:
                        np.testing.assert_array_equal(crop, expected_signal[:16, :16])
                        crop_checks.append("float32 black-subtracted crop exact")
                    else:
                        raise AssertionError(f"Unexpected candidate array dtype: {crop.dtype}")
            assert "uint16 raw crop exact" in crop_checks
            report["crop_checks"] = crop_checks
            candidate_files = {str(path): sha(path) for path in candidate_dir.rglob("*") if path.is_file()}
            immutable.update(candidate_files)
            report["checks"].append("protect_trace preserved requested native-CFA 16x16 crop, with recorded artifact SHA256 verified")

            before = sorted(str(path.relative_to(output)) for path in output.rglob("*"))
            report["stage"] = "invalid_requests"
            save(report)
            rejections = []
            for name, arguments, reason in (
                ("protect_trace", {"audit_dir": directory.name, "rectangle": [False, 0, 16, 16]}, "boolean rectangle element"),
                ("protect_trace", {"audit_dir": directory.name, "rectangle": [0, 0, 0, 16]}, "zero width"),
                ("inspect_raw", {"source_path": str(INPUT_DIR / "frame00001.dng")}, "absolute source outside host input root"),
                ("inspect_raw", {"source_path": "../synthetic-rational.dng"}, "source parent traversal"),
            ):
                result = await session.call_tool(name, arguments)
                assert result.isError, (name, arguments, result)
                rejections.append({"tool": name, "case": reason, "is_error": bool(result.isError),
                                   "message": " ".join(getattr(block, "text", "") for block in result.content)[:1000]})
            assert before == sorted(str(path.relative_to(output)) for path in output.rglob("*"))
            report["rejections"] = rejections
            report["no_new_outputs_after_rejections"] = True
            report["checks"].append("boolean/zero-width rectangles and escaped source paths returned MCP errors without creating candidates or other outputs")

    report["source_unchanged"] = all(sha(Path(path)) == digest for path, digest in originals.items())
    report["audit_and_candidate_unchanged_after_rejections"] = all(sha(Path(path)) == digest for path, digest in immutable.items())
    report["retained_artifact_hashes"] = immutable
    assert report["source_unchanged"] and report["audit_and_candidate_unchanged_after_rejections"]
    report["checks"].append("original numerical DNG, audit manifest/arrays and successful candidate hashes remained unchanged")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--inside-host-job", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if os.name != "nt" or not 20 <= args.timeout <= 60:
        parser.error("Requires Windows; timeout must be 20 to 60 seconds")
    if not args.inside_host_job:
        from starstack.process_guard import guarded_process

        command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--inside-host-job"]
        try:
            with guarded_process(command, stdin=subprocess.DEVNULL, stdout=sys.stdout, stderr=sys.stderr,
                                 cwd=str(ROOT), creationflags=subprocess.CREATE_NO_WINDOW) as child:
                return child.wait(timeout=args.timeout)
        except BaseException as error:
            previous = json.loads(RECORD.read_text(encoding="utf-8")) if RECORD.exists() else {}
            previous.update(acceptance_passed=False, validation_host_error=repr(error),
                            cleanup="closed only the owned validation process subtree")
            save(previous)
            raise

    report = {"schema_version": 1, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "sdk": f"mcp {version('mcp')}", "checks": [],
              "scope": "real official-SDK stdio RAW inspection and manual crop evidence, with invalid-request checks",
              "material": "known-code synthetic rational-black DNG; not a photographic or camera-ADC equivalence test",
              "limits": ["Caller receives local artifact paths and metadata; no original RAW or image bytes are transported to a model.",
                  "Host roots and path validation are not an OS sandbox against concurrent malicious local processes.",
                  "protect_trace records a user-specified rectangle; it does not detect, classify or composite a real meteor."],
              "source_sha256": {name: sha(ROOT / name) for name in (
                  "src/starstack/mcp_server.py", "src/starstack/raw_audit.py", "src/starstack/transients.py",
                  "src/starstack/raw_metadata.py", "scripts/validate_mcp_raw.py")}}
    started, passed = time.monotonic(), False
    try:
        asyncio.run(asyncio.wait_for(exercise(report), timeout=args.timeout - 10))
        passed = True
        return 0
    except BaseException as error:
        report["run_error"] = repr(error)
        return 1
    finally:
        report.update(acceptance_passed=passed, elapsed_seconds=round(time.monotonic() - started, 3))
        if passed:
            report["stage"] = "complete"
        save(report)
        print(json.dumps({"acceptance_passed": passed, "record": str(RECORD),
                          "run_error": report.get("run_error")}, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
