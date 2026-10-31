"""Run the real MCP + Siril synthetic acceptance once, inside an owned host job.

From the repository root:
    .venv/Scripts/python.exe scripts/validate_mcp.py
    .venv/Scripts/python.exe scripts/validate_mcp.py --input-dir data/demo/another-set

The default set is the existing eight-frame numerical fixture, not camera RAW.
Writes docs/diagnostics/mcp.json. No production source or input is modified.
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
RECORD = ROOT / "docs" / "diagnostics" / "mcp.json"
sys.path.insert(0, str(ROOT / "src"))


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(report: dict) -> None:
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    RECORD.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def in_windows_job() -> bool:
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    kernel.IsProcessInJob.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    kernel.IsProcessInJob.restype = ctypes.c_int
    answer = ctypes.c_int()
    if not kernel.IsProcessInJob(kernel.GetCurrentProcess(), None, ctypes.byref(answer)):
        raise ctypes.WinError(ctypes.get_last_error())
    return bool(answer.value)


async def exercise(args, report: dict) -> None:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.os.win32 import utilities as win32_utilities

    source = (ROOT / args.input_dir).resolve()
    mask = (ROOT / args.mask).resolve() if args.mask else source / "sky_mask.png"
    for path in (source, mask):
        if not path.is_relative_to((ROOT / "data").resolve()):
            raise ValueError("Validation inputs must be inside this project's data directory")
    if not source.is_dir() or not mask.is_file():
        raise FileNotFoundError("The numerical fixture directory and sky_mask.png must exist")
    files = sorted(path for path in source.iterdir() if path.suffix.lower() in {".fit", ".fits", ".fts"})
    if len(files) != 8:
        raise ValueError("This acceptance requires exactly eight numerical FITS inputs")
    originals = {str(path): sha(path) for path in [*files, mask]}
    report.update(input_dir=str(source), mask_path=str(mask), input_hashes=originals)
    env = dict(os.environ)
    env.update(PYTHONPATH=str(ROOT / "src"), STARSTACK_INPUT_ROOT=str(ROOT / "data"),
               STARSTACK_OUTPUT_ROOT=str(ROOT / "outputs"))
    parameters = StdioServerParameters(command=sys.executable, args=["-m", "starstack.mcp_server"],
                                       env=env, cwd=str(ROOT))
    report["nested_host_job"] = {
        "explicit_outer_job": True,
        "client_membership_verified": in_windows_job(),
        "official_sdk_pywin32_job_support_available": win32_utilities.win32job is not None,
        "result": "pending",
    }
    if not report["nested_host_job"]["client_membership_verified"]:
        raise RuntimeError("Validation client is not inside the explicit outer Windows job")

    async with stdio_client(parameters) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            async def call(name: str, arguments: dict) -> dict:
                result = await session.call_tool(name, arguments)
                if result.isError:
                    raise RuntimeError(f"{name}: {result.content}")
                if result.structuredContent is None:
                    raise RuntimeError(f"{name} did not return structured output")
                return result.structuredContent

            init = await session.initialize()
            report["protocol_version"] = init.protocolVersion
            report["tools"] = [tool.name for tool in (await session.list_tools()).tools]
            report["checks"].append("official SDK stdio initialize/list_tools")
            report["input_count"] = (await call("inspect_sources", {"input_dir": str(source)}))["count"]
            assert report["input_count"] == 8
            key = "mcp-guard-smoke-" + uuid.uuid4().hex[:12]
            report["idempotency_key"] = key
            submission = {"input_dir": str(source), "sky_mask_path": str(mask), "idempotency_key": key}
            try:
                state = await call("submit_job", submission)
                report["job_dir"] = state["job_dir"]
                save(report)
                states = []
                while state["state"] not in {"succeeded", "failed", "cancelled"}:
                    if state["state"] not in states:
                        states.append(state["state"])
                    await asyncio.sleep(0.5)
                    state = await call("job_status", {"job_dir": report["job_dir"]})
                states.append(state["state"])
                report.update(states_observed=states, final_state=state["state"], error=state.get("error"),
                              artifact_count=len(state["artifacts"]))
                if state["state"] != "succeeded":
                    raise RuntimeError(f"Real guarded Siril job failed: {state}")
                report["checks"].append("real guarded Siril completed eight synthetic FITS plus sky mask")
                report["nested_host_job"]["result"] = "guarded Siril completed under explicit outer host job"
                repeated = await call("submit_job", submission)
                assert repeated["job_dir"] == report["job_dir"] and repeated["state"] == "succeeded"
                report["checks"].append("retry after success returned same completed job")
                report["preview"] = await call("preview", {"job_dir": report["job_dir"], "artifact_index": 0})
                assert report["preview"]["transport"] == "local_path_only"
                assert Path(report["preview"]["path"]).is_file()
                report["checks"].append("registered JPG local path and image dimensions returned")
            except BaseException:
                # Try orderly cancellation while this exact connection is still alive.
                # The owned outer job is the bounded fallback if a protocol call hangs.
                if report.get("job_dir"):
                    try:
                        async with asyncio.timeout(15):
                            state = await call("cancel_job", {"job_dir": report["job_dir"]})
                            while state["state"] not in {"succeeded", "failed", "cancelled"}:
                                await asyncio.sleep(0.5)
                                state = await call("job_status", {"job_dir": report["job_dir"]})
                            report["cleanup"] = {"method": "core cancellation through MCP", "state": state["state"]}
                    except BaseException as cleanup_error:
                        report["cleanup"] = {"method": "owned outer job closes on client exit",
                                             "cancellation_error": repr(cleanup_error)}
                raise

    async with stdio_client(parameters) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            await session.initialize()
            result = await session.call_tool("job_status", {"job_dir": report["job_dir"]})
            assert not result.isError and result.structuredContent["state"] == "succeeded"
            report["checks"].append("new stdio connection read same completed job")
    report["originals_unchanged"] = all(sha(Path(path)) == digest for path, digest in originals.items())
    assert report["originals_unchanged"]
    report["checks"].append("all eight source hashes and mask hash unchanged")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="data/demo/synthetic-fbd621c4")
    parser.add_argument("--mask")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--inside-host-job", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if os.name != "nt":
        parser.error("This acceptance targets Windows nested Job Objects")
    if not 10 <= args.timeout <= 300:
        parser.error("--timeout must be between 10 and 300 seconds")
    if not args.inside_host_job:
        from starstack.process_guard import guarded_process

        # The parent owns this single job handle. Only its validation subtree is
        # terminated on exit/timeout; no global process-name lookup is used.
        command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--inside-host-job"]
        try:
            with guarded_process(command, stdin=subprocess.DEVNULL, stdout=sys.stdout, stderr=sys.stderr,
                                 cwd=str(ROOT), creationflags=subprocess.CREATE_NO_WINDOW) as child:
                return child.wait(timeout=args.timeout + 40)
        except BaseException as error:
            previous = json.loads(RECORD.read_text(encoding="utf-8")) if RECORD.exists() else {}
            previous.update(acceptance_passed=False, validation_host_error=repr(error),
                            cleanup="closed the owned validation Job Object; no global process-name termination")
            save(previous)
            raise

    report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "sdk": f"mcp {version('mcp')}",
              "material": "eight numerical synthetic FITS plus PNG mask; not real camera RAW", "checks": [],
              "source_sha256": {name: sha(ROOT / name) for name in (
                  "src/starstack/process_guard.py", "src/starstack/siril.py", "src/starstack/core.py",
                  "src/starstack/mcp_server.py")},
              "scope": "post-integration real MCP acceptance; normal completion, not owner-death fault injection"}
    began = time.monotonic()
    passed = False
    try:
        asyncio.run(asyncio.wait_for(exercise(args, report), timeout=args.timeout))
        passed = True
        return 0
    except BaseException as error:
        report["run_error"] = repr(error)
        return 1
    finally:
        report.update(acceptance_passed=passed, elapsed_seconds=round(time.monotonic() - began, 3))
        save(report)
        print(json.dumps({"acceptance_passed": passed, "job_dir": report.get("job_dir"),
                          "record": str(RECORD)}, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
