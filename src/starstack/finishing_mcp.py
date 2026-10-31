"""Independent stdio MCP adapter for host-bounded Aester finishing tools."""
import argparse
import asyncio
from contextlib import asynccontextmanager, redirect_stdout
import sys
from typing import Annotated, Any, Literal


def register_tools(server, service):
    """Register the frozen public methods without importing the default service."""
    from pydantic import Field
    from .mcp_replies import structured_tools
    # Every tool declares an object output schema and replies with structuredContent in JSON-native types.
    tool = structured_tools(server)

    @tool()
    async def finishing_capabilities() -> dict[str, Any]:
        """List finishing operations, configured backend availability and explicit limitations."""
        return await asyncio.to_thread(service.capabilities)

    @tool()
    async def finishing_create(source: str, intent: str) -> dict[str, Any]:
        """Start a finishing session from an allowed sRGB16 TIFF; intention is text data."""
        return await asyncio.to_thread(service.create, source=source, intent=intent)

    @tool()
    async def finishing_get(session_dir: str) -> dict[str, Any]:
        """Read a finishing session and its source-bound recipe/candidate history."""
        return await asyncio.to_thread(service.get, session_dir=session_dir)

    @tool()
    async def finishing_add_reference(session_dir: str, path: str,
            kind: Literal["original", "generated", "external"], note: str = "") -> dict[str, Any]:
        """Register a local aesthetic reference and its provenance; it never becomes render pixels."""
        return await asyncio.to_thread(service.add_reference, session_dir=session_dir,
                                       path=path, kind=kind, note=note)

    @tool()
    async def finishing_add_mask(session_dir: str, path: str, mask_id: str,
                                 source_sha256: str, note: str = "") -> dict[str, Any]:
        """Register an allowed full-size grayscale mask bound to the source hash; no human claim."""
        return await asyncio.to_thread(service.add_mask, session_dir=session_dir, path=path,
                                       mask_id=mask_id, source_sha256=source_sha256, note=note)

    @tool()
    async def finishing_add_layers(session_dir: str, bundle_id: str, background_path: str,
            diffuse_path: str, pure_sky_mask_id: str, transient_support_mask_id: str,
            source_sha256: str, provenance: dict) -> dict[str, Any]:
        """Snapshot source-bound B/D layers and binary support regions with explicit provenance.

        Structural checks do not certify the reported separation lineage or quality.
        Existing aesthetic reference pixels cannot become processing layers.
        """
        return await asyncio.to_thread(service.add_layers, session_dir=session_dir, bundle_id=bundle_id,
            background_path=background_path, diffuse_path=diffuse_path, pure_sky_mask_id=pure_sky_mask_id,
            transient_support_mask_id=transient_support_mask_id, source_sha256=source_sha256, provenance=provenance)

    @tool()
    async def finishing_draw_mask(session_dir: str, mask_id: str, polygons: list,
            feather_px: Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)] = 0.0,
            invert: Annotated[bool, Field(strict=True)] = False) -> dict[str, Any]:
        """Draw a union of source-pixel polygons, feather then optionally invert; inspect the actual mask."""
        return await asyncio.to_thread(service.draw_mask, session_dir=session_dir, mask_id=mask_id,
                                       polygons=polygons, feather_px=feather_px, invert=invert)

    @tool()
    async def finishing_tonal_mask(session_dir: str, mask_id: str, knots: list,
            candidate_id: str | None = None, region_mask_id: str | None = None,
            invert: Annotated[bool, Field(strict=True)] = False) -> dict[str, Any]:
        """Select encoded brightness using four ordered knots; optionally intersect a registered region.

        Uses actual source or verified candidate pixels, never a reference. The fixed
        mask has smooth tonal ramps, not spatial feathering. Inspect its returned overlay.
        """
        return await asyncio.to_thread(service.tonal_mask, session_dir=session_dir, mask_id=mask_id,
            knots=knots, candidate_id=candidate_id, region_mask_id=region_mask_id, invert=invert)

    @tool()
    async def finishing_revise(session_dir: str, recipe: dict,
            expected_revision: Annotated[int, Field(strict=True, ge=0)]) -> dict[str, Any]:
        """Append a validated recipe revision; a stale expected revision fails without overwriting history."""
        return await asyncio.to_thread(service.revise, session_dir=session_dir, recipe=recipe,
                                       expected_revision=expected_revision)

    @tool()
    async def finishing_submit_render(session_dir: str,
            expected_revision: Annotated[int, Field(strict=True, ge=0)],
            timeout: Annotated[float, Field(strict=True, gt=0, le=3600, allow_inf_nan=False)] = 600.0) -> dict[str, Any]:
        """Start a render and return its job_id immediately; poll status before using a validated candidate."""
        return service.submit_render(session_dir=session_dir, expected_revision=expected_revision, timeout=timeout)

    @tool()
    async def finishing_render_status(job_id: str) -> dict[str, Any]:
        """Read a submitted finishing render's state; submission is not success."""
        return service.render_status(job_id=job_id)

    @tool()
    async def finishing_cancel_render(job_id: str) -> dict[str, Any]:
        """Request cancellation; continue checking status until execution has stopped."""
        return service.cancel_render(job_id=job_id)

    @tool()
    async def finishing_preview(session_dir: str, candidate_id: str | None = None,
                                crop: list | None = None) -> dict[str, Any]:
        """Return a local viewing JPEG and mapping; optional crop is [x,y,width,height] in source pixels."""
        return await asyncio.to_thread(service.preview, session_dir=session_dir,
                                       candidate_id=candidate_id, crop=crop)

    @tool()
    async def finishing_feedback(session_dir: str, candidate_id: str, text: str,
            decision: Literal["observe", "revise", "reject"] = "observe") -> dict[str, Any]:
        """Record agent feedback on a real candidate; relayed user opinion is not authenticated acceptance."""
        return await asyncio.to_thread(service.feedback, session_dir=session_dir, candidate_id=candidate_id,
                                       text=text, decision=decision)

    @tool()
    async def finishing_export(session_dir: str, candidate_id: str) -> dict[str, Any]:
        """Export a verified candidate's TIFF, JPEG and editable XCF to the host-configured output root."""
        return await asyncio.to_thread(service.export, session_dir=session_dir, candidate_id=candidate_id)

    @tool()
    async def finishing_recover(session_dir: str) -> dict[str, Any]:
        """After acquiring ownership, mark abandoned running candidates interrupted; never auto-revive them."""
        return await asyncio.to_thread(service.recover, session_dir=session_dir)

    return server


def create_server(service=None):
    import anyio
    from mcp.server.fastmcp import FastMCP
    if service is None:
        from .mcp_server import WorkspaceTools
        from .finishing_service import FinishingTools
        service = FinishingTools(WorkspaceTools())
    @asynccontextmanager
    async def lifespan(_server):
        try:
            yield {}
        finally:
            close = getattr(service, "close", None)
            if callable(close):
                # The service joins its own render pool; keep the protocol loop
                # free and allow cleanup to finish when the connection closes.
                with anyio.CancelScope(shield=True):
                    await asyncio.to_thread(close)

    server = FastMCP("AesterRetouch", log_level="WARNING", lifespan=lifespan, instructions=(
        "Local replayable finishing of sRGB16 TIFFs. Start with finishing_capabilities, create a session, "
        "inspect its preview, then revise the explicit recipe. Submit rendering, poll its job_id, and "
        "inspect a validated candidate before feedback/export. References are aesthetic guidance only. "
        "Tool calls are agent actions and cannot certify human acceptance. GIMP and filesystem roots "
        "are host configuration. Returned image paths are local; these tools do not upload images."))
    return register_tools(server, service)


def main(argv=None) -> int:
    from .finishing_cli import add_host_arguments, configured_service
    parser = argparse.ArgumentParser(description="Aester local finishing MCP server (stdio)")
    add_host_arguments(parser)
    args = parser.parse_args(argv)
    try:
        with redirect_stdout(sys.stderr):
            server = create_server(configured_service(args))
    except Exception as exc:
        print(f"Finishing MCP startup failed: {exc}", file=sys.stderr)
        return 1
    # Stdout belongs to the protocol; do not redirect it while the server runs.
    server.run(transport="stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
