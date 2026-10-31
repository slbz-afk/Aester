"""Shared public finishing service with host-owned paths and GIMP configuration."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, CancelledError
import json
from pathlib import Path
import threading
import uuid

from . import finishing_session as sessions, finishing_candidates as candidates
from .finishing_assets import verify_source
from .finishing_visuals import draw_coverage, photo_preview, reference_view
from .finishing_masks import tonal_coverage
from .mcp_server import WorkspaceTools, _bounded, _no_link_components, AccessDenied


class FinishingTools:
    def __init__(self, workspace=None, gimp_path=None):
        self.workspace = workspace or WorkspaceTools()
        from .retouch_integration import get_retouch_info
        self._gimp_argument = gimp_path
        self._retouch_info = get_retouch_info(gimp_path=gimp_path)
        self.gimp_path = self._retouch_info["gimp"].get("selected_path")
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="aester-finishing")
        self._jobs = {}
        self._job_lock = threading.Lock()

    def _discover_if_missing(self):
        # A connected Agent may guide the user through installing GIMP. Pick
        # it up on the next request, but never replace an explicit bad choice.
        if self.gimp_path is None and self._gimp_argument is None:
            from .retouch_integration import get_retouch_info
            self._retouch_info = get_retouch_info()
            self.gimp_path = self._retouch_info["gimp"].get("selected_path")

    def _root(self):
        root = self.workspace._output() / "finishing"
        _no_link_components(root)
        root.mkdir(exist_ok=True)
        return root

    def _session(self, directory):
        root = self._root()
        path = _bounded(directory, root)
        if path.parent != root:
            raise AccessDenied("请选择此服务创建的修图会话")
        value = sessions.read_session(path)
        self.workspace._reference_path(value["source"]["path"])
        for record in [*value["masks"].values(), *value["references"]]:
            _bounded(record["path"], path / "assets")
        for record in value["candidates"]:
            _bounded(record["directory"], path / "renders")
        for record in value.get("layer_bundles", {}).values():
            for item in [record["background"], record["diffuse"], *record["provenance"]["evidence"]]:
                _bounded(item["path"], path / "assets")
        return value

    def capabilities(self):
        self._discover_if_missing()
        configured = bool(self.gimp_path and Path(self.gimp_path).is_file())
        return {"schema": "aester-finishing-capabilities-1", "gimp_configured": configured,
                "runtime_verified": False, "runtime_check": "Every render enforces GIMP 3.2.6 and validates products",
                "required_gimp_version": "3.2.6", "platform": "Windows",
                "source": "single-page encoded sRGB RGB16 classic uncompressed strip TIFF, FillOrder=1",
                "mask": "bound grayscale8/16, classic uncompressed strip TIFF FillOrder1 or noninterlaced PNG",
                "recipe_schema": "aester-finishing-recipe-1", "operations_limit": 32,
                "recipe_schemas": ["aester-finishing-recipe-1", "aester-finishing-recipe-2", "aester-finishing-recipe-3"],
                "operations": {"exposure": {"params": {"ev": [-2, 2]}, "black_level": 0},
                  "saturation": {"params": {"scale": [0, 2]}, "colorspace": "Native"},
                  "curves": {"params": {"channel": ["value", "red", "green", "blue"],
                      "points": "2..32 [x,y] in [0,1]; x strictly increasing 0..1; y nondecreasing"},
                      "implementation": "gimp-free-256-v1", "max_sampling_error": 1 / 255}},
                "v2_operations": {"local_contrast_unsharp": {"params": {"sigma_px": [4, 80], "amount": [0, .6]},
                    "implementation": "gimp-3.2.6-unsharp-v2-xcf-float32", "threshold": 0,
                    "warning": "Spatial contrast also amplifies stars/noise; prefer suitable separated diffuse material"}},
                "v3_operations": {"astrophotographic_layers": {
                    "placement": "first and unique; outer mask_id must be null",
                    "params": {"bundle_id": "registered source-bound bundle", "bundle_sha256": "exact bundle digest",
                        "star_strength": [0, 1], "milkyway_amount": [0, 3],
                        "milkyway_mask_id": "registered region or null only at zero amount",
                        "meteor_core_extra": [0, 2], "meteor_glow_amount": [0, 4]},
                    "implementation": "aester-layer-stage-teacher-1",
                    "order": ["stars", "milkyway", "meteor"],
                    "domain": "signed encoded-sRGB differences; not physical flux",
                    "status": "experimental development; check case-specific validation before use",
                    "editability": "Cumulative RGB16 snapshot; parameter changes require recipe rerun",
                    "lineage": "Caller-reported chain with stored evidence; structural checks do not certify separation quality"}},
                "mask_creation": "source-coordinate polygon union, optional feather/invert; Agent provenance",
                "tonal_mask": {"implementation": "encoded-srgb-tonal-range-v1",
                    "knots": "0<=a<=b<=c<=d<=1, a<d; smooth ramps with full selection b..c",
                    "image": "original or verified candidate; fixed RGB16 coverage, not physical luminance",
                    "intersection": "optional registered region; invert range before intersection"},
                "missing_operations": ["denoise", "star_shape_repair", "automatic_layer_separation_in_finishing"],
                "outputs": ["JPEG", "RGB16 TIFF", "XCF"], "reference_pixels_used": False,
                "feedback_origin": "agent; user acceptance is not authenticated by MCP",
                "image_transport": "local_paths", "server_mode": "shared product and standalone finishing interface",
                "skill_required": False, "skill_recommendation": self._retouch_info["recommendation"]}

    def create(self, source, intent):
        path = self.workspace._reference_path(source)
        return sessions.create_session(path, self._root(), intent=intent)

    def get(self, session_dir):
        return self._session(session_dir)

    def add_reference(self, session_dir, path, kind, note=""):
        value = self._session(session_dir)
        reference = self.workspace._reference_path(path)
        return sessions.register_reference(value["directory"], reference, kind=kind, note=note)

    def add_mask(self, session_dir, path, mask_id, source_sha256, note=""):
        value = self._session(session_dir)
        mask = self.workspace._reference_path(path)
        return sessions.register_mask(value["directory"], mask, mask_id=mask_id, source_sha256=source_sha256,
                                      origin="agent", note=note)

    def add_layers(self, session_dir, bundle_id, background_path, diffuse_path,
                   pure_sky_mask_id, transient_support_mask_id, source_sha256, provenance):
        from .finishing_layers import register_layers
        value = self._session(session_dir)
        background = self.workspace._reference_path(background_path)
        diffuse = self.workspace._reference_path(diffuse_path)
        if type(provenance) is not dict:
            raise ValueError("Layer provenance must be a JSON object")
        provenance = json.loads(json.dumps(provenance, allow_nan=False))
        if type(provenance.get("evidence")) is not list:
            raise ValueError("Layer provenance requires evidence records")
        for record in provenance["evidence"]:
            if type(record) is not dict or "path" not in record:
                raise ValueError("Layer evidence requires a path")
            record["path"] = str(self.workspace._reference_path(record["path"]))
        return register_layers(value["directory"], bundle_id=bundle_id,
            background_path=background, diffuse_path=diffuse, pure_sky_mask_id=pure_sky_mask_id,
            transient_support_mask_id=transient_support_mask_id, source_sha256=source_sha256,
            provenance=provenance)

    def draw_mask(self, session_dir, mask_id, polygons, feather_px=0, invert=False):
        value = self._session(session_dir)
        sessions._asset_id(mask_id)
        verify_source(value["source"])
        if mask_id in value["masks"]:
            raise ValueError("Mask IDs cannot be replaced; choose a new ID")
        drawing = draw_coverage(value["directory"], value["source"], polygons,
                                feather_px=feather_px, invert=invert)
        value = sessions.register_mask(value["directory"], drawing["path"], mask_id=mask_id,
                    source_sha256=value["source"]["sha256"], origin="agent", note="Agent polygon coverage",
                    geometry=drawing["geometry"])
        preview = photo_preview(value["directory"], value["source"], mask_record=value["masks"][mask_id])
        return {"session": value, "geometry": drawing["geometry"], "mask_preview": preview}

    def revise(self, session_dir, recipe, expected_revision):
        value = self._session(session_dir)
        return sessions.revise_recipe(value["directory"], recipe, expected_revision=expected_revision)

    def tonal_mask(self, session_dir, mask_id, knots, candidate_id=None, region_mask_id=None, invert=False):
        value = self._session(session_dir)
        sessions._asset_id(mask_id)
        if mask_id in value["masks"]:
            raise ValueError("Mask IDs cannot be replaced; choose a new ID")
        verify_source(value["source"])
        asset = value["source"] if candidate_id is None else candidates.verified_candidate(
            value["directory"], candidate_id)["execution"]["products"]["tiff"]
        region = None
        if region_mask_id is not None:
            sessions._asset_id(region_mask_id)
            if region_mask_id not in value["masks"]:
                raise ValueError("Unknown region mask")
            region = value["masks"][region_mask_id]
        drawing = tonal_coverage(value["directory"], value["source"], asset, knots,
                                 invert=invert, region=region)
        drawing["geometry"]["candidate_id"] = candidate_id
        dependencies = [{"kind": "image", "asset": asset}]
        if region is not None:
            dependencies.append({"kind": "region", "asset": region})
        value = sessions.register_mask(value["directory"], drawing["path"], mask_id=mask_id,
            source_sha256=value["source"]["sha256"], origin="agent", note="Agent tonal range selection",
            geometry=drawing["geometry"], dependencies=dependencies)
        return {"session": value, "geometry": drawing["geometry"],
            "mask_preview": photo_preview(value["directory"], asset, mask_record=value["masks"][mask_id])}

    def render(self, session_dir, expected_revision, timeout=600, *, _cancel=None):
        value = self._session(session_dir)
        self._discover_if_missing()
        if not self.gimp_path:
            raise ValueError("请由主机配置 AESTER_GIMP，指向已有 GIMP 3.2.6 console 程序")
        return candidates.render_candidate(value["directory"], expected_revision=expected_revision,
                    gimp_path=self.gimp_path, timeout=timeout, cancel=_cancel)

    def submit_render(self, session_dir, expected_revision, timeout=600):
        value = self._session(session_dir)
        sessions._expect(value, expected_revision)
        session_dir = value["directory"]
        with self._job_lock:
            if len(self._jobs) >= 256:
                raise ValueError("This service reached its render request limit")
            if any(job["session"] == session_dir and not job["future"].done() for job in self._jobs.values()):
                raise ValueError("This session already has an active render request")
            job_id = "render-" + uuid.uuid4().hex
            cancelled = threading.Event()
            future = self._pool.submit(self.render, session_dir, expected_revision, timeout, _cancel=cancelled.is_set)
            self._jobs[job_id] = {"future": future, "cancel": cancelled, "session": session_dir}
        return {"job_id": job_id, "session_dir": session_dir, "status": "submitted"}

    def render_status(self, job_id):
        with self._job_lock:
            if type(job_id) is not str or job_id not in self._jobs:
                raise ValueError("Unknown render request; inspect the persisted session after a service restart")
            job = self._jobs[job_id]
        future = job["future"]
        if not future.done():
            return {"job_id": job_id, "status": "cancelling" if job["cancel"].is_set() else "running"}
        try:
            candidate = future.result()
        except Exception as exc:
            return {"job_id": job_id, "status": "cancelled" if isinstance(exc, (candidates.RenderCancelled, CancelledError)) else "failed",
                    "error": str(exc), "session_dir": job["session"]}
        return {"job_id": job_id, "status": "validated", "candidate": candidate}

    def cancel_render(self, job_id):
        with self._job_lock:
            if type(job_id) is not str or job_id not in self._jobs:
                raise ValueError("Unknown render request")
            self._jobs[job_id]["cancel"].set()
        return self.render_status(job_id)

    def preview(self, session_dir, candidate_id=None, crop=None):
        value = self._session(session_dir)
        if candidate_id is None:
            verify_source(value["source"])
            asset = value["source"]
        else:
            candidate = candidates.verified_candidate(value["directory"], candidate_id)
            asset = candidate["execution"]["products"]["tiff"]
        return photo_preview(value["directory"], asset, crop=crop)

    def feedback(self, session_dir, candidate_id, text, decision="observe"):
        value = self._session(session_dir)
        return candidates.candidate_feedback(value["directory"], candidate_id=candidate_id,
                                            text=text, origin="agent", decision=decision)

    # These three methods are local host UI actions, deliberately not exposed
    # by the Agent CLI or MCP schema. The host owns the actual user click.
    def feedback_user(self, session_dir, candidate_id, text, decision="observe"):
        value = self._session(session_dir)
        return candidates.candidate_feedback(value["directory"], candidate_id=candidate_id,
                                            text=text, origin="user", decision=decision)

    def set_intent(self, session_dir, intent, expected_intent_revision):
        value = self._session(session_dir)
        return sessions.update_intent(value["directory"], intent,
                                      expected_intent_revision=expected_intent_revision)

    def reference_preview(self, session_dir, reference_id):
        value = self._session(session_dir)
        record = next((item for item in value["references"] if item["id"] == reference_id), None)
        if record is None:
            raise ValueError("Unknown reference")
        return reference_view(value["directory"], record)

    def export(self, session_dir, candidate_id):
        value = self._session(session_dir)
        root = self.workspace.delivery_root or self.workspace._output()
        _no_link_components(root)
        root.mkdir(parents=True, exist_ok=True)
        return candidates.export_candidate(value["directory"], candidate_id, root)

    def recover(self, session_dir):
        value = self._session(session_dir)
        return candidates.recover_interrupted(value["directory"])

    def close(self):
        with self._job_lock:
            for job in self._jobs.values():
                job["cancel"].set()
        self._pool.shutdown(wait=True, cancel_futures=True)
