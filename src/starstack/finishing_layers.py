"""Immutable, structurally checked display-layer bundles; no lineage certification.

Only S/B/D photographs and binary supports are execution inputs. Signed R/T are
derived with int32 arithmetic, never accepted as photographs. The service owns
external path authorization; all persisted execution assets stay in the session.
"""
from __future__ import annotations

from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import re

from .finishing_assets import (inspect_source, inspect_mask, ordinary_path,
                               recipe_source, sha256_file)

SCHEMA = "aester-finishing-layer-bundle-1"
MAX_BUNDLES = 16
_ROWS = 32
_HASH = re.compile(r"[0-9a-f]{64}")
_BUNDLE_KEYS = {"schema", "id", "source", "background", "diffuse", "pure_sky",
                "transient_support", "provenance", "lineage_verified",
                "structural_checks", "created_at", "bundle_sha256"}


def _hash_text(value):
    if type(value) is not str or _HASH.fullmatch(value) is None:
        raise ValueError("Expected a lowercase SHA256 identity")
    return value


def _digest(value):
    try:
        payload = json.dumps({k: v for k, v in value.items() if k != "bundle_sha256"},
                             sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ValueError("Layer bundle must be canonical JSON") from exc
    return hashlib.sha256(payload).hexdigest()


def _rgb(path, source=None):
    import tifffile
    record = inspect_source(path, encoding="srgb16")
    with tifffile.TiffFile(record["path"]) as tf:
        page = tf.pages[0]
        if (tf.is_bigtiff or page.is_tiled or int(page.compression) != 1
                or int(page.fillorder) != 1 or not page.is_memmappable):
            raise ValueError("Layers require classic uncompressed memmappable strip TIFF, FillOrder=1")
    if source is not None and any(record[key] != source[key]
            for key in ("width", "height", "encoding", "icc_sha256")):
        raise ValueError("Layer geometry or ICC differs from source")
    return record


def _asset_path(value, path):
    current = ordinary_path(path)
    if current.parent != Path(value["directory"]) / "assets":
        raise ValueError("Layer assets must remain in this session's assets")
    return current


def _mask(value, mask_id):
    from .finishing_session import _asset_id
    _asset_id(mask_id)
    record = value["masks"].get(mask_id)
    if not isinstance(record, dict):
        raise ValueError("Unregistered layer support mask")
    _asset_path(value, record["path"])
    actual = inspect_mask(record["path"], value["source"],
                          source_sha256=record["source_sha256"])
    if any(record.get(key) != item for key, item in actual.items()):
        raise ValueError("Registered layer support mask changed")
    return {"mask_id": mask_id, "asset": actual}


def _read_binary(record):
    import numpy as np
    import tifffile
    from PIL import Image
    if Path(record["path"]).suffix.lower() in {".tif", ".tiff"}:
        pixels = tifffile.imread(record["path"])
    else:
        with Image.open(record["path"]) as im:
            pixels = np.array(im)
    maximum = 2 ** record["bits"] - 1
    if (pixels.shape != (record["height"], record["width"])
            or np.any((pixels != 0) & (pixels != maximum))):
        raise ValueError("Layer support masks must be strictly binary")
    return pixels != 0


def _reject_reference_layers(value, *records):
    forbidden = {record["sha256"] for record in value.get("references", [])
                 if record.get("kind") in {"generated", "external"}}
    if any(record["sha256"] in forbidden for record in records):
        raise ValueError("Generated/external reference identities cannot be execution layers")


def _evidence_file(path, expected_hash=None):
    path = ordinary_path(path)
    before = path.stat()
    digest = sha256_file(path)
    after = path.stat()
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(getattr(before, key) != getattr(after, key) for key in fields):
        raise ValueError("Layer evidence changed while inspected")
    if expected_hash is not None and digest != _hash_text(expected_hash):
        raise ValueError("Layer evidence hash differs")
    return {"path": str(path), "sha256": digest, "bytes": before.st_size}


def _provenance(value):
    from .finishing_session import _text
    if type(value) is not dict or set(value) != {"kind", "description", "evidence"}:
        raise ValueError("Provenance requires exactly kind, description, evidence")
    if value["kind"] != "caller_reported_layer_chain":
        raise ValueError("Only caller_reported_layer_chain provenance is supported")
    description = _text(value["description"], "Layer provenance description")
    if not description.strip():
        raise ValueError("Layer provenance description must not be empty")
    evidence = value["evidence"]
    if type(evidence) is not list or not 1 <= len(evidence) <= 16:
        raise ValueError("Layer provenance requires 1..16 evidence files")
    records = []
    for item in evidence:
        if type(item) is not dict or set(item) != {"path", "sha256"}:
            raise ValueError("Evidence requires exactly path and sha256")
        if type(item["path"]) is not str or not Path(item["path"]).is_absolute():
            raise ValueError("Evidence path must be an absolute string")
        records.append(_evidence_file(item["path"], _hash_text(item["sha256"])))
    return {"kind": value["kind"], "description": description, "evidence": records}


def _structure(source, background, diffuse, pure_sky, transient_support):
    """Check every native pixel with bounded RGB temporaries, no resampling."""
    import numpy as np
    import tifffile
    pure = _read_binary(pure_sky["asset"])
    support = _read_binary(transient_support["asset"])
    hashes = {name: hashlib.sha256() for name in ("source", "background", "diffuse")}
    extrema = {name: [65535, -65535] for name in ("R", "T")}
    with ExitStack() as stack:
        arrays = []
        for record in (source, background, diffuse):
            array = tifffile.memmap(record["path"], mode="r")
            stack.callback(array._mmap.close)
            if array.shape != (source["height"], source["width"], 3):
                raise ValueError("Mapped layer geometry changed")
            arrays.append(array)
        for y in range(0, source["height"], _ROWS):
            section = slice(y, y + _ROWS)
            s, b, d = [array[section].astype(np.int32) for array in arrays]
            r, t = b - d, s - b
            if np.any(t[~support[section]]):
                raise ValueError("Transient difference is nonzero outside transient support")
            if np.any(t[~pure[section]]):
                raise ValueError("Transient difference is nonzero outside pure sky")
            if np.any(r[~pure[section]]):
                raise ValueError("Star difference is nonzero outside pure sky")
            for name, values in zip(hashes, (s, b, d)):
                # A fixed little-endian raster digest, independent of TIFF tags.
                hashes[name].update(values.astype("<u2").tobytes(order="C"))
            for name, values in (("R", r), ("T", t)):
                extrema[name][0] = min(extrema[name][0], int(values.min()))
                extrema[name][1] = max(extrema[name][1], int(values.max()))
    return {"implementation": "signed-display-layer-structure-v1",
            "checked_pixels": source["width"] * source["height"],
            "pixel_hash_encoding": "row-major interleaved RGB little-endian uint16",
            "pixel_sha256": {name: digest.hexdigest() for name, digest in hashes.items()},
            "pure_sky_pixels": int(pure.sum()), "transient_support_pixels": int(support.sum()),
            "T_zero_outside_support": True, "T_zero_outside_pure_sky": True,
            "R_zero_outside_pure_sky": True, "outside_pure_S_B_D_equal": True,
            "residual_dtype": "int32", "R_range": extrema["R"], "T_range": extrema["T"],
            "lineage_or_separation_quality_proven": False}


def _unchanged_rgb(record, source=None):
    if _rgb(record["path"], source) != record:
        raise ValueError("Layer image changed before verification/publication")


def register_layers(directory, *, bundle_id, background_path, diffuse_path,
                    pure_sky_mask_id, transient_support_mask_id, source_sha256,
                    provenance: dict) -> dict:
    """Snapshot caller-reported layers and evidence, without certifying lineage."""
    from . import finishing_session as sessions
    sessions._asset_id(bundle_id)
    _hash_text(source_sha256)
    with sessions._locked(directory):
        value = sessions.read_session(directory)
        bundles = value.get("layer_bundles", {})
        if type(bundles) is not dict or bundle_id in bundles or len(bundles) >= MAX_BUNDLES:
            raise ValueError("Layer bundle ID exists or bundle limit reached")
        source = value["source"]
        if source_sha256 != source["sha256"]:
            raise ValueError("Layer bundle belongs to a different source")
        _unchanged_rgb(source)
        background = _rgb(background_path, source)
        diffuse = _rgb(diffuse_path, source)
        _reject_reference_layers(value, background, diffuse)
        pure = _mask(value, pure_sky_mask_id)
        support = _mask(value, transient_support_mask_id)
        provenance_record = _provenance(provenance)
        created = []

        def copy(record):
            snapshot = sessions._copy_asset(value["directory"], record)
            created.append(Path(snapshot["path"]))
            return snapshot

        try:
            b, d = copy(background), copy(diffuse)
            evidence = [copy(item) for item in provenance_record["evidence"]]
            checks = _structure(source, b, d, pure, support)
            record = {"schema": SCHEMA, "id": bundle_id, "source": dict(source),
                      "background": b, "diffuse": d, "pure_sky": pure,
                      "transient_support": support,
                      "provenance": {**provenance_record, "evidence": evidence},
                      "lineage_verified": False, "structural_checks": checks,
                      "created_at": sessions._now()}
            record["bundle_sha256"] = _digest(record)
            # Recheck original inputs as well as copied assets at publication.
            for item in (source, background, diffuse, b, d):
                _unchanged_rgb(item, source)
            if _mask(value, pure_sky_mask_id) != pure or _mask(value, transient_support_mask_id) != support:
                raise ValueError("Layer support changed before publication")
            for item in [*provenance_record["evidence"], *evidence]:
                if _evidence_file(item["path"], item["sha256"]) != item:
                    raise ValueError("Layer evidence changed before publication")
            value["layer_bundles"] = {**bundles, bundle_id: record}
            sessions._save(Path(value["directory"]) / "session.json", value)
            return value
        except Exception:
            # Only unique files created by this call, never caller-owned inputs.
            for path in created:
                path.unlink(missing_ok=True)
            raise


def verify_bundle(value, bundle_id, bundle_sha256=None) -> dict:
    """Recheck bytes, geometry and every support relation; never certify lineage."""
    from .finishing_session import _asset_id, _text
    _asset_id(bundle_id)
    record = value.get("layer_bundles", {}).get(bundle_id)
    if type(record) is not dict or set(record) != _BUNDLE_KEYS:
        raise ValueError("Unregistered or malformed layer bundle")
    if (record["schema"] != SCHEMA or record["id"] != bundle_id
            or record["lineage_verified"] is not False):
        raise ValueError("Invalid layer bundle identity or lineage claim")
    digest = _hash_text(record["bundle_sha256"])
    if digest != _digest(record) or (bundle_sha256 is not None and _hash_text(bundle_sha256) != digest):
        raise ValueError("Layer bundle digest changed or differs from recipe")
    source = value["source"]
    if record["source"] != source:
        raise ValueError("Layer bundle source binding differs")
    _unchanged_rgb(source)
    for key in ("background", "diffuse"):
        _asset_path(value, record[key]["path"])
        _unchanged_rgb(record[key], source)
    _reject_reference_layers(value, record["background"], record["diffuse"])
    for key in ("pure_sky", "transient_support"):
        item = record[key]
        if type(item) is not dict or set(item) != {"mask_id", "asset"} or _mask(value, item["mask_id"]) != item:
            raise ValueError("Layer support binding changed")
    provenance = record["provenance"]
    if (type(provenance) is not dict or set(provenance) != {"kind", "description", "evidence"}
            or provenance["kind"] != "caller_reported_layer_chain"):
        raise ValueError("Invalid layer provenance")
    if not _text(provenance["description"], "Layer provenance description").strip():
        raise ValueError("Layer provenance description is empty")
    evidence = provenance["evidence"]
    if type(evidence) is not list or not 1 <= len(evidence) <= 16:
        raise ValueError("Invalid layer evidence")
    for item in evidence:
        if type(item) is not dict or set(item) != {"path", "sha256", "bytes"}:
            raise ValueError("Invalid layer evidence record")
        _asset_path(value, item["path"])
        if _evidence_file(item["path"], item["sha256"]) != item:
            raise ValueError("Layer evidence changed")
    if _structure(source, record["background"], record["diffuse"],
                  record["pure_sky"], record["transient_support"]) != record["structural_checks"]:
        raise ValueError("Layer structural checks differ")
    for item in (source, record["background"], record["diffuse"]):
        _unchanged_rgb(item, source)
    for key in ("pure_sky", "transient_support"):
        if _mask(value, record[key]["mask_id"]) != record[key]:
            raise ValueError("Layer support changed during verification")
    for item in evidence:
        if _evidence_file(item["path"], item["sha256"]) != item:
            raise ValueError("Layer evidence changed during verification")
    return json.loads(json.dumps(record, ensure_ascii=False, allow_nan=False))


def recipe_bundles(value, recipe) -> dict:
    """Resolve exactly referenced bundles after recipe normalization, even disabled."""
    if type(recipe) is not dict or recipe.get("source") != recipe_source(value["source"]):
        raise ValueError("Layer recipe belongs to a different source")
    operations = recipe.get("operations")
    if type(operations) is not list:
        raise ValueError("Recipe operations must be a list")
    result = {}
    for operation in operations:
        if type(operation) is not dict:
            raise ValueError("Invalid recipe operation")
        if operation.get("type") != "astrophotographic_layers":
            continue
        params = operation.get("params")
        if type(params) is not dict or not {"bundle_id", "bundle_sha256"} <= set(params):
            raise ValueError("Layer operation requires bundle ID and digest")
        identity, digest = params["bundle_id"], _hash_text(params["bundle_sha256"])
        if type(identity) is not str:
            raise ValueError("Invalid layer bundle ID")
        if identity in result:
            if result[identity]["bundle_sha256"] != digest:
                raise ValueError("Conflicting layer bundle digests")
        else:
            result[identity] = verify_bundle(value, identity, digest)
    return result


def verify_render_bundle(source, bundle) -> dict:
    """Recheck a frozen render bundle without loading or trusting session state.

    This checks bytes and a common owned assets directory, not permission to use
    that directory or the caller's reported lineage. The public service/session
    must first authorize and resolve the bundle with ``verify_bundle``.
    """
    if type(bundle) is not dict or set(bundle) != _BUNDLE_KEYS:
        raise ValueError("Malformed frozen layer bundle")
    if type(bundle["background"]) is not dict or "path" not in bundle["background"]:
        raise ValueError("Malformed frozen background record")
    assets = ordinary_path(bundle["background"]["path"]).parent
    if assets.name != "assets":
        raise ValueError("Frozen layer assets require their owned assets directory")
    masks = {}
    for key in ("pure_sky", "transient_support"):
        item = bundle[key]
        if type(item) is not dict or set(item) != {"mask_id", "asset"}:
            raise ValueError("Malformed frozen layer support")
        identity = item["mask_id"]
        if type(identity) is not str or type(item["asset"]) is not dict:
            raise ValueError("Malformed frozen layer support identity")
        if identity in masks and masks[identity] != item["asset"]:
            raise ValueError("Conflicting frozen mask identities")
        masks[identity] = item["asset"]
    from .finishing_session import _asset_id
    _asset_id(bundle["id"])
    context = {"directory": str(assets.parent), "source": source, "masks": masks,
               "references": [], "layer_bundles": {bundle["id"]: bundle}}
    return verify_bundle(context, bundle["id"], bundle["bundle_sha256"])
