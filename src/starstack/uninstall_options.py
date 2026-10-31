"""Optional, conservative uninstall cleanup; usable from a stdlib-only bootstrap.

The caller owns the installation lifecycle exclusion for the entire execution.
This module never imports product/runtime libraries, executes image code, follows
links, deletes a cache root, or recursively clears a registry namespace.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from pathlib import Path, PurePosixPath
import base64
import hashlib
import json
import os
import re
import stat
import uuid

CLEANUP_MARKER = ".aester-cache-cleaned.json"
PLAN_VERSION = "aester-uninstall-cache-1"
SETTINGS_ROOT = r"Software\StarStack\StarStack"
SETTINGS_KEYS = (
    "appearance/theme", "appearance/language", "output_directory", "siril_path", "stack_engine_path", "star_model_path",
    "stack_backend_mode", "dng_converter_path",
    "model/channel_order", "model/value_range", "model/layout", "model/stitch",
    "storage/cache_root", "experience/welcome_seen", "mcp/input_root",
    # 2026-09-27: the GIMP choice for natural-language retouching and the 「拓星」 viewing conditions.
    "retouch/gimp_path", "appearance/surround", "appearance/night",
    # 2026-10-09: the JPEG quality slider.
    "output/jpeg_quality",
)
_HASH = re.compile(r"[a-f0-9]{64}\Z")
# Same fixed allowlist as storage.EXTRA_IMAGES (kept local: stdlib-only bootstrap).
_EXTRA_IMAGES = {"starless_srgb16.tif": "starless", "stars_add_srgb16.tif": "stars_add",
                 "stars_subtract_srgb16.tif": "stars_subtract", "trails_layer_rgba16.tif": "trails_layer"}


def _path(value, *, exists=True):
    if not isinstance(value, (str, os.PathLike)) or not str(value):
        raise ValueError("A nonempty absolute path is required")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError("An absolute path is required")
    path = Path(os.path.abspath(path))
    for part in (path, *path.parents):
        try:
            info = part.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("Linked or reparse cleanup path is unsupported: " + str(part))
    if exists and not path.exists():
        raise ValueError("Cleanup path is missing: " + str(path))
    return path


def _relative(value):
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ValueError("Invalid owned relative path")
    rel = PurePosixPath(value)
    if rel.is_absolute() or any(p in {".", "..", ""} for p in value.split("/")):
        raise ValueError("Invalid owned relative path")
    return rel


def _json(path):
    path = _path(path)
    if not path.is_file() or path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("Invalid cleanup metadata file")
    data = json.loads(path.read_bytes())
    if not isinstance(data, dict):
        raise ValueError("Cleanup metadata must be an object")
    return data


def _digest(path):
    with _path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _file(path, *, relative_to=None):
    path = _path(path)
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError("Cleanup requires an unlinked ordinary file")
    return {"file": path.relative_to(relative_to).as_posix() if relative_to else str(path),
            "bytes": info.st_size, "sha256": _digest(path)}


def _entry(entry):
    if (not isinstance(entry, dict) or set(entry) != {"file", "bytes", "sha256"}
            or not isinstance(entry["file"], str) or type(entry["bytes"]) is not int
            or entry["bytes"] < 0 or not isinstance(entry["sha256"], str)
            or not _HASH.fullmatch(entry["sha256"])):
        raise ValueError("Invalid file identity")
    return entry


def _verify(entry, *, root=None):
    _entry(entry)
    path = root / str(_relative(entry["file"])) if root else _path(entry["file"])
    if _file(path, relative_to=root) != entry:
        raise ValueError("File changed since cleanup was planned: " + str(path))
    return path


@contextmanager
def _idle_lock(path, *, create=False):
    """Hold an existing worker lock without creating new unowned cache files."""
    path = _path(path, exists=False)
    if not path.exists() and not create:
        yield
        return
    with path.open("a+b" if create else "r+b") as stream:
        if create and path.stat().st_size == 0:
            stream.write(b"0")
            stream.flush()
        try:
            if os.name == "nt":
                import msvcrt
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise ValueError("A processing job is active") from error
        try:
            yield
        finally:
            if os.name == "nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def _record(directory):
    record = _json(directory / "workflow.json")
    request = record.get("request")
    if (record.get("version") != "workflow-1" or record.get("workflow_dir") != str(directory)
            or record.get("state") != "succeeded" or not isinstance(request, dict)
            or not request.get("delivery_root")):
        raise ValueError("Only successful separately delivered workflows can be cleaned")
    signature = hashlib.sha256(json.dumps(request, sort_keys=True, ensure_ascii=False,
                                          allow_nan=False).encode()).hexdigest()
    if record.get("signature") != signature:
        raise ValueError("Workflow request identity changed")
    if (directory / CLEANUP_MARKER).exists():
        raise ValueError("This workflow cache was already partially or fully cleaned")
    inputs = request.get("input_files")
    if not isinstance(inputs, list) or not inputs or any(not isinstance(p, str) for p in inputs):
        raise ValueError("Invalid original file list")
    for value in inputs:
        original = _path(value, exists=False)
        if original == directory or original.is_relative_to(directory):
            raise ValueError("An original is inside the cache")
    stack = record.get("stack_job_dir")
    if stack:
        stack = _path(stack)
        if not stack.is_relative_to(directory / "stack"):
            raise ValueError("Stack job escaped its workflow")
        child = _json(stack / "job.json")
        if child.get("state") != "succeeded":
            raise ValueError("Stack job is not successful")
    return record


def _deliveries(record, directory):
    """Verify frozen records for all current, base and reviewed publications."""
    paths = {record.get("delivery_dir"), record.get("base_delivery_dir", record.get("delivery_dir"))}
    histories = record.get("review_deliveries", [])
    if not isinstance(histories, list) or any(not isinstance(i, dict) for i in histories):
        raise ValueError("Invalid review publication list")
    paths.update(item.get("delivery_dir") for item in histories)
    verified = []
    for value in sorted(paths, key=str):
        delivery = _path(value)
        if delivery == directory or delivery.is_relative_to(directory) or directory.is_relative_to(delivery):
            raise ValueError("A publication is still embedded in processing cache")
        receipt_path = delivery / "delivery.json"
        image_only = record["request"].get("delivery_format") == "aester-delivery-2"
        if image_only:
            # New image-only publications keep audit receipts in the work area.
            receipts = record.get("publication_receipts", {})
            item = receipts.get(str(delivery)) if isinstance(receipts, dict) else None
            if not isinstance(item, dict):
                raise ValueError("A publication has no verified delivery receipt")
            receipt_path = _verify({"file": item.get("path"), "sha256": item.get("sha256"), "bytes": item.get("bytes")})
            if receipt_path.parent != directory / "publication-receipts":
                raise ValueError("Image-only receipt must belong to its workflow cache")
        receipt = _json(receipt_path)
        if (receipt.get("version") not in {"aester-delivery-1", "aester-delivery-2"}
                or receipt.get("workflow_signature") != record["signature"]
                or receipt.get("workflow_dir") != str(directory)):
            raise ValueError("Publication belongs to another workflow")
        if image_only and (receipt.get("version") != "aester-delivery-2" or receipt.get("delivery_dir") != str(delivery)):
            raise ValueError("Image-only receipt does not bind this exact publication")
        if not image_only and receipt.get("version") != "aester-delivery-1":
            raise ValueError("Publication version differs from the frozen workflow")
        entries = receipt.get("files")
        if not isinstance(entries, list) or not entries:
            raise ValueError("Publication has no files")
        names = []
        for entry in entries:
            _entry(entry)
            name = str(_relative(entry["file"]))
            if "/" in name:
                raise ValueError("Publication files must be plain filenames")
            names.append(name)
            path = _verify(entry, root=delivery)
            verified.append(_file(path))
        if len(set(names)) != len(names):
            raise ValueError("Duplicate publication filenames")
        if image_only:
            master = receipt.get("master_file")
            extras = receipt.get("extras", [])
            if (not isinstance(extras, list)
                    or any(not isinstance(item, dict) or _EXTRA_IMAGES.get(item.get("file")) != item.get("role") for item in extras)):
                raise ValueError("Image-only publication declares an unknown extra image")
            declared = {item["file"] for item in extras}
            if (len(declared) != len(extras) or master not in {"master_scene_linear.tif", "master_input_linear.tif"}
                    or set(names) != {master, "final.jpg", "final_srgb16.tif"} | declared):
                raise ValueError("Image-only publication must contain the three required images")
        verified.append(_file(receipt_path))
    return sorted({item["file"]: item for item in verified}.values(), key=lambda item: item["file"])


def _get(data, *keys):
    for key in keys:
        if not isinstance(data, dict):
            raise ValueError("Invalid produced artifact record")
        data = data.get(key)
        if data is None:
            return None
    return data


def _raise_walk_error(error):
    raise error


def _generated_files(record, directory):
    """Closed whitelist of actual output fields, never arbitrary JSON paths."""
    checkpoints = record.get("checkpoints")
    if not isinstance(checkpoints, dict):
        raise ValueError("No output checkpoints")
    output = {}
    sources = {_path(p, exists=False) for p in record["request"]["input_files"]}
    child = _path(record["stack_job_dir"]) if record.get("stack_job_dir") else None
    fields = {
        "stack": (("colour", "master", "path"), ("colour", "photo", "tiff"),
                  ("colour", "photo", "jpeg"),
                  ("job", "diagnostics", "sky_validity", "compose_weight", "path"),
                  ("job", "diagnostics", "sky_validity", "compose_weight", "composite")),
        "masks": (("sky_mask",), ("protected_mask",)),
        "milkyway_region": (("path",),), "legacy_display": (("path",),),
    }
    for stage, locations in fields.items():
        checkpoint = checkpoints.get(stage)
        if checkpoint is None:
            continue
        if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("files"), list):
            raise ValueError("Invalid output checkpoint")
        data = checkpoint.get("data")
        digest = hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()
        if digest != checkpoint.get("data_sha256"):
            raise ValueError("Output checkpoint data changed")
        if stage == "stack" and _get(data, "job", "job_dir") != (str(child) if child else None):
            raise ValueError("Stack checkpoint belongs to another child job")
        bound = {}
        for entry in checkpoint["files"]:
            if not isinstance(entry, dict):
                raise ValueError("Invalid checkpoint identity")
            frozen = {"file": entry.get("path"), "bytes": entry.get("bytes"), "sha256": entry.get("sha256")}
            _entry(frozen)
            if frozen["file"] in bound:
                raise ValueError("Duplicate checkpoint identity")
            bound[frozen["file"]] = frozen
        for location in locations:
            value = _get(data, *location)
            if value is None:
                continue
            path = _path(value)
            if not path.is_relative_to(directory) or path in sources:
                raise ValueError("Produced artifact escaped workflow or names an original")
            if stage == "stack" and (child is None or not path.is_relative_to(child / "products")):
                raise ValueError("Stack output is outside its owned products directory")
            if path.suffix.lower() not in {".tif", ".tiff", ".fit", ".fits", ".npy", ".jpg", ".jpeg", ".png"}:
                raise ValueError("Only explicit image/array products can be cleaned")
            if str(path) not in bound:
                raise ValueError("Produced artifact has no frozen hash identity")
            _verify(bound[str(path)])
            output[str(path)] = _file(path, relative_to=directory)
    if not output:
        raise ValueError("No verifiable generated image cache is available")
    return sorted(output.values(), key=lambda e: e["file"])


def _tree_names(directory):
    """Inspect links without hashing unrelated files; RAW copies are never selected."""
    names = []
    for current, folders, files in os.walk(directory, followlinks=False, onerror=_raise_walk_error):
        for name in folders:
            _path(Path(current) / name)
        for name in files:
            path = _path(Path(current) / name)
            names.append(path.relative_to(directory).as_posix())
    return sorted(names)


def prepare_cache_cleanup(workflow_dir, *, cache_root, install_root=None):
    """Read-only plan for explicit hashed image products, preserving other files."""
    directory, root = _path(workflow_dir), _path(cache_root)
    if (directory == root or not directory.is_relative_to(root)
            or not re.fullmatch(r"workflow-[a-f0-9]+", directory.name)):
        raise ValueError("Only one identified workflow inside its cache root may be cleaned")
    record = _record(directory)
    with ExitStack() as locks:
        paths = [directory / name for name in ("worker.lock", "edit.lock", "metadata.lock")]
        if record.get("stack_job_dir"):
            paths += [Path(record["stack_job_dir"]) / name for name in ("worker.lock", "metadata.lock")]
        for path in paths:
            locks.enter_context(_idle_lock(path))
        record = _record(directory)
        delivery_root = _path(record["request"]["delivery_root"])
        if delivery_root == root or delivery_root.is_relative_to(root) or root.is_relative_to(delivery_root):
            raise ValueError("Publication root is not independent of the processing cache")
        preserved = _deliveries(record, directory)
        generated = _generated_files(record, directory)
        names = _tree_names(directory)
        removed = {item["file"] for item in generated}
        return {"version": PLAN_VERSION, "workflow_dir": str(directory), "cache_root": str(root),
                "files": generated, "preserved_files": preserved, "workflow_record": _file(directory / "workflow.json"),
                "workflow_signature": record["signature"], "retained_cache_files": sorted(set(names) - removed),
                "originals": record["request"]["input_files"], "bytes": sum(e["bytes"] for e in generated),
                "loses_workflow_resume_and_review": True}


def _write_marker(directory, result):
    target = _path(directory / CLEANUP_MARKER, exists=False)
    temporary = directory / (CLEANUP_MARKER + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(target)


def _lock_paths(directory, record):
    paths = [directory / name for name in ("worker.lock", "edit.lock", "metadata.lock")]
    if record.get("stack_job_dir"):
        paths += [Path(record["stack_job_dir"]) / name for name in ("worker.lock", "metadata.lock")]
    return paths


def execute_cache_cleanup(plan, *, exclusive=False):
    """Recheck everything under caller's lifecycle exclusion, then mark+delete.

    Unknown files, control JSON, receipts and workflow roots are always kept.
    Manual filesystem edits are not transactional; a late failure is reported
    and the marker prevents an incompletely cleaned job claiming resumability.
    """
    if exclusive is not True:
        raise ValueError("Installation lifecycle exclusion is required")
    if not isinstance(plan, dict) or plan.get("version") != PLAN_VERSION:
        raise ValueError("Invalid cache cleanup plan")
    current = prepare_cache_cleanup(plan.get("workflow_dir"), cache_root=plan.get("cache_root"))
    if current != plan:
        raise ValueError("Cache or preserved photographs changed; no deletion performed")
    directory = _path(plan["workflow_dir"])
    result = {"version": "aester-cache-cleaned-1", "workflow_dir": str(directory),
              "workflow_signature": plan["workflow_signature"], "state": "cleanup_started",
              "removed_files": [], "removed_bytes": 0, "errors": [],
              "loses_workflow_resume_and_review": True}
    with ExitStack() as locks:
        for path in _lock_paths(directory, _record(directory)):
            locks.enter_context(_idle_lock(path, create=True))
        _verify(plan["workflow_record"])
        _write_marker(directory, result)
        for entry in plan["files"]:
            try:
                path = _verify(entry, root=directory)
                path.unlink()
                result["removed_files"].append(entry["file"])
                result["removed_bytes"] += entry["bytes"]
            except (OSError, ValueError) as error:
                result["errors"].append(str(error))
                break
        result["state"] = "partial" if result["errors"] else "completed"
        _write_marker(directory, result)
    return result


def _registry():
    if os.name != "nt":
        raise ValueError("Native settings cleanup is supported only on Windows")
    import winreg
    return winreg


def _setting_location(key):
    if key not in SETTINGS_KEYS:
        raise ValueError("This setting is not owned by Aester")
    head, _, name = key.rpartition("/")
    return SETTINGS_ROOT + ("\\" + head.replace("/", "\\") if head else ""), name


def _setting_digest(value, kind):
    if isinstance(value, bytes):
        value = {"base64": base64.b64encode(value).decode("ascii")}
    return hashlib.sha256(json.dumps({"kind": kind, "value": value}, sort_keys=True).encode()).hexdigest()


def plan_settings_cleanup():
    """Snapshot only explicit known values; plan never contains private values."""
    reg, values = _registry(), []
    for key in SETTINGS_KEYS:
        subkey, name = _setting_location(key)
        try:
            with reg.OpenKey(reg.HKEY_CURRENT_USER, subkey, 0, reg.KEY_READ) as handle:
                value, kind = reg.QueryValueEx(handle, name)
        except FileNotFoundError:
            continue
        values.append({"key": key, "sha256": _setting_digest(value, kind)})
    return {"version": "aester-uninstall-settings-1", "root": SETTINGS_ROOT, "values": values}


def execute_settings_cleanup(plan, *, exclusive=False):
    if exclusive is not True:
        raise ValueError("Installation lifecycle exclusion is required")
    if (not isinstance(plan, dict) or set(plan) != {"version", "root", "values"}
            or plan.get("version") != "aester-uninstall-settings-1" or plan.get("root") != SETTINGS_ROOT
            or not isinstance(plan.get("values"), list)):
        raise ValueError("Invalid settings cleanup plan")
    keys = []
    for item in plan["values"]:
        if (not isinstance(item, dict) or set(item) != {"key", "sha256"}
                or item.get("key") not in SETTINGS_KEYS or not isinstance(item.get("sha256"), str)
                or not _HASH.fullmatch(item["sha256"])):
            raise ValueError("Invalid setting identity")
        keys.append(item["key"])
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate setting identity")
    reg = _registry()
    result = {"removed": [], "preserved_changed": [], "errors": []}
    for item in plan["values"]:
        key = item["key"]
        subkey, name = _setting_location(key)
        try:
            with reg.OpenKey(reg.HKEY_CURRENT_USER, subkey, 0, reg.KEY_READ | reg.KEY_SET_VALUE) as handle:
                value, kind = reg.QueryValueEx(handle, name)
                if _setting_digest(value, kind) != item["sha256"]:
                    result["preserved_changed"].append(key)
                    continue
                reg.DeleteValue(handle, name)
                result["removed"].append(key)
        except FileNotFoundError:
            continue
        except OSError as error:
            result["errors"].append({"key": key, "error": str(error)})
    return result


def _known_cache_roots():
    if os.name == "nt":
        default = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData/Local"))) / "Aester/Cache"
    else:
        default = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "Aester/Cache"
    roots = [default]
    if os.name == "nt":
        reg = _registry()
        subkey, name = _setting_location("storage/cache_root")
        try:
            with reg.OpenKey(reg.HKEY_CURRENT_USER, subkey, 0, reg.KEY_READ) as handle:
                value, _ = reg.QueryValueEx(handle, name)
            if isinstance(value, str) and value:
                roots.append(Path(value))
        except FileNotFoundError:
            pass
    return list(dict.fromkeys(roots))


def _size_without_links(directory):
    total = 0
    for current, dirs, names in os.walk(directory, followlinks=False, onerror=_raise_walk_error):
        for name in dirs:
            _path(Path(current) / name)
        for name in names:
            path = _path(Path(current) / name)
            if path.is_file():
                total += path.stat().st_size
    return total


def scan_options(install_root):
    """Inventory only known cache locations; no ownership is inferred by scan."""
    owner = _path(install_root)
    plans, preserved = [], []
    for candidate in _known_cache_roots():
        try:
            root = _path(candidate, exists=False)
            if not root.exists():
                continue
            if not root.is_dir() or root == Path(root.anchor):
                raise ValueError("Cache root is not a dedicated directory")
            jobs = sorted(root.glob("workflow-*"))
            if len(jobs) > 1000:
                raise ValueError("Too many cache jobs for bounded uninstall inspection")
            for directory in jobs:
                if not directory.is_dir():
                    continue
                try:
                    plans.append(prepare_cache_cleanup(directory, cache_root=root, install_root=owner))
                except (OSError, ValueError, TypeError, KeyError) as error:
                    try:
                        size = _size_without_links(_path(directory))
                    except (OSError, ValueError):
                        size = None
                    preserved.append({"path": str(directory), "reason": str(error), "bytes": size})
        except (OSError, ValueError) as error:
            preserved.append({"path": str(candidate), "reason": str(error), "bytes": None})
    try:
        settings = plan_settings_cleanup()
        settings_error = None
    except (OSError, ValueError) as error:
        settings, settings_error = None, str(error)
    size = sum(plan["bytes"] for plan in plans)
    message = (f"可清理已完成作业的大文件缓存：{len(plans)}个作业，共{size:,}字节。保留{len(preserved)}项无法安全确认的作业。"
               "仅删除生产记录明确归属、哈希吻合且独立作品校验通过的图像中间产物。"
               "未完成作业、控制记录、未知文件和无输出哈希的旧缓存保留，不保证工作目录清空。"
               "清理后这些作业不能恢复或复核亮迹；原片和独立作品保留。软件设置仅删除已知偏好项。")
    return {"version": "aester-uninstall-options-1", "install_root": str(owner), "cache_plans": plans,
            "preserved": preserved, "settings_plan": settings, "settings_error": settings_error,
            "summary_zh": message, "default_clear_safe_cache": False, "default_clear_settings": False}
